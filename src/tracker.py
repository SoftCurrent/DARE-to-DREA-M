"""
Head tracking, on its own thread so it never blocks the display loop.

Uses picamera2 + OpenCV Haar-cascade face detection to find the viewer's
head each frame, then converts the pixel offset from image centre into
azimuth/elevation angles using the camera's field of view. Applies
exponential smoothing, a dead-zone, velocity-based prediction (to cancel
capture+detect+servo latency), and outlier rejection.

TODO(calibration): CAMERA_HFOV_DEG / CAMERA_VFOV_DEG in config.py must match
the real NoIR camera + lens combination, or the angle conversion will be
wrong even though detection itself works fine.
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import config


@dataclass
class HeadPose:
    """Smoothed, latency-compensated head pose, in degrees."""

    azimuth_deg: float
    elevation_deg: float
    face_detected: bool
    timestamp: float


class HeadTracker:
    """Runs face detection on a background thread and exposes the latest
    smoothed/predicted HeadPose via get_pose().

    Uses picamera2 when available (real Pi hardware); falls back to
    OpenCV's VideoCapture (e.g. a USB webcam, or for desktop development)
    if picamera2 can't be imported.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)

        # Raw (unsmoothed) last-seen angles, used for velocity estimation.
        self._raw_azimuth = 0.0
        self._raw_elevation = (
            config.ELEVATION_MIN_DEG + config.ELEVATION_MAX_DEG
        ) / 2.0
        self._raw_timestamp = time.monotonic()

        # Smoothed angles exposed to the caller.
        self._smoothed_azimuth = self._raw_azimuth
        self._smoothed_elevation = self._raw_elevation

        self._velocity_azimuth = 0.0
        self._velocity_elevation = 0.0

        self._last_detection_time = 0.0
        self._face_detected = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_pose(self) -> HeadPose:
        """Thread-safe read of the current predicted head pose."""
        with self._lock:
            now = time.monotonic()
            lost = (now - self._last_detection_time) > config.TRACKER_LOST_FACE_TIMEOUT_SEC

            if lost:
                # Hold last known position, no prediction (avoid drifting a
                # stale estimate further away from reality).
                return HeadPose(
                    azimuth_deg=self._smoothed_azimuth,
                    elevation_deg=self._smoothed_elevation,
                    face_detected=False,
                    timestamp=now,
                )

            lead = config.TRACKER_PREDICTION_LEAD_SEC
            predicted_azimuth = self._smoothed_azimuth + self._velocity_azimuth * lead
            predicted_elevation = self._smoothed_elevation + self._velocity_elevation * lead

            predicted_elevation = float(
                np.clip(
                    predicted_elevation,
                    config.ELEVATION_MIN_DEG,
                    config.ELEVATION_MAX_DEG,
                )
            )

            return HeadPose(
                azimuth_deg=predicted_azimuth,
                elevation_deg=predicted_elevation,
                face_detected=True,
                timestamp=now,
            )

    def _open_camera(self):
        """Try picamera2 first (real hardware), else fall back to a
        standard OpenCV VideoCapture for development off-Pi.
        """
        try:
            from picamera2 import Picamera2

            picam = Picamera2()
            cfg = picam.create_video_configuration(
                main={"size": config.CAMERA_RESOLUTION, "format": "RGB888"}
            )
            picam.configure(cfg)
            picam.start()

            def read():
                frame = picam.capture_array()
                return True, frame

            return read, picam.stop

        except ImportError:
            cap = cv2.VideoCapture(config.CAMERA_INDEX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])

            def read():
                return cap.read()

            return read, cap.release

    def _run(self):
        read_frame, close_camera = self._open_camera()
        try:
            while self._running:
                ok, frame = read_frame()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                self._process_frame(frame)
        finally:
            close_camera()

    def _process_frame(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=config.FACE_DETECTOR_SCALE_FACTOR,
            minNeighbors=config.FACE_DETECTOR_MIN_NEIGHBORS,
            minSize=config.FACE_DETECTOR_MIN_SIZE,
        )

        if len(faces) == 0:
            return

        # Largest detected face = assumed to be the viewer closest to camera.
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_cx = x + w / 2.0
        face_cy = y + h / 2.0

        frame_h, frame_w = gray.shape[:2]
        offset_x_norm = (face_cx - frame_w / 2.0) / (frame_w / 2.0)   # [-1, 1]
        offset_y_norm = (face_cy - frame_h / 2.0) / (frame_h / 2.0)   # [-1, 1]

        azimuth_deg = offset_x_norm * (config.CAMERA_HFOV_DEG / 2.0)
        # Positive offset_y (face below centre) should decrease elevation.
        elevation_deg = (
            config.ELEVATION_MIN_DEG + config.ELEVATION_MAX_DEG
        ) / 2.0 - offset_y_norm * (config.CAMERA_VFOV_DEG / 2.0)

        now = time.monotonic()

        with self._lock:
            dt = max(now - self._raw_timestamp, 1e-3)

            # Outlier rejection: reject implausible jumps in a single frame.
            jump_azimuth = abs(azimuth_deg - self._raw_azimuth)
            jump_elevation = abs(elevation_deg - self._raw_elevation)
            if (
                self._face_detected
                and (jump_azimuth > config.TRACKER_MAX_JUMP_DEG
                     or jump_elevation > config.TRACKER_MAX_JUMP_DEG)
            ):
                return

            # Dead-zone: ignore sub-threshold changes to kill idle jitter.
            if abs(azimuth_deg - self._smoothed_azimuth) < config.TRACKER_DEAD_ZONE_DEG:
                azimuth_deg = self._smoothed_azimuth
            if abs(elevation_deg - self._smoothed_elevation) < config.TRACKER_DEAD_ZONE_DEG:
                elevation_deg = self._smoothed_elevation

            new_velocity_azimuth = (azimuth_deg - self._raw_azimuth) / dt
            new_velocity_elevation = (elevation_deg - self._raw_elevation) / dt

            alpha = config.TRACKER_SMOOTHING_ALPHA
            self._smoothed_azimuth = (
                alpha * azimuth_deg + (1 - alpha) * self._smoothed_azimuth
            )
            self._smoothed_elevation = (
                alpha * elevation_deg + (1 - alpha) * self._smoothed_elevation
            )
            self._velocity_azimuth = (
                alpha * new_velocity_azimuth + (1 - alpha) * self._velocity_azimuth
            )
            self._velocity_elevation = (
                alpha * new_velocity_elevation + (1 - alpha) * self._velocity_elevation
            )

            self._raw_azimuth = azimuth_deg
            self._raw_elevation = elevation_deg
            self._raw_timestamp = now
            self._last_detection_time = now
            self._face_detected = True
