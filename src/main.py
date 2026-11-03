"""
D:REAM live runtime.

Loads the precomputed hit-maps, starts the head tracker on its own thread,
and runs the main display/servo loop:

    head pose -> nearest precomputed hit-map for that elevation
              -> colourise with current swirl phase
              -> show fullscreen
              -> (throttled) drive servos to face the viewer

No ray-marching happens in this loop — everything expensive was done ahead
of time by precompute.py. If a face isn't currently detected, the last known
head pose is held rather than snapping the display/servos back to centre.
"""

import glob
import os
import re
import time

import cv2
import numpy as np

import config
from disk_texture import colourise
from physics_sim import HitMap
from precompute import elevation_filename
from tracker import HeadTracker

try:
    from servo_control import ServoBus
except ImportError:
    ServoBus = None  # allows running the display loop without pyserial installed


def load_hitmaps() -> dict:
    """Load every precomputed hit-map from config.HITMAP_DIR into memory,
    keyed by elevation in degrees (float).
    """
    pattern = os.path.join(config.HITMAP_DIR, "elev_*_kind.npy")
    kind_files = sorted(glob.glob(pattern))
    if not kind_files:
        raise FileNotFoundError(
            f"No hit-maps found in '{config.HITMAP_DIR}/'. Run precompute.py first."
        )

    hitmaps = {}
    for kind_path in kind_files:
        match = re.search(r"elev_(-?\d+\.\d+)_kind\.npy", kind_path)
        if not match:
            continue
        elevation_deg = float(match.group(1))
        stem = elevation_filename(elevation_deg)

        kind = np.load(os.path.join(config.HITMAP_DIR, f"{stem}_kind.npy"))
        radius = np.load(os.path.join(config.HITMAP_DIR, f"{stem}_radius.npy"))
        angle = np.load(os.path.join(config.HITMAP_DIR, f"{stem}_angle.npy"))

        hitmaps[elevation_deg] = HitMap(kind=kind, radius=radius, angle=angle)

    print(f"Loaded {len(hitmaps)} hit-maps: "
          f"{min(hitmaps)}deg .. {max(hitmaps)}deg")
    return hitmaps


def nearest_elevation(hitmaps: dict, elevation_deg: float) -> float:
    """Pick the closest precomputed elevation key to the requested angle."""
    return min(hitmaps.keys(), key=lambda e: abs(e - elevation_deg))


def main():
    hitmaps = load_hitmaps()
    elevation_keys = np.array(sorted(hitmaps.keys()))

    tracker = HeadTracker()
    tracker.start()

    servo_bus = None
    if ServoBus is not None:
        try:
            servo_bus = ServoBus()
            servo_bus.center()
        except Exception as exc:  # serial port may not be present, e.g. dev machine
            print(f"Servo bus unavailable, running display-only: {exc}")
            servo_bus = None

    if config.FULLSCREEN:
        cv2.namedWindow(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

    frame_period = 1.0 / config.TARGET_FPS
    servo_period = 1.0 / config.SERVO_UPDATE_HZ

    phase_deg = 0.0
    last_frame_time = time.perf_counter()
    last_servo_time = 0.0

    try:
        while True:
            now = time.perf_counter()
            dt = now - last_frame_time
            last_frame_time = now

            pose = tracker.get_pose()

            nearest_elev = nearest_elevation(hitmaps, pose.elevation_deg)
            hitmap = hitmaps[nearest_elev]

            phase_deg = (phase_deg + config.SWIRL_PHASE_SPEED_DEG_PER_SEC * dt) % 360.0
            frame = colourise(hitmap, phase_deg)

            if frame.shape[0] != config.DISPLAY_HEIGHT or frame.shape[1] != config.DISPLAY_WIDTH:
                frame = cv2.resize(
                    frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT),
                    interpolation=cv2.INTER_LINEAR,
                )

            cv2.imshow(config.WINDOW_NAME, frame)

            if servo_bus is not None and (now - last_servo_time) >= servo_period:
                servo_bus.point_at(pose.azimuth_deg, pose.elevation_deg)
                last_servo_time = now

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):  # Esc or 'q' to quit
                break

            elapsed = time.perf_counter() - now
            remaining = frame_period - elapsed
            if remaining > 0:
                time.sleep(remaining)

    finally:
        tracker.stop()
        if servo_bus is not None:
            servo_bus.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
