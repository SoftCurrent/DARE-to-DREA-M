"""
All tunable constants for D:REAM live here. Nothing outside this file should
hardcode a magic number for hardware, physics, or tracking behaviour.

Deadline-driven design note: values marked TODO cannot be correct without the
real hardware in front of you. Everything else is a reasonable starting point
that should run without crashing on a Pi 4B, even if not perfectly tuned.
"""

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
DISPLAY_WIDTH = 480
DISPLAY_HEIGHT = 480
FULLSCREEN = True
WINDOW_NAME = "DREAM"

# ---------------------------------------------------------------------------
# Physics / ray-marcher (physics_sim.py)
# ---------------------------------------------------------------------------
# Render resolution for the hit-map. This is independent of DISPLAY_* — the
# hit-map is upscaled/resized to the display at colourise time if they differ.
SIM_RESOLUTION = 480          # pixels, square hit-map
SIM_STEPS = 400                # ray-march steps per ray
SIM_STEP_SIZE = 0.05            # affine-parameter step size
SIM_CAMERA_DISTANCE = 15.0      # camera radial distance from black hole
SIM_FOV_DEG = 45.0              # camera field of view

SIM_HORIZON_RADIUS = 1.0        # r < this -> horizon hit
SIM_DISK_INNER_RADIUS = 3.0
SIM_DISK_OUTER_RADIUS = 10.0

# ---------------------------------------------------------------------------
# Precompute (precompute.py)
# ---------------------------------------------------------------------------
ELEVATION_MIN_DEG = 12.0
ELEVATION_MAX_DEG = 90.0
ELEVATION_STEP_DEG = 2.0        # resolution of the precomputed elevation grid
HITMAP_DIR = "hitmaps"          # relative output dir for .npy hit-maps

# ---------------------------------------------------------------------------
# Disk texture / swirl (disk_texture.py)
# ---------------------------------------------------------------------------
TEXTURE_PATH = "assets/disk_texture.png"   # optional; None/missing -> procedural
SWIRL_PHASE_SPEED_DEG_PER_SEC = 30.0        # animation speed of the swirl
PROCEDURAL_INNER_COLOR = (200, 240, 255)    # BGR, hot inner edge (pale yellow-white)
PROCEDURAL_OUTER_COLOR = (10, 40, 120)      # BGR, cooler outer edge (deep orange-red)

# ---------------------------------------------------------------------------
# Head tracking (tracker.py)
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_RESOLUTION = (640, 480)
CAMERA_FRAMERATE = 30

# TODO(calibration): measure the NoIR camera module's actual horizontal and
# vertical field of view (depends on lens variant). This directly scales the
# pixel-offset -> angle conversion in tracker.py.
CAMERA_HFOV_DEG = 62.2
CAMERA_VFOV_DEG = 48.8

FACE_DETECTOR_SCALE_FACTOR = 1.2
FACE_DETECTOR_MIN_NEIGHBORS = 5
FACE_DETECTOR_MIN_SIZE = (60, 60)

# Exponential smoothing factor for head position, in [0, 1]. Higher = less
# smoothing (more responsive, more jittery). Lower = smoother but laggier.
TRACKER_SMOOTHING_ALPHA = 0.35

# Dead-zone: ignore position changes smaller than this many degrees, to kill
# idle jitter when the viewer is holding still.
TRACKER_DEAD_ZONE_DEG = 1.0

# Velocity-based prediction: how far ahead (seconds) to extrapolate head
# position to cancel out pipeline latency (capture + detect + servo move).
TRACKER_PREDICTION_LEAD_SEC = 0.08

# Outlier rejection: reject a new detection if it implies an angular jump
# larger than this many degrees within one frame (likely a false detection).
TRACKER_MAX_JUMP_DEG = 25.0

# If no face is detected for longer than this, stop predicting motion and
# just hold the last known position.
TRACKER_LOST_FACE_TIMEOUT_SEC = 1.0

# ---------------------------------------------------------------------------
# Servo control (servo_control.py)
# ---------------------------------------------------------------------------
SERVO_PORT = "/dev/ttyACM0"
SERVO_BAUDRATE = 1_000_000
SERVO_ID_AZIMUTH = 1
SERVO_ID_ELEVATION = 2

SERVO_POSITION_MIN = 0
SERVO_POSITION_MAX = 4095
SERVO_POSITION_CENTER = 2048

SERVO_DEFAULT_SPEED = 800        # 0-4095, TODO(calibration): tune for smoothness
SERVO_DEFAULT_ACCEL = 50         # 0-255, TODO(calibration): tune for smoothness

# TODO(calibration): confirm these against the physical rig. This assumes the
# servo's full 0-4095 range maps linearly to 360 degrees of rotation, and that
# position SERVO_POSITION_CENTER points the display straight at azimuth 0.
SERVO_POSITIONS_PER_DEGREE = (SERVO_POSITION_MAX - SERVO_POSITION_MIN + 1) / 360.0

# TODO(calibration): mechanical limits of the elevation servo's safe travel.
SERVO_ELEVATION_MIN_DEG = 12.0
SERVO_ELEVATION_MAX_DEG = 90.0
SERVO_AZIMUTH_MIN_DEG = -90.0
SERVO_AZIMUTH_MAX_DEG = 90.0

# ---------------------------------------------------------------------------
# Main loop (main.py)
# ---------------------------------------------------------------------------
TARGET_FPS = 30
SERVO_UPDATE_HZ = 20.0           # servos updated less often than display draws
