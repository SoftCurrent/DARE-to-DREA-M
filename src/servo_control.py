"""
Feetech STS bus-servo driver for the 2 ST3215 servos (via a Waveshare bus
servo adapter presenting as a single serial port).

Implements just enough of the STS protocol to set a goal position with a
given speed/acceleration:

    [0xFF, 0xFF, ID, LEN, INSTR, PARAM..., CHECKSUM]

CHECKSUM = ~(ID + LEN + INSTR + sum(PARAM)) & 0xFF

We use the WRITE instruction (0x03) targeting register 0x29 (41, ACC),
writing 7 contiguous bytes covering ACC / Goal Position / Goal Time / Goal
Speed in one packet — this is the same register layout used by Feetech's
own SCServo SDK ("WritePosEx") and by the common hobbyist STS/SCS python
libraries, so it should match the Waveshare adapter's servos as-is. If your
specific firmware differs, this is the one place to adjust.

TODO(calibration): SERVO_POSITIONS_PER_DEGREE and SERVO_POSITION_CENTER in
config.py assume a linear 0-4095 -> 0-360 degree mapping centered on the
"facing forward" position. Verify this against the physical rig — some
horn/gearbox mounts will need an offset or a non-trivial azimuth->position
curve.
"""

import threading

import serial

import config

INSTR_WRITE = 0x03
ADDR_ACC = 0x29  # 41: acceleration register; position/time/speed follow it

_HEADER = bytes([0xFF, 0xFF])


class ServoBus:
    """Owns the serial connection to the Waveshare bus servo adapter and
    sends STS protocol packets to the two servos on it.
    """

    def __init__(self, port: str = None, baudrate: int = None):
        self._port_name = port or config.SERVO_PORT
        self._baudrate = baudrate or config.SERVO_BAUDRATE
        self._serial = serial.Serial(self._port_name, self._baudrate, timeout=0.05)
        self._lock = threading.Lock()

    def close(self):
        self._serial.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def set_position(self, servo_id: int, position: int, speed: int = None, accel: int = None):
        """Command one servo to a goal position (0-4095), with the given
        speed (0-4095) and acceleration (0-255).
        """
        speed = config.SERVO_DEFAULT_SPEED if speed is None else speed
        accel = config.SERVO_DEFAULT_ACCEL if accel is None else accel

        position = int(np_clip(position, config.SERVO_POSITION_MIN, config.SERVO_POSITION_MAX))
        speed = int(np_clip(speed, 0, 4095))
        accel = int(np_clip(accel, 0, 255))

        pos_l, pos_h = position & 0xFF, (position >> 8) & 0xFF
        speed_l, speed_h = speed & 0xFF, (speed >> 8) & 0xFF

        params = [ADDR_ACC, accel, pos_l, pos_h, 0x00, 0x00, speed_l, speed_h]
        packet = self._build_packet(servo_id, INSTR_WRITE, params)

        with self._lock:
            self._serial.write(packet)

    def center(self):
        """Send both servos to their configured centre position."""
        self.set_position(config.SERVO_ID_AZIMUTH, config.SERVO_POSITION_CENTER)
        self.set_position(config.SERVO_ID_ELEVATION, config.SERVO_POSITION_CENTER)

    def point_at(self, azimuth_deg: float, elevation_deg: float) -> "PointResult":
        """Map viewer azimuth/elevation angles to servo goal positions and
        drive both servos there in one call.

        The azimuth servo is treated as coarse: it's commanded to the
        nearest representable position, and the resulting rounding error
        (residual) is returned so the renderer can compensate for it
        optically (e.g. shifting the rendered azimuth-symmetric frame is a
        no-op here since the scene is azimuthally symmetric, but this hook
        exists for elevation-adjacent corrections or future asymmetric
        scenes).
        """
        azimuth_deg = clamp(azimuth_deg, config.SERVO_AZIMUTH_MIN_DEG, config.SERVO_AZIMUTH_MAX_DEG)
        elevation_deg = clamp(
            elevation_deg, config.SERVO_ELEVATION_MIN_DEG, config.SERVO_ELEVATION_MAX_DEG
        )

        az_position, az_residual_deg = self._angle_to_position(azimuth_deg)
        el_position, el_residual_deg = self._angle_to_position(elevation_deg)

        self.set_position(config.SERVO_ID_AZIMUTH, az_position)
        self.set_position(config.SERVO_ID_ELEVATION, el_position)

        return PointResult(
            azimuth_position=az_position,
            elevation_position=el_position,
            azimuth_residual_deg=az_residual_deg,
            elevation_residual_deg=el_residual_deg,
        )

    def _angle_to_position(self, angle_deg: float):
        """Convert a viewer-facing angle (degrees, 0 = centred) into a coarse
        servo position plus the fine leftover angle lost to rounding.
        """
        raw_position = config.SERVO_POSITION_CENTER + angle_deg * config.SERVO_POSITIONS_PER_DEGREE
        coarse_position = int(round(raw_position))
        coarse_position = int(
            np_clip(coarse_position, config.SERVO_POSITION_MIN, config.SERVO_POSITION_MAX)
        )

        achieved_angle_deg = (
            coarse_position - config.SERVO_POSITION_CENTER
        ) / config.SERVO_POSITIONS_PER_DEGREE
        residual_deg = angle_deg - achieved_angle_deg

        return coarse_position, residual_deg

    @staticmethod
    def _build_packet(servo_id: int, instruction: int, params: list) -> bytes:
        length = len(params) + 2
        body = [servo_id, length, instruction] + params
        checksum = (~(servo_id + length + instruction + sum(params))) & 0xFF
        return _HEADER + bytes(body) + bytes([checksum])


class PointResult:
    """Result of ServoBus.point_at(): the coarse positions actually
    commanded, plus the fine residual (in degrees) the render can use to
    compensate for servo rounding error.
    """

    def __init__(self, azimuth_position, elevation_position, azimuth_residual_deg, elevation_residual_deg):
        self.azimuth_position = azimuth_position
        self.elevation_position = elevation_position
        self.azimuth_residual_deg = azimuth_residual_deg
        self.elevation_residual_deg = elevation_residual_deg


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def np_clip(value, lo, hi):
    return max(lo, min(hi, value))
