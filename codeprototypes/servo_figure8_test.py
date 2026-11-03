#!/usr/bin/env python3
"""Small figure-8 to test servo clearance. Stays within ~20 degrees of centre."""

import glob
import math
import os
import time

import serial

PORT = os.environ.get("SERVO_PORT", "/dev/ttyACM0")
BAUD = 1_000_000
SERVO_IDS = [1, 2]  # servo 1 = one axis, servo 2 = other axis
CENTRE = 2548

# ST3215: 4096 steps = 360 deg, so 1 deg = 4096/360 = ~11.4 steps
DEG = 4096 / 360.0
AMPLITUDE_DEG = 10  # max swing from centre
AMP = AMPLITUDE_DEG * DEG

SPEED = 1500
ACCEL = 40

ser = None


def find_serial_port():
    candidates = [
        os.environ.get("SERVO_PORT"),
        "/dev/ttyACM0",
        "/dev/ttyUSB0",
        "/dev/ttyUSB1",
        "/dev/ttyAMA0",
    ]
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen and os.path.exists(candidate):
            seen.add(candidate)
            return candidate

    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "No serial device found. Set SERVO_PORT to the served port such as /dev/ttyACM0 or /dev/ttyUSB0."
    )


def checksum(p):
    return (~sum(p[2:])) & 0xFF


def open_serial():
    port = find_serial_port()
    print(f"Opening serial port {port}")
    return serial.Serial(port, BAUD, timeout=0.1)


def write_pos(sid, position, speed=SPEED, accel=ACCEL):
    if ser is None or not ser.is_open:
        raise RuntimeError("Serial port is not open.")

    position = max(0, min(4095, int(position)))
    pl, ph = position & 0xFF, (position >> 8) & 0xFF
    sl, sh = speed & 0xFF, (speed >> 8) & 0xFF
    body = [sid, 10, 0x03, 42, pl, ph, 0, 0, sl, sh, accel]
    packet = bytearray([0xFF, 0xFF] + body)
    packet.append(checksum(packet))
    ser.write(packet)


def centre_servos():
    for sid in SERVO_IDS:
        write_pos(sid, CENTRE)
    time.sleep(1.0)


def main():
    global ser

    try:
        ser = open_serial()
        centre_servos()

        print("figure-8 (Ctrl-C to stop)")
        t = 0.0
        while True:
            # figure-8: x = sin(t), y = sin(2t)
            x = math.sin(t)
            y = math.sin(2 * t)
            write_pos(SERVO_IDS[0], CENTRE + x * AMP)  # axis 1
            write_pos(SERVO_IDS[1], CENTRE + y * AMP)  # axis 2
            t += 0.08
            time.sleep(0.02)
    except KeyboardInterrupt:
        try:
            for sid in SERVO_IDS:
                write_pos(sid, CENTRE)
            time.sleep(1.0)
        except RuntimeError:
            pass
        print("centred, stopped.")
        return 0
    except (FileNotFoundError, serial.SerialException) as exc:
        print(f"Servo connection error: {exc}")
        return 1
    finally:
        if ser is not None and ser.is_open:
            ser.close()


if __name__ == "__main__":
    raise SystemExit(main())