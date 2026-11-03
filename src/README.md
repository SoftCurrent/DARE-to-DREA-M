# D:REAM

Head-tracked "black hole on a screen" illusion. A Raspberry Pi 4B renders a
gravitationally-lensed black hole on an OLED monitor; a NoIR camera + IR
illuminator track the viewer's head in a dark room; two ST3215 bus servos
rotate the display to face the viewer.

## Why it's split this way

A full Schwarzschild ray-march is too slow to do every frame on a Pi 4B. But
the scene only actually needs to change with the viewer's **elevation**
angle — a black hole + disk is azimuthally symmetric, so horizontal head
movement never changes what the render looks like, only where the display
should physically point.

So the expensive ray-marching is done **once per elevation angle, offline**,
and the live loop just picks the nearest precomputed frame and repaints its
disk colours cheaply. Rotating the display to face the viewer is a separate,
much simpler problem, handled by the servos.

```
precompute.py  (offline, run once)
  physics_sim.py  --render_hitmap(elevation)-->  hitmaps/*.npy

main.py  (live, run every time)
  tracker.py     --head pose (az/el)-->  main loop
  hitmaps/*.npy  --nearest elevation-->  main loop
  disk_texture.py --colourise(hitmap, phase)-->  frame  --cv2.imshow-->  display
  servo_control.py --point_at(az, el)-->  servos
```

## Modules

- **`physics_sim.py`** — numpy-vectorised Schwarzschild ray-marcher using the
  "magic potential" force law. Produces a `HitMap` (per-pixel: sky / disk
  radius+angle / horizon), not a coloured image.
- **`precompute.py`** — loops over elevation angles, calls the ray-marcher
  once per angle, saves each hit-map to `hitmaps/`. Run this once, offline.
- **`disk_texture.py`** — colours a hit-map given a swirl phase, either by
  sampling a texture image or a procedural gradient. Cheap — this is what
  actually runs every displayed frame.
- **`tracker.py`** — head tracking on its own thread (picamera2 + OpenCV
  Haar cascade), with smoothing, a dead-zone, velocity-based latency
  prediction, and outlier rejection. Falls back to `cv2.VideoCapture` if
  `picamera2` isn't importable (e.g. developing off-Pi).
- **`servo_control.py`** — Feetech STS protocol driver for the two ST3215
  servos over the Waveshare bus adapter. `point_at(azimuth, elevation)`
  drives both servos and returns the residual rounding error for optional
  optical compensation.
- **`main.py`** — the live loop: tracker -> nearest hit-map -> colourise ->
  display -> (throttled) servo update.
- **`config.py`** — every tunable constant, in one place.

## Running it

```bash
pip install -r requirements.txt
python precompute.py   # once, offline — writes hitmaps/*.npy
python main.py          # live system
```

Press `q` or `Esc` in the display window to quit `main.py`.

## What still needs real hardware to calibrate

Grep for `TODO(calibration)` in `config.py`, `servo_control.py`, and
`tracker.py`. In particular:

1. **`CAMERA_HFOV_DEG` / `CAMERA_VFOV_DEG`** (config.py) — the NoIR camera's
   actual field of view for your lens. Wrong FOV means detected head
   position maps to the wrong angle even though detection itself works.
2. **`SERVO_POSITIONS_PER_DEGREE` / `SERVO_POSITION_CENTER`** (config.py) —
   assumes a linear 0–4095 -> 0–360° mapping with 2048 as dead-ahead. Verify
   against the physical rig; adjust if the horn mount isn't centred there.
3. **`SERVO_ELEVATION_MIN_DEG` / `_MAX_DEG`, `SERVO_AZIMUTH_MIN_DEG` /
   `_MAX_DEG`** (config.py) — mechanical safe-travel limits, to avoid
   commanding the servo into its own housing.
4. **Servo packet format** (`servo_control.py`) — implemented against the
   common Feetech STS register layout (ACC at 0x29, then Goal
   Position/Time/Speed as one 7-byte write). If your specific ST3215
   firmware disagrees, this is the one function (`ServoBus._build_packet` /
   `set_position`) to adjust.

## Known simplifications (favouring "runs reliably" over "clever")

- Elevation lookup is nearest-neighbour against the precomputed grid, not
  interpolated between adjacent hit-maps — coarser but has zero failure
  modes.
- Face detection uses a Haar cascade (fast, works under IR illumination
  reasonably well) rather than a heavier DNN detector, to keep the tracker
  thread cheap on Pi 4B compute.
- No auto-recovery if the servo serial port disappears mid-run beyond the
  startup try/except in `main.py` — if it raises later, that surfaces as a
  crash rather than silently limping on. Given the time budget, an explicit
  crash is easier to diagnose live than a silent degraded mode.
