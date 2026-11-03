"""
Offline frame-set generator.

Loops over the configured elevation range, ray-marches ONE hit-map per
elevation (the expensive step), and saves each to disk as three .npy arrays
(kind/radius/angle) under config.HITMAP_DIR. main.py loads these at startup
so the live loop never has to ray-march.

Run this once, ahead of time, on the Pi (or on a faster machine and copy the
hitmaps/ directory over — the format is portable numpy).

Usage:
    python precompute.py
"""

import os
import time

import numpy as np

import config
from physics_sim import render_hitmap


def elevation_filename(elevation_deg: float) -> str:
    """Deterministic filename stem for a given elevation, e.g. 'elev_034.5'."""
    return f"elev_{elevation_deg:06.2f}"


def main():
    os.makedirs(config.HITMAP_DIR, exist_ok=True)

    elevations = np.arange(
        config.ELEVATION_MIN_DEG,
        config.ELEVATION_MAX_DEG + 1e-6,
        config.ELEVATION_STEP_DEG,
    )

    print(f"Precomputing {len(elevations)} hit-maps into '{config.HITMAP_DIR}/'")
    print(
        f"resolution={config.SIM_RESOLUTION} steps={config.SIM_STEPS} "
        f"step_size={config.SIM_STEP_SIZE}"
    )

    total_start = time.perf_counter()

    for i, elevation_deg in enumerate(elevations):
        start = time.perf_counter()

        hitmap = render_hitmap(float(elevation_deg))

        stem = elevation_filename(float(elevation_deg))
        np.save(os.path.join(config.HITMAP_DIR, f"{stem}_kind.npy"), hitmap.kind)
        np.save(os.path.join(config.HITMAP_DIR, f"{stem}_radius.npy"), hitmap.radius)
        np.save(os.path.join(config.HITMAP_DIR, f"{stem}_angle.npy"), hitmap.angle)

        elapsed = time.perf_counter() - start
        print(
            f"[{i + 1}/{len(elevations)}] elevation={elevation_deg:6.2f} deg "
            f"-> {stem} ({elapsed:.2f}s)"
        )

    total_elapsed = time.perf_counter() - total_start
    print(f"Done in {total_elapsed:.1f}s. Hit-maps written to '{config.HITMAP_DIR}/'.")


if __name__ == "__main__":
    main()
