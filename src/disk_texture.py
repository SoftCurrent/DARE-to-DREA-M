"""
Cheap per-frame colouriser for precomputed hit-maps.

Takes a HitMap (kind/radius/angle arrays) plus a rotation phase and produces
an RGB image, by sampling a texture image (hit-radius -> texture Y,
hit-angle + phase -> texture X) or falling back to a procedural radial
gradient if no texture is available. Rotating `phase` over time animates the
disk swirl with zero re-rendering cost — this is what main.py calls every
frame.
"""

import os

import cv2
import numpy as np

import config
from physics_sim import HIT_DISK, HIT_HORIZON, HitMap

_texture_cache = None
_texture_load_attempted = False


def _load_texture():
    """Lazily load and cache the disk texture image, if configured."""
    global _texture_cache, _texture_load_attempted
    if _texture_load_attempted:
        return _texture_cache

    _texture_load_attempted = True
    path = config.TEXTURE_PATH
    if path and os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            _texture_cache = img
    return _texture_cache


def colourise(hitmap: HitMap, phase_deg: float) -> np.ndarray:
    """Colour a hit-map into a BGR uint8 image (OpenCV convention).

    phase_deg rotates the disk swirl; advance it over time in the caller to
    animate the disk with no extra ray-marching.
    """
    h, w = hitmap.kind.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)

    disk_mask = hitmap.kind == HIT_DISK
    horizon_mask = hitmap.kind == HIT_HORIZON
    # Everything else stays black (sky).
    output[horizon_mask] = (0, 0, 0)

    if not np.any(disk_mask):
        return output

    texture = _load_texture()
    phase_rad = np.radians(phase_deg)

    inner = config.SIM_DISK_INNER_RADIUS
    outer = config.SIM_DISK_OUTER_RADIUS
    radius_norm = np.clip(
        (hitmap.radius[disk_mask] - inner) / max(outer - inner, 1e-6), 0.0, 1.0
    )
    angle = hitmap.angle[disk_mask] + phase_rad
    # Wrap angle to [0, 2*pi) then normalise to [0, 1) for texture sampling.
    angle_norm = (angle % (2.0 * np.pi)) / (2.0 * np.pi)

    if texture is not None:
        tex_h, tex_w = texture.shape[:2]
        tex_y = np.clip((radius_norm * (tex_h - 1)).astype(np.int32), 0, tex_h - 1)
        tex_x = np.clip((angle_norm * (tex_w - 1)).astype(np.int32), 0, tex_w - 1)
        sampled = texture[tex_y, tex_x]
        output[disk_mask] = sampled
    else:
        output[disk_mask] = _procedural_colour(radius_norm, angle_norm)

    return output


def _procedural_colour(radius_norm: np.ndarray, angle_norm: np.ndarray) -> np.ndarray:
    """Fallback disk colour: radial gradient (hot inner -> cool outer)
    modulated by a swirl brightness stripe so the animation is visible even
    without a texture image.
    """
    inner = np.array(config.PROCEDURAL_INNER_COLOR, dtype=np.float32)
    outer = np.array(config.PROCEDURAL_OUTER_COLOR, dtype=np.float32)

    # Linear blend inner->outer by radius.
    base = inner[None, :] * (1.0 - radius_norm[:, None]) + outer[None, :] * radius_norm[:, None]

    # A few swirl stripes so phase rotation is visually obvious.
    n_stripes = 6
    stripe = 0.5 + 0.5 * np.cos(angle_norm * n_stripes * 2.0 * np.pi)
    brightness = 0.6 + 0.4 * stripe

    colour = np.clip(base * brightness[:, None], 0, 255).astype(np.uint8)
    return colour
