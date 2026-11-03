"""
Numpy-vectorised Schwarzschild geodesic ray-marcher.

For a given camera elevation angle, marches one light ray per output pixel
through the "magic potential" force law from Riccardo Antonelli's Starless
(https://rantonels.github.io/starless/):

    F = -(3/2) * h^2 * r_hat / r^5

where h is each ray's conserved angular momentum per unit mass, computed once
from its initial position and velocity (h = |r x v|).

This module produces a HIT-MAP, not a colour image: for every pixel it
records either
  - a disk hit (radius, angle) if the ray crossed the equatorial plane
    within [DISK_INNER_RADIUS, DISK_OUTER_RADIUS], or
  - HORIZON if the ray fell inside the event horizon, or
  - SKY if the ray escaped without hitting anything.

Colourising the hit-map is disk_texture.py's job, done separately (and
cheaply, per-frame) so this expensive ray-march only has to run once per
elevation angle in precompute.py.
"""

from dataclasses import dataclass

import numpy as np

import config


# Hit-map pixel classification codes, stored in the 'kind' channel.
HIT_SKY = 0
HIT_DISK = 1
HIT_HORIZON = 2


@dataclass
class HitMap:
    """Per-pixel ray-march result for one camera elevation.

    kind:   (H, W) uint8, one of HIT_SKY / HIT_DISK / HIT_HORIZON
    radius: (H, W) float32, disk-plane hit radius (valid where kind == HIT_DISK)
    angle:  (H, W) float32, disk-plane hit angle in radians (valid where kind == HIT_DISK)
    """

    kind: np.ndarray
    radius: np.ndarray
    angle: np.ndarray


def _camera_basis(elevation_deg: float):
    """Build an orthonormal camera basis (forward, right, up) for a camera
    sitting at the given elevation angle above the equatorial (disk) plane,
    looking at the origin from SIM_CAMERA_DISTANCE away.

    Because a Schwarzschild black hole + thin disk is azimuthally symmetric,
    the camera's azimuth is arbitrary — we fix it at 0 (i.e. camera sits in
    the x-z plane) without loss of generality.
    """
    elevation_rad = np.radians(elevation_deg)
    dist = config.SIM_CAMERA_DISTANCE

    cam_pos = np.array(
        [
            dist * np.cos(elevation_rad),
            0.0,
            dist * np.sin(elevation_rad),
        ],
        dtype=np.float32,
    )

    forward = -cam_pos / np.linalg.norm(cam_pos)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    return cam_pos, forward, right, up


def _ray_directions(forward, right, up, resolution: int, fov_deg: float):
    """Build one initial ray direction per pixel, as a (resolution, resolution, 3)
    float32 array, using a simple pinhole camera model.
    """
    fov_rad = np.radians(fov_deg)
    half_extent = np.tan(fov_rad / 2.0)

    # Pixel-space grid in [-half_extent, half_extent], row 0 = top of image.
    axis = np.linspace(-half_extent, half_extent, resolution, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis, -axis)  # grid_y flipped: image row 0 = top

    directions = (
        forward[None, None, :]
        + grid_x[..., None] * right[None, None, :]
        + grid_y[..., None] * up[None, None, :]
    )
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    directions = (directions / norms).astype(np.float32)
    return directions


def render_hitmap(
    elevation_deg: float,
    resolution: int = None,
    steps: int = None,
    step_size: float = None,
) -> HitMap:
    """Ray-march a full hit-map for a camera at the given elevation angle.

    This is the expensive call — intended to be invoked exactly once per
    elevation angle (see precompute.py), not per rendered frame.
    """
    resolution = resolution or config.SIM_RESOLUTION
    steps = steps or config.SIM_STEPS
    step_size = step_size or config.SIM_STEP_SIZE

    cam_pos, forward, right, up = _camera_basis(elevation_deg)
    directions = _ray_directions(forward, right, up, resolution, config.SIM_FOV_DEG)

    # State per ray: position (N,3) and velocity (N,3), N = resolution^2.
    n_rays = resolution * resolution
    pos = np.broadcast_to(cam_pos, (n_rays, 3)).astype(np.float32).copy()
    vel = directions.reshape(n_rays, 3).astype(np.float32).copy()

    # Angular momentum per ray is conserved along a geodesic in this force
    # law; compute it once from the initial conditions.
    h_vec = np.cross(pos, vel)
    h2 = np.sum(h_vec * h_vec, axis=-1).astype(np.float32)  # (N,)

    # Output buffers.
    kind = np.full(n_rays, HIT_SKY, dtype=np.uint8)
    hit_radius = np.zeros(n_rays, dtype=np.float32)
    hit_angle = np.zeros(n_rays, dtype=np.float32)
    active = np.ones(n_rays, dtype=bool)

    horizon_r = config.SIM_HORIZON_RADIUS
    disk_inner = config.SIM_DISK_INNER_RADIUS
    disk_outer = config.SIM_DISK_OUTER_RADIUS

    prev_z = pos[:, 2].copy()

    for _ in range(steps):
        if not np.any(active):
            break

        r = np.linalg.norm(pos, axis=-1)
        r_safe = np.maximum(r, 1e-6)

        # F = -(3/2) h^2 r_hat / r^5 = -(3/2) h^2 pos / r^6
        accel_scale = (-1.5 * h2 / (r_safe**6))[:, None]
        accel = accel_scale * pos

        # Only advance rays that are still active; frozen rays stay put so
        # their buffers don't get corrupted by further stepping.
        step_mask = active[:, None]
        vel += np.where(step_mask, accel * step_size, 0.0)
        new_pos = pos + np.where(step_mask, vel * step_size, 0.0)

        new_r = np.linalg.norm(new_pos, axis=-1)
        new_z = new_pos[:, 2]

        # Horizon capture.
        horizon_mask = active & (new_r < horizon_r)
        kind[horizon_mask] = HIT_HORIZON
        active &= ~horizon_mask

        # Disk-plane crossing: z sign change between consecutive steps.
        crossed = active & (np.sign(new_z) != np.sign(prev_z)) & (new_z != prev_z)
        if np.any(crossed):
            # Linearly interpolate the crossing point between prev and new.
            t = prev_z[crossed] / (prev_z[crossed] - new_z[crossed])
            cross_pos = pos[crossed] + t[:, None] * (new_pos[crossed] - pos[crossed])
            cross_r = np.linalg.norm(cross_pos, axis=-1)

            in_disk = (cross_r >= disk_inner) & (cross_r <= disk_outer)
            crossed_idx = np.nonzero(crossed)[0]
            disk_idx = crossed_idx[in_disk]

            kind[disk_idx] = HIT_DISK
            hit_radius[disk_idx] = cross_r[in_disk]
            hit_angle[disk_idx] = np.arctan2(
                cross_pos[in_disk, 1], cross_pos[in_disk, 0]
            )
            active[disk_idx] = False

        # Escaped to infinity: stop tracking rays that are far out and still
        # moving outward, to save compute on the remaining steps.
        escaped = active & (new_r > config.SIM_CAMERA_DISTANCE * 4.0)
        active &= ~escaped

        pos = new_pos
        prev_z = new_z

    kind_2d = kind.reshape(resolution, resolution)
    radius_2d = hit_radius.reshape(resolution, resolution)
    angle_2d = hit_angle.reshape(resolution, resolution)

    return HitMap(kind=kind_2d, radius=radius_2d, angle=angle_2d)
