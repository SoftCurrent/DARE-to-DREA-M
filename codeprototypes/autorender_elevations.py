import numpy as np
import matplotlib.pyplot as plt
import os
import time

OUT_DIR = "frames"
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 300, 300
fov = 1.2
cam_dist = 20.0
dt = 0.05
num_steps = 1000
horizon_r = 1.0
disk_in, disk_out = 3.0, 10.0

# ballpark range/step - 2 degrees, avoiding the exact poles
ELEV_MIN = 15
ELEV_MAX = 165
ELEV_STEP = 2

def colour_frame_gradient(hit_r, disk_in=3.0, disk_out=10.0):
    img = np.zeros((*hit_r.shape, 3), np.float32)
    mask = hit_r > 0
    t = np.clip((hit_r - disk_in) / (disk_out - disk_in), 0, 1)
    img[..., 0] = np.where(mask, 1.0, 0)
    img[..., 1] = np.where(mask, 0.6 * (1 - t) + 0.3, 0)
    img[..., 2] = np.where(mask, 0.2 * (1 - t), 0)
    return np.clip(img, 0, 1)

def render_for_elevation(elevation_deg):
    elevation = np.radians(elevation_deg)

    sx, sy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    sx, sy = sx.ravel(), sy.ravel()

    cam = np.array([0.0, -cam_dist * np.cos(elevation), cam_dist * np.sin(elevation)])
    forward = -cam / np.linalg.norm(cam)
    right = np.cross(forward, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    dirs = (forward[None, :] + fov * sx[:, None] * right[None, :] + fov * sy[:, None] * up[None, :])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    pos = np.tile(cam, (W * H, 1)).astype(float)
    vel = dirs.copy()
    h2 = np.sum(np.cross(pos, vel) ** 2, axis=1)

    hit_r = np.full(W * H, -1.0)
    alive = np.ones(W * H, dtype=bool)

    for _ in range(num_steps):
        pos_prev = pos.copy()
        r = np.linalg.norm(pos, axis=1)
        alive &= ~(r < horizon_r)

        r_safe = np.where(r < 0.1, 0.1, r)
        fmag = -1.5 * h2 / r_safe ** 5
        F = fmag[:, None] * pos
        F[~alive] = 0
        vel += F * dt
        pos += vel * dt

        crossed = alive & (pos_prev[:, 2] * pos[:, 2] < 0)
        rho = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        hit = crossed & (rho > disk_in) & (rho < disk_out)
        hit_r[hit] = rho[hit]
        alive &= ~hit

        if not alive.any():
            break

    hit_r = hit_r.reshape(H, W)
    return colour_frame_gradient(hit_r, disk_in, disk_out)

# --- run the auto-renderer over all elevations ---
elevations = list(range(ELEV_MIN, ELEV_MAX + 1, ELEV_STEP))
print(f"Rendering {len(elevations)} frames...")

t0 = time.time()
for i, elev in enumerate(elevations):
    img = render_for_elevation(elev)
    filename = os.path.join(OUT_DIR, f"elev_{elev:03d}.png")
    plt.imsave(filename, img, origin='lower')
    elapsed = time.time() - t0
    avg = elapsed / (i + 1)
    remaining = avg * (len(elevations) - i - 1)
    print(f"  [{i+1}/{len(elevations)}] {filename}  "
          f"({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")

print(f"Done. {len(elevations)} frames saved to '{OUT_DIR}/'")