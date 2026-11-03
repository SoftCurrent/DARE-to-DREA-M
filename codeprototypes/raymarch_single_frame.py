import numpy as np
import matplotlib.pyplot as plt

# load texture once (H_t, W_t, 3or4)
texture = plt.imread("disk.jpeg")[:, :, :3]   # drop alpha if present
Ht, Wt = texture.shape[:2]

def colour_frame_gradient(hit_r, hit_a, disk_in=3.0, disk_out=10.0):
    img = np.zeros((*hit_r.shape, 3), np.float32)
    mask = hit_r > 0

    t = np.clip((hit_r - disk_in) / (disk_out - disk_in), 0, 1)  # 0=inner, 1=outer

    # hot white/yellow inside -> deep orange outside
    r_ch = np.where(mask, 1.0, 0)
    g_ch = np.where(mask, 0.6 * (1 - t) + 0.3, 0)
    b_ch = np.where(mask, 0.2 * (1 - t), 0)

    img[..., 0] = r_ch
    img[..., 1] = g_ch
    img[..., 2] = b_ch
    return np.clip(img, 0, 1)

# --- image / camera ---
W, H = 300, 300
fov = 1.2
cam_dist = 20.0          # camera distance from black hole
elevation = np.radians(175)   # viewing angle: 90=top-down, low=edge-on

dt = 0.05
num_steps = 1000
horizon_r = 1.0
disk_in, disk_out = 3.0, 10.0   # accretion disk inner/outer radius

# --- build one ray per pixel ---
# screen coords centered, -1..1
sx, sy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
sx, sy = sx.ravel(), sy.ravel()

# camera position (elevated, looking at origin)
cam = np.array([cam_dist*np.cos(elevation)*0 , cam_dist*np.cos(elevation), cam_dist*np.sin(elevation)])
# simpler: place camera on +z tilted; we work in 3D now
cam = np.array([0.0, -cam_dist*np.cos(elevation), cam_dist*np.sin(elevation)])

# ray directions: point from camera toward a screen grid in front of it
forward = -cam / np.linalg.norm(cam)
right = np.cross(forward, [0,0,1]); right /= np.linalg.norm(right)
up = np.cross(right, forward)

dirs = (forward[None,:]
        + fov*sx[:,None]*right[None,:]
        + fov*sy[:,None]*up[None,:])
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

# state arrays: position and velocity for every pixel-ray
pos = np.tile(cam, (W*H, 1)).astype(float)
vel = dirs.copy()

# angular momentum magnitude per ray (h = |r x v|), constant along each ray
L = np.cross(pos, vel)
h2 = np.sum(L*L, axis=1)          # h squared per ray

# state arrays
pos = np.tile(cam, (W*H, 1)).astype(float)
vel = dirs.copy()
h2 = np.sum(np.cross(pos, vel)**2, axis=1)

hit_r = np.full(W*H, -1.0)     # record WHERE each ray hits the disk
hit_a = np.zeros(W*H)
alive = np.ones(W*H, dtype=bool)

for _ in range(num_steps):
    pos_prev = pos.copy()
    r = np.linalg.norm(pos, axis=1)
    alive &= ~(r < horizon_r)

    r_safe = np.where(r < 0.1, 0.1, r)
    fmag = -1.5 * h2 / r_safe**5
    F = fmag[:, None] * pos
    F[~alive] = 0
    vel += F*dt
    pos += vel*dt

    crossed = alive & (pos_prev[:, 2] * pos[:, 2] < 0)
    rho = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
    hit = crossed & (rho > disk_in) & (rho < disk_out)
    hit_r[hit] = rho[hit]                          # record radius
    hit_a[hit] = np.arctan2(pos[hit, 1], pos[hit, 0])  # record angle
    alive &= ~hit

    if not alive.any():
        break

hit_r = hit_r.reshape(H, W)
hit_a = hit_a.reshape(H, W)

img = colour_frame_gradient(hit_r, hit_a, disk_in=disk_in, disk_out=disk_out)

plt.figure(figsize=(8, 8))
plt.imshow(img, origin='lower')
plt.axis('off')
plt.show()

plt.figure(figsize=(8,8))
plt.imshow(np.clip(img,0,1), origin='lower')
plt.axis('off')
plt.show()
