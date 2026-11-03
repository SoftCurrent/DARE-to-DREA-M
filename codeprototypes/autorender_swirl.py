import numpy as np
import matplotlib.pyplot as plt
import os

# load texture once (H_t, W_t, 3or4)
texture = plt.imread("disk.jpeg")[:, :, :3]   # drop alpha if present
Ht, Wt = texture.shape[:2]

def colour_frame_textured(hit_r, hit_a, swirl_phase, disk_in=3.0, disk_out=10.0):
    img = np.zeros((*hit_r.shape, 3), np.float32)
    mask = hit_r > 0

    # map hit radius -> texture Y (0..1), hit angle+phase -> texture X (0..1)
    t_rad = np.clip((hit_r - disk_in) / (disk_out - disk_in), 0, 1)
    t_ang = ((hit_a + swirl_phase) % (2*np.pi)) / (2*np.pi)   # rotate by phase

    # to texture pixel indices
    tx = np.clip((t_ang * (Wt-1)).astype(int), 0, Wt-1)
    ty = np.clip((t_rad * (Ht-1)).astype(int), 0, Ht-1)

    sampled = texture[ty, tx]                # look up colour per pixel
    img[mask] = sampled[mask]
    return np.clip(img, 0, 1)

def render_hitmap(elevation_deg, W=200, H=200):
    """Run the geodesic march ONCE, return per-pixel disk hit (radius, angle)."""
    elevation = np.radians(elevation_deg)
    cam_dist, fov = 20.0, 1.2
    dt, num_steps = 0.05, 1000
    horizon_r, disk_in, disk_out = 1.0, 3.0, 10.0

    sx, sy = np.meshgrid(np.linspace(-1,1,W), np.linspace(-1,1,H))
    sx, sy = sx.ravel(), sy.ravel()
    cam = np.array([0.0, -cam_dist*np.cos(elevation), cam_dist*np.sin(elevation)])
    forward = -cam/np.linalg.norm(cam)
    right = np.cross(forward,[0,0,1]); right/=np.linalg.norm(right)
    up = np.cross(right,forward)
    dirs = forward[None,:]+fov*sx[:,None]*right[None,:]+fov*sy[:,None]*up[None,:]
    dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)

    pos = np.tile(cam,(W*H,1)).astype(np.float32)
    vel = dirs.astype(np.float32)
    h2 = np.sum(np.cross(pos,vel)**2,axis=1)

    hit_r = np.full(W*H, -1.0, np.float32)     # -1 = no disk hit
    hit_a = np.zeros(W*H, np.float32)
    alive = np.ones(W*H, bool)

    for _ in range(num_steps):
        pos_prev = pos.copy()
        r = np.linalg.norm(pos,axis=1)
        alive &= ~(r < horizon_r)
        r_safe = np.where(r<0.1,0.1,r)
        fmag = -1.5*h2/r_safe**5
        F = fmag[:,None]*pos; F[~alive]=0
        vel += F*dt; pos += vel*dt
        crossed = alive & (pos_prev[:,2]*pos[:,2] < 0)
        rho = np.sqrt(pos[:,0]**2+pos[:,1]**2)
        hit = crossed & (rho>disk_in) & (rho<disk_out)
        hit_r[hit] = rho[hit]
        hit_a[hit] = np.arctan2(pos[hit,1], pos[hit,0])   # angle on disk
        alive &= ~hit
        if not alive.any(): break

    return hit_r.reshape(H,W), hit_a.reshape(H,W)

def colour_frame(hit_r, hit_a, swirl_phase, disk_in=3.0, disk_out=10.0):
    """Cheap: colour the disk given a hit-map and a swirl rotation. No physics."""
    img = np.zeros((*hit_r.shape,3), np.float32)
    mask = hit_r > 0
    t = np.clip((hit_r-disk_in)/(disk_out-disk_in),0,1)
    # swirl: brightness varies with angle + rotating phase
    swirl = 0.6 + 0.4*np.sin(hit_a*3 + swirl_phase)
    img[...,0] = np.where(mask, 1.0*swirl, 0)
    img[...,1] = np.where(mask, (0.6*(1-t)+0.3)*swirl, 0)
    img[...,2] = np.where(mask, 0.2*(1-t)*swirl, 0)
    return np.clip(img,0,1)

# --- render one latitude, many swirl frames ---
os.makedirs("frames", exist_ok=True)
elevation_deg = 80
hit_r, hit_a = render_hitmap(elevation_deg)   # expensive: ONCE

n_frames = 60
for i in range(n_frames):
    phase = 2*np.pi*i/n_frames
    frame = colour_frame(hit_r, hit_a, phase)
    plt.imsave(f"frames/lat{elevation_deg}_f{i:03d}.png", frame, origin='lower')
    print(f"frame {i+1}/{n_frames}")
print("done")