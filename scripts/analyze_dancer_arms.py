import numpy as np

d = np.load('data/salsa_dl_mocap_9000_frames.npz')
lms = d['landmarks_3d_meters']
N = len(lms)

# 11: left_shoulder, 12: right_shoulder
# 13: left_elbow, 14: right_elbow
# 15: left_wrist, 16: right_wrist
# 23: left_hip, 24: right_hip

lw = lms[:, 15]
rw = lms[:, 16]
le = lms[:, 13]
re = lms[:, 14]
ls = lms[:, 11]
rs = lms[:, 12]
lh = lms[:, 23]
rh = lms[:, 24]

# Torso vectors
v_spine = 0.5 * (ls + rs) - 0.5 * (lh + rh)
u_up = v_spine / (np.linalg.norm(v_spine, axis=-1, keepdims=True) + 1e-6)
v_sh = ls - rs
u_lat = v_sh / (np.linalg.norm(v_sh, axis=-1, keepdims=True) + 1e-6)
u_fwd = np.cross(u_lat, u_up)
u_fwd = u_fwd / (np.linalg.norm(u_fwd, axis=-1, keepdims=True) + 1e-6)

# Elbow flexion angle (angle between upper arm and forearm)
v_l_up = le - ls
v_l_fore = lw - le
cos_el_l = np.sum(v_l_up * v_l_fore, axis=-1) / (np.linalg.norm(v_l_up, axis=-1) * np.linalg.norm(v_l_fore, axis=-1) + 1e-6)
angle_el_l = np.arccos(np.clip(cos_el_l, -1.0, 1.0)) # 0 = straight arm, pi = completely folded

v_r_up = re - rs
v_r_fore = rw - re
cos_el_r = np.sum(v_r_up * v_r_fore, axis=-1) / (np.linalg.norm(v_r_up, axis=-1) * np.linalg.norm(v_r_fore, axis=-1) + 1e-6)
angle_el_r = np.arccos(np.clip(cos_el_r, -1.0, 1.0))

# Upper arm direction relative to torso
# Lateral spread: dot(v_up, u_lat)
# Vertical elevation: dot(v_up, u_up)
# Forward reach: dot(v_up, u_fwd)
lat_l = np.sum(v_l_up * u_lat, axis=-1) / (np.linalg.norm(v_l_up, axis=-1) + 1e-6)
vert_l = np.sum(v_l_up * u_up, axis=-1) / (np.linalg.norm(v_l_up, axis=-1) + 1e-6)
fwd_l = np.sum(v_l_up * u_fwd, axis=-1) / (np.linalg.norm(v_l_up, axis=-1) + 1e-6)

lat_r = -np.sum(v_r_up * u_lat, axis=-1) / (np.linalg.norm(v_r_up, axis=-1) + 1e-6)
vert_r = np.sum(v_r_up * u_up, axis=-1) / (np.linalg.norm(v_r_up, axis=-1) + 1e-6)
fwd_r = np.sum(v_r_up * u_fwd, axis=-1) / (np.linalg.norm(v_r_up, axis=-1) + 1e-6)

# Wrist position relative to shoulder in torso coordinates
w_rel_l = lw - ls
w_lat_l = np.sum(w_rel_l * u_lat, axis=-1)
w_fwd_l = np.sum(w_rel_l * u_fwd, axis=-1)
w_vert_l = np.sum(w_rel_l * u_up, axis=-1)

w_rel_r = rw - rs
w_lat_r = -np.sum(w_rel_r * u_lat, axis=-1)
w_fwd_r = np.sum(w_rel_r * u_fwd, axis=-1)
w_vert_r = np.sum(w_rel_r * u_up, axis=-1)

wrist_dist = np.linalg.norm(lw - rw, axis=-1)

print(f"Total frames: {N}")
print(f"Wrist spread: min={wrist_dist.min():.3f} m, max={wrist_dist.max():.3f} m, p5={np.percentile(wrist_dist, 5):.3f} m, p95={np.percentile(wrist_dist, 95):.3f} m")
print(f"Left elbow bend angle (rad): min={angle_el_l.min():.2f}, max={angle_el_l.max():.2f}, mean={angle_el_l.mean():.2f}")
print(f"Right elbow bend angle (rad): min={angle_el_r.min():.2f}, max={angle_el_r.max():.2f}, mean={angle_el_r.mean():.2f}")
print(f"Left wrist lateral extension (m): min={w_lat_l.min():.3f}, max={w_lat_l.max():.3f}")
print(f"Right wrist lateral extension (m): min={w_lat_r.min():.3f}, max={w_lat_r.max():.3f}")
print(f"Left wrist forward reach (m): min={w_fwd_l.min():.3f}, max={w_fwd_l.max():.3f}")
print(f"Right wrist forward reach (m): min={w_fwd_r.min():.3f}, max={w_fwd_r.max():.3f}")
print(f"Left wrist vertical height (m): min={w_vert_l.min():.3f}, max={w_vert_l.max():.3f}")
print(f"Right wrist vertical height (m): min={w_vert_r.min():.3f}, max={w_vert_r.max():.3f}")
