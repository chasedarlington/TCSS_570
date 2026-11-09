import os
from pathlib import Path
import h5py
import hdf5plugin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# -------------------------------------------------------------------------
# 1. Ingestion: 500,000 Events
# -------------------------------------------------------------------------
h5_path = Path("~/Downloads/mocap-6dof-events_left.h5").expanduser().resolve()

with h5py.File(h5_path, "r") as f:
    events_grp = f["events"] if "events" in f else f["events_left"]
    N = 500_000
    x = events_grp["x"][:N].astype(np.int64)
    y = events_grp["y"][:N].astype(np.int64)
    p = events_grp["p"][:N].astype(np.float32)
    t = events_grp["t"][:N].astype(np.int64)

WIDTH, HEIGHT = 1280, 720
DELTA_T_US = 100_000
t_start = t[0]
tau = (t - t_start) // DELTA_T_US

valid_mask = (tau == 0) & (x >= 0) & (x < WIDTH) & (y >= 0) & (y < HEIGHT)
x_v, y_v, p_v = x[valid_mask], y[valid_mask], p[valid_mask]

# 2D voxel image
linear_indices = y_v * WIDTH + x_v
voxel_img = np.bincount(linear_indices, weights=p_v, minlength=HEIGHT * WIDTH).reshape((HEIGHT, WIDTH))

# -------------------------------------------------------------------------
# 2. Extract Candidate Edges (Pinhole Frame)
# -------------------------------------------------------------------------
fx, fy, cx, cy = 800.0, 800.0, 640.0, 360.0
z_nominal = 1.5

X_cam = (x_v - cx) / fx * z_nominal
Y_cam = (y_v - cy) / fy * z_nominal
Z_cam = np.full_like(X_cam, z_nominal)
points_3d = np.stack([X_cam, Y_cam, Z_cam], axis=1)

TILE_SIZE = 80
raw_segments = []

for tx in range(0, WIDTH, TILE_SIZE):
    for ty in range(0, HEIGHT, TILE_SIZE):
        mask = (x_v >= tx) & (x_v < tx + TILE_SIZE) & (y_v >= ty) & (y_v < ty + TILE_SIZE)
        pts = points_3d[mask]
        if len(pts) < 80:
            continue

        centroid = np.mean(pts, axis=0)
        cov = np.cov(pts - centroid, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        d = eigenvectors[:, -1] / np.linalg.norm(eigenvectors[:, -1])
        m = np.cross(centroid, d)

        projections = np.dot(pts - centroid, d)
        t_min, t_max = np.percentile(projections, [5, 95])
        linearity = eigenvalues[-1] / (eigenvalues[-2] + 1e-6)

        if linearity > 3.5 and (t_max - t_min) > 0.04:
            raw_segments.append({
                "d": d, "m": m, "centroid": centroid,
                "p_start": centroid + t_min * d,
                "p_end": centroid + t_max * d,
                "pts": pts
            })

num_segs = len(raw_segments)

# -------------------------------------------------------------------------
# 3. DSU Collinear Merging
# -------------------------------------------------------------------------
class DisjointSetUnion:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i, root_j = self.find(i), self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
        return False

dsu = DisjointSetUnion(num_segs)
COS_THETA_THRESH = np.cos(np.radians(12.0))
MAX_PERP_DIST = 0.04
MAX_GAP_DIST = 0.20

for i in range(num_segs):
    for j in range(i + 1, num_segs):
        seg_i, seg_j = raw_segments[i], raw_segments[j]
        if abs(np.dot(seg_i["d"], seg_j["d"])) < COS_THETA_THRESH:
            continue
        r_ij = seg_j["centroid"] - seg_i["centroid"]
        if np.linalg.norm(r_ij - np.dot(r_ij, seg_i["d"]) * seg_i["d"]) > MAX_PERP_DIST:
            continue
        ends_i = [seg_i["p_start"], seg_i["p_end"]]
        ends_j = [seg_j["p_start"], seg_j["p_end"]]
        if min(np.linalg.norm(a - b) for a in ends_i for b in ends_j) < MAX_GAP_DIST:
            dsu.union(i, j)

clusters = {}
for i in range(num_segs):
    clusters.setdefault(dsu.find(i), []).append(i)

merged_lines = []
for root, members in clusters.items():
    all_pts = np.vstack([raw_segments[m]["pts"] for m in members])
    new_c = np.mean(all_pts, axis=0)
    cov = np.cov(all_pts - new_c, rowvar=False)
    _, eigvecs = np.linalg.eigh(cov)
    new_d = eigvecs[:, -1] / np.linalg.norm(eigvecs[:, -1])
    
    projs = np.dot(all_pts - new_c, new_d)
    t_min, t_max = np.min(projs), np.max(projs)
    
    p_s, p_e = new_c + t_min * new_d, new_c + t_max * new_d
    u_s = p_s[0] * fx / p_s[2] + cx
    v_s = p_s[1] * fy / p_s[2] + cy
    u_e = p_e[0] * fx / p_e[2] + cx
    v_e = p_e[1] * fy / p_e[2] + cy

    merged_lines.append({
        "p_start_3d": p_s, "p_end_3d": p_e,
        "p_start_2d": (u_s, v_s), "p_end_2d": (u_e, v_e),
        "cluster_size": len(members), "d": new_d
    })

# -------------------------------------------------------------------------
# 4. Side-by-Side Verification Plot
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 7))

ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(voxel_img, cmap="gray", vmin=-2, vmax=2, alpha=0.55)
for line in merged_lines:
    (u1, v1), (u2, v2) = line["p_start_2d"], line["p_end_2d"]
    size = line["cluster_size"]
    is_vertical = abs(line["d"][1]) > abs(line["d"][0])
    color = "cyan" if is_vertical else "yellow"
    lw = 2.5 if size > 1 else 1.2
    ax1.plot([u1, u2], [v1, v2], color=color, lw=lw)

ax1.set_title("2D Voxel Image with DSU Overlay\n(Cyan: Vertical, Yellow: Horizontal)")
ax1.set_xlim(0, WIDTH)
ax1.set_ylim(HEIGHT, 0)
ax1.set_xlabel("u (Column)")
ax1.set_ylabel("v (Row)")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
for line in merged_lines:
    ps, pe = line["p_start_3d"], line["p_end_3d"]
    size = line["cluster_size"]
    is_vertical = abs(line["d"][1]) > abs(line["d"][0])
    color = "cyan" if is_vertical else "yellow"
    lw = 2.5 if size > 1 else 1.2
    ax2.plot([ps[0], pe[0]], [ps[1], pe[1]], [ps[2], pe[2]], color=color, lw=lw)

ax2.set_title(f"3D DSU Wireframe ({len(merged_lines)} Consolidated Lines)")
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.set_zlabel("Z (m)")
ax2.view_init(elev=-90, azim=-90)

plt.tight_layout()
plt.show()
