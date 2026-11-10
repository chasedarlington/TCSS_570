import os
from pathlib import Path
import h5py
import hdf5plugin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import cKDTree

# -------------------------------------------------------------------------
# 1. Event Ingestion & Back-Projection
# -------------------------------------------------------------------------
h5_path = Path("~/Downloads/mocap-6dof-events_left.h5").expanduser().resolve()

with h5py.File(h5_path, "r") as f:
    events_grp = f["events"] if "events" in f else f["events_left"]
    N = 500_000
    x = events_grp["x"][:N].astype(np.int64)
    y = events_grp["y"][:N].astype(np.int64)
    t = events_grp["t"][:N].astype(np.int64)

WIDTH, HEIGHT = 1280, 720
DELTA_T_US = 50_000  # 50 ms
valid_mask = (t - t[0] < DELTA_T_US) & (x >= 0) & (x < WIDTH) & (y >= 0) & (y < HEIGHT)

x_v, y_v = x[valid_mask], y[valid_mask]

fx, fy, cx, cy = 800.0, 800.0, 640.0, 360.0
z_nom = 1.5
X = (x_v - cx) / fx * z_nom
Y = (y_v - cy) / fy * z_nom
Z = np.full_like(X, z_nom)
points_raw = np.stack([X, Y, Z], axis=1)

# Downsample for graph tracing
sample_step = max(1, len(points_raw) // 2500)
pts = points_raw[::sample_step]
num_pts = len(pts)

# -------------------------------------------------------------------------
# 2. k-NN Search & Path Extraction (Map Pattern)
# -------------------------------------------------------------------------
tree = cKDTree(pts)
distances, indices = tree.query(pts, k=5)
MAX_LINK_DIST = 0.08  # 8 cm max edge length

adj = {i: set() for i in range(num_pts)}
for i in range(num_pts):
    for neighbor_idx, dist in zip(indices[i][1:], distances[i][1:]):
        if dist < MAX_LINK_DIST:
            adj[i].add(neighbor_idx)
            adj[neighbor_idx].add(i)

visited = set()
raw_traces = []

for i in range(num_pts):
    if i in visited or len(adj[i]) == 0:
        continue
    
    chain = [i]
    visited.add(i)
    current = i
    
    while True:
        unvisited = [nbr for nbr in adj[current] if nbr not in visited]
        if not unvisited:
            break
        next_node = min(unvisited, key=lambda n: np.linalg.norm(pts[current] - pts[n]))
        visited.add(next_node)
        chain.append(next_node)
        current = next_node

    if len(chain) >= 6:
        raw_traces.append(pts[chain])

print(f"Traced {len(raw_traces)} raw k-NN paths.")

# -------------------------------------------------------------------------
# 3. 1D Stencil Laplacian Smoothing
# -------------------------------------------------------------------------
def laplacian_smooth_polyline(trace, iterations=10, alpha=0.5):
    if len(trace) <= 2:
        return trace
    smoothed = np.copy(trace)
    for _ in range(iterations):
        interior_avg = 0.5 * (smoothed[:-2] + smoothed[2:])
        smoothed[1:-1] = (1.0 - alpha) * smoothed[1:-1] + alpha * interior_avg
    return smoothed

relaxed_traces = [laplacian_smooth_polyline(tr) for tr in raw_traces]

# -------------------------------------------------------------------------
# 4. Comparative Visualizer
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 7))

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
for tr in raw_traces:
    ax1.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="crimson", lw=1.2, alpha=0.85)
ax1.scatter(pts[::4, 0], pts[::4, 1], pts[::4, 2], c="lightgray", s=1, alpha=0.2)
ax1.set_title(f"Step 1: Raw k-NN Traces ({len(raw_traces)} paths)")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_zlabel("Z (m)")
ax1.view_init(elev=-90, azim=-90)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
for tr in relaxed_traces:
    ax2.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="teal", lw=2.0, alpha=0.9)
ax2.scatter(pts[::4, 0], pts[::4, 1], pts[::4, 2], c="lightgray", s=1, alpha=0.2)
ax2.set_title("Step 2: Laplacian Relaxed Wireframes")
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.set_zlabel("Z (m)")
ax2.view_init(elev=-90, azim=-90)

plt.tight_layout()
plt.show()
