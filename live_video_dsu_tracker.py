import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, replace
import cv2
import h5py
import hdf5plugin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Suppress Qt font lookup warnings
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

# =============================================================================
# CONFIGURATION CONTAINER
# =============================================================================
@dataclass
class PipelineConfig:
    # Event Camera Nominal Geometry
    width: int = 1280
    height: int = 720
    fx: float = 800.0
    fy: float = 800.0
    cx: float = 640.0
    cy: float = 360.0
    z_nominal: float = 1.5

    # Dataset Ingestion
    chunk_size: int = 20_000_000
    target_seconds: float = 4.0

    # Temporal Sliding Window (Accumulation)
    window_us: int = 33_333
    step_us: int = 33_333
    min_events_per_window: int = 800

    # Spatial Tiling & Micro-Segment Extraction
    tile_size: int = 80
    min_events_per_tile: int = 60
    linearity_thresh: float = 2.0
    min_proj_percentile: float = 10.0
    max_proj_percentile: float = 90.0
    min_segment_span: float = 0.03

    # Disjoint-Set Union (DSU) Merging
    cos_thresh: float = float(np.cos(np.radians(25.0)))
    max_perp_dist: float = 0.05
    max_gap_dist: float = 0.12

    # Clustering & Rendering Filter
    min_cluster_size: int = 1
    min_render_length_m: float = 0.0

    @property
    def target_duration_us(self) -> int:
        return int(self.target_seconds * 1e6)


# =============================================================================
# FAST DISJOINT-SET UNION (DSU)
# =============================================================================
class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
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


# =============================================================================
# MICRO-SEGMENT EXTRACTION & DSU ENGINE
# =============================================================================
def extract_dsu_lines(x_w: np.ndarray, y_w: np.ndarray, cfg: PipelineConfig):
    # Guard: Reject idle frames
    if len(x_w) < cfg.min_events_per_window:
        return []

    # Nominal projective back-projection
    X = (x_w - cfg.cx) / cfg.fx * cfg.z_nominal
    Y = (y_w - cfg.cy) / cfg.fy * cfg.z_nominal
    Z = np.full_like(X, cfg.z_nominal)
    pts_3d = np.stack([X, Y, Z], axis=1)

    raw_segs = []

    # Map pattern: Spatial tiling
    for tx in range(0, cfg.width, cfg.tile_size):
        for ty in range(0, cfg.height, cfg.tile_size):
            mask = (
                (x_w >= tx) & (x_w < tx + cfg.tile_size) &
                (y_w >= ty) & (y_w < ty + cfg.tile_size)
            )
            pts = pts_3d[mask]
            if len(pts) < cfg.min_events_per_tile:
                continue

            centroid = np.mean(pts, axis=0)
            cov = np.cov(pts - centroid, rowvar=False)
            evals, evecs = np.linalg.eigh(cov)

            d = evecs[:, -1] / np.linalg.norm(evecs[:, -1])
            projs = np.dot(pts - centroid, d)
            t_min, t_max = np.percentile(
                projs, [cfg.min_proj_percentile, cfg.max_proj_percentile]
            )
            linearity = evals[-1] / (evals[-2] + 1e-6)

            # Filtering checks: Linearity and physical segment span
            if linearity > cfg.linearity_thresh and (t_max - t_min) > cfg.min_segment_span:
                raw_segs.append({
                    "d": d,
                    "c": centroid,
                    "p_start": centroid + t_min * d,
                    "p_end": centroid + t_max * d,
                    "pts": pts
                })

    m = len(raw_segs)
    if m == 0:
        return []

    # Equivalence clustering via DSU
    dsu = DSU(m)
    for i in range(m):
        for j in range(i + 1, m):
            s_i, s_j = raw_segs[i], raw_segs[j]
            # 1. Orientation test
            if abs(np.dot(s_i["d"], s_j["d"])) < cfg.cos_thresh:
                continue
            # 2. Perpendicular distance test
            r_ij = s_j["c"] - s_i["c"]
            if np.linalg.norm(r_ij - np.dot(r_ij, s_i["d"]) * s_i["d"]) > cfg.max_perp_dist:
                continue
            # 3. Endpoint distance test
            ends_i = [s_i["p_start"], s_i["p_end"]]
            ends_j = [s_j["p_start"], s_j["p_end"]]
            if min(np.linalg.norm(a - b) for a in ends_i for b in ends_j) < cfg.max_gap_dist:
                dsu.union(i, j)

    clusters = {}
    for i in range(m):
        clusters.setdefault(dsu.find(i), []).append(i)

    merged = []
    for root, members in clusters.items():
        # Render filter: Minimum equivalence cluster size
        if len(members) < cfg.min_cluster_size:
            continue

        all_pts = np.vstack([raw_segs[k]["pts"] for k in members])
        centroid = np.mean(all_pts, axis=0)
        cov = np.cov(all_pts - centroid, rowvar=False)
        _, evecs = np.linalg.eigh(cov)
        d_m = evecs[:, -1] / np.linalg.norm(evecs[:, -1])
        projs = np.dot(all_pts - centroid, d_m)

        ps = centroid + np.min(projs) * d_m
        pe = centroid + np.max(projs) * d_m

        length_3d = float(np.linalg.norm(pe - ps))
        if length_3d < cfg.min_render_length_m:
            continue

        u1 = int(ps[0] * cfg.fx / ps[2] + cfg.cx)
        v1 = int(ps[1] * cfg.fy / ps[2] + cfg.cy)
        u2 = int(pe[0] * cfg.fx / pe[2] + cfg.cx)
        v2 = int(pe[1] * cfg.fy / pe[2] + cfg.cy)

        merged.append({
            "p3d": (ps, pe),
            "p2d": ((u1, v1), (u2, v2)),
            "size": len(members),
            "length": length_3d
        })
    return merged


# =============================================================================
# DATASET LOADER HELPER
# =============================================================================
def load_dataset(h5_path: Path, cfg: PipelineConfig):
    with h5py.File(h5_path, "r") as f:
        events_grp = f["events"] if "events" in f else f["events_left"]
        total_in_file = events_grp["t"].shape[0]
        chunk_sz = min(cfg.chunk_size, total_in_file)
        t0 = int(events_grp["t"][0])

        print(f"Loading up to {chunk_sz:,} events from {h5_path.name}...")
        t_chunk = events_grp["t"][:chunk_sz].astype(np.int64)
        valid_idx = np.searchsorted(t_chunk, t0 + cfg.target_duration_us)

        x = events_grp["x"][:valid_idx].astype(np.int64)
        y = events_grp["y"][:valid_idx].astype(np.int64)
        p = events_grp["p"][:valid_idx].astype(np.float32)
        t = t_chunk[:valid_idx]

    duration = (t[-1] - t[0]) / 1e6
    print(f"Loaded {len(x):,} events spanning {duration:.2f} seconds.")
    return x, y, p, t, duration


# =============================================================================
# SINGLE RUN EXECUTOR (LIVE OR HEADLESS BENCHMARK)
# =============================================================================
def run_pipeline(x, y, p, t, actual_duration, cfg: PipelineConfig, run_title="DSU Tracker", gui=True):
    current_t = t[0]
    t_end_val = t[-1]
    t_start_val = t[0]

    if gui:
        plt.ion()
        fig = plt.figure(figsize=(7, 6))
        ax_3d = fig.add_subplot(1, 1, 1, projection="3d")
        plt.show(block=False)

    frame_count = 0
    total_lines_detected = 0
    start_wall_time = time.time()

    while current_t + cfg.window_us <= t_end_val:
        idx_start = np.searchsorted(t, current_t)
        idx_end = np.searchsorted(t, current_t + cfg.window_us)

        x_w = x[idx_start:idx_end]
        y_w = y[idx_start:idx_end]
        p_w = p[idx_start:idx_end]

        lines = extract_dsu_lines(x_w, y_w, cfg)
        total_lines_detected += len(lines)

        if gui:
            lin_idx = y_w * cfg.width + x_w
            voxel_1d = np.bincount(lin_idx, weights=p_w, minlength=cfg.height * cfg.width)
            voxel_2d = voxel_1d.reshape((cfg.height, cfg.width))

            img_disp = np.clip(128 + voxel_2d * 40, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)

            for l in lines:
                (u1, v1), (u2, v2) = l["p2d"]
                color = (0, 255, 0) if l["size"] > 1 else (255, 180, 0)
                thickness = 2 if l["size"] > 1 else 1
                cv2.line(img_bgr, (u1, v1), (u2, v2), color, thickness)

            rel_sec = (current_t - t_start_val) / 1e6
            cv2.putText(
                img_bgr,
                f"[{run_title}] Time: {rel_sec:.2f}s | Lines: {len(lines)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            cv2.imshow("Neuromorphic Live Video (DSU Overlay)", img_bgr)

            # Refresh 3D Plot every 3 iterations
            if frame_count % 3 == 0:
                ax_3d.cla()
                for l in lines:
                    ps, pe = l["p3d"]
                    color = "forestgreen" if l["size"] > 1 else "dodgerblue"
                    lw = 2.0 if l["size"] > 1 else 1.0
                    ax_3d.plot([ps[0], pe[0]], [ps[1], pe[1]], [ps[2], pe[2]], color=color, lw=lw)

                ax_3d.set_xlim(-1.2, 1.2)
                ax_3d.set_ylim(-0.8, 0.8)
                ax_3d.set_zlim(1.0, 2.0)
                ax_3d.set_title(f"{run_title} | 3D Wireframe | t = {rel_sec:.2f}s")
                ax_3d.view_init(elev=-90, azim=-90)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Early termination requested by user.")
                break

        current_t += cfg.step_us
        frame_count += 1

    elapsed_wall = time.time() - start_wall_time
    avg_lines = total_lines_detected / max(1, frame_count)

    if gui:
        cv2.destroyAllWindows()
        plt.close(fig)

    print(f"Finished {run_title} | Frames: {frame_count} | Avg Lines/Frame: {avg_lines:.2f} | Execution: {elapsed_wall:.2f}s")
    return avg_lines, elapsed_wall


# =============================================================================
# MAIN EXPERIMENT HARNESS: SEQUENTIAL PARAMETER SWEEP
# =============================================================================
if __name__ == "__main__":
    h5_path = Path("/run/media/chase-darlington/Extreme SSD/mocap-6dof-events_left.h5").resolve()
    if not h5_path.exists():
        h5_path = Path("~/Downloads/mocap-6dof-events_left.h5").expanduser().resolve()

    base_cfg = PipelineConfig()
    x, y, p, t, actual_duration = load_dataset(h5_path, base_cfg)

    # Define Parameter Grid for Sequential Sensitivity Sweeps
    experiments = [
        # Baseline Run
        ("Baseline (Default Params)", replace(base_cfg)),

        # Tile Size Sensitivity (Spatial granularity & PCA scope)
        ("Tile Size = 80", replace(base_cfg, tile_size=80)),
        ("Tile Size = 160", replace(base_cfg, tile_size=160)),

        # Minimum Segment Span Sensitivity (Rejecting short stubby clusters)
        ("Min Span = 0.03m (3cm)", replace(base_cfg, min_segment_span=0.03)),
        ("Min Span = 0.10m (10cm)", replace(base_cfg, min_segment_span=0.10)),
        ("Min Span = 0.20m (20cm)", replace(base_cfg, min_segment_span=0.20)),
        ("Min Span = 0.30m (30cm)", replace(base_cfg, min_segment_span=0.30)),

        # Maximum DSU Gap Distance Sensitivity (Stitching range)
        ("Max Gap = 0.06m", replace(base_cfg, max_gap_dist=0.06)),
        ("Max Gap = 0.12m", replace(base_cfg, max_gap_dist=0.12)),
        ("Max Gap = 0.36m", replace(base_cfg, max_gap_dist=0.36)),
        ("Max Gap = 0.72m", replace(base_cfg, max_gap_dist=0.72)),

        # Additional Sweeps for TCSS 570 System Insight:
        # Cluster Size Filtering (Pruning unmerged noise)
        ("Cluster Size >= 3 (Long structural lines only)", replace(base_cfg, min_cluster_size=3)),
        # Linearity Strictness (Sharp vector lines vs arcs)
        ("High Linearity Threshold = 4.0", replace(base_cfg, linearity_thresh=4.0)),
    ]

    print("\n" + "=" * 80)
    print("STARTING SEQUENTIAL PARAMETER SWEEP")
    print("=" * 80)

    summary_table = []
    for idx, (title, exp_cfg) in enumerate(experiments, 1):
        print(f"\n[{idx}/{len(experiments)}] Running: {title}")
        print(f"Config -> tile_size={exp_cfg.tile_size}, min_span={exp_cfg.min_segment_span}, "
              f"max_gap={exp_cfg.max_gap_dist}, min_cluster={exp_cfg.min_cluster_size}, "
              f"linearity={exp_cfg.linearity_thresh}")

        # Set gui=True to inspect visual wireframe behavior, or gui=False for quick benchmark profiling
        avg_lines, elapsed = run_pipeline(x, y, p, t, actual_duration, exp_cfg, run_title=title, gui=True)
        summary_table.append((title, exp_cfg.tile_size, exp_cfg.min_segment_span, exp_cfg.max_gap_dist, avg_lines, elapsed))

    # Print comparative matrix
    print("\n" + "=" * 85)
    print(f"{'Experiment Name':<42} | {'Tile':<4} | {'Span':<5} | {'Gap':<5} | {'Avg Lines':<10} | {'Time (s)':<8}")
    print("-" * 85)
    for row in summary_table:
        print(f"{row[0]:<42} | {row[1]:<4} | {row[2]:<5.2f} | {row[3]:<5.2f} | {row[4]:<10.2f} | {row[5]:<8.2f}")
    print("=" * 85)