import time
import numpy as np
import torch
import mlx.core as mx

# -------------------------------------------------------------
# 1. Dataset Parameters (Matching TUM-VIE Prophesee Gen4)
# -------------------------------------------------------------
N_EVENTS = 5_000_000   # 5M events (~50-100ms burst during high motion)
WIDTH = 1280
HEIGHT = 720
TIME_BUCKET_US = 1000  # 1ms bins

WARMUP_RUNS = 5
BENCHMARK_RUNS = 20

# Generate synthetic raw event stream
np.random.seed(42)
host_x = np.random.randint(0, WIDTH, size=N_EVENTS, dtype=np.int32)
host_y = np.random.randint(0, HEIGHT, size=N_EVENTS, dtype=np.int32)
host_dt = np.random.randint(0, 50_000, size=N_EVENTS, dtype=np.int32)

print(f"Benchmarking 3D Voxel Indexing over {N_EVENTS:,} events")
print(f"Sensor Dim: {WIDTH}x{HEIGHT} | Time Bucket: {TIME_BUCKET_US} µs")
print(f"Platform: Apple Silicon (Unified Memory Architecture)\n" + "-" * 50)

# -------------------------------------------------------------
# 2. PyTorch MPS Implementation
# -------------------------------------------------------------
assert torch.backends.mps.is_available(), "PyTorch MPS backend not found."
mps_device = torch.device("mps")

# Pre-stage tensors to device
t_x = torch.from_numpy(host_x).to(mps_device)
t_y = torch.from_numpy(host_y).to(mps_device)
t_dt = torch.from_numpy(host_dt).to(mps_device)

def run_pytorch_mps():
    # 3D index: y * W + x + (dt // BUCKET) * (W * H)
    dt_bucket = torch.div(t_dt, TIME_BUCKET_US, rounding_mode='floor')
    out = (t_y * WIDTH) + t_x + (dt_bucket * (WIDTH * HEIGHT))
    torch.mps.synchronize()  # Block until GPU execution completes
    return out

# Warmup
for _ in range(WARMUP_RUNS):
    _ = run_pytorch_mps()

times_torch = []
for _ in range(BENCHMARK_RUNS):
    t0 = time.perf_counter()
    _ = run_pytorch_mps()
    times_torch.append(time.perf_counter() - t0)

avg_torch = np.mean(times_torch) * 1000.0
std_torch = np.std(times_torch) * 1000.0
print(f"PyTorch MPS (Vectorized): {avg_torch:6.2f} ms ± {std_torch:.2f} ms")

# -------------------------------------------------------------
# 3. MLX Standard Vectorized Implementation
# -------------------------------------------------------------
mx_x = mx.array(host_x)
mx_y = mx.array(host_y)
mx_dt = mx.array(host_dt)

def run_mlx_vectorized():
    dt_bucket = mx_dt // TIME_BUCKET_US
    out = (mx_y * WIDTH) + mx_x + (dt_bucket * (WIDTH * HEIGHT))
    mx.eval(out)  # Force lazy evaluation on GPU
    return out

# Warmup
for _ in range(WARMUP_RUNS):
    _ = run_mlx_vectorized()

times_mlx_vec = []
for _ in range(BENCHMARK_RUNS):
    t0 = time.perf_counter()
    _ = run_mlx_vectorized()
    times_mlx_vec.append(time.perf_counter() - t0)

avg_mlx_vec = np.mean(times_mlx_vec) * 1000.0
std_mlx_vec = np.std(times_mlx_vec) * 1000.0
print(f"MLX Native Vectorized:    {avg_mlx_vec:6.2f} ms ± {std_mlx_vec:.2f} ms")

# -------------------------------------------------------------
# 4. MLX Custom JIT Metal Kernel (Fused Single-Pass)
# -------------------------------------------------------------
# Fuses math directly into registers without intermediate buffer allocations
msl_source = """
    uint id = thread_position_in_grid.x;
    uint dt_bucket = dt[id] / 1000;
    out[id] = (y[id] * 1280) + x[id] + (dt_bucket * (1280 * 720));
"""

voxel_kernel = mx.fast.metal_kernel(
    name="fast_voxel_indexer",
    input_names=["x", "y", "dt"],
    output_names=["out"],
    source=msl_source
)

def run_mlx_custom_metal():
    out = voxel_kernel(
        inputs=[mx_x, mx_y, mx_dt],
        grid=(N_EVENTS, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(N_EVENTS,)],
        output_dtypes=[mx.uint32]
    )[0]
    mx.eval(out)
    return out

# Warmup & JIT compilation pass
for _ in range(WARMUP_RUNS):
    _ = run_mlx_custom_metal()

times_mlx_metal = []
for _ in range(BENCHMARK_RUNS):
    t0 = time.perf_counter()
    _ = run_mlx_custom_metal()
    times_mlx_metal.append(time.perf_counter() - t0)

avg_mlx_metal = np.mean(times_mlx_metal) * 1000.0
std_mlx_metal = np.std(times_mlx_metal) * 1000.0
print(f"MLX Custom Metal Kernel:  {avg_mlx_metal:6.2f} ms ± {std_mlx_metal:.2f} ms")

# -------------------------------------------------------------
# 5. Output Verification
# -------------------------------------------------------------
ref = run_pytorch_mps().cpu().numpy()
res_mlx = np.array(run_mlx_custom_metal())
assert np.array_equal(ref.astype(np.uint32), res_mlx), "Verification failed: output mismatch!"
print("-" * 50)
print(f"Verification: PASS (Outputs identical across PyTorch and MLX)")
print(f"Speedup (MLX Custom Metal vs PyTorch MPS): {avg_torch / avg_mlx_metal:.2f}x")
