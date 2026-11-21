import mlx.core as mx

# 1. You write ONLY the inner loop body (MLX auto-generates the signature)
source = """
    uint id = thread_position_in_grid.x;
    out[id] = (y[id] * 1280) + x[id] + ((dt[id] / 1000) * 1280 * 720);
"""

# 2. Build the JIT-compiled kernel wrapper once
voxel_kernel = mx.fast.metal_kernel(
    name="voxel_indexer",
    input_names=["x", "y", "dt"],
    output_names=["out"],
    source=source
)

# 3. Call directly with native arrays
def run_indexing(x_arr, y_arr, dt_arr):
    n_events = x_arr.size
    return voxel_kernel(
        inputs=[x_arr, y_arr, dt_arr],
        template=[("T", mx.uint32)],
        grid=(n_events, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n_events,)],
        output_dtypes=[mx.uint32]
    )[0]
