#include <metal_stdlib>
using namespace metal;

// Output structure for compact corner point extraction
struct CornerRecord {
    float x;
    float y;
    float response;
    uint  bin_idx;
};

// Uniform parameters passed to both kernels
struct HarrisParams {
    uint width;
    uint height;
    uint depth;
    float kappa;          // Typically 0.04 - 0.06
    float threshold;      // Minimum response threshold to consider
};

// Helper to access a flat 3D voxel grid: [B, H, W]
inline uint voxel_index(uint x, uint y, uint b, uint width, uint height) {
    return b * (width * height) + y * width + x;
}

// -----------------------------------------------------------------------------
// PASS 1: Compute Gradients, Structure Tensor, and Harris Response
// -----------------------------------------------------------------------------
kernel void compute_structure_tensor_and_harris(
    device const float*       voxel_grid    [[buffer(0)]], // [B x H x W]
    device float*             harris_map    [[buffer(1)]], // [B x H x W]
    constant HarrisParams&    params        [[buffer(2)]],
    uint3                     gid           [[thread_position_in_grid]])
{
    const uint x = gid.x;
    const uint y = gid.y;
    const uint b = gid.z;

    // Boundary check (need 2-pixel margin for Sobel + 3x3 Gaussian window)
    if (x < 2 || x >= params.width - 2 || y < 2 || y >= params.height - 2 || b >= params.depth) {
        if (x < params.width && y < params.height && b < params.depth) {
            harris_map[voxel_index(x, y, b, params.width, params.height)] = 0.0f;
        }
        return;
    }

    // 3x3 Gaussian weights for structure tensor integration
    // [1 2 1] / 16 outer product
    const float G[3][3] = {
        { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f },
        { 2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f },
        { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f }
    };

    float Mxx = 0.0f;
    float Myy = 0.0f;
    float Mxy = 0.0f;

    // Integrate over a 3x3 local neighborhood
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const uint cx = x + dx;
            const uint cy = y + dy;

            // Compute Sobel Ix at (cx, cy)
            float v00 = voxel_grid[voxel_index(cx - 1, cy - 1, b, params.width, params.height)];
            float v01 = voxel_grid[voxel_index(cx,     cy - 1, b, params.width, params.height)];
            float v02 = voxel_grid[voxel_index(cx + 1, cy - 1, b, params.width, params.height)];

            float v10 = voxel_grid[voxel_index(cx - 1, cy,     b, params.width, params.height)];
            float v12 = voxel_grid[voxel_index(cx + 1, cy,     b, params.width, params.height)];

            float v20 = voxel_grid[voxel_index(cx - 1, cy + 1, b, params.width, params.height)];
            float v21 = voxel_grid[voxel_index(cx,     cy + 1, b, params.width, params.height)];
            float v22 = voxel_grid[voxel_index(cx + 1, cy + 1, b, params.width, params.height)];

            // Sobel 3x3 filters
            float Ix = (v02 + 2.0f * v12 + v22) - (v00 + 2.0f * v10 + v20);
            float Iy = (v20 + 2.0f * v21 + v22) - (v00 + 2.0f * v01 + v02);

            float weight = G[dy + 1][dx + 1];
            Mxx += weight * (Ix * Ix);
            Myy += weight * (Iy * Iy);
            Mxy += weight * (Ix * Iy);
        }
    }

    // Evaluate Harris Metric: R = det(M) - kappa * (trace(M))^2
    float det = (Mxx * Myy) - (Mxy * Mxy);
    float trace = Mxx + Myy;
    float response = det - (params.kappa * trace * trace);

    harris_map[voxel_index(x, y, b, params.width, params.height)] = (response > params.threshold) ? response : 0.0f;
}

// -----------------------------------------------------------------------------
// PASS 2: Non-Maximum Suppression (3x3) & Stream Compaction
// -----------------------------------------------------------------------------
kernel void non_maximum_suppression(
    device const float*        harris_map       [[buffer(0)]], // [B x H x W]
    device atomic_uint*        corner_count     [[buffer(1)]], // Single counter
    device CornerRecord*       corners_out      [[buffer(2)]], // Compact output array
    constant HarrisParams&     params           [[buffer(3)]],
    constant uint&             max_corners      [[buffer(4)]],
    uint3                      gid              [[thread_position_in_grid]])
{
    const uint x = gid.x;
    const uint y = gid.y;
    const uint b = gid.z;

    if (x < 1 || x >= params.width - 1 || y < 1 || y >= params.height - 1 || b >= params.depth) {
        return;
    }

    const uint center_idx = voxel_index(x, y, b, params.width, params.height);
    const float center_val = harris_map[center_idx];

    // Early exit if below threshold or suppressed in Pass 1
    if (center_val <= params.threshold) {
        return;
    }

    // Check if strictly greater than all neighboring cells in the 3x3 spatial window
    bool is_max = true;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;

            float neighbor = harris_map[voxel_index(x + dx, y + dy, b, params.width, params.height)];

            // Use strictly greater on prior scanlines and greater-or-equal on subsequent 
            // scanlines to prevent duplicates for plateaus
            if (dy < 0 || (dy == 0 && dx < 0)) {
                if (neighbor >= center_val) {
                    is_max = false;
                    break;
                }
            } else {
                if (neighbor > center_val) {
                    is_max = false;
                    break;
                }
            }
        }
        if (!is_max) break;
    }

    // Atomic stream compaction into linear corner buffer
    if (is_max) {
        uint insert_idx = atomic_fetch_add_explicit(corner_count, 1, memory_order_relaxed);
        if (insert_idx < max_corners) {
            corners_out[insert_idx] = CornerRecord{
                .x = static_cast<float>(x),
                .y = static_cast<float>(y),
                .response = center_val,
                .bin_idx = b
            };
        }
    }
}
