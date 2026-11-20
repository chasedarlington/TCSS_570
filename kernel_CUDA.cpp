#include <iostream>
#include <vector>
#include <cmath>
#include <atomic>
#include <cstdint>
#include <omp.h>

void accumulateEventsCPU(
    const uint16_t* d_x,
    const uint16_t* d_y,
    const uint64_t* d_t,
    const float*    d_p,
    std::vector<std::atomic<float>>& d_V, // Thread-safe atomic linear buffer of size B * H * W
    int numEvents,
    int width,
    int height,
    int B,
    double deltaT,
    uint64_t t0
) {
    size_t slice_stride = (size_t)height * width;
    double scale = (double)(B - 1) / deltaT;

    // Parallelize event loop across CPU cores using OpenMP
    #omp parallel for schedule(static)
    for (int idx = 0; idx < numEvents; ++idx) {
        int x_k = d_x[idx];
        int y_k = d_y[idx];
        uint64_t t_k = d_t[idx];
        float p_k = d_p[idx];

        // Compute continuous temporal coordinate
        double t_star = scale * (double)(t_k - t0);

        // Find the lower temporal slice index
        int b = static_cast<int>(std::floor(t_star));

        // Check boundary safety for trilinear time interpolation
        if (b >= 0 && b < B - 1) {
            float weight_b  = static_cast<float>(1.0 - (t_star - b));
            float weight_b1 = static_cast<float>(t_star - b);

            size_t offset_b  = static_cast<size_t>(b)       * slice_stride + (size_t)y_k * width + x_k;
            size_t offset_b1 = static_cast<size_t>(b + 1) * slice_stride + (size_t)y_k * width + x_k;

            // Thread-safe atomic float additions to prevent data races across CPU threads
            // (Equivalent to CUDA's atomicAdd)
            atomicAddFloat(d_V[offset_b], p_k * weight_b);
            atomicAddFloat(d_V[offset_b1], p_k * weight_b1);
        }
    }
}

// Helper utility for atomic float addition using std::atomic
inline void atomicAddFloat(std::atomic<float>& addr, float val) {
    float expected = addr.load(std::memory_order_relaxed);
    while (!addr.compare_exchange_weak(expected, expected + val,
                                       std::memory_order_relaxed,
                                       std::memory_order_relaxed));
}