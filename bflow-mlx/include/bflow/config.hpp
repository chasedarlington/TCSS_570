#pragma once

#include <array>
#include <cstddef>
#include <string_view>

namespace bflow {

// Shape and channel layout of the published MultiFlow event-only checkpoint.
struct BFlowConfig {
    static constexpr std::string_view model_name = "E_LU5_BD10_lowpyramid";
    static constexpr int bezier_degree = 10;
    static constexpr int control_channels = 2 * bezier_degree;
    static constexpr int context_bins = 41;
    static constexpr int correlation_bins = 25;
    static constexpr int input_bins = context_bins + correlation_bins - 1;
    static constexpr int hidden_channels = 128;
    static constexpr int context_channels = 128;
    static constexpr int feature_channels = 256;
    static constexpr int motion_channels = 128;
    static constexpr int scale = 8;
    static constexpr int radius = 4;
    static constexpr int default_iterations = 12;
    static constexpr int minimum_image_side = 128;

    static constexpr std::array target_indices{8, 16, 24, 32, 40};
    static constexpr std::array pyramid_levels{1, 1, 1, 1, 4};
    static constexpr auto timestamps = [] {
        std::array<double, target_indices.size()> result{};
        for (std::size_t index = 0; index < result.size(); ++index) {
            result[index] = static_cast<double>(target_indices[index]) / (context_bins - 1);
        }
        return result;
    }();

    static constexpr int correlation_channels = [] {
        int levels = 0;
        for (const int count : pyramid_levels) {
            levels += count;
        }
        const int side = 2 * radius + 1;
        return levels * side * side;
    }();
};

static_assert(BFlowConfig::correlation_channels == 648);

} // namespace bflow
