#pragma once

#include "bflow/bezier.hpp"
#include "bflow/layers.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bflow {

class CorrelationPyramid {
public:
    // Every target receives a base volume. The checkpoint gives only its
    // final target all four levels; earlier targets stop at level zero.
    CorrelationPyramid(
        const mx::array& reference,
        const std::vector<mx::array>& targets,
        std::span<const int> levels,
        int radius)
        : window_size_(window_size(radius)),
          offsets_(mx::subtract(coordinate_grid(window_size_, window_size_),
                                mx::array(static_cast<float>(radius)))) {
        if (reference.ndim() != 4 || targets.empty() || levels.size() != targets.size()) {
            throw std::invalid_argument("correlation needs NHWC maps and one level count per target");
        }
        batch_ = reference.shape(0);
        height_ = reference.shape(1);
        width_ = reference.shape(2);
        const int channels = reference.shape(3);
        if (batch_ <= 0 || height_ <= 0 || width_ <= 0 || channels <= 0 ||
            static_cast<std::int64_t>(height_) * width_ >
                std::numeric_limits<int>::max() / batch_) {
            throw std::invalid_argument("correlation feature dimensions are invalid");
        }

        const int pixels = height_ * width_;
        const mx::Shape feature_shape{batch_, pixels, channels};
        const mx::Shape volume_shape{batch_ * pixels, height_, width_, 1};
        const auto ref = mx::reshape(mx::astype(reference, mx::float32), feature_shape);
        const auto normalization = mx::array(std::sqrt(static_cast<float>(channels)));
        pyramids_.reserve(targets.size());
        for (std::size_t target = 0; target < targets.size(); ++target) {
            if (targets[target].shape() != reference.shape() || levels[target] < 1) {
                throw std::invalid_argument("correlation target shapes or levels are invalid");
            }
            auto& pyramid = pyramids_.emplace_back();
            const auto feature = mx::reshape(mx::astype(targets[target], mx::float32),
                                             feature_shape);
            const auto dot = mx::matmul(ref, mx::transpose(feature, {0, 2, 1}));
            pyramid.push_back(mx::reshape(mx::divide(dot, normalization), volume_shape));
            for (int level = 1; level < levels[target]; ++level) {
                pyramid.push_back(pool2d(pyramid.back()));
            }
            max_levels_ = std::max(max_levels_, levels[target]);
        }
    }

    // Pixel coordinates [K,B,H,W,2] select one neighborhood per reference
    // pixel at each surviving level. The update block expects channels in
    // level-major, target-major, dy-major, dx-minor order.
    [[nodiscard]] mx::array lookup(const mx::array& coordinates) const {
        const mx::Shape expected{
            static_cast<int>(pyramids_.size()), batch_, height_, width_, 2,
        };
        if (coordinates.shape() != expected) {
            throw std::invalid_argument("correlation coordinates must be [K,B,H,W,2]");
        }
        const int reference_pixels = batch_ * height_ * width_;
        const int neighbors = window_size_ * window_size_;
        const auto centres = mx::unstack(coordinates, 0);
        std::vector<mx::array> samples;
        float scale = 1.0F;
        for (int level = 0; level < max_levels_; ++level, scale *= 2.0F) {
            for (std::size_t target = 0; target < pyramids_.size(); ++target) {
                const auto& pyramid = pyramids_[target];
                if (static_cast<std::size_t>(level) >= pyramid.size()) {
                    continue;
                }
                const auto center = mx::reshape(centres[target], {reference_pixels, 1, 1, 2});
                const auto query = mx::add(mx::divide(center, mx::array(scale)), offsets_);
                const auto sampled = bilinear_sample(pyramid[level], query);
                samples.push_back(mx::reshape(sampled, {batch_, height_, width_, neighbors}));
            }
        }
        return mx::concatenate(std::move(samples), -1);
    }

private:
    static int window_size(int radius) {
        const std::int64_t side = 2LL * radius + 1;
        if (radius < 0 || side > std::numeric_limits<int>::max() / side) {
            throw std::invalid_argument("correlation radius is invalid");
        }
        return static_cast<int>(side);
    }

    static mx::array pool2d(const mx::array& volume) {
        const int n = volume.shape(0);
        const int height = volume.shape(1) / 2;
        const int width = volume.shape(2) / 2;
        if (height < 1 || width < 1) {
            throw std::invalid_argument("correlation feature map is too small for its pyramid");
        }
        // MLX C++ exposes the pooling primitives as core operations. Trim
        // incomplete tiles and reduce non-overlapping groups of four.
        const auto trimmed = mx::slice(volume, {0, 0, 0, 0}, {n, 2 * height, 2 * width, 1});
        const auto tiles = mx::reshape(trimmed, {n, height, 2, width, 2, 1});
        return mx::mean(tiles, std::vector<int>{2, 4});
    }

    int window_size_;
    mx::array offsets_;
    int batch_ = 0;
    int height_ = 0;
    int width_ = 0;
    int max_levels_ = 0;
    std::vector<std::vector<mx::array>> pyramids_;
};

}  // namespace bflow
