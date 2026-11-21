#pragma once

#include <string>

#include "bflow/config.hpp"
#include "bflow/layers.hpp"

namespace bflow {

class BFlowNet {
public:
    explicit BFlowNet(const std::string& weights_path);

    [[nodiscard]] const auto& weight_metadata() const noexcept { return weights_.metadata(); }

    // Input [B,H,W,65]. Output [B,H,W,20]: x1..x10, y1..y10 in full-resolution pixels.
    // The graph stays lazy. Evaluate the returned array at the inference boundary.
    [[nodiscard]] mlx::core::array forward(
        const mlx::core::array& voxels,
        int iterations = BFlowConfig::default_iterations) const;

private:
    Weights weights_;
};

} // namespace bflow
