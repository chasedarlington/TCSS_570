// C++ / MLX port of the event-only RAFTSpline model from uzh-rpg/bflow.
#include "bflow/bflow_net.hpp"

#include <mlx/ops.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bflow/bezier.hpp"
#include "bflow/corr.hpp"

namespace bflow {
namespace {

namespace mx = mlx::core;
using mx::array;
using Config = BFlowConfig;

array channels(const array& input, int begin, int end) {
    return mx::slice(input, {0, 0, 0, begin}, {input.shape(0), input.shape(1), input.shape(2), end});
}

array encode_motion(const Weights& weights, const array& corr, const array& controls) {
    const std::string prefix = "update_block.encoder.";
    auto correlation = relu(conv2d(corr, weights, prefix + "convc1", 1, 0));
    correlation = relu(conv2d(correlation, weights, prefix + "convc2", 1, 1));
    auto motion = relu(conv2d(controls, weights, prefix + "convf1", 1, 3));
    motion = relu(conv2d(motion, weights, prefix + "convf2", 1, 1));
    auto encoded = relu(conv2d(mx::concatenate({correlation, motion}, -1), weights, prefix + "conv", 1, 1));
    return mx::concatenate({encoded, controls}, -1);
}

array gru_pass(const Weights& weights, const array& hidden, const array& input,
               const std::string& suffix, std::pair<int, int> padding) {
    const std::string prefix = "update_block.gru.";
    const auto joined = mx::concatenate({hidden, input}, -1);
    const auto z = mx::sigmoid(conv2d(joined, weights, prefix + "convz" + suffix, {1, 1}, padding));
    const auto r = mx::sigmoid(conv2d(joined, weights, prefix + "convr" + suffix, {1, 1}, padding));
    const auto candidate_input = mx::concatenate({mx::multiply(r, hidden), input}, -1);
    const auto q = mx::tanh(conv2d(candidate_input, weights, prefix + "convq" + suffix, {1, 1}, padding));
    return mx::add(mx::multiply(mx::subtract(array(1.0f), z), hidden), mx::multiply(z, q));
}

array control_delta(const Weights& weights, const array& hidden) {
    const std::string prefix = "update_block.bezier_head.";
    const auto features = relu(conv2d(hidden, weights, prefix + "conv1", 1, 1));
    return conv2d(features, weights, prefix + "conv2", 1, 1);
}

array upsample_mask(const Weights& weights, const array& hidden) {
    const std::string prefix = "update_block.mask.";
    const auto features = relu(conv2d(hidden, weights, prefix + "0", 1, 1));
    return mx::multiply(array(0.25f), conv2d(features, weights, prefix + "2", 1, 0));
}

array convex_upsample(const array& controls, const array& mask_logits) {
    const int batch = controls.shape(0);
    const int height = controls.shape(1);
    const int width = controls.shape(2);
    const int channels = controls.shape(3);
    constexpr int scale = Config::scale;
    constexpr int neighbors = 9;
    const auto mask = mx::softmax(mx::reshape(mask_logits, {batch, height, width, neighbors, scale, scale, 1}), 3);

    // Match the PyTorch unfold order (dy first, then dx). Pixel displacement scales by eight.
    const std::vector<std::pair<int, int>> pad = {{0, 0}, {1, 1}, {1, 1}, {0, 0}};
    const auto padded = mx::pad(mx::multiply(controls, array(static_cast<float>(scale))), pad);
    std::vector<array> neighborhood;
    neighborhood.reserve(neighbors);
    for (int dy = 0; dy < 3; ++dy) {
        for (int dx = 0; dx < 3; ++dx) {
            neighborhood.push_back(mx::slice(
                padded, {0, dy, dx, 0}, {batch, dy + height, dx + width, channels}));
        }
    }
    const auto values = mx::reshape(mx::stack(neighborhood, 3), {batch, height, width, neighbors, 1, 1, channels});
    const auto upsampled = mx::sum(mx::multiply(mask, values), 3);
    return mx::reshape(mx::transpose(upsampled, {0, 1, 3, 2, 4, 5}),
                       {batch, height * scale, width * scale, channels});
}

} // namespace

BFlowNet::BFlowNet(const std::string& weights_path) : weights_(weights_path) {
    const auto& metadata = weights_.metadata();
    const auto model = metadata.find("model");
    if (model != metadata.end() && model->second != Config::model_name) {
        throw std::invalid_argument("BFlow checkpoint profile must be " + std::string(Config::model_name));
    }
}

mlx::core::array BFlowNet::forward(const mlx::core::array& voxels, int iterations) const {
    if (voxels.ndim() != 4 || voxels.shape(0) <= 0 || voxels.shape(3) != Config::input_bins) {
        throw std::invalid_argument("BFlow needs nonempty NHWC voxels with 65 channels");
    }
    if (voxels.shape(1) < Config::minimum_image_side || voxels.shape(2) < Config::minimum_image_side ||
        voxels.shape(1) % Config::scale != 0 || voxels.shape(2) % Config::scale != 0) {
        throw std::invalid_argument("BFlow image dimensions must be multiples of 8 and at least 128");
    }
    if (iterations <= 0) {
        throw std::invalid_argument("BFlow iterations must be positive");
    }
    if (voxels.dtype() != mx::float32) {
        throw std::invalid_argument("BFlow voxels must be float32");
    }

    const auto encode_features = [&](int first) {
        return basic_encoder(channels(voxels, first, first + Config::correlation_bins),
                             weights_, "fnet_ev", EncoderNorm::Instance);
    };
    const auto reference = encode_features(0);
    std::vector<array> targets;
    targets.reserve(Config::target_indices.size());
    for (const int first : Config::target_indices) {
        targets.push_back(encode_features(first));
    }
    const CorrelationPyramid pyramid(reference, targets, Config::pyramid_levels, Config::radius);

    const auto context_voxels = channels(voxels, Config::input_bins - Config::context_bins, Config::input_bins);
    const auto context_features = basic_encoder(context_voxels, weights_, "cnet", EncoderNorm::FoldedBatch);
    auto hidden = mx::tanh(channels(context_features, 0, Config::hidden_channels));
    const auto context = relu(channels(context_features, Config::hidden_channels,
                                      Config::hidden_channels + Config::context_channels));

    const int height = reference.shape(1);
    const int width = reference.shape(2);
    const auto coords0 = coordinate_grid(height, width);
    const BezierBasis bezier(Config::bezier_degree, Config::timestamps);
    auto control_points = mx::zeros({voxels.shape(0), height, width, Config::control_channels}, mx::float32);

    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto lookup = pyramid.lookup(mx::add(coords0, bezier.evaluate(control_points)));
        const auto motion = encode_motion(weights_, lookup, control_points);
        const auto gru_input = mx::concatenate({context, motion}, -1);
        hidden = gru_pass(weights_, hidden, gru_input, "1", {0, 2});
        hidden = gru_pass(weights_, hidden, gru_input, "2", {2, 0});
        control_points = mx::add(control_points, control_delta(weights_, hidden));
    }

    return convex_upsample(control_points, upsample_mask(weights_, hidden));
}

} // namespace bflow
