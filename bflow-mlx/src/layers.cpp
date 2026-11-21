#include "bflow/layers.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mlx/io.h>
#include <mlx/ops.h>

namespace bflow {
namespace mx = mlx::core;

namespace {

void check_metadata(
    const std::unordered_map<std::string, std::string>& metadata,
    const std::string& key,
    const std::string& expected) {
  const auto found = metadata.find(key);
  if (found != metadata.end() && found->second != expected) {
    throw std::invalid_argument(
        "Weight metadata " + key + " is " + found->second + ", expected " + expected);
  }
}

mx::array encoder_normalize(const mx::array& input, EncoderNorm norm) {
  if (norm != EncoderNorm::Instance) {
    // Context encoder BatchNorm affine transforms are folded into its convolutions.
    return input;
  }

  constexpr float epsilon = 1e-5f;
  const std::vector<int> spatial_axes{1, 2};
  const auto centered = input - mx::mean(input, spatial_axes, true);
  const auto variance = mx::mean(mx::square(centered), spatial_axes, true);
  return centered * mx::rsqrt(variance + epsilon);
}

mx::array residual_block(
    const mx::array& input,
    const Weights& weights,
    const std::string& prefix,
    EncoderNorm norm,
    int stride) {
  auto output = relu(encoder_normalize(conv2d(input, weights, prefix + ".conv1", stride, 1), norm));
  output = relu(encoder_normalize(conv2d(output, weights, prefix + ".conv2", 1, 1), norm));

  auto skip = input;
  if (stride != 1) {
    skip = encoder_normalize(conv2d(input, weights, prefix + ".downsample.0", stride), norm);
  }
  return relu(skip + output);
}

}  // namespace

Weights::Weights(const std::string& safetensors_path) {
  auto [tensors, metadata] = mx::load_safetensors(safetensors_path);
  if (tensors.empty()) {
    throw std::invalid_argument("No tensors found in " + safetensors_path);
  }
  check_metadata(metadata, "format", "bflow-mlx-weights-v1");
  check_metadata(metadata, "conv_layout", "OHWI");
  check_metadata(metadata, "context_batch_norm", "folded");
  check_metadata(metadata, "feature_instance_norm", "input_statistics");
  tensors_ = std::move(tensors);
  metadata_ = std::move(metadata);
}

const mx::array& Weights::at(const std::string& name) const {
  const auto found = tensors_.find(name);
  if (found == tensors_.end()) {
    throw std::out_of_range("Missing BFlow tensor " + name);
  }
  return found->second;
}

const std::unordered_map<std::string, std::string>& Weights::metadata() const noexcept {
  return metadata_;
}

mx::array conv2d(
    const mx::array& input,
    const Weights& weights,
    const std::string& prefix,
    std::pair<int, int> stride,
    std::pair<int, int> padding) {
  const auto& kernel = weights.at(prefix + ".weight");
  const auto& bias = weights.at(prefix + ".bias");
  if (input.ndim() != 4 || kernel.ndim() != 4 || bias.ndim() != 1 ||
      kernel.shape(3) != input.shape(3) || kernel.shape(0) != bias.shape(0) ||
      kernel.dtype() != mx::float32 || bias.dtype() != mx::float32) {
    throw std::invalid_argument("Invalid NHWC input or float32 convolution tensors for " + prefix);
  }
  return mx::conv2d(input, kernel, stride, padding) + bias;
}

mx::array conv2d(
    const mx::array& input,
    const Weights& weights,
    const std::string& prefix,
    int stride,
    int padding) {
  return conv2d(input, weights, prefix, {stride, stride}, {padding, padding});
}

mx::array relu(const mx::array& input) {
  return mx::maximum(input, mx::array(0, input.dtype()));
}

mx::array basic_encoder(
    const mx::array& input,
    const Weights& weights,
    const std::string& prefix,
    EncoderNorm norm) {
  auto output = relu(encoder_normalize(conv2d(input, weights, prefix + ".conv1", 2, 3), norm));
  for (int layer = 1; layer <= 3; ++layer) {
    const auto stage = prefix + ".layer" + std::to_string(layer);
    const int stride = layer == 1 ? 1 : 2;
    output = residual_block(output, weights, stage + ".0", norm, stride);
    output = residual_block(output, weights, stage + ".1", norm, 1);
  }
  return conv2d(output, weights, prefix + ".conv2");
}

mx::array bilinear_sample(const mx::array& image, const mx::array& coordinates) {
  if (image.ndim() != 4 || coordinates.ndim() < 2 || coordinates.shape(-1) != 2 ||
      image.shape(0) != coordinates.shape(0)) {
    throw std::invalid_argument("Bilinear sampling requires [N,H,W,C] and [N,...,2] inputs");
  }

  const auto batch = image.shape(0);
  const auto height = image.shape(1);
  const auto width = image.shape(2);
  const auto channels = image.shape(3);
  if (batch <= 0 || height <= 0 || width <= 0 || channels <= 0 ||
      static_cast<std::int64_t>(height) * width > std::numeric_limits<mx::ShapeElem>::max() / batch) {
    throw std::invalid_argument("Invalid bilinear sampling image size");
  }

  const auto x = mx::take(coordinates, 0, -1);
  const auto y = mx::take(coordinates, 1, -1);
  const auto x0 = mx::floor(x);
  const auto y0 = mx::floor(y);
  const auto x1 = x0 + 1;
  const auto y1 = y0 + 1;
  const auto dx = x - x0;
  const auto dy = y - y0;

  const auto xs = mx::stack({x0, x1, x0, x1}, -1);
  const auto ys = mx::stack({y0, y0, y1, y1}, -1);
  const auto coefficients = mx::stack({(1 - dx) * (1 - dy), dx * (1 - dy),
                                       (1 - dx) * dy, dx * dy}, -1);
  const auto valid_x = mx::logical_and(xs >= 0, xs < width);
  const auto valid_y = mx::logical_and(ys >= 0, ys < height);
  const auto valid = mx::logical_and(valid_x, valid_y);

  // Map all four neighbor coordinates to one gather over flattened spatial data.
  const auto ix = mx::astype(mx::where(valid, xs, mx::array(0, xs.dtype())), mx::int32);
  const auto iy = mx::astype(mx::where(valid, ys, mx::array(0, ys.dtype())), mx::int32);
  mx::Shape batch_shape(coordinates.ndim(), 1);
  batch_shape[0] = batch;
  const auto batch_base = mx::reshape(mx::arange(batch), batch_shape) * (height * width);
  const auto indices = batch_base + iy * width + ix;
  const auto neighbors = mx::take(mx::reshape(image, {-1, channels}), indices, 0);
  const auto zero = mx::array(0, image.dtype());
  const auto masked = mx::where(mx::expand_dims(valid, -1), neighbors, zero);
  const auto weights = mx::where(valid, coefficients, mx::array(0, coefficients.dtype()));
  return mx::sum(masked * mx::expand_dims(weights, -1), -2);
}

}  // namespace bflow
