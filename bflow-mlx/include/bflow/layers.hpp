#pragma once

#include <string>
#include <unordered_map>
#include <utility>

#include <mlx/array.h>

namespace bflow {

class Weights {
 public:
  explicit Weights(const std::string& safetensors_path);

  [[nodiscard]] const mlx::core::array& at(const std::string& name) const;
  [[nodiscard]] const std::unordered_map<std::string, std::string>& metadata() const noexcept;

 private:
  std::unordered_map<std::string, mlx::core::array> tensors_;
  std::unordered_map<std::string, std::string> metadata_;
};

enum class EncoderNorm { Instance, FoldedBatch };

// Activations are NHWC. Weights are OHWI and use upstream state-dict names.
mlx::core::array conv2d(
    const mlx::core::array& input,
    const Weights& weights,
    const std::string& prefix,
    std::pair<int, int> stride = {1, 1},
    std::pair<int, int> padding = {0, 0});

mlx::core::array conv2d(
    const mlx::core::array& input,
    const Weights& weights,
    const std::string& prefix,
    int stride,
    int padding = 0);

mlx::core::array relu(const mlx::core::array& input);

mlx::core::array basic_encoder(
    const mlx::core::array& input,
    const Weights& weights,
    const std::string& prefix,
    EncoderNorm norm);

// Pixel coordinates [N,...,2] are ordered (x,y). Out-of-bounds pixels are zero.
// The output shape is [N,...,C] for an image of shape [N,H,W,C].
mlx::core::array bilinear_sample(
    const mlx::core::array& image,
    const mlx::core::array& coordinates);

}  // namespace bflow
