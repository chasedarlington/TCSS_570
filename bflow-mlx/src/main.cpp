#include "bflow/bflow_net.hpp"
#include "bflow/config.hpp"
#include "bflow/voxelizer.hpp"

#include <mlx/mlx.h>

#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace {

namespace mx = mlx::core;
using bflow::BFlowConfig;
using bflow::PolarityEncoding;
using Path = std::filesystem::path;
using Clock = std::chrono::steady_clock;

constexpr int64_t training_bin_step_us = 12'500;

struct Options {
  Path weights;
  Path input;
  Path events;
  Path parity;
  Path reference;
  Path output;
  std::optional<int64_t> time_us;
  std::optional<int> iterations;
  std::optional<PolarityEncoding> polarity;
  int width = 1280;
  int height = 720;
  float atol = 2e-3f;
  float rtol = 2e-3f;
};

void usage(std::ostream& out) {
  out << R"(Usage:
  bflow-mlx --weights MODEL.safetensors --parity PARITY.safetensors
  bflow-mlx --weights MODEL.safetensors --input VOXELS.npy [--reference CONTROL_POINTS.npy]
  bflow-mlx --weights MODEL.safetensors --events EVENTS.h5 --time-us REFERENCE_TIME

Options:
  --input PATH            NHWC voxel tensor (NPY or safetensors key voxel_grid)
  --reference PATH        NHWC parity target (NPY or safetensors key control_points)
  --iterations N          GRU iterations (bundle value in parity mode, otherwise 12)
  --atol VALUE            Parity absolute tolerance (default 0.002)
  --rtol VALUE            Parity relative tolerance (default 0.002)
  --output PATH           Save NHWC control points (.npy or .safetensors)

Event input options:
  --events PATH           One TUM-VIE HDF5 camera stream, or x,y,t_us,p CSV
  --time-us INTEGER       Reference timestamp on the event stream clock (microseconds)
  --width N               Image width (default 1280)
  --height N              Image height (default 720)
  --polarity ENCODING     tum-vie, dsec, or signed (default tum-vie for HDF5, signed for CSV)

The 65-bin raw-event input extends 300 ms before and 500 ms after the
reference bin, with one extra bin of event support on either side.
)";
}

template <typename Integer>
Integer parse_integer(std::string_view value, std::string_view flag) {
  Integer parsed{};
  const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
    throw std::invalid_argument(std::string(flag) + " requires an integer");
  }
  return parsed;
}

float parse_tolerance(const char* value, std::string_view flag) {
  errno = 0;
  char* end = nullptr;
  const float parsed = std::strtof(value, &end);
  if (end == value || *end != '\0' || errno == ERANGE || !std::isfinite(parsed) || parsed < 0) {
    throw std::invalid_argument(std::string(flag) + " requires a finite nonnegative number");
  }
  return parsed;
}

PolarityEncoding parse_polarity(std::string_view value) {
  if (value == "tum-vie") {
    return PolarityEncoding::tum_vie;
  }
  if (value == "dsec") {
    return PolarityEncoding::dsec;
  }
  if (value == "signed") {
    return PolarityEncoding::signed_values;
  }
  throw std::invalid_argument("--polarity must be tum-vie, dsec, or signed");
}

Options parse_options(int argc, char** argv) {
  Options options;
  std::unordered_set<std::string_view> seen;
  for (int i = 1; i < argc; i += 2) {
    const std::string_view flag = argv[i];
    if (i + 1 == argc || std::string_view(argv[i + 1]).starts_with("--")) {
      throw std::invalid_argument("missing value for " + std::string(flag));
    }
    if (!seen.insert(flag).second) {
      throw std::invalid_argument("duplicate option " + std::string(flag));
    }
    const char* value = argv[i + 1];
    if (flag == "--weights") {
      options.weights = value;
    } else if (flag == "--input") {
      options.input = value;
    } else if (flag == "--events") {
      options.events = value;
    } else if (flag == "--parity") {
      options.parity = value;
    } else if (flag == "--reference") {
      options.reference = value;
    } else if (flag == "--output") {
      options.output = value;
    } else if (flag == "--time-us") {
      options.time_us = parse_integer<int64_t>(value, flag);
    } else if (flag == "--iterations") {
      options.iterations = parse_integer<int>(value, flag);
    } else if (flag == "--width") {
      options.width = parse_integer<int>(value, flag);
    } else if (flag == "--height") {
      options.height = parse_integer<int>(value, flag);
    } else if (flag == "--polarity") {
      options.polarity = parse_polarity(value);
    } else if (flag == "--atol") {
      options.atol = parse_tolerance(value, flag);
    } else if (flag == "--rtol") {
      options.rtol = parse_tolerance(value, flag);
    } else {
      throw std::invalid_argument("unknown option " + std::string(flag));
    }
  }

  if (options.weights.extension() != ".safetensors") {
    throw std::invalid_argument("--weights requires a .safetensors path");
  }
  const int input_count = !options.input.empty() + !options.events.empty() + !options.parity.empty();
  if (input_count != 1) {
    throw std::invalid_argument("choose exactly one of --input, --events, and --parity");
  }
  if (!options.parity.empty() && (!options.reference.empty() || options.parity.extension() != ".safetensors")) {
    throw std::invalid_argument("--parity takes a .safetensors bundle with its own reference");
  }
  if (options.events.empty() &&
      (options.time_us || options.polarity || seen.contains("--width") || seen.contains("--height"))) {
    throw std::invalid_argument("--time-us, --polarity, --width, and --height require --events");
  }
  if (!options.events.empty() && !options.time_us) {
    throw std::invalid_argument("--events requires --time-us");
  }
  if (options.iterations && *options.iterations <= 0) {
    throw std::invalid_argument("--iterations must be positive");
  }
  if (!options.output.empty() && options.output.extension() != ".npy" &&
      options.output.extension() != ".safetensors") {
    throw std::invalid_argument("--output requires .npy or .safetensors");
  }
  if (!options.output.empty()) {
    const auto canonical_output = std::filesystem::weakly_canonical(options.output);
    for (const auto* source : {&options.weights, &options.input, &options.events,
                               &options.parity, &options.reference}) {
      if (source->empty()) {
        continue;
      }
      std::error_code error;
      if (std::filesystem::weakly_canonical(*source) == canonical_output ||
          std::filesystem::equivalent(*source, options.output, error)) {
        throw std::invalid_argument("--output aliases input file " + source->string());
      }
    }
  }
  return options;
}

mx::array get_tensor(const mx::SafetensorsLoad& archive, const std::string& key) {
  const auto it = archive.first.find(key);
  if (it == archive.first.end()) {
    throw std::runtime_error("missing safetensors tensor " + key);
  }
  return it->second;
}

mx::array load_tensor(const Path& path, const std::string& key) {
  if (path.extension() == ".npy") {
    return mx::load(path.string());
  }
  if (path.extension() == ".safetensors") {
    return get_tensor(mx::load_safetensors(path.string()), key);
  }
  throw std::invalid_argument("expected .npy or .safetensors at " + path.string());
}

mx::array load_event_voxels(const Options& options) {
  const int width = options.width;
  const int height = options.height;
  if (width < BFlowConfig::minimum_image_side || height < BFlowConfig::minimum_image_side ||
      width % BFlowConfig::scale != 0 || height % BFlowConfig::scale != 0) {
    throw std::invalid_argument("event width and height must be multiples of 8 and at least 128");
  }

  constexpr int64_t history = (BFlowConfig::correlation_bins - 1) * training_bin_step_us;
  constexpr int64_t future = (BFlowConfig::context_bins - 1) * training_bin_step_us;
  const int64_t reference_time = *options.time_us;
  if (reference_time < std::numeric_limits<int64_t>::min() + history + training_bin_step_us ||
      reference_time > std::numeric_limits<int64_t>::max() - future - training_bin_step_us) {
    throw std::invalid_argument("--time-us is outside the supported timestamp range");
  }
  const int64_t first_center = reference_time - history;
  const int64_t last_center = reference_time + future;
  const auto [event_start, event_end] =
      bflow::extended_time_window(first_center, last_center, BFlowConfig::input_bins);

  const auto extension = options.events.extension();
  bflow::EventSoA events;
  if (extension == ".csv") {
    events = bflow::read_events_csv(options.events,
        options.polarity.value_or(PolarityEncoding::signed_values));
  } else if (extension == ".h5" || extension == ".hdf5") {
    events = bflow::read_events_hdf5(options.events, event_start, event_end,
        options.polarity.value_or(PolarityEncoding::tum_vie));
  } else {
    throw std::invalid_argument("--events requires .h5, .hdf5, or .csv");
  }

  const auto voxels = bflow::voxelize(events, width, height, BFlowConfig::input_bins,
      first_center, last_center);
  return mx::array(voxels.begin(), {1, height, width, BFlowConfig::input_bins}, mx::float32);
}

struct Input {
  mx::array voxels;
  std::optional<mx::array> reference;
  int iterations;
};

void validate_parity_metadata(const mx::SafetensorsLoad& archive, const bflow::BFlowNet& model) {
  const auto matches_if_present = [&](const std::string& key, std::string_view expected) {
    const auto found = archive.second.find(key);
    if (found != archive.second.end() && found->second != expected) {
      throw std::invalid_argument("unrecognized parity metadata " + key + ": " + found->second);
    }
  };
  matches_if_present("format", "bflow-mlx-parity-v1");
  matches_if_present("model", BFlowConfig::model_name);
  matches_if_present("tensor_layout", "NHWC");

  const auto parity_sha = archive.second.find("source_sha256");
  const auto& weights = model.weight_metadata();
  const auto weights_sha = weights.find("source_sha256");
  if (parity_sha != archive.second.end() && weights_sha != weights.end() &&
      parity_sha->second != weights_sha->second) {
    throw std::invalid_argument("parity bundle and weights were exported from different checkpoints");
  }
}

Input prepare_input(const Options& options, const bflow::BFlowNet& model) {
  if (!options.parity.empty()) {
    // The iteration count is metadata for the host, independent of the GPU graph.
    auto archive = mx::load_safetensors(options.parity.string());
    validate_parity_metadata(archive, model);
    auto count = get_tensor(archive, "iterations");
    if (count.size() != 1 || count.dtype() != mx::int32) {
      throw std::runtime_error("parity iterations must be one int32 value");
    }
    const int iterations = count.item<int32_t>();
    if (iterations <= 0 || (options.iterations && *options.iterations != iterations)) {
      throw std::runtime_error("--iterations must match the positive parity bundle iteration count");
    }
    return {get_tensor(archive, "voxel_grid"), get_tensor(archive, "control_points"), iterations};
  }

  auto voxels = options.events.empty() ? load_tensor(options.input, "voxel_grid") : load_event_voxels(options);
  std::optional<mx::array> reference;
  if (!options.reference.empty()) {
    reference.emplace(load_tensor(options.reference, "control_points"));
  }
  return {std::move(voxels), std::move(reference), options.iterations.value_or(BFlowConfig::default_iterations)};
}

std::string shape_string(const mx::array& array) {
  std::string shape = "[";
  for (const auto extent : array.shape()) {
    if (shape.size() > 1) {
      shape += ", ";
    }
    shape += std::to_string(extent);
  }
  return shape + "]";
}

bool verify_parity(const mx::array& result, const mx::array& target, const Options& options) {
  if (result.shape() != target.shape()) {
    throw std::runtime_error("parity shape mismatch: result " + shape_string(result) +
        ", reference " + shape_string(target));
  }

  const auto reference = mx::astype(target, mx::float32);
  const auto prediction = mx::astype(result, mx::float32);
  const auto abs_error = mx::abs(prediction - reference);
  const auto tolerance = options.atol + options.rtol * mx::abs(reference);
  auto pass = mx::all(mx::isfinite(reference) && mx::isfinite(prediction) && (abs_error <= tolerance));
  auto max_error = mx::max(abs_error);
  auto mean_error = mx::mean(abs_error);
  mx::eval(pass, max_error, mean_error);

  const bool passed = pass.item<bool>();
  std::cout << "parity=" << (passed ? "pass" : "fail") << std::scientific
            << " max_abs=" << max_error.item<float>() << " mean_abs=" << mean_error.item<float>()
            << " atol=" << options.atol << " rtol=" << options.rtol << '\n';
  return passed;
}

void save_output(const Path& path, const mx::array& result) {
  if (path.extension() == ".npy") {
    mx::save(path.string(), result);
  } else {
    mx::save_safetensors(path.string(), {{"control_points", result}},
        {{"model", std::string(BFlowConfig::model_name)}, {"tensor_layout", "NHWC"}});
  }
}

double milliseconds(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
    usage(std::cout);
    return EXIT_SUCCESS;
  }
  try {
    const auto options = parse_options(argc, argv);
    if (!mx::metal::is_available()) {
      throw std::runtime_error("MLX could not access an Apple Metal GPU");
    }
    mx::set_default_device(mx::Device::gpu);

    const auto start = Clock::now();
    const auto model = bflow::BFlowNet(options.weights.string());
    const auto input = prepare_input(options, model);
    const auto inference_start = Clock::now();
    auto control_points = model.forward(input.voxels, input.iterations);
    mx::eval(control_points);
    const auto inference_end = Clock::now();

    std::cout << "control_points=" << shape_string(control_points) << " iterations=" << input.iterations
              << std::fixed << std::setprecision(3)
              << " preparation_ms=" << milliseconds(inference_start - start)
              << " forward_and_eval_ms=" << milliseconds(inference_end - inference_start) << '\n';
    if (input.reference && !verify_parity(control_points, *input.reference, options)) {
      return EXIT_FAILURE;
    }
    if (!options.output.empty()) {
      save_output(options.output, control_points);
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "bflow-mlx: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
