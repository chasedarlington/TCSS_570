#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <utility>
#include <vector>

namespace bflow {

struct EventSoA {
  std::vector<float> x;
  std::vector<float> y;
  std::vector<std::int64_t> t_us;
  std::vector<float> polarity;

  [[nodiscard]] std::size_t size() const noexcept { return t_us.size(); }
  [[nodiscard]] bool empty() const noexcept { return t_us.empty(); }
};

enum class PolarityEncoding {
  signed_values,  // -1 is OFF and +1 is ON.
  dsec,          // 0 is OFF and 1 is ON.
  tum_vie,       // The TUM example labels 0 as ON and 1 as OFF.
};

// Reads x,y,t_us,p from comma- or whitespace-separated records. An optional
// header and lines starting with # are accepted. Returned polarities are ±1.
[[nodiscard]] EventSoA read_events_csv(
    const std::filesystem::path& path,
    PolarityEncoding encoding = PolarityEncoding::signed_values);

// Reads /events/{x,y,t,p} for the half-open time interval [start_us, end_us).
// HDF5 must be enabled at build time, with the file's filters available at run time.
[[nodiscard]] EventSoA read_events_hdf5(
    const std::filesystem::path& path, std::int64_t start_us,
    std::int64_t end_us, PolarityEncoding encoding = PolarityEncoding::tum_vie);

// Source bounds include one additional bin spacing outside the endpoint centers.
[[nodiscard]] std::pair<std::int64_t, std::int64_t> extended_time_window(
    std::int64_t start_us, std::int64_t end_us, int bins);

// Outputs float32 HWC for use as a single NHWC batch. The first and last bin
// centers are start_us and end_us. Only nonzero voxels enter normalization.
[[nodiscard]] std::vector<float> voxelize(
    const EventSoA& events, int width, int height, int bins,
    std::int64_t start_us, std::int64_t end_us, bool normalize = true);

}  // namespace bflow
