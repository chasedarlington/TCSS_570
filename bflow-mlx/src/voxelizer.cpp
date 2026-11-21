#include "bflow/voxelizer.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#ifdef BFLOW_WITH_HDF5
#include <hdf5.h>
#endif

namespace bflow {
namespace {

float signed_polarity(float value, PolarityEncoding encoding) {
  switch (encoding) {
    case PolarityEncoding::signed_values:
      if (value == -1.0F || value == 1.0F) {
        return value;
      }
      throw std::invalid_argument("Signed polarity must be -1 or +1");
    case PolarityEncoding::dsec:
      if (value == 0.0F || value == 1.0F) {
        return 2.0F * value - 1.0F;
      }
      break;
    case PolarityEncoding::tum_vie:
      if (value == 0.0F || value == 1.0F) {
        return 1.0F - 2.0F * value;
      }
      break;
  }
  throw std::invalid_argument("Binary polarity must be 0 or 1");
}

// Unsigned subtraction gives the exact distance between ordered int64_t
// timestamps, including intervals that straddle zero.
std::uint64_t time_distance(std::int64_t end, std::int64_t start) {
  return static_cast<std::uint64_t>(end) - static_cast<std::uint64_t>(start);
}

std::size_t checked_product(std::size_t first, std::size_t second) {
  if (first > std::numeric_limits<std::size_t>::max() / second) {
    throw std::length_error("Voxel grid shape exceeds addressable memory");
  }
  return first * second;
}

void normalize_nonzero(std::vector<float>& grid) {
  double sum = 0.0;
  std::size_t count = 0;
  for (float value : grid) {
    if (value != 0.0F) {
      sum += value;
      ++count;
    }
  }
  if (count == 0) {
    return;
  }
  const double mean = sum / static_cast<double>(count);
  double squared_deviations = 0.0;
  for (float value : grid) {
    if (value != 0.0F) {
      const double delta = static_cast<double>(value) - mean;
      squared_deviations += delta * delta;
    }
  }
  // PyTorch's std uses the sample denominator. The upstream fallback
  // subtracts the mean when the standard deviation is zero or undefined.
  const double scale =
      count > 1 && squared_deviations > 0.0
          ? std::sqrt(squared_deviations / static_cast<double>(count - 1))
          : 1.0;
  for (float& value : grid) {
    if (value != 0.0F) {
      value = static_cast<float>((static_cast<double>(value) - mean) / scale);
    }
  }
}

#ifdef BFLOW_WITH_HDF5

template <herr_t (*Close)(hid_t)>
class HdfHandle {
 public:
  HdfHandle(hid_t value, std::string_view description) : value_(value) {
    if (value_ < 0) {
      throw std::runtime_error("HDF5 could not open " + std::string(description));
    }
  }
  HdfHandle(const HdfHandle&) = delete;
  HdfHandle& operator=(const HdfHandle&) = delete;
  ~HdfHandle() { Close(value_); }

  [[nodiscard]] hid_t get() const noexcept { return value_; }

 private:
  hid_t value_;
};

using HdfFile = HdfHandle<H5Fclose>;
using HdfDataset = HdfHandle<H5Dclose>;
using HdfSpace = HdfHandle<H5Sclose>;
using HdfType = HdfHandle<H5Tclose>;

hsize_t dataset_length(hid_t dataset, std::string_view name) {
  const HdfSpace space(H5Dget_space(dataset), name);
  hsize_t length = 0;
  if (H5Sget_simple_extent_ndims(space.get()) != 1 ||
      H5Sget_simple_extent_dims(space.get(), &length, nullptr) < 0) {
    throw std::runtime_error("Expected a one-dimensional HDF5 dataset at " +
                             std::string(name));
  }
  return length;
}

void read_slice(hid_t dataset, hid_t memory_type, hsize_t start, hsize_t count,
                void* output, std::string_view name) {
  const HdfSpace file_space(H5Dget_space(dataset), name);
  const HdfSpace memory_space(H5Screate_simple(1, &count, nullptr), name);
  if (H5Sselect_hyperslab(file_space.get(), H5S_SELECT_SET, &start, nullptr,
                          &count, nullptr) < 0) {
    throw std::runtime_error("Could not select the HDF5 slice at " +
                             std::string(name));
  }
  herr_t status = -1;
  H5E_BEGIN_TRY {
    status = H5Dread(dataset, memory_type, memory_space.get(), file_space.get(),
                     H5P_DEFAULT, output);
  } H5E_END_TRY;
  if (status < 0) {
    throw std::runtime_error(
        "Could not read HDF5 dataset " + std::string(name) +
        ". For compressed TUM-VIE files, set HDF5_PLUGIN_PATH to the "
        "directory containing the Blosc HDF5 filter.");
  }
}

std::int64_t read_time_offset(hid_t file) {
  const htri_t exists = H5Lexists(file, "/t_offset", H5P_DEFAULT);
  if (exists < 0) {
    throw std::runtime_error("Could not check HDF5 /t_offset");
  }
  if (exists == 0) {
    return 0;
  }
  const HdfDataset dataset(H5Dopen2(file, "/t_offset", H5P_DEFAULT), "/t_offset");
  const HdfSpace space(H5Dget_space(dataset.get()), "/t_offset");
  std::int64_t offset = 0;
  if (H5Sget_simple_extent_npoints(space.get()) != 1 ||
      H5Dread(dataset.get(), H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              &offset) < 0) {
    throw std::runtime_error("Expected a single timestamp at HDF5 /t_offset");
  }
  return offset;
}

std::int64_t apply_time_offset(std::int64_t time, std::int64_t offset) {
  if ((offset > 0 && time > std::numeric_limits<std::int64_t>::max() - offset) ||
      (offset < 0 && time < std::numeric_limits<std::int64_t>::min() - offset)) {
    throw std::overflow_error("HDF5 timestamp plus t_offset exceeds int64");
  }
  return time + offset;
}

hsize_t timestamp_lower_bound(hid_t timestamps, hsize_t left, hsize_t right,
                              std::int64_t target, std::int64_t offset) {
  while (left < right) {
    const hsize_t middle = left + (right - left) / 2;
    std::int64_t time = 0;
    read_slice(timestamps, H5T_NATIVE_INT64, middle, 1, &time, "/events/t");
    if (apply_time_offset(time, offset) < target) {
      left = middle + 1;
    } else {
      right = middle;
    }
  }
  return left;
}

#endif

}  // namespace

EventSoA read_events_csv(const std::filesystem::path& path,
                         PolarityEncoding encoding) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Could not open event file " + path.string());
  }

  EventSoA events;
  std::istringstream fields;
  fields.imbue(std::locale::classic());
  std::string line;
  std::size_t line_number = 0;
  bool allow_header = true;
  while (std::getline(input, line)) {
    ++line_number;
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') {
      continue;
    }
    for (char& character : line) {
      if (character == ',') {
        character = ' ';
      }
    }
    fields.clear();
    fields.str(line);
    if (allow_header && line[first] == 'x') {
      std::string x, y, time, polarity, trailing;
      if (fields >> x >> y >> time >> polarity && !(fields >> trailing) &&
          x == "x" && y == "y" &&
          (time == "t_us" || time == "t") &&
          (polarity == "p" || polarity == "polarity")) {
        allow_header = false;
        continue;
      }
      fields.clear();
      fields.seekg(0);
    }
    allow_header = false;
    float x = 0.0F;
    float y = 0.0F;
    float polarity = 0.0F;
    std::int64_t time = 0;
    std::string trailing;
    if (!(fields >> x >> y >> time >> polarity) || fields >> trailing ||
        !std::isfinite(x) || !std::isfinite(y)) {
      throw std::runtime_error("Expected x,y,t_us,p in " + path.string() +
                               " at line " + std::to_string(line_number));
    }
    try {
      polarity = signed_polarity(polarity, encoding);
    } catch (const std::invalid_argument& error) {
      throw std::runtime_error(std::string(error.what()) + " in " +
                               path.string() + " at line " +
                               std::to_string(line_number));
    }
    events.x.push_back(x);
    events.y.push_back(y);
    events.t_us.push_back(time);
    events.polarity.push_back(polarity);
  }
  if (input.bad()) {
    throw std::runtime_error("Error reading event file " + path.string());
  }
  return events;
}

EventSoA read_events_hdf5(const std::filesystem::path& path,
                          std::int64_t start_us, std::int64_t end_us,
                          PolarityEncoding encoding) {
#ifdef BFLOW_WITH_HDF5
  if (end_us <= start_us) {
    throw std::invalid_argument("HDF5 event interval must have positive duration");
  }
  const HdfFile file(H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT),
                     path.string());
  const HdfDataset x(H5Dopen2(file.get(), "/events/x", H5P_DEFAULT), "/events/x");
  const HdfDataset y(H5Dopen2(file.get(), "/events/y", H5P_DEFAULT), "/events/y");
  const HdfDataset t(H5Dopen2(file.get(), "/events/t", H5P_DEFAULT), "/events/t");
  const HdfDataset p(H5Dopen2(file.get(), "/events/p", H5P_DEFAULT), "/events/p");
  const hsize_t length = dataset_length(t.get(), "/events/t");
  if (dataset_length(x.get(), "/events/x") != length ||
      dataset_length(y.get(), "/events/y") != length ||
      dataset_length(p.get(), "/events/p") != length) {
    throw std::runtime_error("HDF5 /events columns have unequal lengths");
  }
  const auto offset = read_time_offset(file.get());
  const hsize_t first =
      timestamp_lower_bound(t.get(), 0, length, start_us, offset);
  const hsize_t last =
      timestamp_lower_bound(t.get(), first, length, end_us, offset);
  const hsize_t count = last - first;
  EventSoA events;
  if (count == 0) {
    return events;
  }
  if (count > std::numeric_limits<std::size_t>::max()) {
    throw std::length_error("HDF5 event slice exceeds addressable memory");
  }
  events.x.resize(count);
  events.y.resize(count);
  events.t_us.resize(count);
  events.polarity.resize(count);
  read_slice(x.get(), H5T_NATIVE_FLOAT, first, count, events.x.data(), "/events/x");
  read_slice(y.get(), H5T_NATIVE_FLOAT, first, count, events.y.data(), "/events/y");
  read_slice(t.get(), H5T_NATIVE_INT64, first, count, events.t_us.data(),
             "/events/t");
  const HdfType polarity_type(H5Dget_type(p.get()), "/events/p");
  if (H5Tget_class(polarity_type.get()) == H5T_ENUM) {
    std::vector<std::int8_t> values(count);
    read_slice(p.get(), H5T_NATIVE_INT8, first, count, values.data(), "/events/p");
    for (std::size_t i = 0; i < events.size(); ++i) {
      events.polarity[i] = signed_polarity(values[i], encoding);
    }
  } else {
    read_slice(p.get(), H5T_NATIVE_FLOAT, first, count, events.polarity.data(),
               "/events/p");
    for (float& value : events.polarity) {
      value = signed_polarity(value, encoding);
    }
  }
  if (offset != 0) {
    for (auto& time : events.t_us) {
      time = apply_time_offset(time, offset);
    }
  }
  return events;
#else
  (void)path;
  (void)start_us;
  (void)end_us;
  (void)encoding;
  throw std::runtime_error("This build has no HDF5 support. Enable HDF5 or "
                           "use the x,y,t_us,p CSV reader.");
#endif
}

std::pair<std::int64_t, std::int64_t> extended_time_window(
    std::int64_t start_us, std::int64_t end_us, int bins) {
  if (bins < 2 || end_us <= start_us) {
    throw std::invalid_argument("A voxel window needs at least two bins and "
                                "a positive duration");
  }
  const std::uint64_t duration = time_distance(end_us, start_us);
  const auto intervals = static_cast<std::uint64_t>(bins - 1);
  const std::uint64_t padding = duration / intervals + (duration % intervals != 0);
  constexpr auto minimum = std::numeric_limits<std::int64_t>::min();
  constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
  if (padding > time_distance(start_us, minimum) ||
      padding > time_distance(maximum, end_us)) {
    throw std::overflow_error("Extended voxel window exceeds int64 timestamps");
  }
  return {
      static_cast<std::int64_t>(static_cast<std::uint64_t>(start_us) - padding),
      static_cast<std::int64_t>(static_cast<std::uint64_t>(end_us) + padding),
  };
}

std::vector<float> voxelize(const EventSoA& events, int width, int height,
                            int bins, std::int64_t start_us, std::int64_t end_us,
                            bool normalize) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("Voxel dimensions must be positive");
  }
  if (events.x.size() != events.size() || events.y.size() != events.size() ||
      events.polarity.size() != events.size()) {
    throw std::invalid_argument("Event columns have unequal lengths");
  }
  const auto [first, last] = extended_time_window(start_us, end_us, bins);
  const float duration = static_cast<float>(time_distance(end_us, start_us));
  const auto channel_count = static_cast<std::size_t>(bins);
  const auto image_width = static_cast<std::size_t>(width);
  const auto area = checked_product(image_width, static_cast<std::size_t>(height));
  std::vector<float> grid(checked_product(area, channel_count), 0.0F);

  for (std::size_t i = 0; i < events.size(); ++i) {
    const auto time = events.t_us[i];
    const float x = events.x[i];
    const float y = events.y[i];
    if (time < first || time >= last ||
        !(x > -1.0F && x < static_cast<float>(width) &&
          y > -1.0F && y < static_cast<float>(height))) {
      continue;
    }
    const float polarity = signed_polarity(events.polarity[i],
                                            PolarityEncoding::signed_values);
    const float relative_time =
        time >= start_us ? static_cast<float>(time_distance(time, start_us))
                         : -static_cast<float>(time_distance(start_us, time));
    const float bin_time =
        (relative_time / duration) * static_cast<float>(bins - 1);
    const auto base_bin = static_cast<std::int64_t>(std::floor(bin_time));
    const int base_x = static_cast<int>(std::floor(x));
    const int base_y = static_cast<int>(std::floor(y));
    const float bin_weight = bin_time - static_cast<float>(base_bin);
    const float x_weight = x - static_cast<float>(base_x);
    const float y_weight = y - static_cast<float>(base_y);
    const int x_neighbors = x_weight == 0.0F ? 1 : 2;
    const int y_neighbors = y_weight == 0.0F ? 1 : 2;
    for (int dx = 0; dx < x_neighbors; ++dx) {
      const int pixel_x = base_x + dx;
      if (pixel_x < 0 || pixel_x >= width) {
        continue;
      }
      const float weight_x = dx == 0 ? 1.0F - x_weight : x_weight;
      for (int dy = 0; dy < y_neighbors; ++dy) {
        const int pixel_y = base_y + dy;
        if (pixel_y < 0 || pixel_y >= height) {
          continue;
        }
        const float weight_y = dy == 0 ? 1.0F - y_weight : y_weight;
        const float spatial_weight = polarity * weight_x * weight_y;
        const auto pixel = (static_cast<std::size_t>(pixel_y) * image_width +
                            static_cast<std::size_t>(pixel_x)) * channel_count;
        if (base_bin >= 0 && base_bin < bins) {
          grid[pixel + static_cast<std::size_t>(base_bin)] +=
              spatial_weight * (1.0F - bin_weight);
        }
        if (bin_weight != 0.0F && base_bin >= -1 && base_bin < bins - 1) {
          grid[pixel + static_cast<std::size_t>(base_bin + 1)] +=
              spatial_weight * bin_weight;
        }
      }
    }
  }
  if (normalize) {
    normalize_nonzero(grid);
  }
  return grid;
}

}  // namespace bflow
