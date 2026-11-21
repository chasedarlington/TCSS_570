cmake_minimum_required(VERSION 3.25)

project(bflow-mlx VERSION 0.1.0 LANGUAGES C CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$" OR
   (CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64"))
  message(FATAL_ERROR "bflow-mlx requires native arm64 build tools on Apple silicon macOS.")
endif()

option(BFLOW_FETCH_MLX "Build the pinned native MLX if it is not installed" ON)
option(BFLOW_WITH_HDF5 "Read TUM-VIE event HDF5 files directly" ON)

if(BFLOW_WITH_HDF5)
  find_package(HDF5 REQUIRED COMPONENTS C)
endif()

find_package(MLX 0.32.4 CONFIG QUIET)
if(TARGET mlx)
  if(NOT MLX_BUILD_METAL)
    message(FATAL_ERROR "The installed MLX package needs the Metal backend.")
  endif()
  set(BFLOW_METALLIB "${MLX_LIBRARY_DIRS}/mlx.metallib")
  if(NOT EXISTS "${BFLOW_METALLIB}")
    message(FATAL_ERROR "The installed MLX package is missing ${BFLOW_METALLIB}.")
  endif()
elseif(BFLOW_FETCH_MLX)
  include(FetchContent)
  set(MLX_BUILD_METAL ON)
  set(MLX_BUILD_PYTHON_BINDINGS OFF)
  set(MLX_BUILD_TESTS OFF)
  set(MLX_BUILD_EXAMPLES OFF)
  set(MLX_BUILD_GGUF OFF)
  set(MLX_BUILD_SAFETENSORS ON)
  set(MLX_METAL_PATH "${CMAKE_CURRENT_BINARY_DIR}/native-mlx-metal")
  file(MAKE_DIRECTORY "${MLX_METAL_PATH}")
  FetchContent_Declare(mlx
    GIT_REPOSITORY https://github.com/ml-explore/mlx.git
    GIT_TAG v0.32.4
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(mlx)
  set(BFLOW_METALLIB "${MLX_METAL_PATH}/mlx.metallib")
else()
  message(FATAL_ERROR "Install MLX 0.32.4+ with Metal or enable BFLOW_FETCH_MLX.")
endif()

find_library(BFLOW_METAL_FRAMEWORK Metal REQUIRED)

add_executable(bflow-mlx
  src/main.cpp
  src/voxelizer.cpp
  src/layers.cpp
  src/bflow_net.cpp
)
target_compile_features(bflow-mlx PRIVATE cxx_std_20)
set_target_properties(bflow-mlx PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(bflow-mlx PRIVATE include)
# MLX exports the CMake target `mlx` for its mlx::core C++ API.
target_link_libraries(bflow-mlx PRIVATE mlx "${BFLOW_METAL_FRAMEWORK}")

if(BFLOW_WITH_HDF5)
  target_compile_definitions(bflow-mlx PRIVATE BFLOW_WITH_HDF5=1)
  target_link_libraries(bflow-mlx PRIVATE HDF5::HDF5)
endif()

add_custom_command(TARGET bflow-mlx POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${BFLOW_METALLIB}" "$<TARGET_FILE_DIR:bflow-mlx>/mlx.metallib"
  VERBATIM
)
