# bflow-mlx

Native Apple-silicon inference for the event-only UZH RPG BFlow `E_LU5_BD10_lowpyramid` checkpoint. The C++ executable uses MLX and Metal. Python and PyTorch are used for offline checkpoint export and reference generation.

## Build on Apple silicon

Use arm64 macOS 14 or later, Xcode with its Metal toolchain, and CMake 3.25 or later. Install native HDF5 for direct TUM-VIE ingestion.

```sh
brew install cmake hdf5
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

CMake uses an installed Metal-enabled MLX package when available or builds the pinned MLX v0.32.4 with Python bindings disabled. Keep `mlx.metallib` beside the executable. For NumPy, safetensors, and CSV input only, configure with `-DBFLOW_WITH_HDF5=OFF`.

## Export the checkpoint and check parity

Run the exporter once using Python 3.11 or later. The resulting safetensors files can be copied to the inference machine. The example uses a deterministic 128 × 128 input for 12 iterations.

```sh
python3 -m venv .export-env
source .export-env/bin/activate
python -m pip install 'torch>=2.6' numpy scipy numba omegaconf safetensors
git clone https://github.com/uzh-rpg/bflow.git upstream-bflow
git -C upstream-bflow checkout 7149d2d4e36c26f15c5bee014c8a98fa0ad8cfd4
curl -fL https://download.ifi.uzh.ch/rpg/bflow/checkpoints/multiflow/E_LU5_BD10.ckpt \
  -o E_LU5_BD10.ckpt
python scripts/export_and_verify.py \
  --upstream upstream-bflow --checkpoint E_LU5_BD10.ckpt \
  --weights bflow.safetensors --reference parity.safetensors
deactivate

./build/bflow-mlx --weights bflow.safetensors --parity parity.safetensors
```

The exporter checks the context encoder’s BatchNorm folding against the original PyTorch output. Convolution kernels are stored in MLX’s `[out, kernel_h, kernel_w, in]` order. The feature encoder’s InstanceNorm uses per-image spatial statistics at inference, so its normalization runs in MLX. The native parity command compares final control points against the PyTorch fixture and exits with an error if any value exceeds the tolerance. Use `--voxels input.npy` at export to generate a fixture from your own float32 `[B,H,W,65]` voxel tensor.

## Run inference

Use voxel arrays or a single TUM-VIE event camera stream. `--time-us` sets the reference time on that stream’s published microsecond clock.

The released `mocap-6dof` event files use HDF5 filter 32001 (Blosc with Zstandard). If the filter is unavailable, build the native plugin and point HDF5 at its build directory.

```sh
git clone https://github.com/Blosc/hdf5-blosc.git
cmake -S hdf5-blosc -B build-blosc -DBUILD_TESTS=OFF \
  -DCMAKE_PREFIX_PATH="$(brew --prefix hdf5)"
cmake --build build-blosc --parallel
export HDF5_PLUGIN_PATH="$PWD/build-blosc"
```

```sh
./build/bflow-mlx --weights bflow.safetensors --input voxels.npy \
  --reference control_points.npy --output predicted.npy

./build/bflow-mlx --weights bflow.safetensors \
  --events mocap-6dof-events_left.h5 --time-us 2000000 \
  --output predicted.safetensors
```

Omit `--reference` for inference without a NumPy parity check. CSV event input uses `x,y,t_us,p` rows with signed polarity `-1` or `+1`. The HDF5 polarity default follows the TUM example’s labels. Use `--help` for explicit encoding, image dimensions, iteration count, and tolerance options.

The raw path keeps events in sensor coordinates. It constructs one normalized 65-bin NHWC volume with 12,500 µs between centres, from `t_ref−300000` through `t_ref+500000`, with an extra bin interval read at each boundary for interpolation. Channels `0:25` encode the reference. The five correlation targets start at channels 8, 16, 24, 32, and 40, while the context encoder takes channels `24:65`.

Output is float32 `[B,H,W,20]` in full-resolution pixel units, with channels `x1…x10,y1…y10`. The initial control point is implicit and zero. The model returns a lazy graph and the CLI evaluates it once per inference. Inputs must have sides divisible by 8 and at least 128 pixels. At 1280 × 720 and batch size one, the float32 all-pairs correlation pyramids alone require about 4.42 GB before activations and other working memory. Allow additional unified memory for full-resolution inference.

The native model is a port of the [UZH RPG BFlow implementation](https://github.com/uzh-rpg/bflow), distributed under [Apache 2.0](LICENSE).
