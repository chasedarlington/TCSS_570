#!/usr/bin/env python3
"""Export the official BFlow MultiFlow events-only model and a PyTorch reference.

Install the offline dependencies with:
    python -m pip install torch numpy scipy numba omegaconf safetensors

Pass a checkout of https://github.com/uzh-rpg/bflow at
7149d2d4e36c26f15c5bee014c8a98fa0ad8cfd4 to --upstream, and obtain the
checkpoint at https://download.ifi.uzh.ch/rpg/bflow/checkpoints/multiflow/E_LU5_BD10.ckpt.
"""

import argparse
from collections.abc import Mapping
import hashlib
from pathlib import Path
import pickle
import sys


MODEL_NAME = "E_LU5_BD10_lowpyramid"
OFFICIAL_SHA256 = "51650b2ea3910dd5b2bd05768882b99fd2d9288f8247a1d22a997fb3acb7c649"
CONTEXT_BINS = 41
CORRELATION_BINS = 25
INPUT_BINS = CONTEXT_BINS + CORRELATION_BINS - 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--upstream", required=True, type=Path, help="UZH BFlow source checkout")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Official .ckpt or tensor state_dict")
    parser.add_argument("--weights", required=True, type=Path, help="Output OHWI weights .safetensors")
    parser.add_argument("--reference", required=True, type=Path, help="Output NHWC parity .safetensors")
    parser.add_argument("--voxels", type=Path, help="Optional float32 NHWC input .npy, shape [B,H,W,65]")
    parser.add_argument("--iterations", type=int, default=12, help="PyTorch recurrence iterations (default 12)")
    parser.add_argument("--height", type=int, default=128, help="Synthetic fixture height (default 128)")
    parser.add_argument("--width", type=int, default=128, help="Synthetic fixture width (default 128)")
    parser.add_argument("--seed", type=int, default=0, help="Synthetic fixture seed")
    args = parser.parse_args()
    if args.iterations < 1 or args.iterations > 2**31 - 1:
        parser.error("--iterations must be positive and fit in int32")
    if args.weights.suffix != ".safetensors" or args.reference.suffix != ".safetensors":
        parser.error("--weights and --reference must end in .safetensors")
    if args.weights.resolve() == args.reference.resolve():
        parser.error("--weights and --reference must be different files")
    input_paths = {args.checkpoint.resolve()}
    if args.voxels is not None:
        input_paths.add(args.voxels.resolve())
    if input_paths & {args.weights.resolve(), args.reference.resolve()}:
        parser.error("output files must be different from the inputs")
    return args


def upstream_config():
    # Resolved model config for the upstream MultiFlow events-only experiment.
    return {
        "num_bins": {"context": CONTEXT_BINS, "correlation": CORRELATION_BINS},
        "bezier_degree": 10,
        "detach_bezier": False,
        "use_boundary_images": False,
        "use_events": True,
        "correlation": {
            "use_cosine_sim": False,
            "ev": {
                "target_indices": [8, 16, 24, 32, 40],
                "levels": [1, 1, 1, 1, 4],
                "radius": [4, 4, 4, 4, 4],
            },
        },
        "hidden": {"dim": 128},
        "context": {"dim": 128, "norm": "batch"},
        "feature": {"dim": 256, "norm": "instance"},
        "motion": {"dim": 128},
    }


def read_state_dict(torch, path):
    # The released archive includes pickled optimizer/scheduler state. Permit
    # the full loader only for the byte-for-byte known official checkpoint.
    with path.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        try:
            archive = torch.load(stream, map_location="cpu", weights_only=sha256 != OFFICIAL_SHA256)
        except pickle.UnpicklingError as exc:
            raise ValueError(
                "Checkpoint cannot be read safely. Use the official release or a plain tensor state_dict."
            ) from exc

    if not isinstance(archive, Mapping):
        raise ValueError("checkpoint must contain a state dictionary")
    state = archive.get("state_dict", archive)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint state_dict is missing or empty")
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): tensor for key, tensor in state.items()}
    if any(key.startswith("net.") for key in state):
        state = {key.removeprefix("net."): tensor for key, tensor in state.items() if key.startswith("net.")}
    return state, sha256


def fixture_shape_is_valid(height, width):
    # At level 3 the upstream align_corners sampler needs H,W >= 2.
    return all(size >= 128 and size % 8 == 0 for size in (height, width))


def make_fixture(np, args):
    if args.voxels is None:
        if not fixture_shape_is_valid(args.height, args.width):
            raise ValueError("synthetic height and width must be multiples of 8 and at least 128")
        rng = np.random.default_rng(args.seed)
        shape = (1, args.height, args.width, INPUT_BINS)
        voxels = rng.standard_normal(shape, dtype=np.float32)
        voxels[rng.random(shape, dtype=np.float32) >= 0.1] = 0
    else:
        voxels = np.load(args.voxels, allow_pickle=False)
        if not isinstance(voxels, np.ndarray):
            raise ValueError("--voxels must be a single .npy array")
    if (
        voxels.dtype != np.float32
        or voxels.ndim != 4
        or voxels.shape[0] < 1
        or voxels.shape[-1] != INPUT_BINS
        or not fixture_shape_is_valid(*voxels.shape[1:3])
        or not np.isfinite(voxels).all()
    ):
        raise ValueError("input must be finite float32 [B,H,W,65], with H and W multiples of 8 and at least 128")
    return np.ascontiguousarray(voxels)


def fold_context_batch_norm(nn, model):
    def replace(parent, conv_name, norm_name):
        conv = getattr(parent, conv_name)
        norm = getattr(parent, norm_name)
        if not isinstance(conv, nn.Conv2d) or not isinstance(norm, nn.BatchNorm2d):
            raise ValueError("upstream context encoder architecture does not match the checkpoint")
        setattr(parent, conv_name, nn.utils.fuse_conv_bn_eval(conv, norm))
        setattr(parent, norm_name, nn.Identity())

    encoder = model.cnet
    replace(encoder, "conv1", "norm1")
    for layer in (encoder.layer1, encoder.layer2, encoder.layer3):
        for block in layer:
            replace(block, "conv1", "norm1")
            replace(block, "conv2", "norm2")
            if block.downsample is not None:
                replace(block.downsample, "0", "1")
                # The upstream residual block also registers this same norm as norm3.
                block.norm3 = block.downsample[1]

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            raise ValueError("unfolded BatchNorm found in the upstream model")
        if isinstance(module, nn.InstanceNorm2d) and (module.affine or module.track_running_stats):
            raise ValueError("the native model expects non-affine InstanceNorm using input statistics")


def export_convolutions(torch, model):
    weights = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        # PyTorch filters are [out,in,kh,kw]. MLX expects [out,kh,kw,in].
        weights[name + ".weight"] = module.weight.detach().float().permute(0, 2, 3, 1).contiguous().numpy()
        if module.bias is not None:
            weights[name + ".bias"] = module.bias.detach().float().contiguous().numpy()
    if set(weights) != set(model.state_dict()):
        raise ValueError("upstream contains tensors without a native export rule")
    return weights


def predict(model, inputs, iterations):
    low, full = model(voxel_grid=inputs, iters=iterations, test_mode=True)
    return low.get_params(), full.get_params()


def main():
    args = parse_args()
    import numpy as np
    import torch
    from safetensors.numpy import save_file

    upstream = args.upstream.resolve()
    if not (upstream / "models/raft_spline/raft.py").is_file():
        raise ValueError("--upstream must point to a BFlow source checkout")
    sys.path.insert(0, str(upstream))
    from models.raft_spline.raft import RAFTSpline

    state, sha256 = read_state_dict(torch, args.checkpoint)
    model = RAFTSpline(upstream_config()).float().eval()
    model.load_state_dict(state, strict=True)
    voxels = make_fixture(np, args)
    inputs = torch.from_numpy(voxels).permute(0, 3, 1, 2).contiguous()
    with torch.inference_mode():
        reference = predict(model, inputs, args.iterations)
        fold_context_batch_norm(torch.nn, model)
        folded = predict(model, inputs, args.iterations)
        for expected, actual in zip(reference, folded):
            if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
                raise ValueError("PyTorch reference or folded model returned nonfinite control points")
            torch.testing.assert_close(actual, expected, atol=1e-3, rtol=5e-4)

    low, full = (tensor.permute(0, 2, 3, 1).contiguous().numpy() for tensor in reference)
    ref_tensors = {
        "voxel_grid": voxels,
        "control_points_low": low,
        "control_points": full,
        "iterations": np.array([args.iterations], dtype=np.int32),
    }
    weights = export_convolutions(torch, model)
    for path in (args.weights, args.reference):
        path.parent.mkdir(parents=True, exist_ok=True)
    common = {"model": MODEL_NAME, "source_sha256": sha256}
    save_file(weights, args.weights, metadata={
        **common,
        "format": "bflow-mlx-weights-v1",
        "conv_layout": "OHWI",
        "context_batch_norm": "folded",
        "feature_instance_norm": "input_statistics",
    })
    save_file(ref_tensors, args.reference, metadata={
        **common,
        "format": "bflow-mlx-parity-v1",
        "tensor_layout": "NHWC",
    })
    print(f"Verified folded PyTorch parity for {args.iterations} iterations on {list(voxels.shape)}.")
    print(f"Exported {len(weights)} weight tensors to {args.weights}.")
    print(f"Wrote the original PyTorch control points to {args.reference}.")


if __name__ == "__main__":
    main()
