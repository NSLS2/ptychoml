# ptychoml

Neural network inference for ptychography. Runs PtychoViT models via TensorRT.

## About PtychoViT

PtychoViT is a Vision Transformer ([ViT](https://arxiv.org/abs/2010.11929)) adapted for ptychographic reconstruction. It takes a batch of diffraction patterns and directly predicts amplitude and phase estimates — orders of magnitude faster than iterative (DM, ML) methods, enabling real-time feedback during live scans.

The model is developed at Argonne National Laboratory (ANL). Training code lives in the `ptycho-vit` repo (private, maintained at ANL).

`ptychoml` handles the **inference** side only — taking a trained model exported to ONNX, converting it to a TensorRT engine, and running fast batched inference on a GPU.

## Architecture

**What this repo is**: a pure computation library for ML-based ptychographic reconstruction. Loads pre-built TensorRT engines and runs inference on diffraction patterns.

**What this repo is not**:
- Pipeline orchestration → see [`NSLS2/holoptycho`](https://github.com/NSLS2/holoptycho)
- Iterative (DM) reconstruction → see [`NSLS2/ptycho`](https://github.com/NSLS2/ptycho)
- Model training → see `ptycho-vit` (PyTorch training code maintained by ANL)

**Design principle**: no I/O, no framework deps (Holoscan, MPI, etc.). Return data to the caller; let the caller decide where it goes.

## Install

```bash
git clone git@github.com:NSLS2/ptychoml.git
cd ptychoml
pixi install
```

Requires an NVIDIA GPU with CUDA 12 driver and [pixi](https://pixi.sh).

## Usage

**Python API:**

```python
from ptychoml import PtychoViTInference

with PtychoViTInference(engine_path="model.engine", gpu=0) as session:
    pred, indices = session.predict(diff_amp, image_indices)
    # pred.shape == (B, 2, H, W) or (B, H, W)
```

**Build a TensorRT engine from ONNX:**

```bash
pixi run build-engine --onnx model.onnx --output model.engine
# or
ptychoml-build-engine --onnx model.onnx --output model.engine
```

```python
from ptychoml import build_engine, save_engine

engine = build_engine("model.onnx", fp16=False, tf32=True)
save_engine(engine, "model.engine")
```

**Run inference on an HDF5 dataset:**

```bash
pixi run predict --engine model.engine --data scan_1234.h5 --output results.h5
# or
ptychoml-predict --engine model.engine --data scan_1234.h5 --output results.h5
```

By default, diffraction amplitudes are read from the `diffamp` dataset key (matching the format used by [holoptycho](https://github.com/NSLS2/holoptycho)). Use `--dataset` to specify a different key:

```bash
pixi run predict --engine model.engine --data scan.h5 --output results.h5 --dataset entry/data/data
```

Additional options:

| Flag | Description |
|---|---|
| `--gpu N` | CUDA device ordinal (default: 0) |
| `--shifted` | Set if input data has been fftshift'd |
| `--dataset KEY` | HDF5 dataset key for diffraction amplitudes (default: `diffamp`) |

The output HDF5 file contains a `predictions` dataset with shape `(N, 2, H, W)` or `(N, H, W)` depending on the model. If the input file has a `points` dataset (scan positions), it is copied through to the output.

## Preprocessing utilities

Array-in / array-out helpers for preparing diffraction data and reconstructions before inference. Importable from the top-level package:

```python
from ptychoml import (
    apply_intensity_floor,
    adjust_object_for_pad,
    compute_sample_pixel_size,
    crop_to_roi,
    inpaint_bad_pixels,
    mask_hot_pixels,
    resize_diffraction_patterns,
)
```

| Function | Purpose |
|---|---|
| `crop_to_roi(arr, roi)` | Crop the last two axes to a fixed `[[y0, y1], [x0, x1]]` window. Use when the crop region is calibrated and identical for every frame. |
| `resize_diffraction_patterns(dp, target_n)` | Crop each pattern around its per-frame argmax or zero-pad to `target_n × target_n`. Mask hot pixels first if the detector has saturated outliers. |
| `adjust_object_for_pad(obj, scale_y, scale_x, obj_pad)` | Trim or zero-pad an object's last two axes by `obj_pad * (scale - 1)` after a pixel-grid rescale, to match a backend's fixed padding allocation. |
| `mask_hot_pixels(arr, threshold, fill=0.0)` | Replace values above `threshold` with `fill` (saturated/dead-pixel masking). Returns a copy. |
| `inpaint_bad_pixels(arr, coords, radius=1)` | Replace each `(row, col)` in `coords` with the median of a `(2*radius+1)²` neighborhood. Operates on the last two axes. Returns a copy. |
| `apply_intensity_floor(arr, threshold)` | Zero values strictly below `threshold` (noise-floor cutoff). Returns a copy. |
| `compute_sample_pixel_size(wavelength_m, detector_distance_m, ccd_pixel_size_m, n_pixels)` | Far-field pixel size at the sample plane: `λ z / (N · dx_detector)`. |

## Run tests

```bash
pixi run test
```

Tests run without GPU/TRT (via `pytest.importorskip` gates).

## Related repos

| Repo | Role |
|---|---|
| [`NSLS2/holoptycho`](https://github.com/NSLS2/holoptycho) | Streaming Holoscan pipeline that uses ptychoml for live inference |
| [`NSLS2/ptycho`](https://github.com/NSLS2/ptycho) | Iterative DM reconstruction kernels |
| `ptycho-vit` | PyTorch training code, produces ONNX files consumed by ptychoml |
