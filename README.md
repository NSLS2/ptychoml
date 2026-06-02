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
    auto_detect_roi_offsets,
    compute_sample_pixel_size,
    crop_to_roi,
    detect_dc_at_corner,
    estimate_roi,
    fourier_shift,
    inpaint_bad_pixels,
    mask_hot_pixels,
    normalize_intensity,
    resize_diffraction_patterns,
    zero_pad_to_target,
)
```

Each function's docstring includes a `Source:` line naming the upstream
file/function it was lifted from (holoptycho, ptycho_gui, ptycho-vit,
or HXN h5_conv). Some functions are kept as side-by-side variants; they
will be deduped in a follow-up once call sites are unified.

**GPU support:** the in-place mutating functions (`mask_hot_pixels`,
`apply_intensity_floor`), `crop_to_roi`, `normalize_intensity`, and
`inpaint_bad_pixels` work transparently on `cupy` arrays. Functions
that use `scipy.fft` (`fourier_shift`) remain numpy-only for now.

Functions are grouped into four families so variants can be evaluated
side-by-side. The same grouping is used in
[`ptychoml/preprocess.py`](ptychoml/preprocess.py).

### 1. ROI detection

Find where the signal lives in a frame; these return coordinates and do
not modify the input. Pair them with `crop_to_roi` to actually crop.

| Function | Purpose |
|---|---|
| `auto_detect_roi_offsets(frames, nx, ny)` | Center an `nx × ny` crop on the diffraction-pattern center of mass after masking saturated pixels. Returns `(bx0, by0)`. |
| `estimate_roi(image, threshold=0.1)` | Variant using normalized intensity projections and edge-of-signal detection instead of COM. Returns `(x0, y0, w, h)`. |

### 2. Crop / pad / resize

Change the spatial extent of frames. Three variants by use case.

| Function | Purpose |
|---|---|
| `crop_to_roi(arr, roi)` | Crop the last two axes to a fixed `[[y0, y1], [x0, x1]]` window. Use when the crop region is calibrated and identical for every frame. |
| `zero_pad_to_target(image, target_size)` | Strict centered zero-pad of a 2D image to `target_size × target_size`; raises if input is larger. |
| `resize_diffraction_patterns(dp, target_n)` | Combined per-frame argmax-crop and zero-pad to `target_n × target_n`. Mask hot pixels first if the detector has saturated outliers. |

### 3. Bad-pixel masking, inpainting & threshold cleanup

| Function | Purpose |
|---|---|
| `mask_hot_pixels(arr, threshold, fill=0.0)` | Replace values above `threshold` with `fill`. **Mutates in place** and returns `arr`. |
| `mask_hot_pixels_by_count(arr, count_threshold, kind)` | Photon-count threshold variant; pass `kind='amplitude'` to apply `sqrt(count_threshold)` instead. **Mutates in place.** Used inside `preprocess_diffraction`. |
| `apply_intensity_floor(arr, threshold)` | Zero values strictly below `threshold` (noise-floor cutoff). Symmetric to `mask_hot_pixels`. **Mutates in place.** |
| `inpaint_bad_pixels(arr, coords, radius=1)` | Replace each `(row, col)` in `coords` with the median of a `(2*radius+1)²` neighborhood. **Mutates in place.** Available for holoptycho-style live streaming with known bad-pixel maps; not part of `preprocess_diffraction`. |

### 4. Intensity & geometric transforms

| Function | Purpose |
|---|---|
| `normalize_intensity(arr, normalization, scale=10000.0)` | Scale `arr` by `scale / normalization`. Default `scale` matches ptycho-vit's `config.yaml`; the dataset class default in ptycho-vit (100000.0) is overridden by every HXN config to 10000.0. |
| `detect_dc_at_corner(arr)` | Return True if the central beam currently sits at the corners (i.e. an `fftshift` is needed to land it at the center). Used internally by `preprocess_diffraction` and `PtychoViTInference.predict` for auto DC-convention detection. |
| `fourier_shift(images, shifts)` | Sub-pixel shift each `(H, W)` plane by `shifts[i] = (dy, dx)` via FFT phase-ramp multiplication. |
| `compute_sample_pixel_size(wavelength_m, detector_distance_m, ccd_pixel_size_m, n_pixels)` | Far-field pixel size at the sample plane: `λ z / (N · dx_detector)`. |

### How these map onto holoptycho's pipeline

holoptycho currently runs equivalent inline code rather than importing
from ptychoml — the helpers above were lifted from those inline copies.
The map below shows where each one fits in the live streaming flow, so
you can match a ptychoml function to its real-world call site:

- **Per-frame** (`ImageBatchOp` in `holoptycho/preprocess.py`):
  `crop_to_roi` for the detector window, then `mask_hot_pixels` with a
  detector-specific saturation threshold.
- **Per-batch** (`ImagePreprocessorOp`): `inpaint_bad_pixels` for known
  bad-pixel coordinates, `apply_intensity_floor` for the optional noise
  threshold. The same operator also runs inline `np.rot90`,
  `np.fft.fftshift`, and `np.sqrt` for orientation and intensity →
  amplitude conversion (numpy one-liners; not exposed as ptychoml
  helpers).
- **Inference** (`vit_inference.py` →
  `ptychoml.PtychoViTInference.predict`): the diffraction amplitudes
  produced above are the input to the ViT model.
- **Post-inference** (`holoptycho/mosaic_stitch.py`): `fourier_shift`
  places each predicted ViT patch at its sub-pixel scan position when
  assembling the live mosaic.
- **Replay scripts** (`holoptycho/scripts/replay_from_tiled.py`):
  `auto_detect_roi_offsets` picks a sensible default ROI when no
  user-supplied calibration is available.

The remaining functions in this section
(`resize_diffraction_patterns`, `mask_hot_pixels`,
`compute_sample_pixel_size`, `estimate_roi`, `zero_pad_to_target`,
`normalize_intensity`) come from offline tools (HXN h5_conv,
ptycho_gui, ptycho-vit training data prep) and aren't called by
holoptycho today.

## Run tests

```bash
pixi run test                         # default (gpu) env — installs CUDA + TensorRT
pixi run --environment ci-py312 test  # CPU-only, no GPU/CUDA required
pixi run --environment ci-py313 test  # CPU-only, no GPU/CUDA required
```

The test code itself runs without a GPU or a real `.engine` file (GPU/TRT
paths are gated by `pytest.importorskip`). Note, however, that the
`default` environment still pulls CUDA + TensorRT at install time because
it bundles the `gpu` feature. On a machine without an NVIDIA GPU/CUDA
driver, use the `ci-py312` / `ci-py313` environments to run the suite
without installing the GPU stack.

## Related repos

| Repo | Role |
|---|---|
| [`NSLS2/holoptycho`](https://github.com/NSLS2/holoptycho) | Streaming Holoscan pipeline that uses ptychoml for live inference |
| [`NSLS2/ptycho`](https://github.com/NSLS2/ptycho) | Iterative DM reconstruction kernels |
| `ptycho-vit` | PyTorch training code, produces ONNX files consumed by ptychoml |
