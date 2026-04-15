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
