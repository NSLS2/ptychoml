# ptychoml Agent Notes

## Project overview

Neural network inference library for ptychography using PtychoViT (Vision
Transformer) models exported to TensorRT. Wraps the preprocessing,
orientation detection, and TRT inference steps that were previously
scattered across `ptycho-vit`, `holoptycho`, and `ptycho_gui`.

## Running tests

The pixi environment is Linux-only (`linux-64`). Do not attempt to add
macOS platforms — the `gpu` environment pulls CUDA packages that have no
macOS builds and will fail to solve.

Run tests on a Linux machine with:

```
pixi run test                        # default env (needs GPU for TRT)
pixi run --environment ci-py312 test # CPU-only, no GPU required
pixi run --environment ci-py313 test # CPU-only, no GPU required
```

All tests in `tests/` are pure numpy — none require a GPU or a real
`.engine` file. GPU/TRT tests do not exist yet.

## Codebase notes

### `ptychoml/preprocess.py`

The composed pipeline entry point is `preprocess_diffraction`. It runs in
this order: hot-pixel mask → normalize+scale → sqrt → D4 → fftshift.

**Known issues flagged in docstrings (not yet fixed):**
- `crop_to_roi`: no bounds checking on ROI indices
- `resize_diffraction_patterns`: per-frame argmax centering can be misled
  by hot pixels; mixed-axis crop+pad can lose edge data
- `auto_detect_roi_offsets` and `crop_to_roi` use opposite axis-order
  conventions (x,y vs y,x) — composing them silently transposes the crop

**Streaming compatibility:**
- `preprocess_diffraction` with `fftshift=None` (auto-detect) must not be
  used mid-scan. The DC convention should be determined once (from the
  first batch or from scan config) and then locked in as `fftshift=True`
  or `fftshift=False` for the rest of the scan.
- `compute_intensity_normalization` requires the full DP stack — not
  streaming-safe. In streaming mode, pass `normalization` from scan config.

**`normalize_intensity` scale parameter:**
- Default `scale` is `10000.0` to match ptycho-vit `config.yaml`. Always
  pass `scale=` explicitly if you are unsure — relying on the default
  silently produces inputs at the wrong magnitude if the default ever changes.

### `ptychoml/orientation.py`

`autodetect_orientation` sweeps all 8 D4 transforms and scores each using
`_score_forward_consistency` (forward-physics NCC). Lower score = better.

**Do not pass `fftshift=None` in `preprocess_kwargs` to `autodetect_orientation`.**
The scorer derives `apply_fftshift` from
`bool(preprocess_kwargs.get('fftshift', False))`. `bool(None) == False`, so
the scorer always assumes no fftshift regardless of what
`preprocess_diffraction` actually applied. This degrades all scores equally
so ranking still works, but absolute scores are wrong. Always pass an
explicit `fftshift=True` or `fftshift=False` when calling
`autodetect_orientation`.

**`autodetect_orientation` is not streaming-safe.** It holds the full
`intensity_batch` in RAM and runs 8× inference passes. Use it once on a
representative subset at the start of a scan, then lock in the winning
`dp_orient` for the rest.

### `ptychoml/trt.py`

Default TRT workspace size is 2 GiB. If engine build fails with a
serialization error, increase `--workspace-size` (e.g. `4294967296` for
4 GiB).
