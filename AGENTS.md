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
  or `fftshift=False` for the rest of the scan. This is documented in the
  `fftshift` parameter docstring.
- `compute_intensity_normalization` requires the full DP stack — not
  streaming-safe. In streaming mode, pass `normalization` from scan config.

**`normalize_intensity` scale parameter:**
- Default `scale` is `10000.0` to match ptycho-vit `config.yaml`. Always
  pass `scale=` explicitly if you are unsure — relying on the default
  silently produces inputs at the wrong magnitude if the default ever changes.

**`remap_positions` sign convention:**
- When `swap_xy=True`, signs are applied to the *output* columns after the
  swap, not the input columns before. So `signs=(sx, sy)` with `swap_xy=True`
  means `sx` scales what was originally y, and `sy` scales what was originally
  x. This matches `hxn_to_vit.py:POSITION_MAPS` but is easy to get wrong —
  verify with a known input/output pair before wiring into a pipeline.

### `ptychoml/orientation.py`

`autodetect_orientation` sweeps all 8 D4 transforms and scores each using
`_score_forward_consistency` (forward-physics NCC). Lower score = better.

**`autodetect_orientation` is not streaming-safe.** It holds the full
`intensity_batch` in RAM and runs 8× inference passes. Use it once on a
representative subset at the start of a scan, then lock in the winning
`dp_orient` for the rest.

**`fftshift=None` in `preprocess_kwargs` is safe to pass** — `autodetect_orientation`
will resolve the DC convention once from the batch via `detect_dc_at_corner`
and lock it in for the entire sweep. Pass an explicit `fftshift=True` or
`fftshift=False` if you already know the convention.

### `ptychoml/trt.py`

Default TRT workspace size is 2 GiB. If engine build fails with a
serialization error, increase `--workspace-size` (e.g. `4294967296` for
4 GiB).

### `ptychoml/stitch.py`

Patch-placement / mosaic helpers lifted verbatim from
`holoptycho/mosaic_stitch.py` so holoptycho can re-export them under the
original names (zero behavior change). Three strategies:
`place_patches_fourier_shift` / `stitch_batch_into` (sub-pixel Fourier
shift), `stitch_batch_livestitch_into` (nearest-integer + touched bbox),
and `stitch_batch_nearest` (plain nearest-integer, edge-clamped).

`canvas` and `counts` accumulate **in place**; callers normalize as
`canvas / np.maximum(counts, 1)` — these functions never normalize. The
Fourier-shift and livestitch paths flip each patch up-down before
placement; `stitch_batch_nearest` does not. All edge handling clamps to
the canvas (no wrap-around). Pure numpy + `scipy.fft`; depends only on
`fourier_shift` from `preprocess.py`.

**All three are streaming-safe** — per-patch placement with no
batch-global reduction, so per-batch stitching equals one-shot stitching
regardless of how frames are chunked. Two gotchas in a streaming loop:
the nearest and livestitch paths are bit-exact across batchings while the
Fourier path is associative only up to FFT round-off; and the Fourier
path may **reallocate** the canvas on edge straddle, so always assign from
the return value (`canvas, counts = stitch_batch_into(...)`) rather than
relying on in-place mutation.
