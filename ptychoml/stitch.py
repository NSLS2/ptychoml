"""Patch-placement / mosaic stitching for ViT ptychography output.

Array-in / array-out stitching helpers that accumulate a batch of
reconstructed patches into a running ``(canvas, counts)`` pair. Like the
rest of ptychoml these operate on plain numpy arrays so they can be reused
by any caller — holoptycho's streaming Holoscan pipeline, offline mosaic
builders, notebook one-offs — without dragging in a framework.

Three placement strategies are provided side-by-side so callers can pick
the accuracy/speed trade-off they need:

    place_patches_fourier_shift / stitch_batch_into
        Sub-pixel Fourier-shift placement. Each patch is over-extracted by
        ``pad`` pixels, phase-ramp shifted by its fractional position, then
        centre-cropped before scatter-add. Highest placement accuracy.
    stitch_batch_livestitch_into
        Nearest-integer scatter-add that also returns the bounding box of
        the region it touched — useful for incremental/live display where
        the writer only needs to repaint the changed area.
    stitch_batch_nearest
        Plain nearest-integer scatter-add. Simplest; clamps at canvas
        edges (no wrap-around). Handy as a JIT/cache warm-up kernel.

Both ``canvas`` and ``counts`` accumulate in place; the displayed/written
mosaic is ``canvas / np.maximum(counts, 1)``. No normalization happens
here — the caller decides the min-overlap threshold.

Implemented in numpy (not torch): the algorithm is FFT + scatter-add +
divide, no autograd needed, which keeps inference containers light.

Performance notes (Fourier-shift path):
* The FFT uses ``scipy.fft`` with ``workers=-1`` (multithreaded
  pocketfft) in ``complex64`` — ~25-50x faster than ``numpy.fft`` on a
  (64, 128, 128) batch, with no observable accuracy loss after the
  over-extract + crop.
* The counts canvas is updated with a dedicated scatter-add (no FFT). A
  Fourier shift of an all-ones patch returns ones (only DC is non-zero,
  and DC is unchanged by the phase ramp), so the FFT round-trip the
  reference does on ``np.ones_like(patches)`` is pure waste.

Provenance
----------
Source: holoptycho/mosaic_stitch.py (lifted verbatim so call sites there
can re-export under the original names). That implementation in turn was
adapted from the ptycho-vit reference
(``utils/ptychi_utils.py:place_patches_fourier_shift`` on the
holostitching branch, https://github.com/SYNAPS-I/ptycho-vit), itself
adapted from Ming Du's pty-chi
(https://github.com/AdvancedPhotonSource/pty-chi).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Sub-pixel FFT shift lives in ptychoml.preprocess so it can be shared with
# any caller. Aliased to the original local name to keep the lifted code
# below byte-for-byte identical to holoptycho/mosaic_stitch.py.
from .preprocess import fourier_shift as _fourier_shift


def _placement_indices(
    image_shape: Tuple[int, int],
    positions: np.ndarray,
    patch_shape: Tuple[int, int],
    pad: int,
):
    """Compute integer scatter indices and any required boundary padding.

    Shared by the patch-placement and counts-update paths so they always
    place over the same region. Returns
    ``(sys, sxs, ph_eff, pw_eff, pad_lengths, fractional)`` where ``sys``
    and ``sxs`` are already shifted into the (possibly padded) canvas
    coordinate system.
    """
    ph, pw = patch_shape
    sys_float = positions[:, 0] - (ph - 1.0) / 2.0
    sxs_float = positions[:, 1] - (pw - 1.0) / 2.0

    sys = np.floor(sys_float).astype(np.int64) + pad
    sxs = np.floor(sxs_float).astype(np.int64) + pad
    eys = sys + ph - 2 * pad
    exs = sxs + pw - 2 * pad

    fractional = np.stack(
        [sys_float - sys + pad, sxs_float - sxs + pad], axis=-1
    ).astype(np.float64)

    pad_lengths = (
        max(int(-sys.min()), 0),
        max(int(eys.max() - image_shape[0]), 0),
        max(int(-sxs.min()), 0),
        max(int(exs.max() - image_shape[1]), 0),
    )
    if any(pad_lengths):
        sys = sys + pad_lengths[0]
        sxs = sxs + pad_lengths[2]

    ph_eff = ph - 2 * pad if pad > 0 else ph
    pw_eff = pw - 2 * pad if pad > 0 else pw

    return sys, sxs, ph_eff, pw_eff, pad_lengths, fractional


def place_patches_fourier_shift(
    image: np.ndarray,
    positions: np.ndarray,
    patches: np.ndarray,
    pad: int = 1,
) -> np.ndarray:
    """Add patches into ``image`` with sub-pixel Fourier shifts.

    Mirrors ``ptycho-vit:place_patches_fourier_shift`` with ``op="add"`` and
    ``adjoint_mode=False``: each patch is over-extracted by ``pad`` pixels,
    Fourier-shifted by its fractional position, then center-cropped back to
    its original size before scatter-add.

    Source: holoptycho/mosaic_stitch.py ``place_patches_fourier_shift``.
    """
    ph, pw = patches.shape[-2:]
    sys, sxs, ph_eff, pw_eff, pad_lengths, fractional = _placement_indices(
        image.shape, positions, (ph, pw), pad,
    )

    if any(pad_lengths):
        image = np.pad(
            image,
            ((pad_lengths[0], pad_lengths[1]), (pad_lengths[2], pad_lengths[3])),
            mode="constant",
        )

    # Up-down flip the patch before FFT
    patches = patches[:, ::-1, :]

    if not np.allclose(fractional, 0.0, atol=1e-7):
        patches = _fourier_shift(patches, fractional)

    if pad > 0:
        patches = patches[:, pad:ph - pad, pad:pw - pad]

    for i in range(len(patches)):
        image[sys[i]:sys[i] + ph_eff, sxs[i]:sxs[i] + pw_eff] += patches[i]

    if any(pad_lengths):
        image = image[
            pad_lengths[0]: image.shape[0] - pad_lengths[1],
            pad_lengths[2]: image.shape[1] - pad_lengths[3],
        ]
    return image


def _add_ones_at(
    canvas: np.ndarray,
    positions: np.ndarray,
    patch_shape: Tuple[int, int],
    pad: int,
) -> np.ndarray:
    """Counts-update fast path: scatter-add ones over the same regions
    ``place_patches_fourier_shift`` would, but without FFTs.

    A Fourier shift of an all-ones patch returns ones (only the DC bin is
    non-zero, and DC is unchanged by the phase ramp), so the round-trip in
    the original counts path was pure overhead.
    """
    ph, pw = patch_shape
    sys, sxs, ph_eff, pw_eff, pad_lengths, _ = _placement_indices(
        canvas.shape, positions, (ph, pw), pad,
    )

    if any(pad_lengths):
        canvas = np.pad(
            canvas,
            ((pad_lengths[0], pad_lengths[1]), (pad_lengths[2], pad_lengths[3])),
            mode="constant",
        )

    for i in range(len(positions)):
        canvas[sys[i]:sys[i] + ph_eff, sxs[i]:sxs[i] + pw_eff] += 1.0

    if any(pad_lengths):
        canvas = canvas[
            pad_lengths[0]: canvas.shape[0] - pad_lengths[1],
            pad_lengths[2]: canvas.shape[1] - pad_lengths[3],
        ]
    return canvas


def stitch_batch_into(
    canvas: np.ndarray,
    counts: np.ndarray,
    patches: np.ndarray,
    positions_px: np.ndarray,
    *,
    pad: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Accumulate one batch of cropped patches into (canvas, counts).

    ``patches`` should already be center-cropped (the caller decides
    ``inner_crop``). ``positions_px`` is (N, 2) in canvas pixel coordinates,
    (y, x), pointing at the patch centers.

    Scatter-add is associative, so per-batch accumulation gives the same
    result (up to FFT noise) as one-shot stitching of all patches.

    Source: holoptycho/mosaic_stitch.py ``stitch_batch_into``.
    """
    canvas = place_patches_fourier_shift(canvas, positions_px, patches, pad=pad)
    counts = _add_ones_at(counts, positions_px, patches.shape[-2:], pad=pad)
    return canvas, counts


def stitch_batch_livestitch_into(
    canvas: np.ndarray,
    counts: np.ndarray,
    patches: np.ndarray,
    positions_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """LiveStitch-style batched nearest-integer patch placement.

    ``canvas`` stores the running sum of all patch values, ``counts`` the
    running occupancy count; the displayed/written mosaic is
    ``canvas / np.maximum(counts, 1)``. No in-place normalization is
    performed here.

    Returns ``(canvas, counts, bbox)`` where ``bbox`` is the
    ``(y0, y1, x0, x1)`` bounding box of the region touched this batch — a
    live writer can repaint just that sub-rectangle. ``bbox`` is
    ``(0, 0, 0, 0)`` when no patch overlapped the canvas.

    Source: holoptycho/mosaic_stitch.py ``stitch_batch_livestitch_into``.
    """

    if patches.ndim != 3:
        raise ValueError(f"patches must be [B, ph, pw], got {patches.shape}")

    B, ph, pw = patches.shape
    H, W = canvas.shape

    y0 = np.rint(positions_px[:, 0] - ph / 2).astype(np.int64)
    x0 = np.rint(positions_px[:, 1] - pw / 2).astype(np.int64)
    y1 = y0 + ph
    x1 = x0 + pw

    # Up-down flip the patch before placement
    patches = patches[:, ::-1, :]

    valid = (y1 > 0) & (x1 > 0) & (y0 < H) & (x0 < W)

    if not np.any(valid):
        return canvas, counts, (0, 0, 0, 0)

    bbox_y0 = H
    bbox_x0 = W
    bbox_y1 = 0
    bbox_x1 = 0

    for i in np.where(valid)[0]:
        cy0 = max(y0[i], 0)
        cx0 = max(x0[i], 0)
        cy1 = min(y1[i], H)
        cx1 = min(x1[i], W)

        py0 = cy0 - y0[i]
        px0 = cx0 - x0[i]
        py1 = py0 + (cy1 - cy0)
        px1 = px0 + (cx1 - cx0)

        canvas[cy0:cy1, cx0:cx1] += patches[i, py0:py1, px0:px1]
        counts[cy0:cy1, cx0:cx1] += 1.0

        bbox_y0 = min(bbox_y0, cy0)
        bbox_y1 = max(bbox_y1, cy1)
        bbox_x0 = min(bbox_x0, cx0)
        bbox_x1 = max(bbox_x1, cx1)

    return canvas, counts, (bbox_y0, bbox_y1, bbox_x0, bbox_x1)


def stitch_batch_nearest(
    canvas: np.ndarray,
    counts: np.ndarray,
    patches: np.ndarray,
    positions_px: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Accumulate patches using nearest-integer placement (no Fourier shift).

    Simpler than ``stitch_batch_into`` — rounds each position to the nearest
    pixel and does a plain scatter-add. Patches near the canvas edge are
    clamped rather than wrapped.

    Source: holoptycho/mosaic_stitch.py ``stitch_batch_nearest``.
    """
    ph, pw = patches.shape[-2:]
    ch, cw = canvas.shape
    for i in range(len(patches)):
        ry = int(round(positions_px[i, 0]))
        rx = int(round(positions_px[i, 1]))
        sy = ry - ph // 2
        sx = rx - pw // 2
        sy0 = max(0, sy);  sy1 = min(ch, sy + ph)
        sx0 = max(0, sx);  sx1 = min(cw, sx + pw)
        py0 = sy0 - sy;    py1 = py0 + (sy1 - sy0)
        px0 = sx0 - sx;    px1 = px0 + (sx1 - sx0)
        if sy1 > sy0 and sx1 > sx0:
            canvas[sy0:sy1, sx0:sx1] += patches[i, py0:py1, px0:px1]
            counts[sy0:sy1, sx0:sx1] += 1.0
    return canvas, counts
