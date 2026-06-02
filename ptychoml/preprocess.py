"""Array-in / array-out preprocessing utilities for ptychography data.

These helpers operate on plain numpy arrays so they can be reused by any
caller — HXN HDF5 pipelines, holoptycho's streaming Holoscan operators,
notebook one-offs — without dragging in HDF5, MPI, or filesystem
dependencies.

Functions are grouped into four sections so variants can be evaluated
side-by-side:

    1. ROI detection            — find a region without modifying the data
    2. Crop / pad / resize      — change the spatial extent of the data
    3. Bad-pixel & threshold    — mask, inpaint, or threshold-clip values
    4. Intensity & geometry     — scale, shift, geometry helpers

Provenance
----------
Each function below has a ``Source:`` line in its docstring naming the
upstream file/function it was lifted from. Four upstreams contribute:

* ``holoptycho`` — https://github.com/NSLS2/holoptycho (live streaming
  Holoscan pipeline). Inline array ops have been pulled out of Operator
  ``compute()`` methods into pure functions.
* ``ptycho_gui`` — https://github.com/NSLS2/ptycho_gui (offline GUI for
  iterative reconstruction). Source files cited as ``ptycho_gui/...``.
* ``ptycho-vit`` — private PyTorch training repo at ANL. Numpy-only
  preprocessing methods on the dataset class are lifted; torch ops are
  skipped since this library is numpy-only.
* HXN h5_conv (offline HDF5-to-HDF5 converter, provided to this PR via a
  one-off ``temp_code`` script — not a public repo).

Some functions are *variants* of each other (e.g. ``estimate_roi`` uses
intensity projections while ``auto_detect_roi_offsets`` uses
centre-of-mass). They are kept side-by-side for now and will be deduped
in a follow-up once call sites are unified.

GPU support
-----------
Functions that only use array methods + slicing + comparisons (the
in-place mutating ones, ``crop_to_roi``, ``normalize_intensity``) work
transparently on ``cupy`` arrays because cupy is API-compatible with
numpy at the method level. ``inpaint_bad_pixels`` uses the small
``_get_xp`` helper to dispatch ``median`` against the right module.
Functions that use ``scipy.fft`` (``fourier_shift``) remain numpy-only
for now — cupy callers can move data to host or use the cupy-native
equivalent directly.

Per-frame argmax centering note
-------------------------------
``resize_diffraction_patterns`` finds the crop center independently for
each frame using ``np.argmax``. Saturated / hot pixels can therefore
mislead the centering. Mask them with ``mask_hot_pixels`` (or pre-crop
to a detector ROI with ``crop_to_roi``) before calling.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

import numpy as np
import scipy.fft

ArrayLike = Union[np.ndarray, Iterable[np.ndarray]]


def detect_dc_at_corner(arr: np.ndarray) -> bool:
    """Return True if the central beam of ``arr`` sits at the corners.

    A diffraction pattern's central (zero-frequency) beam is by far the
    brightest feature. We sum intensity in the central half-by-half
    window and compare against the sum in the four matching corner
    windows; whichever wins tells us where the central beam currently
    lives. Returns True when the corners dominate (so the caller should
    ``fftshift`` to put the central beam at the center).

    ``arr`` may be ``(H, W)`` or ``(N, H, W)``; in the stacked case the
    mean across the batch is used so a single hot frame can't flip the
    decision.
    """
    a = arr.mean(axis=0) if arr.ndim == 3 else arr
    a = np.abs(a)
    H, W = a.shape[-2:]
    qh, qw = H // 4, W // 4
    center_sum = float(a[qh:H - qh, qw:W - qw].sum())
    corner_sum = float(
        a[:qh, :qw].sum()
        + a[:qh, W - qw:].sum()
        + a[H - qh:, :qw].sum()
        + a[H - qh:, W - qw:].sum()
    )
    return corner_sum > center_sum


def _get_xp(arr):
    """Return the array's module (numpy or cupy).

    Falls back to numpy when cupy is not installed. Used by functions
    that need module-level calls (e.g. ``xp.median``) so they work on
    both numpy and cupy arrays without duplicating the implementation.
    """
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp
    except ImportError:
        pass
    return np


# ============================================================================
# 1. ROI detection
# ----------------------------------------------------------------------------
# Find where the signal lives in a frame — these return coordinates and do
# *not* modify the input. Two variants here:
#   - auto_detect_roi_offsets: intensity-weighted COM with saturation masking
#   - estimate_roi: normalized intensity projections + edge-of-signal threshold
# Pair them with crop_to_roi to actually crop the data.
# ============================================================================


def auto_detect_roi_offsets(
    frames: np.ndarray,
    nx: int,
    ny: int,
    n_sample: int = 50,
    saturation_threshold: float | None = None,
) -> Tuple[int, int]:
    """Auto-detect detector ROI offsets from the diffraction-pattern center.

    Averages up to ``n_sample`` frames, optionally masks pixels above
    ``saturation_threshold`` (which would otherwise drag the COM off
    course), then computes the intensity-weighted center of mass and
    returns ``(bx0, by0)`` such that an ``nx × ny`` crop is centered on
    it. Returns ``(0, 0)`` if the masked frame has zero total intensity.

    ``saturation_threshold`` defaults to ``None`` (no masking). Supply a
    calibrated value when your detector saturates at a known intensity.
    The previous implicit dtype-max sentinel only worked for raw
    integer-dtype reads and was useless once data was converted to
    float32, so callers must now opt in explicitly.

    Source: holoptycho/scripts/replay_from_tiled.py ``_auto_batch_offsets``.

    POTENTIAL BUG: returns ``(bx0, by0)`` in (x, y) / (col, row) order,
    but ``crop_to_roi`` takes ``[[y0, y1], [x0, x1]]`` in (row, col)
    order. The two ROI helpers in this section therefore disagree on
    axis order; a caller wiring them together can silently transpose
    the crop window. Pick one convention and migrate both. Until then,
    callers must remember to swap when composing the two.
    """
    sample = frames[:min(n_sample, len(frames))].astype(np.float64)
    mean_frame = sample.mean(axis=0)
    if saturation_threshold is None:
        masked = mean_frame
    else:
        sat_mask = (sample > saturation_threshold).any(axis=0)
        masked = np.where(sat_mask, 0.0, mean_frame)
    total = masked.sum()
    if total <= 0:
        return 0, 0
    ys, xs = np.indices(masked.shape)
    cy = float((ys * masked).sum() / total)
    cx = float((xs * masked).sum() / total)
    h, w = mean_frame.shape
    bx0 = max(0, min(w - nx, round(cx - nx / 2)))
    by0 = max(0, min(h - ny, round(cy - ny / 2)))
    return int(bx0), int(by0)


def _project_on_x(image: np.ndarray) -> np.ndarray:
    """Sum along axis 0. Source: ptycho_gui/.../imgTools.py ``project_on_x``."""
    return np.cumsum(image, axis=0)[-1]


def _project_on_y(image: np.ndarray) -> np.ndarray:
    """Sum along axis 1. Source: ptycho_gui/.../imgTools.py ``project_on_y``."""
    return np.cumsum(image, axis=1)[:, -1]


def _find_start_end(arr: np.ndarray, threshold_weight: float = 0.3) -> Tuple[int, int]:
    """Edge-of-signal indices in a 1D projection.

    Source: ptycho_gui/.../imgTools.py ``find_start_end``.

    POTENTIAL BUG: when ``arr`` is constant ``diff`` is all-zero and the
    ``< threshold_weight * mean(diff)`` test becomes ``0 < 0`` everywhere
    (all False). ``argmin`` of an all-False array is 0, so ``start = -2``
    and ``end = len(arr) + 1`` — both out of range. The estimate_roi
    fallback hides this with a clip, but the function itself returns
    nonsense on flat inputs. Revisit when we have time to also rework
    this whole projection-based ROI detector.
    """
    diff = np.abs(arr[:-1] - arr[1:])
    diff = diff < threshold_weight * np.mean(diff)
    start = np.argmin(diff) - 2
    end = len(arr) - np.argmin(diff[::-1]) - 1 + 2
    return start, end


def estimate_roi(image: np.ndarray, threshold: float = 0.1) -> Tuple[int, int, int, int]:
    """Estimate a rectangular ROI ``(x0, y0, w, h)`` via intensity projection.

    Variant of :func:`auto_detect_roi_offsets` that normalises the image
    to ``[0, 1]``, projects onto each axis, and uses an edge-of-signal
    threshold to pick start/end positions. Falls back to the full image
    if the detected box is degenerate.

    Source: ptycho_gui/nsls2ptycho/core/widgets/imgTools.py ``estimate_roi``.
    """
    height, width = image.shape
    image_f = image.astype(np.float64, copy=False)
    _image = (image_f - image_f.min()) / np.ptp(image_f)

    proj_x = _project_on_x(_image) / height
    proj_y = _project_on_y(_image) / width

    x0, x1 = _find_start_end(proj_x, threshold)
    y0, y1 = _find_start_end(proj_y, threshold)

    x0 = int(np.clip(x0, 0, width - 1))
    x1 = int(np.clip(x1, 0, width - 1))
    y0 = int(np.clip(y0, 0, height - 1))
    y1 = int(np.clip(y1, 0, height - 1))

    w = x1 - x0
    h = y1 - y0

    if w <= 0 or h <= 0:
        x0 = 0
        y0 = 0
        w = width
        h = height

    return x0, y0, w, h


# ============================================================================
# 2. Crop / pad / resize
# ----------------------------------------------------------------------------
# Change the spatial extent of frames. Three variants by use case:
#   - crop_to_roi:                 fixed window, identical for every frame
#   - zero_pad_to_target:          strict centered pad; errors if input is too big
#   - resize_diffraction_patterns: combined per-frame argmax-crop + zero-pad
# ============================================================================


def crop_to_roi(arr: np.ndarray, roi) -> np.ndarray:
    """Crop the last two axes of ``arr`` to a fixed ``[[y0, y1], [x0, x1]]`` ROI.

    Used when the crop window is known from detector calibration and should
    be applied identically to every frame (e.g. holoptycho streaming). The
    ROI uses Python half-open ranges: ``[y0, y1)`` rows, ``[x0, x1)`` cols.
    Works on numpy or cupy arrays (slicing is method-based).

    Source: holoptycho/preprocess.py ``ImageBatchOp.compute`` inline crop.

    POTENTIAL BUG: no bounds checking. Negative or out-of-range indices
    silently produce a wrong-shaped output via numpy slicing semantics
    (e.g. ``y0=-1`` slices from the last row, an empty range gives an
    empty axis). Validating against ``arr.shape[-2:]`` and raising on
    invalid ROIs would catch caller mistakes loudly instead of feeding
    a malformed array downstream. Also see the axis-order mismatch
    flagged on ``auto_detect_roi_offsets``.
    """
    roi = np.asarray(roi)
    y0, y1 = int(roi[0, 0]), int(roi[0, 1])
    x0, x1 = int(roi[1, 0]), int(roi[1, 1])
    return arr[..., y0:y1, x0:x1]


def zero_pad_to_target(image: np.ndarray, target_size: int) -> np.ndarray:
    """Zero-pad a 2D image to ``target_size × target_size``, keeping content centered.

    Strict variant of :func:`resize_diffraction_patterns`'s pad branch:
    raises ``ValueError`` if the input is larger than ``target_size`` on
    either axis. Returns the input unchanged if already at target;
    otherwise allocates and returns a new array.

    Source: ptycho-vit ``data.py:_zero_pad_to_target``.
    """
    h, w = image.shape
    if h == target_size and w == target_size:
        return image
    if h > target_size or w > target_size:
        raise ValueError(
            f"Image size ({h}, {w}) larger than target size ({target_size})"
        )

    pad_h = target_size - h
    pad_w = target_size - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )


def resize_diffraction_patterns(dp: ArrayLike, target_n: int) -> np.ndarray:
    """Crop or zero-pad each diffraction pattern to ``target_n × target_n``.

    For each pattern in the input stack:
      * if larger than ``target_n`` on any axis, crop a window of size
        ``target_n`` around the per-frame argmax (clamped to image bounds);
      * if (still) smaller than ``target_n`` on any axis, zero-pad the
        result symmetrically out to ``target_n × target_n``.

    The two branches compose: a crop that gets clamped near an edge will
    fall through to the pad branch, so the final shape is always
    ``(N, target_n, target_n)``.

    Source: HXN h5_conv ``_resize_dp`` (provided via temp_code).

    POTENTIAL BUG (per-frame argmax centering): the crop window is
    centered on each frame's argmax independently. A single saturated
    or hot pixel anywhere off-center can therefore swing the window
    for that one frame, breaking inter-frame alignment and changing
    where the central beam lands relative to the model's expected
    frame. Mask hot pixels first (``mask_hot_pixels``) or pre-crop to
    a detector ROI (``crop_to_roi``) — and consider replacing the
    argmax with a COM (``auto_detect_roi_offsets``-style) batch
    estimate so every frame in a scan gets the same window.

    POTENTIAL BUG (mixed-axis crop+pad): when one axis is over
    ``target_n`` and the other is under, the crop branch runs only
    along the over-sized axis (the under-sized axis is sliced by its
    own argmax range — which can clip valid signal at the edge),
    then the pad branch enlarges the under-sized axis back to
    ``target_n``. The under-sized axis is not really "cropped"
    deliberately, just sliced by happenstance. Result is shaped
    correctly but the content can lose data near the edges. Rare in
    practice (square detectors), but worth a real fix.
    """
    resized = []
    for pattern in dp:
        if pattern.shape[-1] > target_n or pattern.shape[-2] > target_n:
            peak_y, peak_x = np.unravel_index(np.argmax(pattern), pattern.shape)
            start_x = max(peak_x - target_n // 2, 0)
            end_x = min(peak_x + target_n // 2, pattern.shape[-1])
            start_y = max(peak_y - target_n // 2, 0)
            end_y = min(peak_y + target_n // 2, pattern.shape[-2])
            pattern = crop_to_roi(pattern, [[start_y, end_y], [start_x, end_x]])

        if pattern.shape[-1] < target_n or pattern.shape[-2] < target_n:
            pattern = zero_pad_to_target(pattern, target_n)

        resized.append(pattern)

    return np.array(resized)


# ============================================================================
# 3. Bad-pixel masking, inpainting & threshold cleanup
# ----------------------------------------------------------------------------
# Two sub-styles share this section:
#   (a) Threshold-based masks (single-pass, value test):
#         - mask_hot_pixels:        value > threshold → fill
#         - mask_hot_pixels_by_count: photon-count threshold w/ amp/int kind
#         - apply_intensity_floor:  value < threshold → 0    (symmetric)
#   (b) Median inpainting at known coords:
#         - inpaint_bad_pixels:     coords as (K, 2); 3×3 (or larger) median
# All masking ops mutate in place (zero allocation, suitable for streaming).
# ============================================================================


def mask_hot_pixels(
    arr: np.ndarray,
    threshold: float,
    fill: float = 0.0,
) -> np.ndarray:
    """Replace values strictly greater than ``threshold`` with ``fill``, in place.

    Mutates ``arr`` and returns it (no allocation), so this is safe to use
    in streaming hot paths. Callers wanting a copy should pass
    ``arr.copy()`` explicitly. Works transparently on numpy or cupy arrays.

    Source: HXN h5_conv ``load_ptycho_data`` inline ``raw_counts > 60000``
    handler (provided via temp_code).
    """
    arr[arr > threshold] = fill
    return arr


def mask_hot_pixels_by_count(
    arr: np.ndarray,
    count_threshold: float,
    kind: str = "intensity",
    fill: float = 0.0,
) -> np.ndarray:
    """Zero pixels whose photon count exceeds ``count_threshold``, in place.

    The threshold is always specified in photon-count units. How it is
    applied to ``arr`` depends on whether ``arr`` already represents counts
    (``kind='intensity'``) or has been square-rooted to amplitude
    (``kind='amplitude'``); in the latter case the comparison threshold is
    ``sqrt(count_threshold)``. This lets the same caller-facing parameter
    apply consistently whether the filter runs before or after a sqrt.

    Pass ``count_threshold=None`` (or a non-finite value) to skip filtering
    entirely — returns ``arr`` unchanged so callers can wire this in
    unconditionally and gate via a single config field.

    Source: ptycho-vit ``scripts/hxn_to_vit.py:_apply_hot_pixel_filter``.
    """
    if count_threshold is None or not np.isfinite(count_threshold):
        return arr
    if kind == "amplitude":
        raw_threshold = float(np.sqrt(count_threshold))
    elif kind == "intensity":
        raw_threshold = float(count_threshold)
    else:
        raise ValueError(
            f"kind must be 'amplitude' or 'intensity'; got {kind!r}"
        )
    arr[arr > raw_threshold] = fill
    return arr


def apply_intensity_floor(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Zero values strictly below ``threshold`` (noise-floor cutoff), in place.

    Symmetric to ``mask_hot_pixels`` (which zeros values *above* a
    threshold). Mutates ``arr`` and returns it (no allocation), so this
    is safe to use in streaming hot paths. Works on numpy or cupy arrays.

    Source: holoptycho/preprocess.py ``ImagePreprocessorOp.compute``
    ``detmap_threshold`` block (also in eiger_test cupy variant).
    """
    arr[arr < threshold] = 0
    return arr


def inpaint_bad_pixels(
    arr: np.ndarray,
    coords,
    radius: int = 1,
) -> np.ndarray:
    """Replace known bad-pixel coordinates with the median of their neighbourhood, in place.

    For each ``(row, col)`` in ``coords``, replaces the pixel at that
    location with the median of the surrounding ``(2*radius+1) × (2*radius+1)``
    window. Operates on the last two axes; works for both 2D arrays and
    stacks of shape ``(N, H, W)``. Mutates ``arr`` and returns it. The
    loop is sequential, so a later coord's median is computed against any
    earlier replacement that overlaps its window — matching upstream
    behavior. Works on numpy or cupy arrays (median is dispatched against
    the array's module).

    Source: holoptycho/preprocess.py ``ImagePreprocessorOp.compute`` inline
    bad-pixel inpainting loop (also in eiger_test cupy variant).
    """
    xp = _get_xp(arr)
    h, w = arr.shape[-2], arr.shape[-1]
    coords = np.asarray(coords).reshape(-1, 2)
    for r, c in coords:
        r, c = int(r), int(c)
        r0 = max(r - radius, 0)
        r1 = min(r + radius + 1, h)
        c0 = max(c - radius, 0)
        c1 = min(c + radius + 1, w)
        window = arr[..., r0:r1, c0:c1]
        arr[..., r, c] = xp.median(window, axis=(-2, -1))
    return arr


# ============================================================================
# 4. Intensity & geometric transforms
# ----------------------------------------------------------------------------
# Single-purpose helpers that don't fit the masking/cropping families:
#   - normalize_intensity:       scalar rescale to match training-time normalization
#   - fourier_shift:             FFT-based sub-pixel shift (mosaic stitching)
#   - compute_sample_pixel_size: pure scalar geometry helper
# ============================================================================


def normalize_intensity(
    arr: np.ndarray,
    normalization: float,
    scale: float = 10000.0,
) -> np.ndarray:
    """Scale ``arr`` by ``scale / normalization``.

    The PtychoViT model is trained with diffraction patterns rescaled by
    a per-dataset ``(scale / normalization)`` factor where
    ``normalization`` is the max intensity across all DPs in the scan
    (with hot pixels excluded) and ``scale`` is the global factor from
    ptycho-vit's ``config.yaml`` (10000.0). Inference callers must
    apply the same scaling; passing a different value silently produces
    inputs at the wrong magnitude and the model output is meaningless.
    Returns a new array (does not mutate).

    See ``compute_intensity_normalization`` for the canonical way to
    derive the ``normalization`` value from a DP stack.

    Source: ptycho-vit ``data.py:PtychographyDataset.normalize``.
    """
    return (arr / normalization) * scale


def compute_intensity_normalization(
    intensity: np.ndarray,
    hot_pixel_count_threshold: float | None = None,
) -> float:
    """Return the per-scan normalization constant for ``preprocess_diffraction``.

    Convention is "max intensity across the full DP stack, with hot
    pixels excluded" — the same number ``hxn_to_vit.py`` writes per scan
    into its ``{scan_id}_max_intensity.pkl``. Use this in offline
    pipelines (and in the ``ptychoml-predict`` CLI) where the full stack
    is in hand; in streaming mode where it is not, the caller should
    supply the value out-of-band (e.g. from a scan-config JSON).

    Args:
        intensity:                  ``(..., H, W)`` raw detector counts.
                                    May span the whole scan or a
                                    representative subset.
        hot_pixel_count_threshold:  Photon-count threshold above which
                                    pixels are excluded from the max.
                                    ``None`` includes everything; use it
                                    when you know your data has no hot
                                    pixels (e.g. a simulation).

    Raises:
        ValueError: every pixel exceeds the hot-pixel threshold.

    Source: matches ``hxn_to_vit.py:write_outputs``'s running-max-after-
    hot-pixel-filter behaviour.
    """
    arr = np.asarray(intensity)
    if hot_pixel_count_threshold is None:
        return float(arr.max())
    mask = arr <= float(hot_pixel_count_threshold)
    if not mask.any():
        raise ValueError(
            "All pixels exceed hot_pixel_count_threshold; cannot compute "
            "normalization. Raise the threshold or check the data."
        )
    return float(arr[mask].max())


def fourier_shift(images: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Sub-pixel shift each ``(H, W)`` plane of ``images`` by ``shifts[i] = (dy, dx)``.

    FFT-based phase-ramp multiplication. Runs in ``complex64`` via
    ``scipy.fft`` with worker threads for speed; output is cast back to
    the input dtype. Used by holoptycho's mosaic stitcher to place ViT
    output patches at fractional positions.

    Source: holoptycho/mosaic_stitch.py ``_fourier_shift``.
    """
    h, w = images.shape[-2:]
    images_c = np.asarray(images, dtype=np.complex64)
    ft = scipy.fft.fft2(images_c, workers=-1)

    shifts_f32 = np.asarray(shifts, dtype=np.float32)
    fy = np.fft.fftfreq(h).astype(np.float32)
    fx = np.fft.fftfreq(w).astype(np.float32)
    two_pi_neg = -2.0 * np.float32(np.pi)
    arg_y = (two_pi_neg * shifts_f32[:, 0, None]) * fy[None, :]
    arg_x = (two_pi_neg * shifts_f32[:, 1, None]) * fx[None, :]
    ramp_y = np.exp((1j * arg_y).astype(np.complex64))
    ramp_x = np.exp((1j * arg_x).astype(np.complex64))
    ft *= ramp_y[:, :, None]
    ft *= ramp_x[:, None, :]

    out = scipy.fft.ifft2(ft, workers=-1)
    return out.real.astype(images.dtype, copy=False)


# 8 D4 symmetries acting on the last two axes of an array. The keys match
# the names used by the orientation auto-detector and by
# ``ptycho-vit/scripts/hxn_to_vit.py`` so winning combinations can be
# round-tripped through human-readable config or report files.
D4_TRANSFORMS = {
    'identity':      lambda a: a,
    'fliplr':        lambda a: np.flip(a, axis=-1),
    'flipud':        lambda a: np.flip(a, axis=-2),
    'rot180':        lambda a: np.flip(np.flip(a, axis=-1), axis=-2),
    'transpose':     lambda a: np.swapaxes(a, -1, -2),
    'rot90_ccw':     lambda a: np.flip(np.swapaxes(a, -1, -2), axis=-2),
    'rot90_cw':      lambda a: np.flip(np.swapaxes(a, -1, -2), axis=-1),
    'antitranspose': lambda a: np.flip(np.flip(np.swapaxes(a, -1, -2), axis=-1), axis=-2),
}

D4_NAMES = tuple(D4_TRANSFORMS.keys())


def apply_d4(arr: np.ndarray, name: str) -> np.ndarray:
    """Apply a named D4 symmetry to the last two axes of ``arr``.

    Returns a view where possible. The eight names are the standard D4
    group elements: identity, the three axis-and-180° flips, and the four
    transpose-based rotations and reflections. See ``D4_NAMES`` for the
    full set.

    Works on numpy and cupy arrays (both expose ``flip`` / ``swapaxes``).

    Source: ptycho-vit ``scripts/hxn_to_vit.py:ORIENTATIONS``.
    """
    try:
        fn = D4_TRANSFORMS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown D4 transform: {name!r}; valid names are {D4_NAMES!r}"
        ) from exc
    return fn(arr)


def remap_positions(
    positions: np.ndarray,
    signs: Tuple[int, int] = (1, 1),
    swap_xy: bool = False,
) -> np.ndarray:
    """Apply a sign-and-swap remapping to scan positions.

    ``positions`` is an ``(N, 2)`` array; column 0 is x, column 1 is y.
    ``signs = (sx, sy)`` flips each axis independently (each must be ±1);
    ``swap_xy`` exchanges the two columns. Returns a new array — the input
    is not modified. Pairs with ``apply_d4`` for the orientation auto-
    detector's search space: the same combination of sign-flip and
    transpose must be applied consistently to detector data and scan
    positions for the forward model to be self-consistent.

    **Convention note:** when ``swap_xy=True``, the sign ``sx`` is applied
    to the output x column (which holds the original y values after the
    swap), and ``sy`` is applied to the output y column (original x). In
    other words, signs are applied *after* the swap, to the output columns,
    not to the input columns before swapping. This matches
    ``hxn_to_vit.py:POSITION_MAPS`` but is easy to get wrong — if in doubt,
    verify with a known (x, y) → expected output pair before wiring into a
    pipeline.

    Source: ptycho-vit ``scripts/hxn_to_vit.py:POSITION_MAPS``.
    """
    sx, sy = signs
    if sx not in (-1, 1) or sy not in (-1, 1):
        raise ValueError(f"signs must each be ±1; got {signs!r}")
    out = np.empty_like(positions)
    if swap_xy:
        out[:, 0] = sx * positions[:, 1]
        out[:, 1] = sy * positions[:, 0]
    else:
        out[:, 0] = sx * positions[:, 0]
        out[:, 1] = sy * positions[:, 1]
    return out


def compute_sample_pixel_size(
    wavelength_m: float,
    detector_distance_m: float,
    ccd_pixel_size_m: float,
    n_pixels: int,
) -> float:
    """Far-field (Fraunhofer) pixel size at the sample plane.

    ``dx_sample = λ * z / (N * dx_detector)``

    Source: extracted from inline formula reused 4× across HXN h5_conv
    (provided via temp_code).
    """
    return wavelength_m * detector_distance_m / (n_pixels * ccd_pixel_size_m)


# ============================================================================
# 5. Composed pipeline
# ----------------------------------------------------------------------------
# A single entry point that composes the individual utilities above in the
# order ptycho-vit applies them during training. Callers wire this in once
# and configure behaviour via the parameters rather than re-implementing the
# sequence each time. Geometric site-specific transforms (axis flips,
# rotations, transposes) are intentionally NOT part of this — they belong to
# the caller because each beamline lays out its detector differently.
# ============================================================================


def preprocess_diffraction(
    intensity: np.ndarray,
    *,
    normalization: float,
    scale: float = 10000.0,
    hot_pixel_count_threshold: Optional[float] = 50000.0,
    dp_orient: str = 'identity',
    fftshift: Optional[bool] = None,
    out_dtype=np.float32,
) -> np.ndarray:
    """Convert raw detector intensity to a model-ready diffraction amplitude.

    Composes the ptycho-vit training-time preprocessing on a batch of raw
    detector frames. Steps run in this order:

      1. ``mask_hot_pixels_by_count`` (intensity-domain comparison)
      2. ``(intensity / normalization) * scale``  (matches training scale)
      3. ``sqrt`` (intensity → amplitude)
      4. ``apply_d4`` (rotate/flip the detector frame into the model frame)
      5. ``fftshift`` on the last two axes (auto-detected by default —
         applied iff the central beam currently sits at the corners)

    Step 1 mutates a working copy so the caller's input is not modified.
    Steps 2–5 allocate a new float array of ``out_dtype``.

    Args:
        intensity:                  Raw detector counts, shape ``(..., H, W)``.
                                    May be any numeric dtype; the working
                                    copy is promoted to float64 before
                                    rescaling and cast to ``out_dtype`` at
                                    the end.
        normalization:              Per-scan max intensity (hot pixels
                                    excluded) — what ptycho-vit's training
                                    pipeline divides DPs by to land them on
                                    the model's amplitude scale. Compute
                                    with ``compute_intensity_normalization``
                                    when the full DP stack is in hand
                                    (offline / CLI). In streaming mode where
                                    it is not, take the value from the scan
                                    config and pass it in.
        scale:                      Global scale factor (10000.0 — matches
                                    ptycho-vit's ``config.yaml``). The
                                    PtychographyDataset class-level default
                                    in ptycho-vit is 100000.0, but every
                                    HXN config overrides to 10000.0.
        hot_pixel_count_threshold:  Photon-count threshold; pixels strictly
                                    above this are zeroed before rescaling.
                                    Default 50000.0 matches hxn_to_vit's
                                    conversion default. Set ``None`` to
                                    disable (e.g. when data is already
                                    filtered upstream).
        dp_orient:                  D4 transform name (see ``D4_NAMES``)
                                    that rotates/flips the detector frame
                                    into the orientation the ViT model
                                    expects. ``'identity'`` is a no-op.
                                    Pick by hand from a known config, or
                                    let the orientation auto-detector pick
                                    one and pass the winning name here.
        fftshift:                   Tri-state DC-convention control:
                                      * ``None`` (default): auto-detect via
                                        ``detect_dc_at_corner`` — applies
                                        fftshift iff the central beam is
                                        currently at the corners. The model
                                        always sees central-beam-at-center.
                                      * ``True``: force fftshift.
                                      * ``False``: skip fftshift.

                                    **Streaming warning:** do not use
                                    ``fftshift=None`` mid-scan. The DC
                                    convention must be consistent across
                                    all batches — determine it once from
                                    the first batch or from scan config,
                                    then pass ``True`` or ``False``
                                    explicitly for the rest of the scan.
                                    A frame with an unusual intensity
                                    distribution could flip the
                                    auto-detect decision and silently
                                    shift some batches but not others.
        out_dtype:                  Output dtype. Default ``float32``,
                                    matching the ONNX/TRT engine's expected
                                    input dtype.

    Returns:
        Amplitude array shape ``intensity.shape`` and dtype ``out_dtype``.

    Source: composes utilities lifted from ptycho-vit ``data.py:normalize``
    + ``data.py:_load_hdf5_pattern`` (sqrt + zero-pad omitted; padding is a
    spatial concern handled separately) and ``hxn_to_vit.py``'s hot-pixel
    filter + D4 sweep.
    """
    working = np.asarray(intensity).copy()

    if hot_pixel_count_threshold is not None:
        mask_hot_pixels_by_count(
            working, hot_pixel_count_threshold, kind="intensity",
        )

    amplitude = np.sqrt(
        (working.astype(np.float64) / float(normalization)) * float(scale)
    ).astype(out_dtype, copy=False)

    if dp_orient != 'identity':
        amplitude = apply_d4(amplitude, dp_orient)

    do_shift = detect_dc_at_corner(amplitude) if fftshift is None else bool(fftshift)
    if do_shift:
        amplitude = np.fft.fftshift(amplitude, axes=(-2, -1))

    return np.ascontiguousarray(amplitude)
