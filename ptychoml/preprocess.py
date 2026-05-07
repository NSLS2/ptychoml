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
numpy at the method level. ``inpaint_bad_pixels`` and
``mask_saturated_pixels`` use the small ``_get_xp`` helper to dispatch
``median`` / ``iinfo`` against the right module. Functions that use
``scipy.fft`` or ``scipy.ndimage`` (``fourier_shift``,
``find_outlier_pixels``) remain numpy-only for now — cupy callers can
move data to host or use the cupy-native equivalents directly.

Per-frame argmax centering note
-------------------------------
``resize_diffraction_patterns`` finds the crop center independently for
each frame using ``np.argmax``. Saturated / hot pixels can therefore
mislead the centering. Mask them with ``mask_hot_pixels`` (or pre-crop
to a detector ROI with ``crop_to_roi``) before calling.
"""
from __future__ import annotations

from typing import Iterable, Tuple, Union

import numpy as np
import scipy.fft
from scipy.ndimage import median_filter

ArrayLike = Union[np.ndarray, Iterable[np.ndarray]]


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
    _image = (image - np.min(image)) / np.ptp(image)

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
        w = width - 1
        h = height - 1

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
# Three sub-styles share this section:
#   (a) Threshold-based masks (single-pass, value test):
#         - mask_hot_pixels:        value > threshold → fill
#         - apply_intensity_floor:  value < threshold → 0    (symmetric)
#   (b) Median inpainting at known coords:
#         - inpaint_bad_pixels:     coords as (K, 2); 3×3 (or larger) median
#   (c) Auto-detection (no caller-supplied coords):
#         - find_outlier_pixels:    median-filter difference, σ-based threshold
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


def find_outlier_pixels(
    data: np.ndarray,
    tolerance: int = 3,
    worry_about_edges: bool = True,
    get_fixed_image: bool = False,
):
    """Detect hot/dead pixels in a 2D array via median-filter difference.

    Returns ``hot_pixels`` (a ``(2, K)`` array of ``[rows, cols]``). When
    ``get_fixed_image=True``, also returns a copy of ``data`` with the
    detected pixels replaced by the median-filtered value, including
    edge / corner cases when ``worry_about_edges=True``.

    Note: faithfully copied — the ``tolerance`` parameter is currently
    unused upstream (the threshold is hard-coded to ``10*std(diff)``).
    Kept in the signature for compatibility.

    Source: ptycho_gui/nsls2ptycho/core/widgets/imgTools.py ``find_outlier_pixels``.
    """
    data = data.astype(float)
    blurred = median_filter(data, size=2)
    difference = data - blurred
    threshold = 10 * np.std(difference)

    hot_pixels = np.nonzero(np.abs(difference[1:-1, 1:-1]) > threshold)
    hot_pixels = np.array(hot_pixels) + 1

    if get_fixed_image:
        fixed_image = np.copy(data)
        for y, x in zip(hot_pixels[0], hot_pixels[1]):
            fixed_image[y, x] = blurred[y, x]

        if worry_about_edges:
            height, width = np.shape(data)

            for index in range(1, height - 1):
                med = np.median(data[index - 1:index + 2, 0:2])
                if np.abs(data[index, 0] - med) > threshold:
                    hot_pixels = np.hstack((hot_pixels, [[index], [0]]))
                    fixed_image[index, 0] = med

                med = np.median(data[index - 1:index + 2, -2:])
                if np.abs(data[index, -1] - med) > threshold:
                    hot_pixels = np.hstack((hot_pixels, [[index], [width - 1]]))
                    fixed_image[index, -1] = med

            for index in range(1, width - 1):
                med = np.median(data[0:2, index - 1:index + 2])
                if np.abs(data[0, index] - med) > threshold:
                    hot_pixels = np.hstack((hot_pixels, [[0], [index]]))
                    fixed_image[0, index] = med

                med = np.median(data[-2:, index - 1:index + 2])
                if np.abs(data[-1, index] - med) > threshold:
                    hot_pixels = np.hstack((hot_pixels, [[height - 1], [index]]))
                    fixed_image[-1, index] = med

            for (cy, cx, py, px) in (
                (0, 0, slice(0, 2), slice(0, 2)),
                (0, -1, slice(0, 2), slice(-2, None)),
                (-1, 0, slice(-2, None), slice(0, 2)),
                (-1, -1, slice(-2, None), slice(-2, None)),
            ):
                med = np.median(data[py, px])
                if np.abs(data[cy, cx] - med) > threshold:
                    row = height - 1 if cy == -1 else 0
                    col = width - 1 if cx == -1 else 0
                    hot_pixels = np.hstack((hot_pixels, [[row], [col]]))
                    fixed_image[cy, cx] = med

        return hot_pixels, fixed_image
    return hot_pixels


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
    scale: float = 1.0,
) -> np.ndarray:
    """Scale ``arr`` by ``scale / normalization``.

    The PtychoViT model is trained with diffraction patterns rescaled by
    a per-dataset ``(scale / normalization)`` factor; inference callers
    must apply the same scaling. Returns a new array (does not mutate).

    Source: ptycho-vit ``data.py:PtychographyDataset.normalize``.
    """
    return (arr / normalization) * scale


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
