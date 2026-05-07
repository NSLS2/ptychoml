"""Array-in / array-out preprocessing utilities for ptychography data.

These helpers operate on plain numpy arrays so they can be reused by any
caller — HXN HDF5 pipelines, holoptycho's streaming Holoscan operators,
notebook one-offs — without dragging in HDF5, MPI, or filesystem
dependencies.

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

Some functions are *variants* of each other (e.g. an HXN angle-correction
routine that flips sign for ``angle <= -45°`` versus a holoptycho one that
does not). They are kept side-by-side for now and will be deduped in a
follow-up once the call sites are unified.

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
            pattern = pattern[start_y:end_y, start_x:end_x]

        if pattern.shape[-1] < target_n or pattern.shape[-2] < target_n:
            padded = np.zeros((target_n, target_n), dtype=pattern.dtype)
            px = (target_n - pattern.shape[-1]) // 2
            py = (target_n - pattern.shape[-2]) // 2
            padded[py:py + pattern.shape[-2], px:px + pattern.shape[-1]] = pattern
            pattern = padded

        resized.append(pattern)

    return np.array(resized)


def adjust_object_for_pad(
    obj: np.ndarray,
    scale_y: float,
    scale_x: float,
    obj_pad: int,
) -> np.ndarray:
    """Correct an object's last two axes after a pixel-grid rescale.

    When an object is rescaled by ``(scale_y, scale_x)`` to match a new
    diffraction-pattern pixel size, the per-axis padding region (which is
    ``obj_pad`` pixels in the unscaled object) is also rescaled. Most
    iterative ptycho backends, however, allocate a *fixed* ``obj_pad``
    pixels of padding regardless of grid size, so the rescaled object
    needs to be trimmed (``scale > 1``) or zero-padded (``scale < 1``) by
    ``obj_pad * (scale - 1)`` pixels, split symmetrically across each
    axis.

    Source: HXN h5_conv ``adjust_obj_for_backend`` (provided via temp_code).
    """
    corr_h = int(round(obj_pad * (scale_y - 1)))
    corr_w = int(round(obj_pad * (scale_x - 1)))

    if corr_h > 0:
        top = corr_h // 2
        bot = corr_h - top
        obj = obj[:, top:obj.shape[-2] - bot, :]
    elif corr_h < 0:
        pad = -corr_h
        top = pad // 2
        obj = np.pad(obj, ((0, 0), (top, pad - top), (0, 0)), mode="constant")

    if corr_w > 0:
        lft = corr_w // 2
        rgt = corr_w - lft
        obj = obj[:, :, lft:obj.shape[-1] - rgt]
    elif corr_w < 0:
        pad = -corr_w
        lft = pad // 2
        obj = np.pad(obj, ((0, 0), (0, 0), (lft, pad - lft)), mode="constant")

    return obj


def mask_hot_pixels(
    arr: np.ndarray,
    threshold: float,
    fill: float = 0.0,
) -> np.ndarray:
    """Replace values strictly greater than ``threshold`` with ``fill``, in place.

    Mutates ``arr`` and returns it (no allocation), so this is safe to use
    in streaming hot paths. Callers wanting a copy should pass
    ``arr.copy()`` explicitly.

    Source: HXN h5_conv ``load_ptycho_data`` inline ``raw_counts > 60000``
    handler (provided via temp_code).
    """
    arr[arr > threshold] = fill
    return arr


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


def crop_to_roi(arr: np.ndarray, roi) -> np.ndarray:
    """Crop the last two axes of ``arr`` to a fixed ``[[y0, y1], [x0, x1]]`` ROI.

    Used when the crop window is known from detector calibration and should
    be applied identically to every frame (e.g. holoptycho streaming). The
    ROI uses Python half-open ranges: ``[y0, y1)`` rows, ``[x0, x1)`` cols.

    Source: holoptycho/preprocess.py ``ImageBatchOp.compute`` inline crop.
    """
    roi = np.asarray(roi)
    y0, y1 = int(roi[0, 0]), int(roi[0, 1])
    x0, x1 = int(roi[1, 0]), int(roi[1, 1])
    return arr[..., y0:y1, x0:x1]


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
    behavior.

    Source: holoptycho/preprocess.py ``ImagePreprocessorOp.compute`` inline
    bad-pixel inpainting loop.
    """
    h, w = arr.shape[-2], arr.shape[-1]
    coords = np.asarray(coords).reshape(-1, 2)
    for r, c in coords:
        r, c = int(r), int(c)
        r0 = max(r - radius, 0)
        r1 = min(r + radius + 1, h)
        c0 = max(c - radius, 0)
        c1 = min(c + radius + 1, w)
        window = arr[..., r0:r1, c0:c1]
        arr[..., r, c] = np.median(window, axis=(-2, -1))
    return arr


def apply_intensity_floor(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Zero values strictly below ``threshold`` (noise-floor cutoff), in place.

    Symmetric to ``mask_hot_pixels`` (which zeros values *above* a
    threshold). Mutates ``arr`` and returns it (no allocation), so this
    is safe to use in streaming hot paths.

    Source: holoptycho/preprocess.py ``ImagePreprocessorOp.compute``
    ``detmap_threshold`` block.
    """
    arr[arr < threshold] = 0
    return arr


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


def compute_object_shape_from_scan(
    x_range_um: float,
    y_range_um: float,
    nx_prb: int,
    ny_prb: int,
    x_pixel_m: float,
    y_pixel_m: float,
    obj_pad: int,
) -> Tuple[int, int]:
    """Compute the object array shape required to cover a scan region.

    Adds the probe size and a fixed pad to the scan range expressed in
    pixels, then rounds each dimension up to the next even integer so
    downstream FFT-based kernels prefer real-FFT-friendly sizes.

    Returns ``(nx_obj, ny_obj)``.

    Source: holoptycho/streaming_recon.py
    ``StreamingReconOp._required_object_shape`` (factored out of the
    class so it doesn't depend on operator state).
    """
    if x_pixel_m <= 0 or y_pixel_m <= 0:
        raise ValueError("Pixel sizes must be positive.")
    nx_obj = int(nx_prb + np.ceil(abs(x_range_um) * 1e-6 / x_pixel_m) + obj_pad)
    ny_obj = int(ny_prb + np.ceil(abs(y_range_um) * 1e-6 / y_pixel_m) + obj_pad)
    nx_obj += nx_obj % 2
    ny_obj += ny_obj % 2
    return nx_obj, ny_obj


def apply_angle_correction_x(value, angle_deg: float):
    """Rescale an x-axis quantity (range or position) by the rotation angle.

    Multiplies by ``|cos(angle)|`` for ``|angle| <= 45°`` and by
    ``|sin(angle)|`` otherwise. ``value`` may be a scalar or an array;
    returned as the same type. Does *not* apply the additional sign flip
    used in some HXN flows for ``angle <= -45°``; callers needing that
    should apply it separately.

    Source: holoptycho/ptycho_holo.py ``X-axis rescale by rotation angle``
    block (``self.angle_correction_flag`` branch).
    """
    if np.abs(angle_deg) <= 45.0:
        return value * np.abs(np.cos(angle_deg * np.pi / 180.0))
    return value * np.abs(np.sin(angle_deg * np.pi / 180.0))


def auto_detect_roi_offsets(
    frames: np.ndarray,
    nx: int,
    ny: int,
    n_sample: int = 50,
) -> Tuple[int, int]:
    """Auto-detect detector ROI offsets from the diffraction-pattern center.

    Averages up to ``n_sample`` frames, masks pixels saturated at the
    dtype max (hot pixels / detector artifacts that drag the COM off
    course), then computes the intensity-weighted center of mass and
    returns ``(bx0, by0)`` such that an ``nx × ny`` crop is centered on
    it. Returns ``(0, 0)`` if the masked frame has zero total intensity.

    Source: holoptycho/scripts/replay_from_tiled.py ``_auto_batch_offsets``.
    """
    sample = frames[:min(n_sample, len(frames))].astype(np.float64)
    mean_frame = sample.mean(axis=0)
    sat_mask = (sample == np.iinfo(frames.dtype).max).any(axis=0)
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


def rm_outlier_pixels(
    data: np.ndarray,
    rows,
    cols,
    set_to_zero: bool = False,
) -> np.ndarray:
    """Replace outlier pixels at known ``(rows[i], cols[i])`` locations, in place.

    Variant of :func:`inpaint_bad_pixels` that uses parallel ``rows`` and
    ``cols`` arrays (rather than a ``(K, 2)`` coords array) and offers a
    ``set_to_zero`` shortcut. Mutates ``data`` and returns it.

    Note: faithfully copied from upstream — the median window is
    ``data[x-1:x+1, y-1:y+1]`` (a 2×2 upper-left, *not* a 3×3 centered
    window). This is a minor quirk of the upstream implementation.

    Source: ptycho_gui/nsls2ptycho/core/widgets/imgTools.py ``rm_outlier_pixels``.
    """
    if set_to_zero:
        data[rows, cols] = 0.0
    else:
        assert len(rows) == len(cols)
        for x, y in zip(rows, cols):
            data[x, y] = np.median(data[x - 1:x + 1, y - 1:y + 1])
    return data


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


def array_ensure_positive_elements(arr: np.ndarray) -> None:
    """Replace zero / negative values in a 1D array with the closest *following* positive value.

    Iterates the array in reverse so a non-positive entry is filled with
    the next valid value to its right (more likely to belong to the same
    sub-scan than the preceding one). Mutates ``arr`` in place; returns
    ``None``. If the array contains no positive values, it is left
    unchanged — callers should validate ``np.any(arr > 0)`` themselves
    when that matters.

    Source: ptycho_gui/nsls2ptycho/core/HXN_databroker.py
    ``array_ensure_positive_elements`` (upstream ``name`` parameter and
    diagnostic prints dropped — library code should be quiet).
    """
    n_items_to_replace = int(np.sum(arr <= 0))
    if not n_items_to_replace:
        return

    v_closest_positive = None
    for v in np.flip(arr):
        if v > 0:
            v_closest_positive = v
            break

    if v_closest_positive is None:
        return

    n_replaced = 0
    for n in reversed(range(arr.size)):
        if arr[n] <= 0:
            arr[n] = v_closest_positive
            n_replaced += 1
            if n_replaced == n_items_to_replace:
                break
        else:
            v_closest_positive = arr[n]


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
