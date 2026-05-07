"""Array-in / array-out preprocessing utilities for ptychography data.

These helpers operate on plain numpy arrays so they can be reused by any
caller — HXN HDF5 pipelines, holoptycho's streaming Holoscan operators,
notebook one-offs — without dragging in HDF5, MPI, or filesystem
dependencies.

Provenance
----------
Each function below has a ``Source:`` line in its docstring naming the
upstream file/function it was lifted from. Two upstreams contribute:

* ``holoptycho`` — https://github.com/NSLS2/holoptycho (live streaming
  Holoscan pipeline). Inline array ops have been pulled out of Operator
  ``compute()`` methods into pure functions.
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
    """Replace values strictly greater than ``threshold`` with ``fill``.

    Returns a copy; the input array is not modified.

    Source: HXN h5_conv ``load_ptycho_data`` inline ``raw_counts > 60000``
    handler (provided via temp_code).
    """
    out = arr.copy()
    out[out > threshold] = fill
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
    """Replace known bad-pixel coordinates with the median of their neighbourhood.

    For each ``(row, col)`` in ``coords``, replaces the pixel at that
    location with the median of the surrounding ``(2*radius+1) × (2*radius+1)``
    window. Operates on the last two axes; works for both 2D arrays and
    stacks of shape ``(N, H, W)``. Returns a copy.

    Source: holoptycho/preprocess.py ``ImagePreprocessorOp.compute`` inline
    bad-pixel inpainting loop.
    """
    out = arr.copy()
    h, w = out.shape[-2], out.shape[-1]
    coords = np.asarray(coords).reshape(-1, 2)
    for r, c in coords:
        r, c = int(r), int(c)
        r0 = max(r - radius, 0)
        r1 = min(r + radius + 1, h)
        c0 = max(c - radius, 0)
        c1 = min(c + radius + 1, w)
        window = out[..., r0:r1, c0:c1]
        out[..., r, c] = np.median(window, axis=(-2, -1))
    return out


def apply_intensity_floor(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Zero values strictly below ``threshold`` (noise-floor cutoff).

    Returns a copy; the input array is not modified. Symmetric to
    ``mask_hot_pixels`` (which zeros values *above* a threshold).

    Source: holoptycho/preprocess.py ``ImagePreprocessorOp.compute``
    ``detmap_threshold`` block.
    """
    out = arr.copy()
    out[out < threshold] = 0
    return out


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
