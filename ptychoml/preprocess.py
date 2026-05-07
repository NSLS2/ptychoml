"""Array-in / array-out preprocessing utilities for ptychography data.

These helpers operate on plain numpy arrays so they can be reused by any
caller — HXN HDF5 pipelines, holoptycho's streaming Holoscan operators,
notebook one-offs — without dragging in HDF5, MPI, or filesystem
dependencies.

Per-frame argmax centering note
-------------------------------
``resize_diffraction_patterns`` finds the crop center independently for
each frame using ``np.argmax``. Saturated / hot pixels can therefore
mislead the centering. Mask them with ``mask_hot_pixels`` (or pre-crop
to a detector ROI) before calling.
"""
from __future__ import annotations

from typing import Iterable, Union

import numpy as np

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

    Parameters
    ----------
    dp : sequence or ndarray
        Iterable of 2D patterns, or a 3D ndarray of shape ``(N, H, W)``.
    target_n : int
        Output edge length.

    Returns
    -------
    ndarray
        Stacked output of shape ``(N, target_n, target_n)`` with the
        input dtype preserved.
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

    Parameters
    ----------
    obj : ndarray
        Object array of shape ``(S, H, W)``.
    scale_y, scale_x : float
        The rescale factors that were applied to H and W respectively.
    obj_pad : int
        Number of fixed padding pixels the downstream backend allocates.

    Returns
    -------
    ndarray
        Adjusted object with corrected H and W.
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
    """
    return wavelength_m * detector_distance_m / (n_pixels * ccd_pixel_size_m)


def crop_to_roi(arr: np.ndarray, roi) -> np.ndarray:
    """Crop the last two axes of ``arr`` to a fixed ``[[y0, y1], [x0, x1]]`` ROI.

    Used when the crop window is known from detector calibration and should
    be applied identically to every frame (e.g. holoptycho streaming). The
    ROI uses Python half-open ranges: ``[y0, y1)`` rows, ``[x0, x1)`` cols.

    Parameters
    ----------
    arr : ndarray
        Array of shape ``(..., H, W)``. Leading dims are preserved.
    roi : array-like of shape (2, 2)
        ``[[y0, y1], [x0, x1]]``.

    Returns
    -------
    ndarray
        View into ``arr`` for the requested window.
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

    Parameters
    ----------
    arr : ndarray
        Array of shape ``(..., H, W)``.
    coords : array-like of shape (K, 2)
        ``(row, col)`` pairs of bad-pixel locations.
    radius : int
        Half-window size. Default 1 (3×3 neighborhood).

    Returns
    -------
    ndarray
        A copy of ``arr`` with the bad-pixel locations replaced.
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
    """
    out = arr.copy()
    out[out < threshold] = 0
    return out
