"""Tests for ptychoml.preprocess utilities."""
import numpy as np
import pytest

from ptychoml.preprocess import (
    adjust_object_for_pad,
    compute_sample_pixel_size,
    mask_hot_pixels,
    resize_diffraction_patterns,
)


# ----- resize_diffraction_patterns ------------------------------------------

def test_resize_diffraction_patterns_crop():
    pattern = np.zeros((384, 384), dtype=np.float32)
    peak_y, peak_x = 200, 180
    pattern[peak_y, peak_x] = 100.0
    pattern[peak_y, peak_x + 1] = 50.0  # +x marker for orientation

    out = resize_diffraction_patterns([pattern], target_n=256)

    assert out.shape == (1, 256, 256)
    center = 256 // 2
    assert out[0, center, center] == 100.0
    assert out[0, center, center + 1] == 50.0
    assert out.dtype == np.float32


def test_resize_diffraction_patterns_pad():
    pattern = np.ones((100, 100), dtype=np.float32) * 7.0
    out = resize_diffraction_patterns([pattern], target_n=256)

    assert out.shape == (1, 256, 256)
    # Original content sits centered in the padded output.
    py = (256 - 100) // 2
    px = (256 - 100) // 2
    assert np.all(out[0, py:py + 100, px:px + 100] == 7.0)
    # Borders are zero.
    assert out[0, 0, 0] == 0.0
    assert out[0, -1, -1] == 0.0


def test_resize_diffraction_patterns_no_change():
    rng = np.random.default_rng(42)
    pattern = rng.random((256, 256), dtype=np.float32)
    out = resize_diffraction_patterns([pattern], target_n=256)
    assert out.shape == (1, 256, 256)
    np.testing.assert_array_equal(out[0], pattern)


def test_resize_diffraction_patterns_stacked_input():
    """Function should accept a 3D ndarray, not just a list."""
    stack = np.zeros((3, 384, 384), dtype=np.float32)
    for i in range(3):
        stack[i, 200, 180] = float(i + 1)

    out = resize_diffraction_patterns(stack, target_n=256)
    assert out.shape == (3, 256, 256)
    for i in range(3):
        assert out[i, 128, 128] == float(i + 1)


# ----- adjust_object_for_pad ------------------------------------------------

def test_adjust_object_for_pad_trim():
    obj = np.ones((1, 100, 100), dtype=np.complex64)
    # scale > 1 → trim by obj_pad*(scale-1) on each axis
    out = adjust_object_for_pad(obj, scale_y=2.0, scale_x=2.0, obj_pad=10)
    # corr = round(10 * 1.0) = 10, split 5/5 → trim 10 each axis
    assert out.shape == (1, 90, 90)
    # Center value preserved (still 1+0j).
    assert out[0, 45, 45] == 1.0 + 0j


def test_adjust_object_for_pad_pad():
    obj = np.ones((1, 100, 100), dtype=np.complex64)
    # scale < 1 → zero-pad
    out = adjust_object_for_pad(obj, scale_y=0.5, scale_x=0.5, obj_pad=10)
    # corr = round(10 * -0.5) = -5, pad 5 each axis (split 2/3)
    assert out.shape == (1, 105, 105)
    # Padded edges are zero.
    assert out[0, 0, 0] == 0.0 + 0j
    assert out[0, -1, -1] == 0.0 + 0j
    # Original content preserved somewhere in the middle.
    assert np.any(out[0] == 1.0 + 0j)


def test_adjust_object_for_pad_noop():
    obj = np.arange(1 * 4 * 5, dtype=np.complex64).reshape(1, 4, 5)
    out = adjust_object_for_pad(obj, scale_y=1.0, scale_x=1.0, obj_pad=10)
    np.testing.assert_array_equal(out, obj)


# ----- mask_hot_pixels ------------------------------------------------------

def test_mask_hot_pixels_above_threshold_replaced():
    arr = np.array([[10.0, 100.0], [60001.0, 5.0]], dtype=np.float32)
    out = mask_hot_pixels(arr, threshold=60000.0, fill=0.0)
    np.testing.assert_array_equal(
        out, np.array([[10.0, 100.0], [0.0, 5.0]], dtype=np.float32)
    )


def test_mask_hot_pixels_does_not_mutate_input():
    arr = np.array([60001.0, 1.0], dtype=np.float32)
    original = arr.copy()
    _ = mask_hot_pixels(arr, threshold=60000.0)
    np.testing.assert_array_equal(arr, original)


def test_mask_hot_pixels_custom_fill():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = mask_hot_pixels(arr, threshold=1.5, fill=-1.0)
    np.testing.assert_array_equal(out, np.array([1.0, -1.0, -1.0], dtype=np.float32))


# ----- compute_sample_pixel_size --------------------------------------------

def test_compute_sample_pixel_size_known_value():
    # HXN-typical: λ ≈ 0.124 nm @ 10 keV, z = 1.92 m, ccd = 55 µm, N = 256.
    wavelength_m = 0.124e-9
    detector_distance_m = 1.92
    ccd_pixel_size_m = 55e-6
    n_pixels = 256

    out = compute_sample_pixel_size(
        wavelength_m, detector_distance_m, ccd_pixel_size_m, n_pixels
    )
    expected = wavelength_m * detector_distance_m / (n_pixels * ccd_pixel_size_m)
    assert out == pytest.approx(expected)
    # Sanity: result is in the few-nm range, not absurd.
    assert 1e-9 < out < 1e-7
