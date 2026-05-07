"""Tests for ptychoml.preprocess utilities."""
import numpy as np
import pytest

from ptychoml.preprocess import (
    apply_intensity_floor,
    auto_detect_roi_offsets,
    compute_sample_pixel_size,
    crop_to_roi,
    estimate_roi,
    find_outlier_pixels,
    fourier_shift,
    inpaint_bad_pixels,
    mask_hot_pixels,
    normalize_intensity,
    resize_diffraction_patterns,
    zero_pad_to_target,
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


# ----- mask_hot_pixels ------------------------------------------------------

def test_mask_hot_pixels_above_threshold_replaced():
    arr = np.array([[10.0, 100.0], [60001.0, 5.0]], dtype=np.float32)
    out = mask_hot_pixels(arr, threshold=60000.0, fill=0.0)
    np.testing.assert_array_equal(
        out, np.array([[10.0, 100.0], [0.0, 5.0]], dtype=np.float32)
    )


def test_mask_hot_pixels_mutates_in_place():
    arr = np.array([60001.0, 1.0], dtype=np.float32)
    out = mask_hot_pixels(arr, threshold=60000.0)
    assert out is arr  # same object — no allocation
    np.testing.assert_array_equal(arr, np.array([0.0, 1.0], dtype=np.float32))


def test_mask_hot_pixels_custom_fill():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = mask_hot_pixels(arr, threshold=1.5, fill=-1.0)
    np.testing.assert_array_equal(out, np.array([1.0, -1.0, -1.0], dtype=np.float32))


# ----- cupy compatibility (gated) -------------------------------------------

def test_mask_hot_pixels_works_on_cupy():
    cp = pytest.importorskip("cupy")
    arr = cp.asarray(np.array([60001.0, 1.0], dtype=np.float32))
    out = mask_hot_pixels(arr, threshold=60000.0)
    assert out is arr
    np.testing.assert_array_equal(cp.asnumpy(arr), np.array([0.0, 1.0], dtype=np.float32))


def test_apply_intensity_floor_works_on_cupy():
    cp = pytest.importorskip("cupy")
    arr = cp.asarray(np.array([0.1, 5.0], dtype=np.float32))
    out = apply_intensity_floor(arr, threshold=1.0)
    assert out is arr
    np.testing.assert_array_equal(cp.asnumpy(arr), np.array([0.0, 5.0], dtype=np.float32))


def test_inpaint_bad_pixels_works_on_cupy():
    cp = pytest.importorskip("cupy")
    arr_cpu = np.full((5, 5), 10.0, dtype=np.float32)
    arr_cpu[2, 2] = 999.0
    arr = cp.asarray(arr_cpu)
    out = inpaint_bad_pixels(arr, coords=[(2, 2)])
    assert out is arr
    assert float(arr[2, 2]) == 10.0


def test_crop_to_roi_works_on_cupy():
    cp = pytest.importorskip("cupy")
    arr = cp.asarray(np.arange(20 * 30, dtype=np.float32).reshape(20, 30))
    out = crop_to_roi(arr, [[5, 15], [10, 25]])
    assert out.shape == (10, 15)
    # Output should still be a cupy array.
    assert isinstance(out, cp.ndarray)




# ----- compute_sample_pixel_size --------------------------------------------

# ----- crop_to_roi ----------------------------------------------------------

def test_crop_to_roi_basic():
    arr = np.arange(20 * 30, dtype=np.float32).reshape(20, 30)
    roi = [[5, 15], [10, 25]]
    out = crop_to_roi(arr, roi)
    assert out.shape == (10, 15)
    np.testing.assert_array_equal(out, arr[5:15, 10:25])


def test_crop_to_roi_3d_stack():
    """Leading batch dim is preserved; only last two axes are cropped."""
    arr = np.arange(4 * 20 * 30, dtype=np.float32).reshape(4, 20, 30)
    roi = np.array([[5, 15], [10, 25]])
    out = crop_to_roi(arr, roi)
    assert out.shape == (4, 10, 15)
    np.testing.assert_array_equal(out, arr[:, 5:15, 10:25])


# ----- inpaint_bad_pixels ---------------------------------------------------

def test_inpaint_bad_pixels_replaces_with_median():
    # 5x5 array, all 10s except a "bad pixel" of 999 at (2, 2).
    arr = np.full((5, 5), 10.0, dtype=np.float32)
    arr[2, 2] = 999.0
    out = inpaint_bad_pixels(arr, coords=[(2, 2)], radius=1)
    # 3x3 neighborhood around (2,2) is eight 10s and one 999 → median = 10.
    assert out[2, 2] == 10.0
    # Other pixels untouched.
    assert out[0, 0] == 10.0


def test_inpaint_bad_pixels_3d_stack():
    """Per-frame median across a (N, H, W) stack."""
    stack = np.zeros((3, 5, 5), dtype=np.float32)
    for i in range(3):
        stack[i] = float(i + 1)  # frame i is filled with i+1
        stack[i, 2, 2] = 999.0    # bad pixel in each
    out = inpaint_bad_pixels(stack, coords=[(2, 2)])
    # Each frame's bad pixel takes its own neighborhood median.
    for i in range(3):
        assert out[i, 2, 2] == float(i + 1)


def test_inpaint_bad_pixels_mutates_in_place():
    arr = np.full((5, 5), 10.0, dtype=np.float32)
    arr[2, 2] = 999.0
    out = inpaint_bad_pixels(arr, coords=[(2, 2)])
    assert out is arr  # same object — no allocation
    assert arr[2, 2] == 10.0


# ----- apply_intensity_floor ------------------------------------------------

def test_apply_intensity_floor_below_threshold_zeroed():
    arr = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    out = apply_intensity_floor(arr, threshold=1.5)
    np.testing.assert_array_equal(
        out, np.array([0.0, 0.0, 1.5, 2.0], dtype=np.float32)
    )


def test_apply_intensity_floor_mutates_in_place():
    arr = np.array([0.1, 5.0], dtype=np.float32)
    out = apply_intensity_floor(arr, threshold=1.0)
    assert out is arr  # same object — no allocation
    np.testing.assert_array_equal(arr, np.array([0.0, 5.0], dtype=np.float32))


# ----- fourier_shift --------------------------------------------------------

def test_fourier_shift_integer_shift_matches_roll():
    """An integer Fourier shift should match np.roll on a smooth input."""
    rng = np.random.default_rng(0)
    h, w = 16, 16
    img = rng.standard_normal((1, h, w)).astype(np.float32)
    shifts = np.array([[3, -2]], dtype=np.float32)  # (dy, dx)
    out = fourier_shift(img, shifts)
    expected = np.roll(img, shift=(3, -2), axis=(-2, -1))
    np.testing.assert_allclose(out, expected, atol=1e-3)


def test_fourier_shift_zero_shift_is_identity():
    rng = np.random.default_rng(1)
    img = rng.standard_normal((2, 8, 8)).astype(np.float32)
    out = fourier_shift(img, np.zeros((2, 2), dtype=np.float32))
    np.testing.assert_allclose(out, img, atol=1e-4)


# ----- auto_detect_roi_offsets ----------------------------------------------

def test_auto_detect_roi_offsets_finds_known_center():
    """A bright Gaussian-like blob at a known center should be recovered."""
    H, W = 200, 256
    cy, cx = 130, 90
    ys, xs = np.indices((H, W))
    blob = np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / 50.0)
    frames = (blob * 1000.0).astype(np.uint16)[None].repeat(20, axis=0)
    bx0, by0 = auto_detect_roi_offsets(frames, nx=64, ny=64)
    # Crop should be centered on the blob: bx0 ≈ cx - 32, by0 ≈ cy - 32.
    assert abs(bx0 - (cx - 32)) <= 1
    assert abs(by0 - (cy - 32)) <= 1


def test_auto_detect_roi_offsets_handles_saturation():
    """Pixels above saturation_threshold should be masked and not pull the COM."""
    H, W = 64, 64
    cy, cx = 40, 30
    ys, xs = np.indices((H, W))
    blob = np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / 20.0)
    frames = (blob * 100.0).astype(np.uint16)
    # Inject a saturated pixel that would otherwise drag the COM.
    frames[5, 5] = np.iinfo(np.uint16).max
    frames = frames[None].repeat(10, axis=0)
    bx0, by0 = auto_detect_roi_offsets(
        frames, nx=16, ny=16, saturation_threshold=60000
    )
    # Without masking, the saturated pixel at (5, 5) would skew the center.
    # With masking, we recover something near the true blob center.
    assert abs(bx0 - (cx - 8)) <= 2
    assert abs(by0 - (cy - 8)) <= 2


def test_auto_detect_roi_offsets_zero_frames_returns_origin():
    frames = np.zeros((5, 32, 32), dtype=np.uint16)
    assert auto_detect_roi_offsets(frames, nx=16, ny=16) == (0, 0)


# ----- find_outlier_pixels --------------------------------------------------

def test_find_outlier_pixels_detects_injected_hot_pixel():
    rng = np.random.default_rng(42)
    img = rng.normal(loc=100.0, scale=1.0, size=(20, 20))
    img[8, 12] = 5000.0  # injected hot pixel
    coords = find_outlier_pixels(img, get_fixed_image=False)
    # coords is shape (2, K); should contain (8, 12).
    found = list(zip(coords[0], coords[1]))
    assert (8, 12) in found


def test_find_outlier_pixels_returns_fixed_image():
    rng = np.random.default_rng(0)
    img = rng.normal(loc=100.0, scale=1.0, size=(15, 15))
    img[7, 7] = 5000.0
    _, fixed = find_outlier_pixels(img, get_fixed_image=True, worry_about_edges=False)
    # The hot pixel should be replaced with something near the local mean.
    assert abs(fixed[7, 7] - 100.0) < 10.0


# ----- estimate_roi ---------------------------------------------------------

def test_estimate_roi_finds_central_block():
    """A bright square in a dark image — ROI should be a valid non-empty box."""
    img = np.zeros((100, 100), dtype=np.float32)
    img[30:70, 40:80] = 1.0
    x0, y0, w, h = estimate_roi(img, threshold=0.1)
    assert 0 <= x0 < 100
    assert 0 <= y0 < 100
    assert w > 0
    assert h > 0


# ----- zero_pad_to_target ---------------------------------------------------

def test_zero_pad_to_target_centers_content():
    img = np.ones((100, 100), dtype=np.float32) * 7.0
    out = zero_pad_to_target(img, target_size=256)
    assert out.shape == (256, 256)
    py = (256 - 100) // 2
    px = (256 - 100) // 2
    np.testing.assert_array_equal(out[py:py + 100, px:px + 100], img)
    assert out[0, 0] == 0.0


def test_zero_pad_to_target_no_op_when_at_target():
    rng = np.random.default_rng(0)
    img = rng.random((64, 64), dtype=np.float32)
    out = zero_pad_to_target(img, target_size=64)
    assert out is img  # no allocation when already at target


def test_zero_pad_to_target_raises_when_larger():
    img = np.zeros((300, 300), dtype=np.float32)
    with pytest.raises(ValueError, match="larger than target size"):
        zero_pad_to_target(img, target_size=256)


# ----- normalize_intensity --------------------------------------------------

def test_normalize_intensity_scales_correctly():
    arr = np.array([2.0, 4.0, 6.0], dtype=np.float64)
    out = normalize_intensity(arr, normalization=2.0, scale=3.0)
    np.testing.assert_array_equal(out, np.array([3.0, 6.0, 9.0]))


def test_normalize_intensity_default_scale_is_one():
    arr = np.array([2.0, 4.0], dtype=np.float64)
    out = normalize_intensity(arr, normalization=2.0)
    np.testing.assert_array_equal(out, np.array([1.0, 2.0]))


def test_normalize_intensity_does_not_mutate():
    arr = np.array([1.0, 2.0], dtype=np.float64)
    original = arr.copy()
    _ = normalize_intensity(arr, normalization=2.0, scale=3.0)
    np.testing.assert_array_equal(arr, original)


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
