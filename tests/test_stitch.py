"""Tests for ptychoml.stitch patch-placement / mosaic stitching utilities."""
import numpy as np
import pytest

from ptychoml.stitch import (
    crop_mosaic_border,
    normalize_mosaic,
    place_patches_fourier_shift,
    stitch_batch_into,
    stitch_batch_livestitch_into,
    stitch_batch_nearest,
)


# ----- stitch_batch_nearest -------------------------------------------------

def test_nearest_single_patch_placement():
    """A 4x4 all-ones patch centered at (10, 10) lands at rows/cols 8:12."""
    canvas = np.zeros((20, 20), dtype=np.float32)
    counts = np.zeros((20, 20), dtype=np.float32)
    patches = np.ones((1, 4, 4), dtype=np.float32)
    pos = np.array([[10.0, 10.0]])

    canvas, counts = stitch_batch_nearest(canvas, counts, patches, pos)

    assert canvas[8:12, 8:12].sum() == pytest.approx(16.0)
    assert canvas.sum() == pytest.approx(16.0)
    assert counts[8:12, 8:12].sum() == pytest.approx(16.0)


def test_nearest_does_not_flip():
    """Nearest placement preserves patch orientation (no up-down flip)."""
    canvas = np.zeros((20, 20), dtype=np.float32)
    counts = np.zeros((20, 20), dtype=np.float32)
    patch = np.zeros((4, 4), dtype=np.float32)
    patch[0, :] = 5.0  # top row marker
    patches = patch[None]
    pos = np.array([[10.0, 10.0]])  # -> rows 8:12

    canvas, _ = stitch_batch_nearest(canvas, counts, patches, pos)

    assert canvas[8, 8] == 5.0   # top row stays at the top
    assert canvas[11, 8] == 0.0


def test_nearest_accumulation():
    """Overlapping patches sum in canvas and counts add up."""
    canvas = np.zeros((20, 20), dtype=np.float32)
    counts = np.zeros((20, 20), dtype=np.float32)
    patches = np.ones((2, 4, 4), dtype=np.float32)
    pos = np.array([[10.0, 10.0], [11.0, 10.0]])  # rows 8:12 and 9:13 overlap

    canvas, counts = stitch_batch_nearest(canvas, counts, patches, pos)

    # overlap region rows 9:12 has count 2
    assert counts[9:12, 8:12].min() == 2.0
    assert canvas[9, 8] == 2.0


def test_nearest_edge_clamping_no_wrap():
    """A patch hanging off the top edge is clamped, not wrapped."""
    canvas = np.zeros((20, 20), dtype=np.float32)
    counts = np.zeros((20, 20), dtype=np.float32)
    patches = np.ones((1, 4, 4), dtype=np.float32)
    pos = np.array([[1.0, 10.0]])  # sy = 1 - 2 = -1, rows -1:3 -> clamp to 0:3

    canvas, counts = stitch_batch_nearest(canvas, counts, patches, pos)

    assert counts[0:3, 8:12].min() == 1.0
    assert canvas[17:20, :].sum() == 0.0  # nothing wrapped to the bottom


# ----- stitch_batch_livestitch_into -----------------------------------------

def test_livestitch_bbox_single_patch():
    canvas = np.zeros((20, 20), dtype=np.float32)
    counts = np.zeros((20, 20), dtype=np.float32)
    patches = np.ones((1, 4, 4), dtype=np.float32)
    pos = np.array([[10.0, 10.0]])  # y0 = rint(10-2)=8, y1=12

    canvas, counts, bbox = stitch_batch_livestitch_into(canvas, counts, patches, pos)

    assert bbox == (8, 12, 8, 12)
    assert counts[8:12, 8:12].sum() == pytest.approx(16.0)


def test_livestitch_does_not_flip():
    """LiveStitch preserves patch orientation (no up-down flip), matching nearest."""
    canvas = np.zeros((20, 20), dtype=np.float32)
    counts = np.zeros((20, 20), dtype=np.float32)
    patch = np.zeros((4, 4), dtype=np.float32)
    patch[0, :] = 5.0  # top row marker
    patches = patch[None]
    pos = np.array([[10.0, 10.0]])  # placed at rows 8:12

    canvas, _, _ = stitch_batch_livestitch_into(canvas, counts, patches, pos)

    assert canvas[8, 8] == 5.0   # top row stays at the top (no flip)
    assert canvas[11, 8] == 0.0


def test_livestitch_no_overlap_returns_zero_bbox():
    canvas = np.zeros((20, 20), dtype=np.float32)
    counts = np.zeros((20, 20), dtype=np.float32)
    patches = np.ones((1, 4, 4), dtype=np.float32)
    pos = np.array([[100.0, 100.0]])  # entirely off-canvas

    out_canvas, out_counts, bbox = stitch_batch_livestitch_into(
        canvas, counts, patches, pos
    )

    assert bbox == (0, 0, 0, 0)
    assert out_canvas.sum() == 0.0
    assert out_counts.sum() == 0.0


def test_livestitch_edge_clamping_no_wrap():
    canvas = np.zeros((20, 20), dtype=np.float32)
    counts = np.zeros((20, 20), dtype=np.float32)
    patches = np.ones((1, 4, 4), dtype=np.float32)
    pos = np.array([[1.0, 10.0]])  # y0 = rint(1-2) = -1 -> clamp to 0:3

    canvas, counts, bbox = stitch_batch_livestitch_into(canvas, counts, patches, pos)

    assert bbox[0] == 0 and bbox[1] == 3
    assert counts[0:3, 8:12].min() == 1.0
    assert canvas[17:20, :].sum() == 0.0  # no wrap-around


# ----- place_patches_fourier_shift / stitch_batch_into ----------------------

def test_stitch_into_counts_footprint_odd_patch():
    """With pad=0 and an odd patch at an integer position the fractional shift
    is exactly zero, so the footprint is a clean ph x pw block."""
    canvas = np.zeros((30, 30), dtype=np.float32)
    counts = np.zeros((30, 30), dtype=np.float32)
    patches = np.ones((1, 5, 5), dtype=np.float32)
    pos = np.array([[10.0, 10.0]])  # rows 8:13, cols 8:13

    canvas, counts = stitch_batch_into(canvas, counts, patches, pos, pad=0)

    assert counts[8:13, 8:13].sum() == pytest.approx(25.0)
    assert counts.sum() == pytest.approx(25.0)
    np.testing.assert_allclose(canvas[8:13, 8:13], 1.0, atol=1e-4)


def test_fourier_and_nearest_share_counts_footprint():
    """Fourier-shift (pad=0, odd patch, integer positions) and nearest place
    over the *same* occupancy footprint, so their counts arrays match. (The
    canvas *values* differ — see test_methods_differ_by_updown_flip — so this
    asserts counts only, not canvas.)"""
    pos = np.array([[10.0, 12.0], [18.0, 9.0], [14.0, 20.0]])
    patches = np.ones((len(pos), 5, 5), dtype=np.float32)

    cf = np.zeros((30, 30), dtype=np.float32)
    nf = np.zeros((30, 30), dtype=np.float32)
    cf, nf = stitch_batch_into(cf, nf, patches.copy(), pos, pad=0)

    cn = np.zeros((30, 30), dtype=np.float32)
    nn = np.zeros((30, 30), dtype=np.float32)
    cn, nn = stitch_batch_nearest(cn, nn, patches.copy(), pos)

    np.testing.assert_array_equal(nf, nn)
    # all-ones patch is flip-symmetric, so the *values* also coincide here
    np.testing.assert_allclose(cf, cn, atol=1e-4)


def test_methods_differ_by_updown_flip():
    """nearest and livestitch place patches as-is (no flip); the Fourier path
    flips up-down due to its FFT phase-ramp convention. Verified with a
    non-symmetric patch whose only non-zero row is the top one."""
    patch = np.zeros((5, 5), dtype=np.float32)
    patch[0, :] = 9.0  # top-row marker
    pos = np.array([[15.0, 15.0]])

    def marker_row(fn, **kw):
        canvas = np.zeros((30, 30), dtype=np.float32)
        counts = np.zeros((30, 30), dtype=np.float32)
        out = fn(canvas, counts, patch[None].copy(), pos, **kw)
        canvas = out[0]
        return int(np.argwhere(canvas > 1)[:, 0].mean())

    row_nearest = marker_row(stitch_batch_nearest)
    row_fourier = marker_row(stitch_batch_into, pad=0)
    row_live    = marker_row(stitch_batch_livestitch_into)

    # nearest and livestitch both preserve orientation (no flip); they differ
    # by at most 1 px due to different half-integer rounding conventions
    assert row_nearest == 13   # round(pos) - ph//2
    assert row_live    == 12   # rint(pos - ph/2) -- banker's rounding
    # Fourier path flips, pushing the marker toward the bottom
    assert row_fourier == 17
    assert row_nearest < row_fourier
    assert row_live    < row_fourier


def test_stitch_into_per_batch_equals_one_shot():
    """Scatter-add is associative: splitting a batch gives the same canvas and
    counts (up to FFT noise) as stitching all patches at once."""
    rng = np.random.default_rng(1)
    n = 8
    pos = rng.uniform(20, 80, size=(n, 2))
    patches = rng.standard_normal((n, 8, 8)).astype(np.float32)

    one_c = np.zeros((100, 100), dtype=np.float32)
    one_n = np.zeros((100, 100), dtype=np.float32)
    one_c, one_n = stitch_batch_into(one_c, one_n, patches.copy(), pos)

    split_c = np.zeros((100, 100), dtype=np.float32)
    split_n = np.zeros((100, 100), dtype=np.float32)
    split_c, split_n = stitch_batch_into(
        split_c, split_n, patches[:4].copy(), pos[:4]
    )
    split_c, split_n = stitch_batch_into(
        split_c, split_n, patches[4:].copy(), pos[4:]
    )

    np.testing.assert_allclose(one_c, split_c, atol=1e-3)
    np.testing.assert_array_equal(one_n, split_n)


def test_place_patches_fourier_shift_boundary_padding():
    """Patches straddling the canvas edge are placed via internal padding and
    the returned image keeps the original shape (no wrap-around)."""
    image = np.zeros((20, 20), dtype=np.float32)
    patches = np.ones((1, 6, 6), dtype=np.float32)
    pos = np.array([[1.0, 1.0]])  # patch center near the top-left corner

    out = place_patches_fourier_shift(image, pos, patches, pad=1)

    assert out.shape == (20, 20)
    assert out[10:, 10:].sum() == 0.0  # nothing leaked to the far corner
    assert out.sum() > 0.0  # the in-bounds portion was placed


# ----- normalize_mosaic -----------------------------------------------------

def test_normalize_mosaic_averages_covered_pixels():
    canvas = np.array([[6.0, 0.0], [9.0, 4.0]], dtype=np.float32)
    counts = np.array([[3.0, 0.0], [3.0, 2.0]], dtype=np.float32)

    fill, mosaic = normalize_mosaic(canvas, counts, min_overlap=0.5)

    # covered pixels = canvas / counts
    assert mosaic[0, 0] == pytest.approx(2.0)
    assert mosaic[1, 0] == pytest.approx(3.0)
    assert mosaic[1, 1] == pytest.approx(2.0)
    # the count==0 pixel is under-covered -> NaN
    assert np.isnan(mosaic[0, 1])


def test_normalize_mosaic_masks_undercovered_to_nan():
    canvas = np.array([[5.0, 5.0]], dtype=np.float32)
    counts = np.array([[5.0, 0.0]], dtype=np.float32)

    _, mosaic = normalize_mosaic(canvas, counts)

    assert mosaic[0, 0] == pytest.approx(1.0)
    assert np.isnan(mosaic[0, 1])


def test_normalize_mosaic_fill_is_median_of_covered():
    # covered averaged values are 1, 2, 3 -> median 2; the NaN pixel is excluded
    canvas = np.array([[1.0, 2.0, 3.0, 7.0]], dtype=np.float32)
    counts = np.array([[1.0, 1.0, 1.0, 0.0]], dtype=np.float32)

    fill, _ = normalize_mosaic(canvas, counts)

    assert fill == pytest.approx(2.0)


def test_normalize_mosaic_threshold_is_inclusive():
    # count exactly equal to the threshold counts as covered
    canvas = np.array([[4.0]], dtype=np.float32)
    counts = np.array([[2.0]], dtype=np.float32)

    _, mosaic = normalize_mosaic(canvas, counts, min_overlap=2.0)

    assert mosaic[0, 0] == pytest.approx(2.0)
    assert not np.isnan(mosaic[0, 0])


def test_normalize_mosaic_higher_threshold_excludes_thin_coverage():
    canvas = np.array([[2.0, 6.0]], dtype=np.float32)
    counts = np.array([[1.0, 3.0]], dtype=np.float32)

    _, mosaic = normalize_mosaic(canvas, counts, min_overlap=2.0)

    assert np.isnan(mosaic[0, 0])          # count 1 < 2 -> dropped
    assert mosaic[0, 1] == pytest.approx(2.0)


def test_normalize_mosaic_empty_returns_zero_and_all_nan():
    canvas = np.zeros((3, 3), dtype=np.float32)
    counts = np.zeros((3, 3), dtype=np.float32)

    fill, mosaic = normalize_mosaic(canvas, counts)

    assert fill == 0.0
    assert mosaic.shape == (3, 3)
    assert np.isnan(mosaic).all()


def test_normalize_mosaic_returns_float32():
    canvas = np.ones((2, 2), dtype=np.float64)
    counts = np.ones((2, 2), dtype=np.float64)

    _, mosaic = normalize_mosaic(canvas, counts)

    assert mosaic.dtype == np.float32


def test_normalize_mosaic_does_not_mutate_inputs():
    canvas = np.array([[4.0, 0.0]], dtype=np.float32)
    counts = np.array([[2.0, 0.0]], dtype=np.float32)
    canvas_ref = canvas.copy()
    counts_ref = counts.copy()

    normalize_mosaic(canvas, counts)

    np.testing.assert_array_equal(canvas, canvas_ref)
    np.testing.assert_array_equal(counts, counts_ref)


def test_normalize_mosaic_after_livestitch_recovers_patch_value():
    # stitch two non-overlapping all-ones patches, then normalize: covered
    # region should read back the patch value (1.0), rest NaN.
    canvas = np.zeros((40, 40), dtype=np.float32)
    counts = np.zeros((40, 40), dtype=np.float32)
    patches = np.ones((2, 6, 6), dtype=np.float32)
    positions = np.array([[10.0, 10.0], [10.0, 30.0]])

    canvas, counts, _ = stitch_batch_livestitch_into(canvas, counts, patches, positions)
    fill, mosaic = normalize_mosaic(canvas, counts)

    covered = counts >= 0.5
    assert covered.sum() == 2 * 6 * 6
    np.testing.assert_allclose(mosaic[covered], 1.0, atol=1e-6)
    assert np.isnan(mosaic[~covered]).all()
    assert fill == pytest.approx(1.0)


# ----- crop_mosaic_border ---------------------------------------------------

def test_crop_mosaic_border_explicit_border():
    """Explicit border=64 crops 64 px each side."""
    mosaic = np.ones((700, 700), dtype=np.float32)
    cropped = crop_mosaic_border(mosaic, border=64)
    assert cropped.shape == (700 - 2 * 64, 700 - 2 * 64)


def test_crop_mosaic_border_no_crop_when_border_zero():
    """border=0 → mosaic returned unchanged."""
    mosaic = np.ones((400, 400), dtype=np.float32)
    cropped = crop_mosaic_border(mosaic, border=0)
    assert cropped.shape == mosaic.shape
    assert cropped is mosaic


def test_crop_mosaic_border_preserves_values():
    """Interior values are unchanged after crop."""
    mosaic = np.arange(400, dtype=np.float32).reshape(20, 20)
    cropped = crop_mosaic_border(mosaic, border=4)
    np.testing.assert_array_equal(cropped, mosaic[4:-4, 4:-4])


def test_crop_mosaic_border_default_is_noop():
    """Default border=0 returns mosaic unchanged."""
    mosaic = np.ones((300, 300), dtype=np.float32)
    cropped = crop_mosaic_border(mosaic)
    assert cropped is mosaic
