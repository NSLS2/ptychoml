"""Tests for ptychoml.stitch patch-placement / mosaic stitching utilities."""
import numpy as np
import pytest

from ptychoml.stitch import (
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


def test_livestitch_applies_updown_flip():
    """LiveStitch flips patches up-down before placement (matches Fourier path)."""
    canvas = np.zeros((20, 20), dtype=np.float32)
    counts = np.zeros((20, 20), dtype=np.float32)
    patch = np.zeros((4, 4), dtype=np.float32)
    patch[0, :] = 5.0  # top row marker
    patches = patch[None]
    pos = np.array([[10.0, 10.0]])  # placed at rows 8:12

    canvas, _, _ = stitch_batch_livestitch_into(canvas, counts, patches, pos)

    assert canvas[11, 8] == 5.0  # top row flipped to the bottom
    assert canvas[8, 8] == 0.0


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
    """The three strategies are NOT pixel-interchangeable: nearest places the
    patch as-is, while the Fourier and livestitch paths flip it up-down (and
    use slightly different center-rounding). Verified with a non-symmetric
    patch whose only non-zero row is the top one."""
    patch = np.zeros((5, 5), dtype=np.float32)
    patch[0, :] = 9.0  # top-row marker
    pos = np.array([[15.0, 15.0]])

    def marker_row(fn, *, livestitch=False, **kw):
        canvas = np.zeros((30, 30), dtype=np.float32)
        counts = np.zeros((30, 30), dtype=np.float32)
        out = fn(canvas, counts, patch[None].copy(), pos, **kw)
        canvas = out[0]
        return int(np.argwhere(canvas > 1)[:, 0].mean())

    row_nearest = marker_row(stitch_batch_nearest)
    row_fourier = marker_row(stitch_batch_into, pad=0)
    row_live = marker_row(stitch_batch_livestitch_into)

    # nearest keeps the marker near the top of the footprint (no flip)...
    assert row_nearest == 13
    # ...while both flipping paths push it toward the bottom.
    assert row_fourier == 17
    assert row_live == 16
    assert row_nearest < row_fourier and row_nearest < row_live


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
