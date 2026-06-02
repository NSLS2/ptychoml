"""Tests for ptychoml.orientation: _score_forward_consistency and autodetect_orientation.

Layout
------
1. _score_forward_consistency — unit tests for the scorer formula directly.
   These do not use a session or preprocess_diffraction; they call the
   private function with analytically-constructed inputs so a bug in the
   NCC formula, the fft2 axes, or the fftshift branch fails here rather
   than being hidden inside an end-to-end test.

2. autodetect_orientation (end-to-end) — recovery + ranking tests using
   the module-scoped ``recovery_report`` fixture, plus input-validation
   and candidate-restriction tests using the cheap ``null_session``.
"""
import numpy as np
import pytest

from ptychoml import autodetect_orientation
from ptychoml.orientation import _score_forward_consistency


# ============================================================================
# Helpers shared by scorer unit tests
# ============================================================================

def _make_forward_physics(rng, n_frames=4, patch_size=8):
    """Return (probe, amp, phase, i_sim) built from exact forward physics.

    ``i_sim[n] = |fft2(probe * amp[n] * exp(i*phase[n]))|^2``

    All arrays are float32 / complex64. ``i_sim`` is the ground-truth
    measured intensity the scorer should match perfectly when given the
    correct ``amp`` and ``phase``.
    """
    probe = (
        rng.normal(size=(patch_size, patch_size)).astype(np.float32)
        + 1j * rng.normal(size=(patch_size, patch_size)).astype(np.float32)
    ).astype(np.complex64)

    amp = rng.uniform(0.5, 1.5, size=(n_frames, patch_size, patch_size)).astype(np.float32)
    phase = rng.uniform(-1.0, 1.0, size=(n_frames, patch_size, patch_size)).astype(np.float32)

    psi = amp.astype(np.complex64) * np.exp(1j * phase.astype(np.float32))
    wavefront = probe[None] * psi
    fft = np.fft.fft2(wavefront, axes=(-2, -1))
    i_sim = (fft.real ** 2 + fft.imag ** 2).astype(np.float32)

    return probe, amp, phase, i_sim


# ============================================================================
# 1. _score_forward_consistency unit tests
# ============================================================================

def test_scorer_perfect_match_returns_zero():
    """When amp/phase are the exact ground truth the NCC is 1 → score is 0."""
    rng = np.random.default_rng(0)
    probe, amp, phase, i_sim = _make_forward_physics(rng)
    score = _score_forward_consistency(
        amp, phase, probe, i_sim, apply_fftshift=False,
    )
    assert score == pytest.approx(0.0, abs=1e-5)


def test_scorer_analytical_ncc_value():
    """Score = 1 - NCC against a fully hand-computed, non-trivial NCC.

    Uses a trivial probe (all ones) and a single frame of constant
    amplitude=1, phase=0 so the scorer's internal wavefront is fft2(ones),
    which has a closed form: only the DC bin is nonzero. For a 4x4 patch
    ``fft2(ones)`` has DC value N²=16, so ``i_sim`` is 16²=256 at [0,0] and
    0 everywhere else.

    Scoring that against ``measured = ones`` (16 ones) gives a known NCC
    that exercises the actual 1-NCC arithmetic, not just the degenerate
    perfect-match → 0 case:

        num  = Σ i_sim·measured = 256·1               = 256
        |i_sim|   = √(256²)                            = 256
        |measured| = √(16·1²)                          = 4
        NCC  = 256 / (256·4)                           = 0.25
        score = 1 - NCC                                = 0.75
    """
    patch_size = 4
    n_frames = 1
    probe = np.ones((patch_size, patch_size), dtype=np.complex64)
    amp = np.ones((n_frames, patch_size, patch_size), dtype=np.float32)
    phase = np.zeros((n_frames, patch_size, patch_size), dtype=np.float32)

    # i_sim is DC-only (256 at [0,0]); score it against a uniform measured.
    measured = np.ones((n_frames, patch_size, patch_size), dtype=np.float32)

    score = _score_forward_consistency(amp, phase, probe, measured, apply_fftshift=False)
    assert score == pytest.approx(0.75, abs=1e-6)


@pytest.mark.parametrize("k", [1e-3, 0.5, 7.0, 1e6])
def test_scorer_is_invariant_to_measured_scale(k):
    """A positive global scale on ``measured`` must not change the score.

    This guards the property ``autodetect_orientation`` relies on but never
    sets up explicitly: it feeds ``measured = diff_amp**2`` to the scorer,
    which still carries the global ``(scale / normalization)`` factor from
    ``preprocess_diffraction`` (default ``scale=10000``), while ``i_sim`` is
    built from the model's amplitude-scale patches and carries no such
    factor. NCC normalises both vectors, so the constant cancels — the
    comment at orientation.py "NCC absorbs it" depends on exactly this.

    A regression to a non-normalised metric (e.g. MSE) would still pass
    every other scorer test (they all build ``measured`` at the same scale
    as ``i_sim``) but would silently break real inference. This is the test
    that fails first if that invariance is ever lost.
    """
    rng = np.random.default_rng(6)
    probe, amp, phase, i_sim = _make_forward_physics(rng)

    score_unit = _score_forward_consistency(
        amp, phase, probe, i_sim, apply_fftshift=False,
    )
    score_scaled = _score_forward_consistency(
        amp, phase, probe, (i_sim * k).astype(np.float32), apply_fftshift=False,
    )
    assert score_scaled == pytest.approx(score_unit, abs=1e-6)


def test_scorer_shuffled_patches_increases_score():
    """Passing patches from a different frame than measured raises the score."""
    rng = np.random.default_rng(1)
    probe, amp, phase, i_sim = _make_forward_physics(rng, n_frames=4)

    score_correct = _score_forward_consistency(
        amp, phase, probe, i_sim, apply_fftshift=False,
    )
    # Shuffle frames so patch[n] is scored against measured[n-1].
    amp_shuffled = np.roll(amp, shift=1, axis=0)
    phase_shuffled = np.roll(phase, shift=1, axis=0)
    score_shuffled = _score_forward_consistency(
        amp_shuffled, phase_shuffled, probe, i_sim, apply_fftshift=False,
    )
    assert score_shuffled > score_correct + 0.05


def test_scorer_wrong_probe_increases_score():
    """A random probe that doesn't match the forward model gives a higher score."""
    rng = np.random.default_rng(2)
    probe, amp, phase, i_sim = _make_forward_physics(rng)

    score_correct = _score_forward_consistency(
        amp, phase, probe, i_sim, apply_fftshift=False,
    )
    wrong_probe = (
        rng.normal(size=probe.shape).astype(np.float32)
        + 1j * rng.normal(size=probe.shape).astype(np.float32)
    ).astype(np.complex64)
    score_wrong = _score_forward_consistency(
        amp, phase, wrong_probe, i_sim, apply_fftshift=False,
    )
    assert score_wrong > score_correct + 0.05


def test_scorer_phase_negation_changes_score():
    """|fft2(probe * amp * exp(iφ))|² == |fft2(probe * amp * exp(-iφ))|²
    is NOT generally true, so negating all phases should change the score
    (unless the probe and patches are symmetric). This test verifies the
    scorer is sensitive to phase — i.e. it's actually using phase, not
    discarding it.
    """
    rng = np.random.default_rng(3)
    probe, amp, phase, i_sim = _make_forward_physics(rng)

    score_correct = _score_forward_consistency(
        amp, phase, probe, i_sim, apply_fftshift=False,
    )
    score_negated = _score_forward_consistency(
        amp, -phase, probe, i_sim, apply_fftshift=False,
    )
    # Negating phase should degrade the score for a generic random probe.
    assert score_negated > score_correct + 0.01


def test_scorer_all_zero_predictions_returns_inf():
    """When amp and phase are all zero, norm=0 → score must be inf."""
    patch_size = 8
    probe = np.ones((patch_size, patch_size), dtype=np.complex64)
    amp = np.zeros((2, patch_size, patch_size), dtype=np.float32)
    phase = np.zeros((2, patch_size, patch_size), dtype=np.float32)
    measured = np.ones((2, patch_size, patch_size), dtype=np.float32)

    score = _score_forward_consistency(amp, phase, probe, measured, apply_fftshift=False)
    assert score == float('inf')


def test_scorer_apply_fftshift_false_matches_no_shift_measured():
    """When measured intensity has DC at center (no shift applied), passing
    apply_fftshift=False should give a perfect score; apply_fftshift=True
    should degrade it because i_sim gets shifted but measured does not."""
    rng = np.random.default_rng(4)
    probe, amp, phase, i_sim = _make_forward_physics(rng)
    # i_sim has DC at corner (raw fft2 output). fftshift it to center.
    measured_center = np.fft.fftshift(i_sim, axes=(-2, -1))

    # apply_fftshift=True: scorer shifts i_sim to center → matches measured_center.
    score_correct = _score_forward_consistency(
        amp, phase, probe, measured_center, apply_fftshift=True,
    )
    # apply_fftshift=False: scorer leaves i_sim at corner → mismatches measured_center.
    score_wrong = _score_forward_consistency(
        amp, phase, probe, measured_center, apply_fftshift=False,
    )
    assert score_correct == pytest.approx(0.0, abs=1e-5)
    assert score_wrong > 0.05


def test_scorer_apply_fftshift_true_matches_shifted_measured():
    """Symmetric counterpart: when measured has DC at corner (raw fft2),
    apply_fftshift=False gives perfect score; True degrades it."""
    rng = np.random.default_rng(5)
    probe, amp, phase, i_sim = _make_forward_physics(rng)
    # i_sim already has DC at corner — no shift needed.
    measured_corner = i_sim

    score_correct = _score_forward_consistency(
        amp, phase, probe, measured_corner, apply_fftshift=False,
    )
    score_wrong = _score_forward_consistency(
        amp, phase, probe, measured_corner, apply_fftshift=True,
    )
    assert score_correct == pytest.approx(0.0, abs=1e-5)
    assert score_wrong > 0.05


# ----- end-to-end recovery via forward consistency --------------------------

def test_autodetect_orientation_recovers_truth_dp_orient(
    recovery_fixture, recovery_report,
):
    """The auto-detector picks the truth dp_orient from the oracle fixture.

    The fixture builds detector-frame intensity from forward physics
    (``|fft2(probe · ψ)|²``) and an oracle session that returns the right
    patches under one specific dp_orient. The forward scorer should make
    that dp_orient the winner.
    """
    _, _, _, _, _, truth_dp_orient = recovery_fixture
    assert recovery_report.best.candidate.dp_orient == truth_dp_orient


def test_autodetect_orientation_score_gap_is_wide(recovery_report):
    """Top score must be much smaller than worst — verifies the scorer is
    discriminating rather than emitting near-constants."""
    top = recovery_report.best.score
    worst = recovery_report.ranked[-1].score
    assert worst / max(top, 1e-9) > 3.0


def test_autodetect_orientation_ranked_is_sorted_ascending_by_score(
    recovery_report,
):
    scores = [r.score for r in recovery_report.ranked]
    assert scores == sorted(scores)
    assert recovery_report.best is recovery_report.ranked[0]


# ----- input validation -----------------------------------------------------

def _tiny_kwargs():
    return dict(
        normalization=1.0,
        scale=1.0,
        hot_pixel_count_threshold=None,
        fftshift=False,
    )


def _dummy_probe():
    return np.ones((8, 8), dtype=np.complex64)


def test_autodetect_orientation_probe_is_required(null_session):
    """probe is mandatory — forward consistency needs it, and there is no
    fallback scorer."""
    with pytest.raises(ValueError, match="probe is required"):
        autodetect_orientation(
            np.ones((2, 8, 8), dtype=np.uint32),
            np.array([[0.0, 0.0], [0.1, 0.1]]),
            session=null_session,
            probe=None,
            preprocess_kwargs=_tiny_kwargs(),
        )


def test_autodetect_orientation_dp_orient_in_preprocess_kwargs_raises(null_session):
    with pytest.raises(ValueError, match="dp_orient"):
        autodetect_orientation(
            np.ones((2, 8, 8), dtype=np.uint32),
            np.array([[0.0, 0.0], [0.1, 0.1]]),
            session=null_session,
            probe=_dummy_probe(),
            preprocess_kwargs={**_tiny_kwargs(), "dp_orient": "rot90_cw"},
        )


def test_autodetect_orientation_mismatched_positions_shape_raises(null_session):
    with pytest.raises(ValueError, match="positions_um"):
        autodetect_orientation(
            np.ones((2, 8, 8), dtype=np.uint32),
            np.array([[0.0, 0.0]]),  # only 1 position for 2 frames
            session=null_session,
            probe=_dummy_probe(),
            preprocess_kwargs=_tiny_kwargs(),
        )


def test_autodetect_orientation_intensity_batch_wrong_dim_raises(null_session):
    with pytest.raises(ValueError, match="3D"):
        autodetect_orientation(
            np.ones((8, 8), dtype=np.uint32),  # 2D — must be 3D
            np.array([[0.0, 0.0]]),
            session=null_session,
            probe=_dummy_probe(),
            preprocess_kwargs=_tiny_kwargs(),
        )


# ----- candidate-list restriction -------------------------------------------

def test_autodetect_orientation_restricting_candidate_list_reduces_search_space(
    null_session,
):
    report = autodetect_orientation(
        np.ones((2, 8, 8), dtype=np.uint32),
        np.array([[0.0, 0.0], [0.1, 0.1]]),
        session=null_session,
        probe=_dummy_probe(),
        preprocess_kwargs=_tiny_kwargs(),
        dp_orient_candidates=['identity', 'rot90_cw'],
    )
    assert len(report.ranked) == 2


def test_autodetect_orientation_fftshift_none_resolved_from_batch(null_session):
    """fftshift=None in preprocess_kwargs is resolved once from the batch
    before the sweep — not left as None for the scorer to mishandle.

    We build intensity with DC at the corners (fftshift needed). Passing
    fftshift=None should produce the same report as fftshift=True because
    auto-detect on this batch returns True. If the bug were still present
    (bool(None)==False), the scorer would use the wrong DC convention and
    the two reports would differ.
    """
    from ptychoml import autodetect_orientation
    from ptychoml.preprocess import detect_dc_at_corner

    # Intensity with a strong DC component at the corners.
    rng = np.random.default_rng(7)
    intensity = rng.uniform(0, 1, size=(2, 8, 8)).astype(np.float32)
    intensity[:, 0, 0] = 1e6
    intensity[:, 0, -1] = 1e6
    intensity[:, -1, 0] = 1e6
    intensity[:, -1, -1] = 1e6
    assert detect_dc_at_corner(intensity) is True  # confirm the setup

    positions = np.array([[0.0, 0.0], [0.1, 0.1]])
    probe = np.ones((8, 8), dtype=np.complex64)

    report_none = autodetect_orientation(
        intensity, positions, session=null_session, probe=probe,
        preprocess_kwargs=dict(
            normalization=1.0, scale=1.0,
            hot_pixel_count_threshold=None, fftshift=None,
        ),
        dp_orient_candidates=['identity', 'rot90_cw'],
    )
    report_true = autodetect_orientation(
        intensity, positions, session=null_session, probe=probe,
        preprocess_kwargs=dict(
            normalization=1.0, scale=1.0,
            hot_pixel_count_threshold=None, fftshift=True,
        ),
        dp_orient_candidates=['identity', 'rot90_cw'],
    )

    # Scores should be identical because fftshift=None resolved to True.
    for r_none, r_true in zip(report_none.ranked, report_true.ranked):
        assert r_none.candidate.dp_orient == r_true.candidate.dp_orient
        assert r_none.score == pytest.approx(r_true.score, abs=1e-6)


# ----- scorer drives the ranking --------------------------------------------

def test_autodetect_orientation_best_score_is_near_zero_for_correct_orient(
    recovery_fixture, recovery_report,
):
    """The winning candidate's score should be close to 0 (NCC ≈ 1).

    This verifies the scorer is actually computing forward-physics
    consistency and not just returning an arbitrary value. A score near
    zero means the model's patches reproduce the measured intensity under
    the correct orientation — which is the whole point of the scorer.
    """
    assert recovery_report.best.score == pytest.approx(0.0, abs=0.05)


def test_autodetect_orientation_wrong_orient_score_is_substantially_higher(
    recovery_fixture, recovery_report,
):
    """Every wrong candidate must score meaningfully worse than the winner.

    Guards against a degenerate scorer that returns near-constant values
    for all candidates, making the ranking effectively random.
    """
    best_score = recovery_report.best.score
    for result in recovery_report.ranked[1:]:
        assert result.score > best_score + 0.1, (
            f"{result.candidate.dp_orient} score {result.score:.4f} is not "
            f"sufficiently above best {best_score:.4f}"
        )
