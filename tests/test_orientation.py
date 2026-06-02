"""Tests for ptychoml.orientation.autodetect_orientation.

The recovery + ranking tests share a module-scoped ``recovery_report``
fixture so the synthetic forward-physics sweep only runs once for the
file. Input-validation tests use the much-cheaper ``null_session``
fixture and restrict the candidate space so they finish in milliseconds.
"""
import numpy as np
import pytest

from ptychoml import autodetect_orientation


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
