"""Shared test fixtures for ptychoml.

Holds fake-session scaffolding used by the orientation tests so each test
doesn't re-derive it. The fakes never touch TensorRT or CUDA — they expose
the minimum surface ``autodetect_orientation`` reads (``expected_input_shape``
and ``predict``).
"""
from __future__ import annotations

import numpy as np
import pytest

from ptychoml import D4_NAMES, preprocess_diffraction


class NullSession:
    """Fake session returning all-zero predictions regardless of input.

    Used by tests that only exercise input validation — paths that don't
    depend on what the model actually returns.
    """

    def __init__(self, n_frames: int = 2, patch_size: int = 8):
        self.expected_input_shape = (n_frames, 1, patch_size, patch_size)
        self._pred_shape = (n_frames, 2, patch_size, patch_size)

    def predict(self, diff_amp):
        return np.zeros(self._pred_shape, dtype=np.float32), None


@pytest.fixture
def null_session():
    """A NullSession instance sized for a 2-frame × 8x8 batch."""
    return NullSession(n_frames=2, patch_size=8)


@pytest.fixture(scope="module")
def recovery_fixture():
    """Synthetic ground-truth fixture for the forward-consistency recovery test.

    Builds:
      - A synthetic complex object and a synthetic complex probe, both in
        the probe / canvas frame.
      - True per-frame object patches extracted at a small set of scan
        positions, plus the forward-modelled "detector" intensity
        ``I = |fft2(probe · patch)|²`` for each frame.
      - A FakeSession that, when fed the input under the truth
        ``dp_orient``, returns the true (amp, phase) of the object
        patches; under any other ``dp_orient`` it returns noise. The
        marker for "which dp_orient was applied" is the (0,0,0) pixel of
        the preprocessed input — unique because the detector intensity is
        built from unique-valued integer markers per pixel.

    Returns ``(intensity_detector, positions_um, session, probe,
    preprocess_kwargs, truth_dp_orient)``.
    """
    truth_dp_orient = 'rot180'

    rng = np.random.default_rng(42)
    patch_size = 16
    n_frames = 12
    positions_um = rng.uniform(-0.1, 0.1, size=(n_frames, 2))

    # Synthetic probe and object (canvas frame).
    probe = (
        rng.normal(size=(patch_size, patch_size)).astype(np.float32)
        + 1j * rng.normal(size=(patch_size, patch_size)).astype(np.float32)
    ).astype(np.complex64)

    true_amp = rng.uniform(0.5, 1.5, size=(n_frames, patch_size, patch_size)).astype(np.float32)
    true_phase = rng.uniform(-1.0, 1.0, size=(n_frames, patch_size, patch_size)).astype(np.float32)

    # The "true" detector intensity is the forward physics output: feeding
    # patches with these (amp, phase) through (probe · ψ → fft2 → |·|²)
    # should reproduce the measured I. The fake session returns these
    # exact patches when the right dp_orient input arrives.
    correct_pred = np.stack([true_amp, true_phase], axis=1)
    wrong_pred = np.stack(
        [
            rng.uniform(0.3, 1.5, size=(n_frames, patch_size, patch_size)).astype(np.float32),
            rng.uniform(-np.pi, np.pi, size=(n_frames, patch_size, patch_size)).astype(np.float32),
        ],
        axis=1,
    )

    # Build the detector intensity that corresponds to those patches.
    # We use forward physics directly so the test exercises the same
    # ``|fft2(probe · ψ)|²`` formula the scorer uses internally.
    psi = true_amp.astype(np.complex64) * np.exp(1j * true_phase.astype(np.float32))
    wavefront = probe[None] * psi
    fft = np.fft.fft2(wavefront, axes=(-2, -1))
    true_intensity_model_frame = (fft.real ** 2 + fft.imag ** 2).astype(np.float32)

    # The model was trained with input = apply_d4(detector_intensity, truth_dp_orient).
    # So the detector frame relates to the model-input frame by the inverse
    # of truth_dp_orient. ``D4 inverse(rot180) = rot180`` (self-inverse).
    d4_inverse = {
        'identity': 'identity',
        'fliplr': 'fliplr',
        'flipud': 'flipud',
        'rot180': 'rot180',
        'transpose': 'transpose',
        'antitranspose': 'antitranspose',
        'rot90_ccw': 'rot90_cw',
        'rot90_cw': 'rot90_ccw',
    }
    from ptychoml import apply_d4
    intensity_detector = apply_d4(
        true_intensity_model_frame, d4_inverse[truth_dp_orient],
    )
    # Convert to a unique-marker-per-pixel space by quantising — preserves
    # the (0,0,0) marker idea but keeps the values matched to the physics.
    # The marker is the (0,0,0) value of the preprocessed input, which is
    # uniquely determined by the dp_orient candidate applied to the
    # detector intensity.
    intensity_detector = intensity_detector.astype(np.float32)

    preprocess_kwargs = dict(
        normalization=1.0,
        scale=1.0,
        hot_pixel_count_threshold=None,
        fftshift=False,
    )
    markers = {
        d: float(
            preprocess_diffraction(
                intensity_detector, dp_orient=d, **preprocess_kwargs,
            )[0, 0, 0]
        )
        for d in D4_NAMES
    }

    class OracleSession:
        expected_input_shape = (n_frames, 1, patch_size, patch_size)

        def predict(_self, diff_amp):
            m = float(diff_amp[0, 0, 0])
            # Find the closest marker (markers differ by D4-permutation
            # so they're well-separated for non-degenerate inputs).
            best_name = min(markers, key=lambda d: abs(markers[d] - m))
            pred = correct_pred if best_name == truth_dp_orient else wrong_pred
            return pred.copy(), None

    return (
        intensity_detector,
        positions_um,
        OracleSession(),
        probe,
        preprocess_kwargs,
        truth_dp_orient,
    )


@pytest.fixture(scope="module")
def recovery_report(recovery_fixture):
    """The full orientation sweep report for the recovery fixture.

    Module-scoped because the sweep runs 8 candidates per test —
    re-running per test would re-load the oracle pred arrays and re-do
    the marker lookup repeatedly. The same report serves multiple
    assertions.
    """
    from ptychoml import autodetect_orientation

    intensity, positions_um, session, probe, kwargs, _ = recovery_fixture
    return autodetect_orientation(
        intensity,
        positions_um,
        session=session,
        probe=probe,
        preprocess_kwargs=kwargs,
        phase_channel_index=1,
    )
