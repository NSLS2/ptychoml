"""Orientation auto-detect for the live ptycho-ViT inference pipeline.

The model can be fed detector frames in any one of 8 D4-rotated orientations,
and only one of them matches the orientation the model was trained against.
``hxn_to_vit.py`` solves this offline by sweeping every plausible
combination and scoring each against a forward-model NCC; this module is
the live-pipeline dual, scoped to the only axis the inference path actually
has to choose at run time.

Search space: ``dp_orient`` ∈ ``D4_NAMES`` (8 candidates).

Scoring: forward-physics consistency. For each candidate ``dp_orient``,
the auto-detector

  1. preprocesses the raw intensity with that ``dp_orient`` and runs
     inference,
  2. forward-models the predicted amplitude + phase through the baked
     probe: ``I_sim = |fft2(probe · (amp · e^{iφ}))|²``,
  3. scores the NCC between ``I_sim`` and the measured intensity in the
     model-input frame.

The winning ``dp_orient`` is the one where the model's predictions
reproduce the measured intensity under the actual physics — every other
candidate produces patches that don't FFT back to the right pattern.

**Why no ``patch_flip`` sweep:** the baked probe lives in the model's
output frame (it's the probe that trained the model), so the model
naturally emits patches aligned with the probe. Applying any
``patch_flip ≠ identity`` before the forward FFT mis-aligns probe and
patches, *breaking* the forward score — so the right value is always
``identity`` for inference correctness. ``patch_flip`` is therefore a
pure dashboard / canvas-display preference, configured manually rather
than auto-detected.

**Why no ``position_signs`` / ``swap_xy`` sweep:** the per-frame forward
model is local — ``|fft2(probe · ψ_i)|²`` only uses the patch and probe,
not the scan position. Position sign and swap conventions only affect
canvas stitching layout, not inference correctness; they're operator-side
configuration knobs that the auto-detector has no leverage to choose
between.

Source: this module is the live-mode dual of ptycho-vit's
``scripts/hxn_to_vit.py`` sweep, restricted to what forward-physics with
a baked probe can resolve.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .preprocess import D4_NAMES, preprocess_diffraction

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrientationCandidate:
    """A single point in the ``dp_orient`` search space."""
    dp_orient: str

    def __str__(self) -> str:
        return f"dp_orient={self.dp_orient}"


@dataclass(frozen=True)
class OrientationResult:
    """Score + diagnostics for one candidate."""
    candidate: OrientationCandidate
    score: float


@dataclass(frozen=True)
class OrientationReport:
    """Output of ``autodetect_orientation``: winning candidate + full ranking."""
    best: OrientationResult
    ranked: Tuple[OrientationResult, ...]


def _score_forward_consistency(
    amplitude_patches: np.ndarray,    # (N, H, W) float
    phase_patches: np.ndarray,        # (N, H, W) float
    probe: np.ndarray,                # (H, W) complex — single mode
    measured_intensity: np.ndarray,   # (N, H, W) float, in the model-input frame
    *,
    apply_fftshift: bool,
) -> float:
    """Forward-model NCC: |fft2(probe × ψ)|² vs measured intensity.

    ``measured_intensity`` is what comes out of ``preprocess_diffraction``
    squared back — i.e. the intensity in the model-input frame (after
    ``dp_orient`` and the optional fftshift). ``apply_fftshift`` controls
    whether to fftshift ``I_sim`` so its convention matches; set it to
    the same value as ``preprocess_diffraction``'s ``fftshift`` kwarg.

    Returns ``1 - NCC`` so lower is better (matches a loss convention).
    """
    psi = amplitude_patches.astype(np.complex64) * np.exp(
        1j * phase_patches.astype(np.float32)
    )
    wavefront = probe.astype(np.complex64)[None] * psi
    fft = np.fft.fft2(wavefront, axes=(-2, -1))
    i_sim = (fft.real ** 2 + fft.imag ** 2).astype(np.float32)
    if apply_fftshift:
        i_sim = np.fft.fftshift(i_sim, axes=(-2, -1))

    a = i_sim.ravel().astype(np.float64)
    b = measured_intensity.ravel().astype(np.float64)
    num = float(np.sum(a * b))
    norm = float(np.sqrt(np.sum(a * a))) * float(np.sqrt(np.sum(b * b)))
    if norm <= 0:
        return float('inf')
    return 1.0 - num / norm


def autodetect_orientation(
    intensity_batch: np.ndarray,
    positions_um: np.ndarray,
    *,
    session,
    probe: np.ndarray,
    preprocess_kwargs: dict,
    phase_channel_index: int = 1,
    dp_orient_candidates: Optional[Sequence[str]] = None,
) -> OrientationReport:
    """Sweep ``dp_orient`` candidates and return the most physically consistent.

    Args:
        intensity_batch:    ``(N, H, W)`` raw detector counts on a
                            spatially-diverse subset. Stays in RAM for
                            the duration of the sweep.
        positions_um:       ``(N, 2)`` scan positions ``[x_um, y_um]``.
                            Carried alongside ``intensity_batch`` for
                            input-shape validation; positions don't enter
                            the forward score (it's per-frame).
        session:            ``ptychoml.PtychoViTInference`` (already
                            initialised — ``_init_engine`` called once).
        probe:              ``(H, W)`` complex probe in the object frame.
                            Typically taken from ``session.baked_probe``
                            when the engine was exported with
                            ``convert_pt_to_onnx.py --probe``.
        preprocess_kwargs:  Dict forwarded to ``preprocess_diffraction``.
                            Must NOT contain ``dp_orient`` (that's the
                            sweep variable); must contain everything else
                            (``normalization``, ``scale``,
                            ``hot_pixel_count_threshold``, ``fftshift``).
        phase_channel_index: Which model-output channel is phase. The
                            other channel is treated as amplitude.
        dp_orient_candidates: Restrict the sweep. Default is all 8
                            ``D4_NAMES`` elements.

    Returns:
        ``OrientationReport`` with ``best`` and ``ranked`` (ascending by
        score; lower = better).
    """
    if 'dp_orient' in preprocess_kwargs:
        raise ValueError(
            "preprocess_kwargs must not include 'dp_orient' — it is the "
            "sweep variable."
        )
    if intensity_batch.ndim != 3:
        raise ValueError(
            f"intensity_batch must be 3D (N, H, W); got shape "
            f"{intensity_batch.shape}"
        )
    if positions_um.shape != (intensity_batch.shape[0], 2):
        raise ValueError(
            f"positions_um shape must be (N, 2) with N matching "
            f"intensity_batch[0]; got {positions_um.shape} vs "
            f"{intensity_batch.shape[0]}"
        )
    if probe is None:
        raise ValueError(
            "probe is required. Export the engine with "
            "``convert_pt_to_onnx.py --probe`` so the live session can "
            "expose ``session.baked_probe``, or supply one explicitly."
        )

    dp_orients = (
        tuple(dp_orient_candidates)
        if dp_orient_candidates is not None
        else D4_NAMES
    )
    amp_channel_index = 1 - phase_channel_index

    # Resolve fftshift=None (auto-detect) once from the batch before the
    # sweep so that preprocess_diffraction and the scorer use the same DC
    # convention for every candidate. If left as None, preprocess_diffraction
    # would re-detect per candidate (harmless but wasteful), while the scorer
    # would receive bool(None)==False regardless — causing a DC mismatch when
    # the data actually needs fftshift.
    fftshift_setting = preprocess_kwargs.get('fftshift', False)
    if fftshift_setting is None:
        from .preprocess import detect_dc_at_corner
        resolved_fftshift = detect_dc_at_corner(intensity_batch)
        logger.info(
            "autodetect_orientation: fftshift=None — auto-detected %s from batch",
            resolved_fftshift,
        )
        preprocess_kwargs = {**preprocess_kwargs, 'fftshift': resolved_fftshift}
    apply_fftshift = bool(preprocess_kwargs['fftshift'])
    logger.info(
        "autodetect_orientation: sweep=%d dp_orient candidates on N=%d frames",
        len(dp_orients), intensity_batch.shape[0],
    )

    results = []
    for dp_orient in dp_orients:
        diff_amp = preprocess_diffraction(
            intensity_batch,
            dp_orient=dp_orient,
            **preprocess_kwargs,
        )
        pred, _ = session.predict(diff_amp)
        # Model output shape: (N, C, H, W) for dual-channel models or
        # (N, H, W) for single-output. Normalise to (N, C, H, W).
        if pred.ndim == 3:
            pred = pred[:, None, :, :]
        amp_patches = pred[:, amp_channel_index].astype(np.float32, copy=False)
        ph_patches = pred[:, phase_channel_index].astype(np.float32, copy=False)

        # Measured intensity in the model-input frame is the preprocessed
        # diff_amp squared back. preprocess_diffraction sqrts after
        # scaling, so squaring undoes the sqrt cleanly (with the global
        # (scale / normalization) factor still applied — NCC absorbs it).
        measured = diff_amp.astype(np.float32) ** 2

        score = _score_forward_consistency(
            amp_patches, ph_patches, probe, measured,
            apply_fftshift=apply_fftshift,
        )
        results.append(
            OrientationResult(
                candidate=OrientationCandidate(dp_orient=dp_orient),
                score=score,
            )
        )

    results.sort(key=lambda r: r.score)
    return OrientationReport(
        best=results[0],
        ranked=tuple(results),
    )
