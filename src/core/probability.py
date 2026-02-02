"""
Probability utilities for PROVE pipeline.
Provides anchored sigmoid mapping for detector confidence calibration.
"""

import math
from typing import Tuple


# ==============================================================================
# Detector Confidence Calibration using Anchored Sigmoid Mapping
# ==============================================================================
# Anchor points for detector confidence calibration
# These map raw detector scores to operational probabilities
DETECTOR_ANCHOR_P_LO = 0.1   # Low raw score anchor
DETECTOR_ANCHOR_Q_LO = 0.7   # Maps to 70% operational probability
DETECTOR_ANCHOR_P_HI = 0.5   # High raw score anchor
DETECTOR_ANCHOR_Q_HI = 0.9   # Maps to 90% operational probability


def _logit(p: float) -> float:
    """
    Compute logit (log-odds) of probability p.

    Args:
        p: Probability in (0, 1)

    Returns:
        float: log(p / (1-p))

    Raises:
        ValueError: If p not in (0, 1)
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"Probability must be in (0,1), got {p}")
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    """
    Compute sigmoid function.

    Args:
        x: Real number

    Returns:
        float: 1 / (1 + e^(-x))
    """
    return 1.0 / (1.0 + math.exp(-x))


def _compute_anchor_parameters(
    p_lo: float,
    q_lo: float,
    p_hi: float,
    q_hi: float
) -> Tuple[float, float]:
    """
    Compute anchored sigmoid parameters a and c.

    Solves the system:
        logit(q_lo) = a * logit(p_lo) + c
        logit(q_hi) = a * logit(p_hi) + c

    This yields:
        a = (logit(q_hi) - logit(q_lo)) / (logit(p_hi) - logit(p_lo))
        c = logit(q_lo) - a * logit(p_lo)

    Args:
        p_lo: Low raw score anchor point
        q_lo: Target probability for p_lo
        p_hi: High raw score anchor point
        q_hi: Target probability for p_hi

    Returns:
        Tuple[float, float]: (a, c) parameters for anchored sigmoid

    Example:
        >>> _compute_anchor_parameters(0.1, 0.7, 0.5, 0.9)
        (0.6144, 2.1972)
    """
    logit_p_lo = _logit(p_lo)
    logit_p_hi = _logit(p_hi)
    logit_q_lo = _logit(q_lo)
    logit_q_hi = _logit(q_hi)

    # Solve for a and c
    a = (logit_q_hi - logit_q_lo) / (logit_p_hi - logit_p_lo)
    c = logit_q_lo - a * logit_p_lo

    return a, c


# Pre-compute anchor parameters at module load time for efficiency
_ANCHOR_A, _ANCHOR_C = _compute_anchor_parameters(
    DETECTOR_ANCHOR_P_LO, DETECTOR_ANCHOR_Q_LO,
    DETECTOR_ANCHOR_P_HI, DETECTOR_ANCHOR_Q_HI
)


def calibrate_detector_confidence(raw_score: float) -> float:
    """
    Calibrate raw detector confidence using anchored sigmoid mapping.

    Transforms detector scores p in (0,1) into operational probabilities using:
        p' = 1 / (1 + ((1-p)/p)^a * e^(-c))

    where a and c are computed from anchor points:
        - (0.1 -> 0.7): Low confidence scores map to moderate probabilities
        - (0.5 -> 0.9): Medium confidence scores map to high probabilities

    This probability-space form avoids explicit logit computation and is
    numerically stable for all input values.

    Mathematical Background:
    The mapping enforces two fixed anchor points (p_lo, q_lo) and (p_hi, q_hi),
    solving for parameters in: p' = sigmoid(a * logit(p) + c)

    Args:
        raw_score: Raw detector confidence score in (0, 1)

    Returns:
        float: Calibrated probability in (0, 1)

    Example:
        >>> calibrate_detector_confidence(0.1)   # Low raw score
        0.700
        >>> calibrate_detector_confidence(0.5)   # Medium raw score
        0.900
        >>> calibrate_detector_confidence(0.3)   # Intermediate
        0.831
    """
    # Clamp to valid range with small epsilon to avoid numerical issues
    epsilon = 1e-7
    raw_score = max(epsilon, min(1.0 - epsilon, raw_score))

    # Probability-space form (avoids explicit logit computation)
    # p' = 1 / (1 + ((1-p)/p)^a * e^(-c))
    odds_ratio = (1.0 - raw_score) / raw_score
    calibrated = 1.0 / (1.0 + math.pow(odds_ratio, _ANCHOR_A) * math.exp(-_ANCHOR_C))

    return float(calibrated)
