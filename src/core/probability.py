"""
Binary verification probability extraction for PROVE pipeline.
Provides unified probability calculation from VLM logits using verbalizer summing + softmax.
Also provides anchored sigmoid mapping for detector confidence calibration.
"""

import math
import torch
from typing import List, Tuple


def get_verifier_probability(
    logits_sequence: List[torch.Tensor],
    response: str,
    tokenizer,
    debug: bool = False
) -> float:
    """
    Extract P(statement is true) using verbalizer summing + 2-token softmax.

    Process:
    1. Sum logits for all Yes variants: ["Yes", "yes", "YES"]
    2. Sum logits for all No variants: ["No", "no", "NO"]
    3. Apply 2-token softmax: P(yes) = e^(sum_yes) / (e^(sum_yes) + e^(sum_no))

    This approach:
    - Handles tokenization variants robustly
    - Avoids full-vocabulary probability inflation
    - Returns realistic confidence scores
    - Defaults to 0.5 (neutral) on extraction failure

    Args:
        logits_sequence: List of logit tensors from VLM generation
        response: The actual response text (for validation/debugging)
        tokenizer: Tokenizer for encoding verbalizer variants
        debug: If True, print detailed verbalizer logit breakdown

    Returns:
        float: P(statement is true) between 0.0 and 1.0

    Example:
        >>> prob = get_verifier_probability(logits, "Yes", qwen_tokenizer)
        >>> print(f"Confidence: {prob:.3f}")
        Confidence: 0.837
    """
    if not logits_sequence:
        return 0.5  # Neutral probability for empty logits

    try:
        # Get FIRST generation step logits (where Yes/No token is produced)
        # The VLM generates ["Yes"/"No", "."] so the decision happens in step 0, not -1
        first_logits = logits_sequence[0][0]  # Shape: [vocab_size]

        # Define verbalizer variants for robustness
        yes_verbalizers = ["Yes", "yes", "YES"]
        no_verbalizers = ["No", "no", "NO"]

        # Track individual token logits for debugging
        yes_token_details = []  # List of (verbalizer, token_id, logit_value)
        no_token_details = []   # List of (verbalizer, token_id, logit_value)

        # Sum logits for all Yes verbalizers
        sum_yes_logits = -float('inf')  # Start with log(0) = -inf
        for verbalizer in yes_verbalizers:
            try:
                # Get token IDs for this verbalizer
                token_ids = tokenizer.encode(verbalizer, add_special_tokens=False)

                # Sum logits for all tokens of this verbalizer
                for token_id in token_ids:
                    if token_id < len(first_logits):
                        logit_value = first_logits[token_id].item()

                        # Track for debugging
                        yes_token_details.append((verbalizer, token_id, logit_value))

                        # Use logsumexp for numerical stability: log(a + b) = log(exp(log(a)) + exp(log(b)))
                        if sum_yes_logits == -float('inf'):
                            sum_yes_logits = logit_value
                        else:
                            # logsumexp: log(e^a + e^b) = max(a,b) + log(1 + e^(-|a-b|))
                            max_val = max(sum_yes_logits, logit_value)
                            sum_yes_logits = max_val + math.log(
                                math.exp(sum_yes_logits - max_val) + math.exp(logit_value - max_val)
                            )
            except Exception as e:
                # Skip verbalizers that cause encoding issues
                continue

        # Sum logits for all No verbalizers
        sum_no_logits = -float('inf')  # Start with log(0) = -inf
        for verbalizer in no_verbalizers:
            try:
                # Get token IDs for this verbalizer
                token_ids = tokenizer.encode(verbalizer, add_special_tokens=False)

                # Sum logits for all tokens of this verbalizer
                for token_id in token_ids:
                    if token_id < len(first_logits):
                        logit_value = first_logits[token_id].item()

                        # Track for debugging
                        no_token_details.append((verbalizer, token_id, logit_value))

                        # Use logsumexp for numerical stability
                        if sum_no_logits == -float('inf'):
                            sum_no_logits = logit_value
                        else:
                            max_val = max(sum_no_logits, logit_value)
                            sum_no_logits = max_val + math.log(
                                math.exp(sum_no_logits - max_val) + math.exp(logit_value - max_val)
                            )
            except Exception as e:
                # Skip verbalizers that cause encoding issues
                continue

        # Check if we found valid verbalizer logits
        if sum_yes_logits == -float('inf') or sum_no_logits == -float('inf'):
            print(f"Warning: Could not extract verbalizer logits for response: '{response}'")
            return 0.5  # Neutral probability for extraction failure

        # Apply 2-token softmax: P(yes) = e^(sum_yes) / (e^(sum_yes) + e^(sum_no))
        exp_yes = math.exp(sum_yes_logits)
        exp_no = math.exp(sum_no_logits)

        prob_yes = exp_yes / (exp_yes + exp_no)

        # Debug printing
        if debug:
            print("\n" + "=" * 80)
            print("📊 VERBALIZER LOGITS BREAKDOWN")
            print("=" * 80)

            print("\n🟢 YES Verbalizer Tokens:")
            for verbalizer, token_id, logit_value in yes_token_details:
                print(f"  '{verbalizer}' (token_id={token_id}): logit = {logit_value:.4f}")

            print(f"\n  → Sum of Yes logits (logsumexp): {sum_yes_logits:.4f}")
            print(f"  → exp(sum_yes) = {exp_yes:.6f}")

            print("\n🔴 NO Verbalizer Tokens:")
            for verbalizer, token_id, logit_value in no_token_details:
                print(f"  '{verbalizer}' (token_id={token_id}): logit = {logit_value:.4f}")

            print(f"\n  → Sum of No logits (logsumexp): {sum_no_logits:.4f}")
            print(f"  → exp(sum_no) = {exp_no:.6f}")

            print("\n🎯 SOFTMAX CALCULATION:")
            print(f"  P(Yes) = exp(sum_yes) / (exp(sum_yes) + exp(sum_no))")
            print(f"  P(Yes) = {exp_yes:.6f} / ({exp_yes:.6f} + {exp_no:.6f})")
            print(f"  P(Yes) = {exp_yes:.6f} / {exp_yes + exp_no:.6f}")
            print(f"  P(Yes) = {prob_yes:.4f}")

            print("\n  VLM Response: \"{}\"".format(response))
            print("=" * 80 + "\n")

        return float(prob_yes)

    except Exception as e:
        print(f"Warning: Failed to extract verifier probability: {e}")
        return 0.5  # Neutral probability for unexpected errors


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

    Transforms detector scores p ∈ (0,1) into operational probabilities using:
        p' = 1 / (1 + ((1-p)/p)^a * e^(-c))

    where a and c are computed from anchor points:
        - (0.1 → 0.7): Low confidence scores map to moderate probabilities
        - (0.5 → 0.9): Medium confidence scores map to high probabilities

    This probability-space form avoids explicit logit computation and is
    numerically stable for all input values.

    Mathematical Background:
    The mapping enforces two fixed anchor points (p_lo, q_lo) and (p_hi, q_hi),
    solving for parameters in: p' = σ(a · logit(p) + c)

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
