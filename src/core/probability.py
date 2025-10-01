"""
Binary verification probability extraction for PROVE pipeline.
Provides unified probability calculation from VLM logits using verbalizer summing + softmax.
"""

import math
import torch
from typing import List


def get_verifier_probability(
    logits_sequence: List[torch.Tensor],
    response: str,
    tokenizer
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
        # Get final generation step logits (where Yes/No token is produced)
        final_logits = logits_sequence[-1][0]  # Shape: [vocab_size]

        # Define verbalizer variants for robustness
        yes_verbalizers = ["Yes", "yes", "YES"]
        no_verbalizers = ["No", "no", "NO"]

        # Sum logits for all Yes verbalizers
        sum_yes_logits = -float('inf')  # Start with log(0) = -inf
        for verbalizer in yes_verbalizers:
            try:
                # Get token IDs for this verbalizer
                token_ids = tokenizer.encode(verbalizer, add_special_tokens=False)

                # Sum logits for all tokens of this verbalizer
                for token_id in token_ids:
                    if token_id < len(final_logits):
                        logit_value = final_logits[token_id].item()
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
                    if token_id < len(final_logits):
                        logit_value = final_logits[token_id].item()
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

        return float(prob_yes)

    except Exception as e:
        print(f"Warning: Failed to extract verifier probability: {e}")
        return 0.5  # Neutral probability for unexpected errors
