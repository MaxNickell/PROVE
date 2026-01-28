"""
Count comparison tool for PROVE pipeline.
Direct probabilistic comparisons on Poisson-Binomial distributions.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CountComparisonResult:
    """Result of a count comparison operation."""
    probability: float
    comparison_type: str  # "equal", "greater", "less", "at_least", "at_most", "exactly"
    debug: Dict


class CountComparator:
    """
    Direct probabilistic count comparisons.

    Operates on Poisson-Binomial distributions without ProbLog encoding.
    Following NAVER's approach of specialized tools for specific reasoning types.
    """

    SUPPORTED_COMPARISONS = {
        "equal", "not_equal",
        "greater", "less",
        "greater_or_equal", "less_or_equal",
        "at_least", "at_most", "exactly"
    }

    def compare(
            self,
            dist_a: Dict[int, float],
            dist_b: Dict[int, float],
            comparison: str
    ) -> Tuple[float, Dict]:
        """
        Compare two count distributions.

        Args:
            dist_a: P(count_A = k) distribution
            dist_b: P(count_B = k) distribution
            comparison: One of "equal", "greater", "less", etc.

        Returns:
            (probability, debug_info)
        """
        comparison = comparison.lower().replace(" ", "_")

        if comparison == "equal":
            return self._prob_equal(dist_a, dist_b)
        elif comparison == "not_equal":
            prob, debug = self._prob_equal(dist_a, dist_b)
            return 1.0 - prob, {**debug, "inverted": True}
        elif comparison == "greater":
            return self._prob_greater(dist_a, dist_b)
        elif comparison == "less":
            return self._prob_greater(dist_b, dist_a)  # Swap
        elif comparison == "greater_or_equal":
            p_greater, d1 = self._prob_greater(dist_a, dist_b)
            p_equal, d2 = self._prob_equal(dist_a, dist_b)
            return p_greater + p_equal, {"p_greater": p_greater, "p_equal": p_equal}
        elif comparison == "less_or_equal":
            p_less, d1 = self._prob_greater(dist_b, dist_a)
            p_equal, d2 = self._prob_equal(dist_a, dist_b)
            return p_less + p_equal, {"p_less": p_less, "p_equal": p_equal}
        else:
            raise ValueError(f"Unknown comparison: {comparison}")

    def compare_to_threshold(
            self,
            dist: Dict[int, float],
            threshold: int,
            comparison: str
    ) -> Tuple[float, Dict]:
        """
        Compare count distribution to a fixed threshold.

        Args:
            dist: P(count = k) distribution
            threshold: Integer threshold value
            comparison: "at_least", "at_most", "exactly", "greater", "less"

        Returns:
            (probability, debug_info)
        """
        comparison = comparison.lower().replace(" ", "_")

        if comparison in ("at_least", "greater_or_equal"):
            prob = sum(p for k, p in dist.items() if k >= threshold)
            return prob, {"threshold": threshold, "type": ">="}

        elif comparison in ("at_most", "less_or_equal"):
            prob = sum(p for k, p in dist.items() if k <= threshold)
            return prob, {"threshold": threshold, "type": "<="}

        elif comparison == "exactly":
            prob = dist.get(threshold, 0.0)
            return prob, {"threshold": threshold, "type": "=="}

        elif comparison == "greater":
            prob = sum(p for k, p in dist.items() if k > threshold)
            return prob, {"threshold": threshold, "type": ">"}

        elif comparison == "less":
            prob = sum(p for k, p in dist.items() if k < threshold)
            return prob, {"threshold": threshold, "type": "<"}

        else:
            raise ValueError(f"Unknown comparison: {comparison}")

    def _prob_equal(
            self,
            dist_a: Dict[int, float],
            dist_b: Dict[int, float]
    ) -> Tuple[float, Dict]:
        """P(count_A == count_B) via diagonal convolution."""
        prob = 0.0
        contributions = {}

        common_keys = set(dist_a.keys()) & set(dist_b.keys())
        for k in common_keys:
            contrib = dist_a[k] * dist_b[k]
            prob += contrib
            if contrib > 0.001:  # Track significant contributions
                contributions[k] = round(contrib, 4)

        debug = {
            "comparison": "equal",
            "expected_a": self._expected_value(dist_a),
            "expected_b": self._expected_value(dist_b),
            "contributions": contributions
        }

        return prob, debug

    def _prob_greater(
            self,
            dist_a: Dict[int, float],
            dist_b: Dict[int, float]
    ) -> Tuple[float, Dict]:
        """P(count_A > count_B) via upper triangle sum."""
        prob = 0.0

        for a, p_a in dist_a.items():
            for b, p_b in dist_b.items():
                if a > b:
                    prob += p_a * p_b

        debug = {
            "comparison": "greater",
            "expected_a": self._expected_value(dist_a),
            "expected_b": self._expected_value(dist_b),
        }

        return prob, debug

    def _expected_value(self, dist: Dict[int, float]) -> float:
        """Compute E[count] = Σ k × P(k)."""
        return round(sum(k * p for k, p in dist.items()), 3)

    def get_most_likely_count(self, dist: Dict[int, float]) -> Tuple[int, float]:
        """Return (most_likely_count, probability)."""
        if not dist:
            return 0, 1.0
        k = max(dist, key=dist.get)
        return k, dist[k]