#!/usr/bin/env python3
"""
Fair comparison: CountComparator vs PROVE Agent for count questions.

Both methods start from the SAME question (no subquestion generation).
Compares evidence collection and reasoning approaches.

Usage:
    python count_test.py <image_a> <image_b> <question>

Example:
    python count_test.py img1.jpg img2.jpg "Are there the same number of beers in image A as in image B?"
"""

import sys
from typing import Dict, List
from PIL import Image

from src.core.model_manager import ModelManager
from src.core.knowledge_base import KnowledgeBase
from src.core.types import ImageData
from src.pipeline.detector import Detector
from src.pipeline.unified_agent import UnifiedAgent, EvidenceCollection, CountEvidence
from src.pipeline.problog_executor import ProbLogExecutor
from src.pipeline.problog_builder import ProbLogFactBuilder
from src.pipeline.count_processor2 import CountComparator


def compute_poisson_binomial(probabilities: List[float]) -> Dict[int, float]:
    """Compute Poisson-Binomial distribution using DP."""
    P = [1.0]
    for p in probabilities:
        new_P = [0.0] * (len(P) + 1)
        for k in range(len(P)):
            new_P[k] += P[k] * (1 - p)
            new_P[k + 1] += P[k] * p
        P = new_P
    return {k: P[k] for k in range(len(P))}


def format_distribution(dist: Dict[int, float]) -> str:
    """Pretty print a distribution."""
    items = sorted(dist.items())
    parts = [f"{k}: {p:.3f}" for k, p in items if p > 0.001]
    return "{" + ", ".join(parts) + "}"


def run_shared_detection(image_a_path: str, image_b_path: str, question: str):
    """Run detection once, shared by both methods."""
    print("\n" + "=" * 60)
    print("SHARED: Object Detection")
    print("=" * 60)

    detector = Detector()

    # Detect objects
    detections_a = detector.detect_from_question(image_a_path, question)
    detections_b = detector.detect_from_question(image_b_path, question)

    print(f"\nImage A: {image_a_path}")
    print(f"  Detections: {len(detections_a)}")
    for det in detections_a:
        print(f"    - {det.label} (conf={det.confidence:.3f})")

    print(f"\nImage B: {image_b_path}")
    print(f"  Detections: {len(detections_b)}")
    for det in detections_b:
        print(f"    - {det.label} (conf={det.confidence:.3f})")

    # Build ImageData structures (shared by both methods)
    images = {
        "image_a": ImageData(
            objects=detections_a,
            attributes={},
            relationships=[],
            counts={},
        ),
        "image_b": ImageData(
            objects=detections_b,
            attributes={},
            relationships=[],
            counts={},
        )
    }

    image_paths = {
        "image_a": image_a_path,
        "image_b": image_b_path
    }

    return images, image_paths, detections_a, detections_b


def run_count_comparator(detections_a, detections_b, question: str):
    """Run the direct CountComparator approach."""
    print("\n" + "=" * 60)
    print("METHOD 1: CountComparator (Direct Mathematical)")
    print("=" * 60)
    print(f"\nQuestion: {question}")

    # Get unique classes
    all_classes = set(d.label for d in detections_a) | set(d.label for d in detections_b)

    if not all_classes:
        print("  No objects detected!")
        return 0.0, {}

    comparator = CountComparator()
    results = {}

    for obj_class in all_classes:
        confs_a = [d.confidence for d in detections_a if d.label == obj_class]
        confs_b = [d.confidence for d in detections_b if d.label == obj_class]

        dist_a = compute_poisson_binomial(confs_a) if confs_a else {0: 1.0}
        dist_b = compute_poisson_binomial(confs_b) if confs_b else {0: 1.0}

        print(f"\n  {obj_class}:")
        print(f"    Image A: {len(confs_a)} detections → {format_distribution(dist_a)}")
        print(f"    Image B: {len(confs_b)} detections → {format_distribution(dist_b)}")

        # Compute all comparisons
        prob_equal, debug_eq = comparator.compare(dist_a, dist_b, "equal")
        prob_greater, _ = comparator.compare(dist_a, dist_b, "greater")
        prob_less, _ = comparator.compare(dist_a, dist_b, "less")

        print(f"\n    Comparisons:")
        print(f"      P(A == B) = {prob_equal:.4f}")
        print(f"      P(A > B)  = {prob_greater:.4f}")
        print(f"      P(A < B)  = {prob_less:.4f}")
        print(f"      E[A] = {debug_eq['expected_a']:.3f}, E[B] = {debug_eq['expected_b']:.3f}")

        results[obj_class] = {
            "dist_a": dist_a,
            "dist_b": dist_b,
            "prob_equal": prob_equal,
            "prob_greater": prob_greater,
            "prob_less": prob_less
        }

    # Determine which comparison the question asks about
    question_lower = question.lower()

    if "same" in question_lower or "equal" in question_lower:
        comparison_type = "equal"
        primary_prob = list(results.values())[0]["prob_equal"]
    elif "more" in question_lower and "image a" in question_lower:
        comparison_type = "greater"
        primary_prob = list(results.values())[0]["prob_greater"]
    elif "more" in question_lower and "image b" in question_lower:
        comparison_type = "less"
        primary_prob = list(results.values())[0]["prob_less"]
    elif "fewer" in question_lower or "less" in question_lower:
        comparison_type = "less"
        primary_prob = list(results.values())[0]["prob_less"]
    else:
        comparison_type = "equal"
        primary_prob = list(results.values())[0]["prob_equal"]

    print(f"\n  Question Type: {comparison_type}")
    print(f"  Primary Probability: {primary_prob:.4f}")
    print(f"  Answer: {'TRUE' if primary_prob >= 0.5 else 'FALSE'} (threshold=0.5)")

    return primary_prob, results


def run_prove_agent(images: Dict[str, ImageData], image_paths: Dict[str, str], question: str):
    """Run the PROVE agent approach."""
    print("\n" + "=" * 60)
    print("METHOD 2: PROVE Agent (ProbLog-based)")
    print("=" * 60)
    print(f"\nQuestion: {question}")

    # Run unified agent directly on the question
    print("\n  Agent Evidence Collection:")
    agent = UnifiedAgent(max_iterations=10)  # Limit iterations
    evidence = agent.collect_evidence(
        question=question,
        images=images,
        image_paths=image_paths
    )

    # Show evidence collected
    print(f"\n  Evidence Collected:")
    print(f"    Attributes: {len(evidence.attributes)}")
    for attr in evidence.attributes:
        print(f"      - {attr}")
    print(f"    Relationships: {len(evidence.relationships)}")
    for rel in evidence.relationships:
        print(f"      - {rel}")
    print(f"    Counts: {len(evidence.counts)}")
    for count_ev in evidence.counts:
        print(f"      - {count_ev.query_type}({count_ev.object_class}): p={count_ev.probability:.3f}")

    # Run ProbLog execution
    print("\n  ProbLog Execution:")
    executor = ProbLogExecutor()

    prob_result, det_result = executor.execute_dual(
        question=question,
        evidence=evidence,
        images=images,
        threshold=0.5
    )

    print(f"\n  Results:")
    print(f"    Probabilistic: p={prob_result.probability:.4f}")
    print(f"    Deterministic: p={det_result.probability:.4f}")
    print(f"    Final Answer (Prob): {prob_result.final_answer}")
    print(f"    Final Answer (Det): {det_result.final_answer}")

    # Show ProbLog program
    print(f"\n  ProbLog Program:")
    print("-" * 40)
    for line in prob_result.problog_program.split('\n'):
        if line.strip() and not line.startswith('%'):
            print(f"    {line}")

    return prob_result, det_result, evidence


def main():
    if len(sys.argv) < 4:
        print("Usage: python count_test.py <image_a> <image_b> <question>")
        print(
            'Example: python count_test.py img1.jpg img2.jpg "Are there the same number of beers in image A as in image B?"')
        sys.exit(1)

    image_a_path = sys.argv[1]
    image_b_path = sys.argv[2]
    question = sys.argv[3]

    print("=" * 60)
    print("FAIR COMPARISON: CountComparator vs PROVE Agent")
    print("=" * 60)
    print(f"\nQuestion: {question}")
    print(f"Image A: {image_a_path}")
    print(f"Image B: {image_b_path}")

    # Shared detection (same for both methods)
    images, image_paths, detections_a, detections_b = run_shared_detection(
        image_a_path, image_b_path, question
    )

    # Method 1: CountComparator
    comparator_prob, comparator_results = run_count_comparator(
        detections_a, detections_b, question
    )

    # Method 2: PROVE Agent
    prove_prob_result, prove_det_result, prove_evidence = run_prove_agent(
        images, image_paths, question
    )

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    comparator_answer = "True" if comparator_prob >= 0.5 else "False"
    prove_prob = prove_prob_result.probability
    prove_answer = prove_prob_result.final_answer

    print(f"\nCountComparator (Direct Math):")
    print(f"  Probability: {comparator_prob:.4f}")
    print(f"  Answer: {comparator_answer}")

    print(f"\nPROVE Agent (ProbLog):")
    print(f"  Probability: {prove_prob:.4f}")
    print(f"  Answer: {prove_answer}")

    print(f"\nDifference: {abs(comparator_prob - prove_prob):.4f}")

    if comparator_answer == prove_answer:
        print(f"\n✓ Methods AGREE: {comparator_answer}")
    else:
        print(f"\n✗ Methods DISAGREE:")
        print(f"    CountComparator: {comparator_answer} (p={comparator_prob:.4f})")
        print(f"    PROVE: {prove_answer} (p={prove_prob:.4f})")

    # Analysis of the difference
    print("\n" + "-" * 40)
    print("ANALYSIS")
    print("-" * 40)

    # Check if PROVE collected counts
    if prove_evidence.counts:
        print("\nPROVE collected count evidence:")
        for count_ev in prove_evidence.counts:
            print(f"  {count_ev.query_type}({count_ev.object_class}): p={count_ev.probability:.3f}")
    else:
        print("\nPROVE did NOT collect count evidence!")
        print("  The agent may not have used verify_count action")


if __name__ == "__main__":
    main()
