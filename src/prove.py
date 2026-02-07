"""
PROVE: Probabilistic Reasoning Over Visual Evidence
Main model class encapsulating the entire pipeline.

Unified pipeline that runs both probabilistic and deterministic modes
with shared evidence collection to isolate the effect of perception uncertainty.
"""

from typing import Dict, Any
from pathlib import Path

from .core.knowledge_base import KnowledgeBase
from .core.model_manager import ModelManager
from .core.types import UnifiedResult, SharedEvidence, ModeResult
from .pipeline.detector import Detector
from .pipeline.unified_agent import UnifiedAgent, EvidenceCollection
from .pipeline.problog_executor import ProbLogExecutor


class PROVE:
    """
    PROVE model for visual reasoning over image pairs.

    Unified pipeline that always runs both probabilistic and deterministic modes
    with shared evidence to isolate the effect of perception uncertainty.

    Usage:
        model = PROVE()
        result = model.predict("img1.jpg", "img2.jpg", "Is there a cat in both images?")
        # result contains both probabilistic and deterministic answers
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize PROVE model.

        Args:
            threshold: Threshold for final answer mapping (default 0.5)
                - probability >= threshold → "True"
                - probability < threshold → "False"
        """
        self.threshold = threshold
        # Components initialized lazily on first use
        self._model_manager = None
        self._detector = None
        self._unified_agent = None
        self._problog_executor = None

    def _init_components(self):
        """Initialize pipeline components (lazy loading)."""
        if self._model_manager is None:
            self._model_manager = ModelManager()
        if self._detector is None:
            self._detector = Detector()
        if self._unified_agent is None:
            self._unified_agent = UnifiedAgent(max_iterations=20)
        if self._problog_executor is None:
            self._problog_executor = ProbLogExecutor()

    def predict(
        self,
        image_a_path: str,
        image_b_path: str,
        question: str
    ) -> UnifiedResult:
        """
        Run PROVE inference on image pair (unified pipeline).

        Args:
            image_a_path: Path to first image
            image_b_path: Path to second image
            question: Ultimate question to answer

        Returns:
            UnifiedResult containing both probabilistic and deterministic results

        Raises:
            FileNotFoundError: If image paths don't exist
            RuntimeError: If pipeline execution fails
        """
        return self.predict_with_details(image_a_path, image_b_path, question)

    def predict_with_details(
        self,
        image_a_path: str,
        image_b_path: str,
        question: str,
        save_logs: bool = False,
        log_dir: str = "logs"
    ) -> UnifiedResult:
        """
        Run PROVE unified inference with detailed outputs.

        Runs both probabilistic and deterministic modes with shared evidence
        to isolate the effect of perception uncertainty.

        Args:
            image_a_path: Path to first image
            image_b_path: Path to second image
            question: Ultimate question to answer
            save_logs: Whether to save ProbLog and intermediate results
            log_dir: Base directory for logs (default: "logs")

        Returns:
            UnifiedResult with:
                - threshold: Threshold used for deterministic mapping
                - shared: SharedEvidence (subquestions, evidence, detected objects)
                - probabilistic: ModeResult (subquestion results, final answer, problog program)
                - deterministic: ModeResult (subquestion results, final answer, problog program)

        Raises:
            FileNotFoundError: If image paths don't exist
            RuntimeError: If pipeline execution fails
        """
        # Validate inputs
        if not Path(image_a_path).exists():
            raise FileNotFoundError(f"Image A not found: {image_a_path}")
        if not Path(image_b_path).exists():
            raise FileNotFoundError(f"Image B not found: {image_b_path}")

        # Initialize components
        self._init_components()

        # Initialize knowledge base
        kb = KnowledgeBase(ultimate_question=question)
        image_paths = {"image_a": image_a_path, "image_b": image_b_path}

        try:
            # Print question
            print(f"\nQuestion: \"{question}\"")
            print(f"Threshold: {self.threshold}\n")

            # Step 1: Object Detection
            print("Step 1: Object Detection...")
            for image_id, image_path in image_paths.items():
                detections = self._detector.detect_from_question(image_path, question)
                kb.add_objects(image_id, detections)
                print(f"  {image_id}: {len(detections)} objects detected")

            # Step 2: Evidence Collection
            print("\nStep 2: Evidence Collection...")
            evidence = self._unified_agent.collect_evidence(
                question=question,
                images=kb.images,
                image_paths=image_paths
            )

            # Step 3: Dual ProbLog Execution
            print("\nStep 3: ProbLog Reasoning (dual mode)...")
            prob_result, det_result = self._problog_executor.execute_dual(
                question=question,
                evidence=evidence,
                images=kb.images,
                threshold=self.threshold
            )

            # Build shared evidence
            detected_objects = {
                image_id: list(image_data.objects)
                for image_id, image_data in kb.images.items()
            }
            shared = SharedEvidence(
                question=question,
                evidence_collection=evidence,
                detected_objects=detected_objects
            )

            # Build unified result
            result = UnifiedResult(
                threshold=self.threshold,
                shared=shared,
                probabilistic=prob_result,
                deterministic=det_result
            )

            # Print summary
            print("\n" + "=" * 60)
            print("RESULTS SUMMARY")
            print("=" * 60)
            print(f"\nProbabilistic Mode:")
            print(f"  Probability: {prob_result.probability:.3f}")
            print(f"  → Final Answer: {prob_result.final_answer}")

            print(f"\nDeterministic Mode (threshold={self.threshold}):")
            print(f"  Probability: {det_result.probability:.3f}")
            print(f"  → Final Answer: {det_result.final_answer}")

            agreement = "AGREE" if prob_result.final_answer == det_result.final_answer else "DISAGREE"
            print(f"\nModes {agreement}")
            print("=" * 60)

            # Save logs if requested
            if save_logs:
                from datetime import datetime
                import shutil
                import json

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                question_hash = str(hash(question))[:8]
                example_dir = Path(log_dir) / f"{timestamp}_{question_hash}"
                example_dir.mkdir(parents=True, exist_ok=True)

                # Copy images
                images_dir = example_dir / "images"
                images_dir.mkdir(exist_ok=True)
                shutil.copy(image_a_path, images_dir / "image_a.jpg")
                shutil.copy(image_b_path, images_dir / "image_b.jpg")

                # Save probabilistic ProbLog program
                with open(example_dir / "probabilistic.pl", 'w') as f:
                    f.write(prob_result.problog_program)

                # Save deterministic ProbLog program
                with open(example_dir / "deterministic.pl", 'w') as f:
                    f.write(det_result.problog_program)

                # Save unified results JSON
                with open(example_dir / "results.json", 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)

                print(f"\nLogs saved to: {example_dir}")

            return result

        except Exception as e:
            raise RuntimeError(f"PROVE pipeline failed: {e}") from e
