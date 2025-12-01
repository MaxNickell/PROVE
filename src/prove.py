"""
PROVE: Probabilistic Reasoning Over Visual Evidence
Main model class encapsulating the entire pipeline.
"""

from typing import Dict, Any
from pathlib import Path
from PIL import Image

from .core.knowledge_base import KnowledgeBase
from .core.model_manager import ModelManager
from .pipeline.detector import Detector
from .pipeline.subquestion_generator import SubquestionGenerator
from .pipeline.unified_agent import UnifiedAgent
from .pipeline.problog_executor import ProbLogExecutor


class PROVE:
    """
    PROVE model for visual reasoning over image pairs.

    Usage:
        model = PROVE(verbose=True)
        answer = model.predict("img1.jpg", "img2.jpg", "Is there a cat in both images?")
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize PROVE model.

        Args:
            verbose: Whether to print concise progress updates
        """
        self.verbose = verbose

        # Components initialized lazily on first use
        self._model_manager = None
        self._detector = None
        self._subquestion_generator = None
        self._unified_agent = None
        self._problog_executor = None

    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _init_components(self):
        """Initialize pipeline components (lazy loading)."""
        if self._model_manager is None:
            self._model_manager = ModelManager()
        if self._detector is None:
            self._detector = Detector()
        if self._subquestion_generator is None:
            self._subquestion_generator = SubquestionGenerator()
        if self._unified_agent is None:
            self._unified_agent = UnifiedAgent(max_iterations=20, debug=False)
        if self._problog_executor is None:
            self._problog_executor = ProbLogExecutor()

    def predict(
        self,
        image_a_path: str,
        image_b_path: str,
        question: str
    ) -> str:
        """
        Run PROVE inference on image pair.

        Args:
            image_a_path: Path to first image
            image_b_path: Path to second image
            question: Ultimate question to answer

        Returns:
            Natural language answer

        Raises:
            FileNotFoundError: If image paths don't exist
            RuntimeError: If pipeline execution fails
        """
        result = self.predict_with_details(image_a_path, image_b_path, question)
        return result['answer']

    def predict_with_details(
        self,
        image_a_path: str,
        image_b_path: str,
        question: str,
        save_logs: bool = False,
        log_dir: str = "logs"
    ) -> Dict[str, Any]:
        """
        Run PROVE inference with detailed outputs.

        Args:
            image_a_path: Path to first image
            image_b_path: Path to second image
            question: Ultimate question to answer
            save_logs: Whether to save ProbLog and intermediate results
            log_dir: Base directory for logs (default: "logs")

        Returns:
            Dict with keys:
                - answer: Natural language answer (str)
                - subquestions: List of subquestions with probabilities
                - problog_program: Generated ProbLog program (str)
                - metadata: Object counts, evidence stats, etc.
                - log_path: Path to saved logs (if save_logs=True)

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
            # Step 1: Image Context Generation
            self._log("Step 1: Image Context Generation...")
            florence2 = self._model_manager.get_florence2()
            for image_id, image_path in image_paths.items():
                image = Image.open(image_path)
                caption = florence2.describe_region(image, task="<MORE_DETAILED_CAPTION>")
                kb.add_scene_context(image_id, {"caption": caption, "image_path": image_path})
            self._log("  ✓ Captions generated")

            # Step 2: Object Detection
            self._log("Step 2: Object Detection...")
            for image_id, image_path in image_paths.items():
                # Reuse caption from Step 1 (efficiency improvement)
                caption = kb.images[image_id].scene_context["caption"]
                detections = self._detector.detect_from_caption(image_path, caption)
                kb.add_objects(image_id, detections)
            total_objects = sum(len(img.objects) for img in kb.images.values())
            self._log(f"  ✓ Detected {total_objects} objects")

            # Step 3: Subquestion Generation
            self._log("Step 3: Subquestion Generation...")
            subquestions = self._subquestion_generator.generate_binary_subquestions(
                question, kb.images
            )
            kb.add_subquestions(subquestions)
            self._log(f"  ✓ Generated {len(subquestions)} subquestions")

            # Step 4: Evidence Collection
            self._log("Step 4: Evidence Collection...")
            evidence_by_subquestion = []
            for subquestion in kb.subquestions:
                evidence = self._unified_agent.collect_evidence(
                    subquestion=subquestion,
                    images=kb.images,
                    image_paths=image_paths
                )
                evidence_by_subquestion.append(evidence)

            total_attrs = sum(len(e.attributes) for e in evidence_by_subquestion)
            total_rels = sum(len(e.relationships) for e in evidence_by_subquestion)
            total_counts = sum(len(e.counts) for e in evidence_by_subquestion)
            self._log(f"  ✓ Collected {total_attrs} attributes, {total_rels} relationships, {total_counts} counts")

            # Step 5: ProbLog Reasoning
            self._log("Step 5: ProbLog Reasoning...")
            subquestion_results, ultimate_answer = self._problog_executor.execute_subquestions(
                subquestions=kb.subquestions,
                evidence_collections=evidence_by_subquestion,
                images=kb.images,
                ultimate_question=question
            )
            kb.add_subquestion_results(subquestion_results)
            self._log("  ✓ Reasoning complete")

            # Read generated ProbLog program
            try:
                with open('knowledge_base.pl', 'r') as f:
                    problog_program = f.read()
            except FileNotFoundError:
                problog_program = ""

            # Build result
            result = {
                'answer': ultimate_answer,
                'subquestions': [
                    {
                        'question': sq_result.subquestion,
                        'probability': sq_result.probability
                    }
                    for sq_result in subquestion_results
                ],
                'problog_program': problog_program,
                'metadata': {
                    'total_objects': total_objects,
                    'num_subquestions': len(subquestions),
                    'total_attributes': total_attrs,
                    'total_relationships': total_rels,
                    'total_counts': total_counts
                }
            }

            if self.verbose:
                self._log(f"\nAnswer: {ultimate_answer}")

            # Save logs if requested
            if save_logs:
                from datetime import datetime
                import shutil

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                question_hash = str(hash(question))[:8]
                example_dir = Path(log_dir) / f"{timestamp}_{question_hash}"
                example_dir.mkdir(parents=True, exist_ok=True)

                # Copy images
                images_dir = example_dir / "images"
                images_dir.mkdir(exist_ok=True)
                shutil.copy(image_a_path, images_dir / "image_a.jpg")
                shutil.copy(image_b_path, images_dir / "image_b.jpg")

                # Save ProbLog program
                with open(example_dir / "knowledge_base.pl", 'w') as f:
                    f.write(problog_program)

                # Save results JSON
                import json
                with open(example_dir / "results.json", 'w') as f:
                    json.dump({
                        'question': question,
                        'answer': ultimate_answer,
                        'subquestions': result['subquestions'],
                        'metadata': result['metadata']
                    }, f, indent=2)

                result['log_path'] = str(example_dir)
                self._log(f"Logs saved to: {example_dir}")

            return result

        except Exception as e:
            raise RuntimeError(f"PROVE pipeline failed: {e}") from e
