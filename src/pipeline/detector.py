"""
Object detector using Florence-2 with ModelManager singleton.
Extracts entities from questions and runs open vocabulary detection.
"""

from typing import List, Dict, Any
from PIL import Image

from src.core.model_manager import ModelManager
from src.core.types import ObjectDetection
from src.core.probability import calibrate_detector_confidence
from src.core.image_utils import load_rgb_image


class Detector:
    """
    Object detector using Florence-2 model.

    Flow:
    1. LLM extracts entity nouns from question
    2. Florence-2 runs open vocabulary detection for each entity
    3. Returns calibrated ObjectDetection instances
    """

    def __init__(self):
        """Initialize detector with ModelManager singleton."""
        self.model_manager = ModelManager()

    def detect_from_question(
        self,
        image_path: str,
        question: str,
        visualize: bool = False
    ) -> List[ObjectDetection]:
        """
        Detect objects mentioned in the question.

        Args:
            image_path: Path to input image
            question: Ultimate question text
            visualize: Whether to save annotated image

        Returns:
            List[ObjectDetection]: Detected objects with calibrated confidences
        """
        try:
            image = load_rgb_image(image_path)
            florence2 = self.model_manager.get_florence2()
            llm_client = self.model_manager.get_llm_client()

            # Extract entities from question
            entities = self._extract_entities(question, llm_client)

            if not entities:
                return []

            # Run open vocabulary detection for each entity
            all_detections = []
            for entity_class in entities:
                detections = self._detect_entity(image, entity_class, florence2)
                all_detections.extend(detections)

            # Convert to ObjectDetection instances
            objects = self._convert_to_object_detections(all_detections)

            # Optionally visualize
            if visualize and objects:
                output_path = image_path.rsplit(".", 1)[0] + "_annotated." + image_path.rsplit(".", 1)[1]
                annotated_image = florence2.visualize_detections(image, all_detections)
                annotated_image.save(output_path)

            return objects

        except Exception as err:
            raise RuntimeError(f"Detection failed: {err}")

    def _extract_entities(self, question: str, llm_client) -> List[str]:
        """Extract detectable object nouns from question using LLM."""

        messages = [
            {
                "role": "system",
                "content": "Extract physical objects from questions for object detection. Output singular nouns only."
            },
            {
                "role": "user",
                "content": f"""Extract detectable objects from the question.

RULES:
1. Use singular form (dogs → dog, children → child, people → person)
2. Include compound nouns (cookie dough, traffic light)
3. Exclude attributes (colors, sizes, materials)
4. Exclude non-physical concepts (image, scene, picture, both, same)

EXAMPLES:

Question: Are there more dogs in one image than the other?
Output: {{"entities": ["dog"]}}

Question: Is the silver spoon in the cookie dough?
Output: {{"entities": ["spoon", "cookie dough"]}}

Question: Are all the children wearing red mittens?
Output: {{"entities": ["child", "mitten"]}}

Question: Is there a white bird sitting on a buffalo?
Output: {{"entities": ["bird", "buffalo"]}}

Question: Do both images show people riding bicycles?
Output: {{"entities": ["person", "bicycle"]}}

Question: Are the cars in both images the same color?
Output: {{"entities": ["car"]}}

Question: Is every cat in the picture sleeping on a couch?
Output: {{"entities": ["cat", "couch"]}}

Question: {question}
Output:"""
            }
        ]

        response = llm_client.extract_entities(messages, temperature=0)
        return response.entities  # Already lowercase and deduplicated by Pydantic

    def _detect_entity(
        self,
        image: Image.Image,
        entity_class: str,
        florence2
    ) -> List[Dict[str, Any]]:
        """Run open vocabulary detection for a single entity class."""

        result = florence2.detect_open_vocabulary(image, entity_class)

        bboxes = result.get("bboxes", [])
        labels = result.get("bboxes_labels", [])
        scores = result.get("scores", [])

        detections = []
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            raw_conf = scores[i] if scores and i < len(scores) else None
            calibrated_conf = calibrate_detector_confidence(raw_conf) if raw_conf is not None else 0.5

            detections.append({
                "bbox": bbox,
                "label": label.lower().strip(),
                "confidence": calibrated_conf
            })

        return detections

    def _convert_to_object_detections(
        self,
        raw_detections: List[Dict[str, Any]]
    ) -> List[ObjectDetection]:
        """Convert raw detections to ObjectDetection instances."""

        objects = []
        for i, detection in enumerate(raw_detections):
            try:
                bbox = detection.get('bbox', [0, 0, 0, 0])
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue

                obj = ObjectDetection(
                    object_id=i,
                    label=str(detection.get('label', 'unknown')),
                    bbox=[float(c) for c in bbox],
                    confidence=round(max(0.0, min(1.0, float(detection.get('confidence', 0.5)))), 3)
                )
                objects.append(obj)

            except Exception as e:
                print(f"Warning: Failed to convert detection {i}: {e}")

        return objects
