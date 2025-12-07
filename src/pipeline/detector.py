"""
Object detector using Florence-2 with ModelManager singleton.
"""

from typing import List, Dict, Any
from PIL import Image
import os
import re

from src.core.model_manager import ModelManager
from src.core.types import ObjectDetection
from src.core.probability import calibrate_detector_confidence
from src.core.image_utils import load_rgb_image




class Detector:
    """
    Object detector using Florence-2 model with ModelManager singleton.
    Produces ObjectDetection instances with exact schema compliance.
    """

    def __init__(self):
        """Initialize detector with ModelManager singleton."""
        # Use ModelManager singleton instead of creating Florence2 directly
        self.model_manager = ModelManager()

    def _normalize_label_to_singular(self, label: str) -> str:
        """Normalize plural labels to singular forms."""
        label = label.lower().strip()

        # Common irregular plurals
        plurals = {
            'children': 'child', 'people': 'person', 'men': 'man', 'women': 'woman',
            'feet': 'foot', 'teeth': 'tooth', 'geese': 'goose', 'mice': 'mouse'
        }
        if label in plurals:
            return plurals[label]

        # Simple patterns
        if label.endswith('ies'): return label[:-3] + 'y'
        if label.endswith('es'): return label[:-2]
        if label.endswith('s') and not label.endswith('ss'): return label[:-1]
        return label

    def detect(self, image_path: str, visualize: bool = False) -> List[ObjectDetection]:
        """
        Detect objects using caption-based open vocabulary detection.
        Generates caption internally.

        Args:
            image_path: Path to input image
            visualize: Whether to save annotated image

        Returns:
            List[ObjectDetection]: Detected objects with exact schema compliance

        Raises:
            RuntimeError: If detection fails
        """
        try:
            # Load image and generate caption
            image = load_rgb_image(image_path)
            florence2 = self.model_manager.get_florence2()

            caption = florence2.describe_region(image, task="<MORE_DETAILED_CAPTION>")

            # Delegate to caption-based detection
            return self.detect_from_caption(image_path, caption, visualize)

        except Exception as err:
            raise RuntimeError(f"Object detection failed: {err}")

    def detect_from_caption(self, image_path: str, caption: str, visualize: bool = False) -> List[ObjectDetection]:
        """
        Detect objects using a pre-generated caption.

        Pipeline:
        1. Extract entity nouns from caption using LLM
        2. Run open vocabulary detection for each entity class

        Args:
            image_path: Path to input image
            caption: Pre-generated detailed caption
            visualize: Whether to save annotated image

        Returns:
            List[ObjectDetection]: Detected objects with exact schema compliance

        Raises:
            RuntimeError: If detection fails
        """
        try:
            # Load image (validates existence and converts to RGB)
            image = load_rgb_image(image_path)

            # Get models from singleton ModelManager
            florence2 = self.model_manager.get_florence2()
            llm_client = self.model_manager.get_llm_client()

            # Extract entities (silent)
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at converting an image caption into a list of objects used for downstream object detection."
                },
                {
                    "role": "user",
                    "content": f"""Extract nouns from the caption for object detection.

Rules: Use singular nouns, include compound nouns, exclude modifiers and non-detectable words.

Example:
Caption: A woman cooking on the stove while a child sits on a chair.
Output: {{"entities": ["woman", "stove", "child", "chair"]}}

Caption: {caption}"""
                }
            ]

            entity_response = llm_client.extract_entities(messages, temperature=0)
            entities = entity_response.entities  # Already lowercase and deduplicated by Pydantic

            if not entities:
                return []

            # Run open vocabulary detection for each entity (silent)
            all_detections = []

            for entity_class in entities:
                ovd_result = florence2.detect_open_vocabulary(image, entity_class)

                bboxes = ovd_result.get("bboxes", [])
                labels = ovd_result.get("bboxes_labels", [])  # OVD task returns "bboxes_labels"
                scores = ovd_result.get("scores", [])

                # Process detections for this entity class
                for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                    raw_conf = scores[i] if scores and i < len(scores) else None
                    calibrated_conf = calibrate_detector_confidence(raw_conf) if raw_conf is not None else None

                    # Normalize label to singular form
                    normalized_label = self._normalize_label_to_singular(label)

                    all_detections.append({
                        "bbox": bbox,
                        "label": normalized_label,
                        "confidence": calibrated_conf
                    })

            # Convert to ObjectDetection instances
            objects = self._convert_to_object_detections(all_detections)

            # Optionally visualize
            if visualize and objects:
                output_path = image_path.rsplit(".", 1)[0] + "_annotated." + image_path.rsplit(".", 1)[1]
                annotated_image = florence2.visualize_detections(image, all_detections)
                annotated_image.save(output_path)

            return objects

        except Exception as err:
            raise RuntimeError(f"Caption-based detection failed: {err}")

    def detect_from_question(self, image_path: str, question: str, visualize: bool = False) -> List[ObjectDetection]:
        """
        Detect objects using ultimate question to guide detection.

        Extracts only entities mentioned in the question to ensure:
        - No synonym mismatches (question terms used directly)
        - Only relevant objects detected (efficiency + less noise)

        Args:
            image_path: Path to input image
            question: Ultimate question text
            visualize: Whether to save annotated image

        Returns:
            List[ObjectDetection]: Detected objects

        Raises:
            RuntimeError: If detection fails
        """
        try:
            # Load image (validates existence and converts to RGB)
            image = load_rgb_image(image_path)

            # Get models from singleton ModelManager
            florence2 = self.model_manager.get_florence2()
            llm_client = self.model_manager.get_llm_client()

            # Extract entities (silent)
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at converting a visual reasoning question into a list of objects used for downstream object detection."
                },
                {
                    "role": "user",
                    "content": f"""Extract nouns from the question for object detection.

Rules: Use singular nouns, include compound nouns, exclude modifiers and non-detectable words.

Examples:
Question: A silver spoon has cookie dough in it.
Output: {{"entities": ["spoon", "cookie dough"]}}

Question: There are two pairs of hands wearing gloves.
Output: {{"entities": ["hand", "glove"]}}

Question: {question}"""
                }
            ]

            entity_response = llm_client.extract_entities(messages, temperature=0)
            entities = entity_response.entities  # Already lowercase and deduplicated by Pydantic

            if not entities:
                return []

            # Run open vocabulary detection for each entity (silent)
            all_detections = []

            for entity_class in entities:
                ovd_result = florence2.detect_open_vocabulary(image, entity_class)

                bboxes = ovd_result.get("bboxes", [])
                labels = ovd_result.get("bboxes_labels", [])
                scores = ovd_result.get("scores", [])

                # Process detections for this entity class
                for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                    raw_conf = scores[i] if scores and i < len(scores) else None
                    calibrated_conf = calibrate_detector_confidence(raw_conf) if raw_conf is not None else None

                    # Normalize label to singular form
                    normalized_label = self._normalize_label_to_singular(label)

                    all_detections.append({
                        "bbox": bbox,
                        "label": normalized_label,
                        "confidence": calibrated_conf
                    })

            # Convert to ObjectDetection instances
            objects = self._convert_to_object_detections(all_detections)

            # Optionally visualize
            if visualize and objects:
                output_path = image_path.rsplit(".", 1)[0] + "_annotated." + image_path.rsplit(".", 1)[1]
                annotated_image = florence2.visualize_detections(image, all_detections)
                annotated_image.save(output_path)

            return objects

        except Exception as err:
            raise RuntimeError(f"Question-based detection failed: {err}")

    def detect_with_crops(self, image_path: str, save_crops: bool = False, 
                         crop_dir: str = "crops") -> List[ObjectDetection]:
        """
        Detect objects and optionally save crops for later use.
        
        Args:
            image_path: Path to input image
            save_crops: Whether to save object crops to disk
            crop_dir: Directory to save crops
            
        Returns:
            List[ObjectDetection]: Detected objects with crop paths if saved
        """
        try:
            image = load_rgb_image(image_path)
            florence2 = self.model_manager.get_florence2()
            
            # Use detect_and_describe method for crops
            raw_detections = florence2.detect_and_describe(image, save_crops=save_crops, crop_dir=crop_dir)
            
            # Convert to ObjectDetection instances
            objects = self._convert_to_object_detections(raw_detections)
            
            return objects
            
        except Exception as err:
            raise RuntimeError(f"Florence-2 detection with crops failed: {err}")
    
    def _convert_to_object_detections(self, raw_detections: List[Dict[str, Any]]) -> List[ObjectDetection]:
        """
        Convert Florence-2 raw detection results to ObjectDetection instances.
        
        Args:
            raw_detections: Raw detection results from Florence-2
            
        Returns:
            List[ObjectDetection]: Properly typed object detections
        """
        objects = []
        
        for i, detection in enumerate(raw_detections):
            try:
                # Use sequential object ID assignment (0, 1, 2, ...)
                object_id = i  # Always use sequential index
                label = detection.get('label', 'unknown')
                bbox = detection.get('bbox', [0, 0, 0, 0])
                confidence = detection.get('confidence', 0.5)
                
                # Validate bbox format [x1, y1, x2, y2]
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(f"Invalid bbox format: {bbox}")
                
                # Ensure all bbox coordinates are floats
                bbox = [float(coord) for coord in bbox]
                
                # Validate confidence is between 0 and 1
                confidence = max(0.0, min(1.0, float(confidence)))
                
                # Create ObjectDetection instance
                obj = ObjectDetection(
                    object_id=int(object_id),
                    label=str(label),
                    bbox=bbox,
                    confidence=round(confidence, 3)
                )
                
                objects.append(obj)
                
            except Exception as e:
                print(f"Warning: Failed to convert detection {i}: {e}")
                # Create a fallback object
                fallback_obj = ObjectDetection(
                    object_id=i,
                    label="unknown",
                    bbox=[0.0, 0.0, 10.0, 10.0],
                    confidence=0.1
                )
                objects.append(fallback_obj)
        
        return objects
    
    def validate_detections(self, objects: List[ObjectDetection]) -> bool:
        """
        Validate that all detections conform to expected schema.
        
        Args:
            objects: List of ObjectDetection instances
            
        Returns:
            bool: True if all detections are valid
        """
        try:
            for obj in objects:
                # Check required attributes exist
                assert hasattr(obj, 'object_id')
                assert hasattr(obj, 'label')
                assert hasattr(obj, 'bbox')
                assert hasattr(obj, 'confidence')
                
                # Validate types
                assert isinstance(obj.object_id, int)
                assert isinstance(obj.label, str)
                assert isinstance(obj.bbox, list) and len(obj.bbox) == 4
                assert isinstance(obj.confidence, float)
                
                # Validate value ranges
                assert 0.0 <= obj.confidence <= 1.0
                
            return True
            
        except AssertionError:
            return False
    
    def generate_detailed_captions(self, image_paths: Dict[str, str]) -> Dict[str, str]:
        """
        Generate detailed captions for images using Florence-2.
        
        Args:
            image_paths: Dictionary mapping image_id to image path
            
        Returns:
            Dict[str, str]: Detailed captions per image
            
        Raises:
            DetectorError: If caption generation fails
        """
        try:
            florence2 = self.model_manager.get_florence2()
            captions = {}
            
            for image_id, image_path in image_paths.items():
                # Load image (validates existence and converts to RGB)
                image = load_rgb_image(image_path)

                # Generate detailed caption
                caption = florence2.describe_region(image, task="<MORE_DETAILED_CAPTION>")
                captions[image_id] = caption
            
            return captions
            
        except Exception as err:
            raise RuntimeError(f"Caption generation failed: {err}")
    
    def get_detection_summary(self, objects: List[ObjectDetection]) -> Dict[str, Any]:
        """
        Get summary statistics for detections.
        
        Args:
            objects: List of ObjectDetection instances
            
        Returns:
            Dict[str, Any]: Summary information
        """
        if not objects:
            return {"count": 0, "labels": [], "avg_confidence": 0.0}
        
        labels = [obj.label for obj in objects]
        confidences = [obj.confidence for obj in objects]
        
        return {
            "count": len(objects),
            "labels": list(set(labels)),
            "unique_labels": len(set(labels)),
            "avg_confidence": round(sum(confidences) / len(confidences), 3),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences)
        }
