"""
Qwen verification engine for PROVE pipeline.
Verifies intra-relationships and inter-comparisons using Qwen 2.5-VL-7B with:
- Native bounding box support
- Unconstrained response generation
- Direct logit probability extraction
"""

from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

from src.core.model_manager import ModelManager
from src.core.types import ObjectDetection, IntraQuestion, InterQuestion
from src.vision.qwen_vl import QwenVL, convert_florence_to_qwen_bbox, create_dual_bbox_prompt


class QwenVerificationError(RuntimeError):
    """Custom exception for Qwen verification failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return f"Qwen Verification Error: {self.message}"


class QwenVerifier:
    """
    Qwen 2.5-VL-7B verification engine for visual question answering.
    
    Key features:
    - Full image context with bounding box labels
    - Unconstrained response generation 
    - Direct probability extraction from model logits
    - No union cropping or image manipulation needed
    """
    
    def __init__(self):
        """Initialize verifier with ModelManager singleton."""
        self.model_manager = ModelManager()
    
    def verify_intra_relations(self, image_path: str, objects: List[ObjectDetection], 
                              relation_candidates: Dict[tuple, List[str]]) -> List[Dict[str, Any]]:
        """
        Verify intra-relationships using full image + bounding box labels.
        
        Args:
            image_path: Path to the source image
            objects: List of ObjectDetection instances
            relation_candidates: Dict mapping (obj1_id, obj2_id) to list of relation candidates
            
        Returns:
            List[Dict[str, Any]]: Verification results with specific relations and logit probabilities
            
        Raises:
            QwenVerificationError: If verification fails
        """
        try:
            if not relation_candidates:
                return []
            
            # Get Qwen VL client from ModelManager
            qwen_client = self.model_manager.get_qwen_vl()
            
            # Load image once for all verifications
            image = Image.open(image_path)
            
            results = []
            object_dict = {obj.object_id: obj for obj in objects}
            
            for (obj_id_1, obj_id_2), relations in relation_candidates.items():
                if obj_id_1 not in object_dict or obj_id_2 not in object_dict:
                    continue
                    
                obj_1 = object_dict[obj_id_1]
                obj_2 = object_dict[obj_id_2]
                
                # Verify each relation candidate using binary questions
                for relation in relations:
                    try:
                        # Create binary verification prompt with bounding boxes
                        prompt = f"""In this image, I've marked two objects:
- Object 1: {convert_florence_to_qwen_bbox(obj_1.bbox)}{obj_1.label}
- Object 2: {convert_florence_to_qwen_bbox(obj_2.bbox)}{obj_2.label}

Is Object 1 {relation} Object 2?

Answer: Yes or No"""
                        
                        # Get response with logits for probability extraction
                        response, logits = qwen_client.run_inference_with_logits(image, prompt)
                        
                        # Extract probability from model logits
                        confidence = qwen_client.extract_response_probability(logits)
                        
                        # Determine if relation exists based on response
                        is_positive = response.lower().strip().startswith('yes')
                        
                        # Use confidence directly if positive, complement if negative
                        final_probability = confidence if is_positive else (1.0 - confidence)
                        
                        # Create structured result
                        result = {
                            "relation_id": f"rel_{obj_id_1}_{obj_id_2}_{relation}",
                            "object_1": obj_id_1,
                            "object_2": obj_id_2,
                            "object_1_label": obj_1.label,
                            "object_2_label": obj_2.label,
                            "relation": relation,
                            "probability": final_probability,
                            "raw_response": response.strip(),
                            "relationship_type": "intra"
                        }
                        
                        results.append(result)
                        
                    except Exception as e:
                        print(f"Warning: Failed to verify relation '{relation}' for {obj_1.label}-{obj_2.label}: {e}")
                        continue
            
            return results
            
        except Exception as err:
            raise QwenVerificationError(f"Intra-relation verification failed: {err}")
    
    def verify_inter_comparisons(self, image_a_path: str, image_b_path: str,
                                objects_a: List[ObjectDetection], objects_b: List[ObjectDetection], 
                                attribute_candidates: Dict[tuple, List[str]]) -> List[Dict[str, Any]]:
        """
        Verify inter-comparison attributes by extracting attribute values from individual objects.
        
        Args:
            image_a_path: Path to image A
            image_b_path: Path to image B
            objects_a: List of ObjectDetection instances for image A
            objects_b: List of ObjectDetection instances for image B
            attribute_candidates: Dict mapping (obj_a_id, obj_b_id) to list of attribute names
            
        Returns:
            List[Dict[str, Any]]: Verification results with individual attribute values and confidences
            
        Raises:
            QwenVerificationError: If verification fails
        """
        try:
            if not attribute_candidates:
                return []
            
            # Get Qwen VL client from ModelManager
            qwen_client = self.model_manager.get_qwen_vl()
            
            # Load images
            image_a = Image.open(image_a_path)
            image_b = Image.open(image_b_path)
            
            # Create object lookup dictionaries
            objects_a_dict = {obj.object_id: obj for obj in objects_a}
            objects_b_dict = {obj.object_id: obj for obj in objects_b}
            
            results = []
            
            for (obj_a_id, obj_b_id), attributes in attribute_candidates.items():
                # Get objects for this comparison
                if (obj_a_id not in objects_a_dict or obj_b_id not in objects_b_dict):
                    continue
                
                obj_a = objects_a_dict[obj_a_id]
                obj_b = objects_b_dict[obj_b_id]
                
                # Extract each attribute for both objects
                for attribute in attributes:
                    try:
                        # Extract attribute from Image A (unconstrained)
                        value_a, conf_a = self._extract_single_attribute(
                            qwen_client, image_a, obj_a, attribute
                        )
                        
                        # Extract attribute from Image B (unconstrained)
                        value_b, conf_b = self._extract_single_attribute(
                            qwen_client, image_b, obj_b, attribute
                        )
                        
                        # Create comparison result
                        result = {
                            "comparison_id": f"comp_{obj_a_id}_{obj_b_id}_{attribute}",
                            "image_a_object_id": obj_a_id,
                            "image_b_object_id": obj_b_id,
                            "attribute": attribute,
                            "value_a": value_a,
                            "value_b": value_b,
                            "confidence_a": conf_a,  # Direct from Qwen logits
                            "confidence_b": conf_b,  # Direct from Qwen logits
                            "object_a_label": obj_a.label,
                            "object_b_label": obj_b.label
                        }
                        
                        results.append(result)
                        
                    except Exception as e:
                        print(f"Warning: Failed to verify inter-comparison for attribute '{attribute}': {e}")
                        continue
            
            return results
            
        except Exception as err:
            raise QwenVerificationError(f"Inter-comparison verification failed: {err}")
    
    def _extract_single_attribute(self, qwen_client: QwenVL, image: Image.Image, 
                                 obj: ObjectDetection, attribute: str) -> Tuple[str, float]:
        """
        Extract single attribute value using unified Florence-2 → LLM → VLM binary verification.
        
        Args:
            qwen_client: Qwen VL client instance
            image: PIL Image object
            obj: ObjectDetection instance
            attribute: Attribute to extract (e.g., "color", "size", "state")
            
        Returns:
            Tuple[str, float]: (attribute_value, confidence)
        """
        try:
            # Step 1: Get Florence-2 description for the object region
            florence_desc = self._get_florence_description_for_object(image, obj)
            
            # Step 2: Generate attribute candidates using LLM based on Florence description
            candidates = self._generate_attribute_candidates_for_single(attribute, obj.label, florence_desc)
            
            # Step 3: Use binary VLM verification to find the best candidate
            best_value = "unknown"
            best_confidence = 0.0
            
            for candidate_value in candidates:
                verified_attr = self._verify_attribute_value_binary(
                    qwen_client, image, obj, attribute, candidate_value
                )
                if verified_attr and verified_attr.confidence > best_confidence:
                    best_value = verified_attr.value
                    best_confidence = verified_attr.confidence
            
            return best_value, best_confidence
            
        except Exception as e:
            print(f"Warning: Failed to extract {attribute} for {obj.label}: {e}")
            return "unknown", 0.0  # Use 0.0 to indicate extraction failure
    
    def extract_attribute_for_object(self, image_path: str, obj: ObjectDetection, 
                                   attribute_category: str) -> Tuple[str, float]:
        """
        Extract attribute value for a single object using unconstrained Qwen generation.
        
        Args:
            image_path: Path to the image file
            obj: ObjectDetection instance
            attribute_category: Attribute category to extract
            
        Returns:
            Tuple[str, float]: (attribute_value, confidence_from_logits)
        """
        try:
            # Get Qwen VL client from ModelManager
            qwen_client = self.model_manager.get_qwen_vl()
            
            # Load image
            image = Image.open(image_path)
            
            # Extract attribute using unconstrained generation
            return self._extract_single_attribute(qwen_client, image, obj, attribute_category)
            
        except Exception as e:
            raise QwenVerificationError(f"Attribute extraction failed for {obj.label}: {e}")
    
    def _get_florence_description_for_object(self, image: Image.Image, obj: ObjectDetection) -> str:
        """
        Get Florence-2 dense caption for a specific object region.
        
        Args:
            image: PIL Image object
            obj: ObjectDetection instance
            
        Returns:
            str: Dense caption describing the object region
        """
        try:
            # Get Florence-2 client from ModelManager
            florence2 = self.model_manager.get_florence2()
            
            # Crop object region with padding
            x1, y1, x2, y2 = obj.bbox
            width, height = image.size
            
            # Add small padding while staying within bounds
            padding = 10
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(width, x2 + padding)
            crop_y2 = min(height, y2 + padding)
            
            # Crop object region
            object_crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            
            # Get detailed description for the cropped object
            description = florence2.describe_region(object_crop, task="<MORE_DETAILED_CAPTION>")
            
            return description
            
        except Exception as e:
            print(f"Warning: Failed to get Florence description for {obj.label}: {e}")
            return f"A {obj.label} object in the image"
    
    def _generate_attribute_candidates_for_single(self, attribute_category: str, object_label: str, 
                                                 florence_desc: str) -> List[str]:
        """
        Generate attribute value candidates for a single attribute category using LLM.
        
        Args:
            attribute_category: The attribute category (e.g., "color", "size")
            object_label: Label of the object
            florence_desc: Florence-2 description of the object
            
        Returns:
            List[str]: List of candidate values for the attribute category
        """
        try:
            llm_client = self.model_manager.get_llm_client()
            
            prompt = f"""Based on this description of a {object_label}:
"{florence_desc}"

What are 3-4 possible values for the {attribute_category} of this {object_label}?

Return a JSON array of short, specific values (1-2 words each):
["value1", "value2", "value3"]

Focus on visually observable and specific attributes."""

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at extracting specific attribute values from object descriptions. Return only a JSON array of short attribute values."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            response = llm_client.chat(
                messages,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Parse response and extract candidates
            import json
            try:
                # Try to parse as direct array
                candidates = json.loads(response)
                if isinstance(candidates, list):
                    return [str(c).strip().lower() for c in candidates if c]
                
                # Try to extract array from object
                if isinstance(candidates, dict):
                    for key, value in candidates.items():
                        if isinstance(value, list):
                            return [str(c).strip().lower() for c in value if c]
            except:
                pass
            
            # Fallback: extract from text
            words = response.lower().split()
            return [word.strip('",[]()') for word in words if len(word) > 2][:4]
            
        except Exception as e:
            print(f"Warning: Failed to generate {attribute_category} candidates for {object_label}: {e}")
            # Return common fallback candidates based on category
            return self._get_fallback_candidates(attribute_category)
    
    def _verify_attribute_value_binary(self, qwen_client, image: Image.Image, 
                                     obj: ObjectDetection, category: str, candidate_value: str):
        """
        Verify attribute value using binary Qwen question (imported from AttributeExtractor logic).
        """
        try:
            from src.vision.qwen_vl import convert_florence_to_qwen_bbox
            from src.core.types import AttributeValue
            
            # Create binary verification prompt with bounding box
            prompt = f"""Look at this object in the image:
{convert_florence_to_qwen_bbox(obj.bbox)}{obj.label}

Is this {obj.label} {candidate_value}?

Answer: Yes or No"""
            
            # Get response with logits for probability extraction
            response, logits = qwen_client.run_inference_with_logits(image, prompt)
            
            # Extract confidence from model logits
            confidence = qwen_client.extract_response_probability(logits)
            
            # Determine if verification is positive
            is_positive = response.lower().strip().startswith('yes')
            
            # Use confidence directly if positive, complement if negative
            final_confidence = confidence if is_positive else (1.0 - confidence)
            
            # Only return if verification is positive
            if is_positive:
                return AttributeValue(value=candidate_value, confidence=final_confidence)
            else:
                return None
                
        except Exception as e:
            print(f"Warning: Failed to verify {category} value '{candidate_value}' for {obj.label}: {e}")
            return None
    
    def _get_fallback_candidates(self, attribute_category: str) -> List[str]:
        """Get fallback candidates for common attribute categories."""
        fallbacks = {
            "color": ["brown", "black", "white", "gray"],
            "size": ["small", "medium", "large"],
            "shape": ["round", "square", "long", "wide"],
            "state": ["active", "passive", "moving", "still"],
            "condition": ["good", "poor", "damaged", "clean"],
            "pattern": ["solid", "striped", "spotted", "textured"],
            "material": ["metal", "wood", "plastic", "fabric"],
            "texture": ["smooth", "rough", "soft", "hard"]
        }
        return fallbacks.get(attribute_category, ["unknown"])
    
    def get_qwen_status(self) -> Dict[str, Any]:
        """
        Get status information about the Qwen model.
        
        Returns:
            Dict[str, Any]: Status information including availability and memory usage
        """
        try:
            qwen_client = self.model_manager.get_qwen_vl()
            
            return {
                "available": qwen_client.is_available(),
                "model_name": qwen_client.get_model_name(),
                "memory_info": qwen_client.get_memory_info()
            }
        except Exception as e:
            return {"error": f"Failed to get Qwen status: {e}"}


# Utility functions for bounding box prompts
def create_binary_relation_prompt(obj1_bbox: List[float], obj1_label: str,
                                 obj2_bbox: List[float], obj2_label: str, 
                                 relation: str) -> str:
    """
    Create binary relation verification prompt with bounding boxes.
    
    Args:
        obj1_bbox: First object bounding box
        obj1_label: First object label  
        obj2_bbox: Second object bounding box
        obj2_label: Second object label
        relation: Relation to verify
        
    Returns:
        str: Formatted binary prompt
    """
    return f"""In this image, I've marked two objects:
- Object 1: {convert_florence_to_qwen_bbox(obj1_bbox)}{obj1_label}
- Object 2: {convert_florence_to_qwen_bbox(obj2_bbox)}{obj2_label}

Is Object 1 {relation} Object 2?

Answer: Yes or No"""


def create_attribute_extraction_prompt(obj_bbox: List[float], obj_label: str, 
                                     attribute: str) -> str:
    """
    Create unconstrained attribute extraction prompt with bounding box.
    
    Args:
        obj_bbox: Object bounding box
        obj_label: Object label
        attribute: Attribute to extract
        
    Returns:
        str: Formatted attribute prompt
    """
    return f"""Look at this object in the image:
{convert_florence_to_qwen_bbox(obj_bbox)}{obj_label}

What is the {attribute} of this {obj_label}?

Answer briefly and specifically."""