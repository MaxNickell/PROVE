"""
VLM verification engine for PROVE pipeline.
Verifies intra-relationships and inter-comparisons using visual question answering.
Supports multiple VLMs through abstraction layer.
"""

from typing import List, Dict, Any, Optional, Tuple
import json

from src.core.model_manager import ModelManager
from src.core.types import ObjectDetection, IntraQuestion, InterQuestion
from src.core.vlm_interface import VLMInterface, VLMError, VLMInferenceError, VLMNotAvailableError
from src.vision.image_utils import create_relationship_crop, ImageProcessingError
from PIL import Image


class VLMVerificationError(RuntimeError):
    """Custom exception for VLM verification failures."""
    def __init__(self, message: str, vlm_name: str = "unknown"):
        super().__init__(message)
        self.message = message
        self.vlm_name = vlm_name
    
    def __str__(self):
        return f"VLM Verification Error [{self.vlm_name}]: {self.message}"


class VLMVerifier:
    """
    VLM verification engine for visual question answering.
    Verifies intra-relationships and inter-comparisons using cropped images.
    Supports any VLM that implements VLMInterface (VLM, GPT-4V, Claude Vision, etc.).
    """
    
    def __init__(self):
        """Initialize verifier with ModelManager singleton."""
        self.model_manager = ModelManager()
    
    def _get_vlm_name(self) -> str:
        """Get the name of the current VLM model."""
        try:
            vlm = self.model_manager.get_vlm()
            return vlm.get_model_name()
        except Exception:
            return "unknown"
    
    
    def verify_intra_relations(self, image_path: str, objects: List[ObjectDetection], 
                              relation_candidates: Dict[tuple, List[str]]) -> List[Dict[str, Any]]:
        """
        Verify intra-relationships using LLM-generated relation candidates with binary VLM verification.
        
        Args:
            image_path: Path to the source image
            objects: List of ObjectDetection instances
            relation_candidates: Dict mapping (obj1_id, obj2_id) to list of relation candidates
            
        Returns:
            List[Dict[str, Any]]: Verification results for all relations with probabilities
            
        Raises:
            VLMVerificationError: If verification fails
        """
        try:
            if not relation_candidates:
                return []
            
            # Load image
            image = Image.open(image_path)
            
            # Get VLM client from ModelManager
            vlm_client = self.model_manager.get_vlm()
            
            results = []
            object_dict = {obj.object_id: obj for obj in objects}
            
            for (obj_id_1, obj_id_2), relations in relation_candidates.items():
                if obj_id_1 not in object_dict or obj_id_2 not in object_dict:
                    continue
                    
                obj_1 = object_dict[obj_id_1]
                obj_2 = object_dict[obj_id_2]
                
                # Create relationship crop for VLM
                relationship_crop = create_relationship_crop(
                    image, obj_1, obj_2, 
                    padding=30, use_blackout=True
                )
                
                # Verify each relation candidate
                for relation in relations:
                    try:
                        # Create binary verification prompt
                        prompt = self._create_binary_relation_prompt(relation, obj_1.label, obj_2.label)
                        
                        # Get VLM response
                        llava_response = vlm_client.run_inference(relationship_crop, prompt)
                        
                        # Parse binary response and convert to probability
                        is_positive = self._parse_binary_response(llava_response)
                        probability = 0.9 if is_positive else 0.1
                        
                        # Create result record
                        result = {
                            "relation_id": f"rel_{obj_id_1}_{obj_id_2}_{relation}",
                            "object_1": obj_id_1,
                            "object_2": obj_id_2,
                            "object_1_label": obj_1.label,
                            "object_2_label": obj_2.label,
                            "relation": relation,
                            "probability": probability,
                            "relationship_type": "intra"
                        }
                        
                        results.append(result)
                        
                    except Exception as e:
                        print(f"Warning: Failed to verify relation '{relation}' for {obj_1.label}-{obj_2.label}: {e}")
                        continue
            
            return results
            
        except Exception as err:
            raise VLMVerificationError(f"Intra-relation verification failed: {err}", self._get_vlm_name())
    
    def verify_inter_comparisons(self, image_a_path: str, image_b_path: str,
                                objects_a: List[ObjectDetection], objects_b: List[ObjectDetection], 
                                questions: List[InterQuestion]) -> List[Dict[str, Any]]:
        """
        Verify inter-comparison questions by extracting attribute values from individual objects.
        
        Args:
            image_a_path: Path to image A
            image_b_path: Path to image B
            objects_a: List of ObjectDetection instances for image A
            objects_b: List of ObjectDetection instances for image B
            questions: List of InterQuestion instances to verify
            
        Returns:
            List[Dict[str, Any]]: Verification results with attribute values
            
        Raises:
            VLMVerificationError: If verification fails
        """
        try:
            if not questions:
                return []
            
            # Load images
            image_a = Image.open(image_a_path)
            image_b = Image.open(image_b_path)
            
            # Get VLM client from ModelManager
            vlm_client = self.model_manager.get_vlm()
            
            results = []
            object_dict_a = {obj.object_id: obj for obj in objects_a}
            object_dict_b = {obj.object_id: obj for obj in objects_b}
            
            for question in questions:
                try:
                    # Get objects for this attribute extraction
                    obj_a_id = question.image_a_object_id
                    obj_b_id = question.image_b_object_id
                    
                    if obj_a_id not in object_dict_a or obj_b_id not in object_dict_b:
                        continue
                    
                    obj_a = object_dict_a[obj_a_id]
                    obj_b = object_dict_b[obj_b_id]
                    
                    # Extract attribute from object A
                    crop_a = self._create_object_crop(image_a, obj_a, padding=20)
                    prompt_a = self._create_attribute_extraction_prompt(question.question, obj_a.label)
                    response_a = vlm_client.run_inference(crop_a, prompt_a)
                    value_a = self._parse_attribute_response(response_a)
                    
                    # Extract attribute from object B  
                    crop_b = self._create_object_crop(image_b, obj_b, padding=20)
                    prompt_b = self._create_attribute_extraction_prompt(question.question, obj_b.label)
                    response_b = vlm_client.run_inference(crop_b, prompt_b)
                    value_b = self._parse_attribute_response(response_b)
                    
                    # Determine attribute type from question
                    attribute_type = self._extract_attribute_type(question.question)
                    
                    # Create result record
                    result = {
                        "question_id": f"inter_{obj_a_id}_{obj_b_id}",
                        "question": question.question,
                        "image_a_object_id": obj_a_id,
                        "image_b_object_id": obj_b_id,
                        "image_a_label": obj_a.label,
                        "image_b_label": obj_b.label,
                        "attribute": attribute_type,
                        "value_a": value_a,
                        "value_b": value_b,
                        "confidence_a": 1.0,  # MVP: 100% confidence in extracted value_a
                        "confidence_b": 1.0,  # MVP: 100% confidence in extracted value_b
                        "relationship_type": "inter"
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"Warning: Failed to verify inter-question '{question.question}': {e}")
                    continue
            
            return results
            
        except Exception as err:
            raise VLMVerificationError(f"Inter-comparison verification failed: {err}", self._get_vlm_name())
    
    
    def _create_attribute_extraction_prompt(self, question: str, object_label: str) -> str:
        """
        Create prompt for individual attribute extraction.
        
        Args:
            question: The attribute question (e.g., "What is the size of this carnivore?")
            object_label: Label of the object
            
        Returns:
            str: Formatted prompt for VLM
        """
        prompt = f"""Look at this image and answer the specific question about the {object_label}.

Question: {question}

Answer with a single word or short phrase that describes the attribute value. Examples:
- Size: large, medium, small, tiny, huge
- Color: red, blue, brown, black, white, multicolored
- State: alive, dead, moving, stationary, alert, resting
- Condition: new, old, clean, dirty, damaged, healthy

Answer:"""

        return prompt
    
    def _create_binary_relation_prompt(self, relation: str, obj_1_label: str, obj_2_label: str) -> str:
        """
        Create prompt for binary relation verification.
        
        Args:
            relation: The relation to verify (e.g., "eating", "near", "above")
            obj_1_label: Label of the first object (red box)
            obj_2_label: Label of the second object (yellow box)
            
        Returns:
            str: Formatted prompt for VLM binary verification
        """
        prompt = f"""Look at this image with highlighted objects and answer with only "Yes" or "No".

In this image:
- RED box: {obj_1_label} (object 1)
- YELLOW box: {obj_2_label} (object 2)

Question: Is object 1 {relation} object 2?
(Is the {obj_1_label} {relation} the {obj_2_label}?)

Answer with only "Yes" or "No" based on what you observe in the image."""

        return prompt
    
    def _parse_binary_response(self, response: str) -> bool:
        """
        Parse binary VLM response for relation verification.
        
        Args:
            response: Raw VLM response text
            
        Returns:
            bool: True if relation exists (Yes), False if not (No)
        """
        try:
            response_lower = response.lower().strip()
            
            # Look for clear yes/no indicators
            if any(word in response_lower for word in ['yes', 'true', 'correct', 'indeed']):
                return True
            elif any(word in response_lower for word in ['no', 'false', 'incorrect', 'not']):
                return False
            else:
                # Default to False if unclear
                return False
                
        except Exception as e:
            print(f"Warning: Failed to parse binary response: {e}")
            return False
    
    def _create_inter_verification_prompt(self, question: str, label_a: str, 
                                        label_b: str) -> str:
        """
        Create prompt for inter-comparison verification.
        
        Args:
            question: The comparison question
            label_a: Label of object from image A (left side)
            label_b: Label of object from image B (right side)
            
        Returns:
            str: Formatted prompt for VLM
        """
        prompt = f"""You are comparing two objects to answer a specific question.

In this comparison image:
- LEFT object: {label_a} (from first image)
- RIGHT object: {label_b} (from second image)

Question: {question}

Please compare these objects based on the specified attribute or property. Provide your analysis of how they differ or are similar.

Format your response as:
Comparison: [Your comparative analysis]
Confidence: [High/Medium/Low]
Explanation: [Brief explanation of your reasoning]"""

        return prompt
    
    def _create_object_crop(self, image: Image.Image, obj: ObjectDetection, 
                          padding: int = 20) -> Image.Image:
        """
        Create a crop of a single object with padding.
        
        Args:
            image: PIL Image object
            obj: ObjectDetection instance
            padding: Padding around object
            
        Returns:
            Image.Image: Cropped image of the object
        """
        try:
            width, height = image.size
            x1, y1, x2, y2 = obj.bbox
            
            # Add padding while staying within bounds
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(width, x2 + padding)
            crop_y2 = min(height, y2 + padding)
            
            # Ensure valid crop
            if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
                # Create a small fallback crop
                return Image.new('RGB', (100, 100), color='white')
            
            return image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            
        except Exception as e:
            raise ImageProcessingError(f"Object crop failed: {e}")
    
    def _create_comparison_image(self, crop_a: Image.Image, crop_b: Image.Image) -> Image.Image:
        """
        Create side-by-side comparison image.
        
        Args:
            crop_a: Crop from image A
            crop_b: Crop from image B
            
        Returns:
            Image.Image: Combined comparison image
        """
        try:
            # Resize crops to same height for better comparison
            target_height = 200
            
            # Calculate new widths maintaining aspect ratio
            width_a = int((crop_a.width * target_height) / crop_a.height)
            width_b = int((crop_b.width * target_height) / crop_b.height)
            
            # Resize crops
            crop_a_resized = crop_a.resize((width_a, target_height), Image.Resampling.LANCZOS)
            crop_b_resized = crop_b.resize((width_b, target_height), Image.Resampling.LANCZOS)
            
            # Create combined image
            total_width = width_a + width_b + 10  # 10px separator
            combined = Image.new('RGB', (total_width, target_height), 'white')
            
            # Paste crops side by side
            combined.paste(crop_a_resized, (0, 0))
            combined.paste(crop_b_resized, (width_a + 10, 0))
            
            return combined
            
        except Exception as e:
            raise ImageProcessingError(f"Comparison image creation failed: {e}")
    
    
    def _parse_comparison_response(self, response: str) -> Dict[str, Any]:
        """
        Parse VLM response for inter-comparison verification.
        
        Args:
            response: Raw VLM response text
            
        Returns:
            Dict[str, Any]: Parsed response with comparison and confidence
        """
        try:
            # Initialize default values
            result = {
                "comparison": response,
                "confidence": 0.5,
                "explanation": response
            }
            
            # Parse structured response
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('Comparison:'):
                    result["comparison"] = line.replace('Comparison:', '').strip()
                elif line.startswith('Confidence:'):
                    conf_text = line.replace('Confidence:', '').strip().lower()
                    if 'high' in conf_text:
                        result["confidence"] = 0.9
                    elif 'medium' in conf_text:
                        result["confidence"] = 0.7
                    elif 'low' in conf_text:
                        result["confidence"] = 0.3
                elif line.startswith('Explanation:'):
                    result["explanation"] = line.replace('Explanation:', '').strip()
            
            return result
            
        except Exception as e:
            print(f"Warning: Failed to parse comparison response: {e}")
            return {"comparison": response, "confidence": 0.5, "explanation": response}
    
    def _parse_attribute_response(self, response: str) -> str:
        """
        Parse VLM response for attribute extraction.
        
        Args:
            response: Raw VLM response text
            
        Returns:
            str: Parsed attribute value
        """
        try:
            # Clean the response
            cleaned = response.strip()
            
            # Look for "Answer:" prefix and extract what follows
            if "Answer:" in cleaned:
                cleaned = cleaned.split("Answer:", 1)[1].strip()
            
            # Take the first line or first few words as the attribute value
            lines = cleaned.split('\n')
            first_line = lines[0].strip()
            
            # Remove common prefixes/suffixes
            prefixes_to_remove = ["The", "This", "It is", "It's", "I see", "appears to be", "seems to be"]
            for prefix in prefixes_to_remove:
                if first_line.lower().startswith(prefix.lower()):
                    first_line = first_line[len(prefix):].strip()
            
            # Take first few words (attribute values should be short)
            words = first_line.split()[:3]  # Max 3 words for attribute value
            result = " ".join(words).lower()
            
            # If result is too long or contains common non-attribute words, truncate
            non_attr_words = ["a", "an", "the", "that", "which", "very", "quite", "rather"]
            result_words = [word for word in result.split() if word not in non_attr_words]
            
            return " ".join(result_words) if result_words else result
            
        except Exception as e:
            print(f"Warning: Failed to parse attribute response: {e}")
            return response[:20].lower()  # Fallback: first 20 chars, lowercase
    
    def _extract_attribute_type(self, question: str) -> str:
        """
        Extract attribute type from question text.
        
        Args:
            question: The question text
            
        Returns:
            str: Attribute type (size, color, state, etc.)
        """
        question_lower = question.lower()
        
        attribute_keywords = {
            "size": ["size", "large", "small", "big", "tiny", "huge", "dimensions"],
            "color": ["color", "colour", "red", "blue", "green", "yellow", "black", "white"],
            "state": ["state", "alive", "dead", "moving", "stationary", "condition"],
            "shape": ["shape", "round", "square", "circular", "rectangular"],
            "material": ["material", "metal", "wood", "plastic", "fabric", "made of"],
            "texture": ["texture", "smooth", "rough", "soft", "hard", "furry"],
            "pattern": ["pattern", "striped", "spotted", "solid", "dotted"],
            "condition": ["condition", "new", "old", "damaged", "clean", "dirty", "healthy"],
            "position": ["position", "location", "where", "above", "below", "near"],
            "function": ["function", "purpose", "used for", "doing", "action"]
        }
        
        for attr_type, keywords in attribute_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return attr_type
        
        return "unknown"
    
    def validate_verification_results(self, results: List[Dict[str, Any]]) -> bool:
        """
        Validate verification results format.
        
        Args:
            results: List of verification result dictionaries
            
        Returns:
            bool: True if all results are valid
        """
        try:
            required_fields = ["question_id", "question", "confidence", "relationship_type"]
            
            for result in results:
                # Check required fields
                if not all(field in result for field in required_fields):
                    return False
                
                # Check confidence range
                if not (0 <= result["confidence"] <= 1):
                    return False
                
                # Check relationship type
                if result["relationship_type"] not in ["intra", "inter"]:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_verification_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics for verification results.
        
        Args:
            results: List of verification result dictionaries
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
        if not results:
            return {"total": 0, "intra": 0, "inter": 0, "avg_confidence": 0}
        
        intra_count = sum(1 for r in results if r["relationship_type"] == "intra")
        inter_count = sum(1 for r in results if r["relationship_type"] == "inter")
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        
        return {
            "total": len(results),
            "intra": intra_count,
            "inter": inter_count,
            "avg_confidence": avg_confidence,
            "high_confidence": sum(1 for r in results if r["confidence"] > 0.8),
            "sample_questions": [r["question"] for r in results[:3]]
        }


# Example usage and testing
if __name__ == "__main__":
    # Test VLM verifier
    verifier = VLMVerifier()
    
    # Sample objects and questions for testing
    objects = [
        ObjectDetection(0, "person", [100.0, 50.0, 200.0, 300.0], 0.95),
        ObjectDetection(1, "car", [250.0, 150.0, 450.0, 350.0], 0.88)
    ]
    
    intra_questions = [
        IntraQuestion([0, 1], "Is the person near the car?")
    ]
    
    inter_questions = [
        InterQuestion(0, 0, "Compare the size of these people")
    ]
    
    # Test validation
    sample_results = [{
        "question_id": "intra_0_1",
        "question": "Is the person near the car?",
        "subject_object_id": 0,
        "object_object_id": 1,
        "answer": "Yes",
        "confidence": 0.8,
        "relationship_type": "intra"
    }]
    
    is_valid = verifier.validate_verification_results(sample_results)
    summary = verifier.get_verification_summary(sample_results)
    
    print(f"✓ Verification validation: {is_valid}")
    print(f"✓ Verification summary: {summary}")
    print("✓ VLM verifier ready!")