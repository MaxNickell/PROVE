"""
Attribute extraction component for PROVE pipeline.
Combines Florence region description → LLM candidate generation → Binary VLM verification.
"""

from typing import List, Dict, Any, Tuple
from PIL import Image

from src.core.model_manager import ModelManager
from src.core.types import AttributeRequirement, ObjectDetection, AttributeValue, AttributeData, ImageData
from src.vision.florence2 import Florence2
from src.vision.qwen_vl import QwenVL


class AttributeExtractorError(RuntimeError):
    """Custom exception for attribute extraction failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return self.message


class AttributeExtractor:
    """
    Extract attribute values using combined Florence → LLM → Binary VLM verification.
    Processes attribute requirements to extract specific attribute class/value pairs.
    """
    
    def __init__(self):
        """Initialize extractor with ModelManager singleton."""
        self.model_manager = ModelManager()
    
    def extract_attributes(
        self,
        image_paths: Dict[str, str],  # {"image_a": "/path/to/image", ...}
        images: Dict[str, ImageData],  # Clean ImageData structure
        requirements: List[AttributeRequirement]
    ) -> List[AttributeData]:
        """
        Extract attribute values for objects based on requirements.

        Args:
            image_paths: Paths to images
            images: ImageData structure containing objects and context per image
            requirements: Attribute extraction requirements

        Returns:
            List[AttributeData]: Extracted attributes with confidence scores

        Raises:
            AttributeExtractorError: If extraction fails
        """
        try:
            if not requirements:
                return []
            
            # Get model clients
            florence_client = self.model_manager.get_florence2()
            llm_client = self.model_manager.get_llm_client()
            qwen_client = self.model_manager.get_qwen_vl()
            
            # Load PIL images
            loaded_images = {}
            for image_id, image_path in image_paths.items():
                loaded_images[image_id] = Image.open(image_path)
            
            # Process each attribute requirement
            all_attribute_data = []

            for requirement in requirements:
                attribute_data = self._extract_attributes_for_object(
                    requirement, loaded_images, images,
                    florence_client, llm_client, qwen_client
                )

                if attribute_data:
                    all_attribute_data.append(attribute_data)

            return all_attribute_data
            
        except Exception as err:
            raise AttributeExtractorError(f"Attribute extraction failed: {err}")
    
    def _extract_attributes_for_object(
        self,
        requirement: AttributeRequirement,
        loaded_images: Dict[str, Image.Image],
        images: Dict[str, ImageData],
        florence_client: Florence2,
        llm_client,
        qwen_client: QwenVL
    ) -> AttributeData:
        """
        Extract attributes for a single object based on requirements.

        Args:
            requirement: Attribute requirement for specific object
            loaded_images: Loaded PIL images
            images: ImageData structure containing objects and context
            florence_client: Florence-2 client
            llm_client: LLM client
            qwen_client: Qwen VLM client

        Returns:
            AttributeData: Extracted attributes for the object
        """
        try:
            # Use simple image_id and object_id to find object from ImageData
            obj_image_id = requirement.image_id
            obj_instance = self._find_object_by_simple_ref_imagedata(requirement.image_id, requirement.object_id, images)
            if not obj_instance:
                print(f"Warning: Could not find object {requirement.image_id} object {requirement.object_id}")
                return None

            image = loaded_images[obj_image_id]
            
            # Get Florence region description for the object
            region_description = self._get_region_description(
                florence_client, image, obj_instance
            )
            
            # Extract each required attribute class
            extracted_attributes = {}
            
            for attribute_class in requirement.attribute_classes:
                attribute_values = self._extract_single_attribute_class(
                    attribute_class, region_description, image, obj_instance,
                    llm_client, qwen_client
                )
                
                if attribute_values:
                    extracted_attributes[attribute_class] = attribute_values
            
            # Create simple AttributeData without object reference
            attribute_data = AttributeData(
                attributes=extracted_attributes
            )
            
            return attribute_data
            
        except Exception as e:
            print(f"Warning: Failed to extract attributes for {requirement.image_id} object {requirement.object_id}: {e}")
            return None
    
    def _find_object_by_simple_ref_imagedata(
        self,
        image_id: str,
        object_id: int,
        images: Dict[str, ImageData]
    ) -> ObjectDetection:
        """
        Find object by simple image_id and object_id from ImageData structure.

        Args:
            image_id: Image identifier (e.g., "image_a")
            object_id: Object index within the image
            images: ImageData structure containing objects per image

        Returns:
            ObjectDetection: Found object or None
        """
        if image_id in images:
            for obj in images[image_id].objects:
                if obj.object_id == object_id:
                    return obj
        return None
    
    def _parse_object_id(
        self, 
        object_id: str, 
        all_objects: Dict[str, List[ObjectDetection]]
    ) -> Tuple[str, ObjectDetection]:
        """
        Parse object ID to find the corresponding image and object instance.
        
        Args:
            object_id: Object ID in format "label_imageid_objectid"
            all_objects: All detected objects
            
        Returns:
            Tuple of (image_id, ObjectDetection) or (None, None) if not found
        """
        try:
            # Parse object ID format: label_imageid_objectid
            parts = object_id.split('_')
            if len(parts) < 3:
                return None, None
            
            # Extract image_id and object_index
            simple_image_id = parts[-2]  # Second to last part (e.g., "a")
            object_index = int(parts[-1])  # Last part
            
            # Convert simple image ID back to full key (e.g., "a" -> "image_a")
            image_id = f"image_{simple_image_id}"
            
            # Find the object
            if image_id in all_objects:
                for obj in all_objects[image_id]:
                    if obj.object_id == object_index:
                        return image_id, obj
            
            return None, None
            
        except Exception as e:
            print(f"Warning: Failed to parse object ID {object_id}: {e}")
            return None, None
    
    def _get_region_description(
        self,
        florence_client: Florence2,
        image: Image.Image,
        obj: ObjectDetection
    ) -> str:
        """
        Get detailed Florence-2 description for object region.
        
        Args:
            florence_client: Florence-2 client
            image: PIL Image
            obj: Object detection instance
            
        Returns:
            str: Detailed region description
        """
        try:
            # Crop object region
            x1, y1, x2, y2 = obj.bbox
            cropped_image = image.crop((x1, y1, x2, y2))
            
            # Get detailed description
            description = florence_client.describe_region(cropped_image)
            
            return description
            
        except Exception as e:
            print(f"Warning: Failed to get region description: {e}")
            return f"A {obj.label} object"
    
    def _extract_single_attribute_class(
        self,
        attribute_class: str,
        region_description: str,
        image: Image.Image,
        obj: ObjectDetection,
        llm_client,
        qwen_client: QwenVL
    ) -> List[AttributeValue]:
        """
        Extract values for a single attribute class using Florence → LLM → VLM pipeline.
        
        Args:
            attribute_class: The attribute class to extract (e.g., "muscle_mass")
            region_description: Florence description of object region
            image: Original image for VLM verification
            obj: Object detection instance
            llm_client: LLM client
            qwen_client: Qwen VLM client
            
        Returns:
            List[AttributeValue]: Extracted attribute values with confidence
        """
        try:
            # Step 1: Generate attribute value candidates using LLM
            candidates = self._generate_attribute_candidates(
                llm_client, attribute_class, obj.label, region_description
            )
            
            if not candidates:
                return []
            
            # Step 2: Verify each candidate using binary VLM
            verified_values = []

            for candidate_value in candidates:
                is_verified, prob_statement_true = self._verify_attribute_value(
                    qwen_client, image, obj, attribute_class, candidate_value
                )

                # Store all results with their probability, no filtering
                # ProbLog needs all probabilistic facts, including low-probability ones
                verified_values.append(AttributeValue(
                    value=candidate_value,
                    confidence=prob_statement_true
                ))

            # Sort by confidence and return all results
            verified_values.sort(key=lambda x: x.confidence, reverse=True)
            return verified_values  # Return all verified values for ProbLog
            
        except Exception as e:
            print(f"Warning: Failed to extract {attribute_class}: {e}")
            return []
    
    def _generate_attribute_candidates(
        self,
        llm_client,
        attribute_class: str,
        object_label: str,
        region_description: str
    ) -> List[str]:
        """
        Generate attribute value candidates using LLM based on Florence description.
        
        Args:
            llm_client: LLM client
            attribute_class: Attribute class to generate values for
            object_label: Object label (e.g., "person")
            region_description: Florence region description
            
        Returns:
            List[str]: Candidate attribute values
        """
        prompt = f"""Based on this visual description, generate possible values for the {attribute_class} attribute of this {object_label}:

Visual Description: "{region_description}"
Object: {object_label}
Attribute Class: {attribute_class}

Generate 3-5 most likely attribute values that can be verified visually. Focus on values that are:
- Visually determinable from the description
- Specific and concrete (not vague)
- Commonly used descriptors for this attribute class

Return JSON with this exact format:
{{
  "candidates": ["value1", "value2", "value3", "value4"]
}}

Examples for different attribute classes:
- muscle_mass: ["high", "medium", "low"]
- color: ["brown", "black", "white", "gray"]
- size: ["large", "medium", "small"]  
- condition: ["new", "worn", "damaged"]
- pattern: ["spotted", "striped", "solid"]
- state: ["active", "resting", "moving"]

Focus on the specific attribute class requested and the visual description provided."""

        messages = [
            {
                "role": "system",
                "content": "You are an expert at generating visual attribute candidates based on descriptions. Generate realistic, verifiable attribute values that match the visual description. Return strict JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            # Use Pydantic validation for robust JSON parsing with CandidateResponse model
            response = llm_client.generate_candidates(
                messages,
                temperature=0.3
            )
            
            # Extract candidates from the validated response
            candidates = response.candidates[:5]  # Get up to 5 values
            
            # Clean candidates
            cleaned_candidates = []
            for candidate in candidates:
                if candidate.strip():
                    cleaned_candidates.append(candidate.strip().lower())
            
            return cleaned_candidates[:5]  # Limit to 5 candidates
            
        except Exception as e:
            print(f"Warning: Failed to generate candidates for {attribute_class}: {e}")
            return []
    
    def _verify_attribute_value(
        self,
        qwen_client: QwenVL,
        image: Image.Image,
        obj: ObjectDetection,
        attribute_class: str,
        candidate_value: str
    ) -> Tuple[bool, float]:
        """
        Verify attribute value using binary VLM with bounding box context.

        Args:
            qwen_client: Qwen VLM client
            image: Original image
            obj: Object detection instance
            attribute_class: Attribute class being verified
            candidate_value: Candidate value to verify

        Returns:
            Tuple of (is_verified, confidence)
        """
        try:
            # Create binary verification question with stronger Yes/No compliance
            question = f"""Look at this object: <box>({int(obj.bbox[0])},{int(obj.bbox[1])}),({int(obj.bbox[2])},{int(obj.bbox[3])})</box>{obj.label}

Question: Does this {obj.label} have {candidate_value} {attribute_class}?

IMPORTANT: You must respond with exactly "Yes" or "No" only. Do not include any explanation or additional text.

Answer:"""
            
            # Get VLM response with logits
            response, logits = qwen_client.run_inference_with_logits(image, question)

            # Use proper softmax probability calculation for P(statement is true)
            prob_statement_true = qwen_client.extract_yes_no_probability_with_proper_softmax(logits, response)

            # Validate response format
            is_valid_response = qwen_client.validate_yes_no_response(response)
            if not is_valid_response:
                print(f"Warning: Invalid Yes/No response: '{response}' for attribute verification")

            # Determine if positive (statement is true)
            is_positive = response.lower().strip().startswith('yes')

            return is_positive, prob_statement_true
            
        except Exception as e:
            print(f"Warning: Failed to verify {attribute_class}={candidate_value}: {e}")
            return False, 0.0
    
    def get_extraction_summary(
        self, 
        attribute_data_list: List[AttributeData]
    ) -> Dict[str, Any]:
        """
        Get summary statistics for attribute extraction results.
        
        Args:
            attribute_data_list: List of AttributeData instances
            
        Returns:
            Dict with summary information
        """
        if not attribute_data_list:
            return {
                "total_objects": 0,
                "total_attributes": 0,
                "avg_confidence": 0.0,
                "attribute_classes": {}
            }
        
        total_objects = len(attribute_data_list)
        total_attributes = 0
        all_confidences = []
        attribute_class_counts = {}
        
        for attr_data in attribute_data_list:
            for attr_class, values in attr_data.attributes.items():
                total_attributes += len(values)
                attribute_class_counts[attr_class] = attribute_class_counts.get(attr_class, 0) + len(values)
                
                for value in values:
                    all_confidences.append(value.confidence)
        
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        
        return {
            "total_objects": total_objects,
            "total_attributes": total_attributes,
            "avg_confidence": avg_confidence,
            "attribute_classes": attribute_class_counts,
            "confidence_distribution": {
                "high (>0.8)": len([c for c in all_confidences if c > 0.8]),
                "medium (0.5-0.8)": len([c for c in all_confidences if 0.5 <= c <= 0.8]),
                "low (<0.5)": len([c for c in all_confidences if c < 0.5])
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Test attribute extractor
    extractor = AttributeExtractor()
    
    # Sample data
    from src.core.types import AttributeRequirement, ObjectDetection, ImageData

    requirements = [
        AttributeRequirement(
            image_id="image_a",
            object_id=0,
            attribute_classes=["muscle_mass", "body_size"],
            required_for_subqueries=["Is person_a_0 more muscular than person_b_0?"]
        )
    ]

    image_paths = {
        "image_a": "./test_images/dev-473-3-img0.png"
    }

    images = {
        "image_a": ImageData(
            objects=[
                ObjectDetection(0, "person", [10.0, 20.0, 100.0, 200.0], 0.95)
            ],
            attributes={},
            relationships=[],
            scene_context={}
        )
    }
    
    try:
        # Note: This test requires actual model clients to be available
        print("✓ AttributeExtractor component created")
        print("✓ Ready for integration testing with model clients")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")