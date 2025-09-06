"""
Attribute extractor using Florence-2 and LLM with ModelManager singleton.
Refactored for memory efficiency and exact 10-category compliance.
"""

from typing import List, Dict, Any, Optional
from PIL import Image
import json
import os

from src.core.model_manager import ModelManager
from src.core.types import ObjectDetection, AttributeData, AttributeValue, ATTRIBUTE_CATEGORIES


class AttributeExtractorError(RuntimeError):
    """Custom exception for attribute extraction failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return self.message


class AttributeExtractor:
    """
    Attribute extractor using Florence-2 dense captions and LLM parsing.
    Uses ModelManager singleton and produces exact 10-category compliance.
    """
    
    def __init__(self):
        """Initialize extractor with ModelManager singleton."""
        # Use ModelManager singleton instead of creating models directly
        self.model_manager = ModelManager()
        
        # Ensure we use exact attribute categories from types.py
        self.attribute_categories = ATTRIBUTE_CATEGORIES
    
    def extract_attributes_with_candidates(self, image_path: str, objects: List[ObjectDetection], 
                                          ultimate_question: str = "", save_crops: bool = False, 
                                          crop_dir: str = "crops") -> List[AttributeData]:
        """
        Extract attributes using LLM-generated contextual candidates with VLM verification.
        
        Args:
            image_path: Path to the image file
            objects: List of ObjectDetection instances
            ultimate_question: Main comparative question for context
            save_crops: Whether to save cropped objects
            crop_dir: Directory to save crops
            
        Returns:
            List[AttributeData]: Extracted attributes with individual confidence per value
            
        Raises:
            AttributeExtractorError: If extraction fails
        """
        try:
            if not objects:
                return []
            
            # Load image
            image = Image.open(image_path)
            all_attributes = []
            
            for obj in objects:
                # Step 1: Get Florence description for the object
                florence_desc = self._get_florence_description_for_object(image, obj, image_path)
                
                # Step 2: Generate contextual attribute candidates using LLM
                candidates = self._generate_contextual_attribute_candidates(
                    ultimate_question, obj.label, florence_desc
                )
                
                # Step 3: Use VLM to verify and extract values for candidates
                attributes = self._verify_and_extract_attribute_values(
                    image, obj, candidates, save_crops, crop_dir
                )
                
                all_attributes.append(attributes)
            
            return all_attributes
            
        except Exception as err:
            raise AttributeExtractorError(f"Contextual attribute extraction failed: {err}")
    
    def extract_attributes(self, image_path: str, objects: List[ObjectDetection], 
                         save_crops: bool = False, crop_dir: str = "crops") -> List[AttributeData]:
        """
        Extract normalized attributes for each object in the image.
        
        Args:
            image_path: Path to the image file
            objects: List of ObjectDetection instances
            save_crops: Whether to save cropped objects
            crop_dir: Directory to save crops
            
        Returns:
            List[AttributeData]: Attributes with exact schema compliance
            
        Raises:
            AttributeExtractorError: If extraction fails
        """
        try:
            # Validate input
            if not os.path.exists(image_path):
                raise AttributeExtractorError(f"Image file not found: {image_path}")
            
            if not objects:
                return []
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Get models from singleton ModelManager
            florence2 = self.model_manager.get_florence2()
            llm_client = self.model_manager.get_llm_client()
            
            # Create crop directory if needed
            if save_crops and not os.path.exists(crop_dir):
                os.makedirs(crop_dir)
            
            attributes_list = []
            
            # Process each object
            for obj in objects:
                try:
                    # Crop the object using Florence-2's crop utility
                    cropped = florence2.crop_object(image, obj.bbox)
                    
                    # Save crop if requested
                    if save_crops:
                        crop_filename = f"object_{obj.object_id}_{obj.label.replace(' ', '_')}.jpg"
                        crop_path = os.path.join(crop_dir, crop_filename)
                        cropped.save(crop_path)
                    
                    # Get detailed description using Florence-2
                    description = florence2.describe_region(cropped)
                    
                    # Extract normalized attributes using LLM with strict validation
                    normalized_attrs = self._normalize_attributes(obj.label, description)
                    
                    # Create AttributeData instance with individual confidences
                    attr_data = AttributeData(
                        object_id=obj.object_id,
                        attributes=normalized_attrs  # Already contains AttributeValue objects
                    )
                    
                    attributes_list.append(attr_data)
                    
                except Exception as e:
                    print(f"Warning: Failed to extract attributes for object {obj.object_id}: {e}")
                    # Create fallback attributes
                    fallback_attr_values = {category: [] for category in ATTRIBUTE_CATEGORIES}
                    if obj.label:
                        label_attr = AttributeValue(value=obj.label, confidence=1.0)
                        fallback_attr_values["function"] = [label_attr]
                    
                    fallback_attrs = AttributeData(
                        object_id=obj.object_id,
                        attributes=fallback_attr_values
                    )
                    attributes_list.append(fallback_attrs)
            
            return attributes_list
            
        except Exception as err:
            raise AttributeExtractorError(f"Attribute extraction failed: {err}")
    
    def _normalize_attributes(self, object_label: str, description: str) -> Dict[str, List[str]]:
        """
        Use LLM to extract normalized attributes from description with exact 10-category compliance.
        
        Args:
            object_label: The detected object label
            description: Detailed description from Florence-2
            
        Returns:
            Dict[str, List[AttributeValue]]: Categorized attributes with individual confidences
        """
        # Get LLM client from ModelManager
        llm_client = self.model_manager.get_llm_client()
        
        # Enhanced prompt for strict JSON compliance and exact categories
        prompt = f"""Extract attributes into the 10 categories; return strict JSON with arrays per category; no extra text; unknown → empty list; keep values short.

Categories: color, material, texture, shape, size, state, pattern, style, condition, function

Object: {object_label}
Description: {description}

Return only JSON with these exact keys:
{{
  "color": [],
  "material": [],
  "texture": [],
  "shape": [],
  "size": [],
  "state": [],
  "pattern": [],
  "style": [],
  "condition": [],
  "function": []
}}

Fill in arrays with relevant short attributes (1-2 words). Leave empty arrays for unknown categories."""

        messages = [
            {
                "role": "system", 
                "content": "You are an expert at analyzing objects and extracting structured attributes from descriptions. Return strict JSON only, no markdown or extra text."
            },
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Use strict JSON response format
            response = llm_client.chat(
                messages, 
                temperature=0.1, 
                response_format={"type": "json_object"}
            )
            
            # Parse and validate JSON response
            attributes = json.loads(response)
            
            # Ensure exact 10-category compliance
            validated_attrs = self._validate_attribute_categories(attributes)
            
            return validated_attrs
            
        except Exception as e:
            print(f"Error normalizing attributes for {object_label}: {e}")
            # Return minimal fallback attributes
            return self._get_fallback_attributes(object_label, description)
    
    def _validate_attribute_categories(self, attributes: Dict[str, Any]) -> Dict[str, List[AttributeValue]]:
        """
        Validate and ensure exact 10-category compliance with individual confidences.
        
        Args:
            attributes: Raw attributes from LLM
            
        Returns:
            Dict[str, List[AttributeValue]]: Validated attributes with individual confidences
        """
        validated = {}
        
        # Ensure all 10 categories are present
        for category in self.attribute_categories:
            validated[category] = []
            if category in attributes:
                value = attributes[category]
                # Ensure value is a list
                if isinstance(value, list):
                    for item in value:
                        if item:  # Non-empty value
                            attr_value = AttributeValue(value=str(item), confidence=1.0)
                            validated[category].append(attr_value)
                elif value:  # Non-empty single value
                    attr_value = AttributeValue(value=str(value), confidence=1.0)
                    validated[category].append(attr_value)
        
        # Remove any extra categories not in our standard 10
        return {k: v for k, v in validated.items() if k in self.attribute_categories}
    
    def _get_fallback_attributes(self, object_label: str, description: str) -> Dict[str, List[AttributeValue]]:
        """
        Generate fallback attributes when LLM parsing fails.
        
        Args:
            object_label: Object label
            description: Raw description
            
        Returns:
            Dict[str, List[AttributeValue]]: Minimal fallback attributes
        """
        fallback = {category: [] for category in self.attribute_categories}
        
        # Add basic information
        if object_label:
            attr_value = AttributeValue(value=object_label.lower(), confidence=1.0)
            fallback["function"] = [attr_value]
        
        # Simple keyword matching for basic attributes
        description_lower = description.lower()
        
        # Basic color detection
        colors = ["red", "blue", "green", "yellow", "black", "white", "brown", "gray", "pink", "orange"]
        for color in colors:
            if color in description_lower:
                attr_value = AttributeValue(value=color, confidence=1.0)
                fallback["color"].append(attr_value)
                break
        
        # Basic size detection  
        if any(word in description_lower for word in ["large", "big", "huge"]):
            attr_value = AttributeValue(value="large", confidence=1.0)
            fallback["size"].append(attr_value)
        elif any(word in description_lower for word in ["small", "tiny", "little"]):
            attr_value = AttributeValue(value="small", confidence=1.0)
            fallback["size"].append(attr_value)
        
        return fallback
    
    def validate_attributes(self, attributes_list: List[AttributeData]) -> bool:
        """
        Validate that all attributes conform to expected schema.
        
        Args:
            attributes_list: List of AttributeData instances
            
        Returns:
            bool: True if all attributes are valid
        """
        try:
            for attr_data in attributes_list:
                # Check required fields
                assert hasattr(attr_data, 'object_id')
                assert hasattr(attr_data, 'attributes')
                assert hasattr(attr_data, 'probability')
                
                # Validate types
                assert isinstance(attr_data.object_id, int)
                assert isinstance(attr_data.attributes, dict)
                assert isinstance(attr_data.probability, float)
                
                # Validate probability range
                assert 0.0 <= attr_data.probability <= 1.0
                
                # Validate all 10 categories are present
                for category in self.attribute_categories:
                    assert category in attr_data.attributes
                    assert isinstance(attr_data.attributes[category], list)
            
            return True
            
        except AssertionError:
            return False
    
    def get_attribute_summary(self, attributes_list: List[AttributeData]) -> Dict[str, Any]:
        """
        Get summary statistics for extracted attributes.
        
        Args:
            attributes_list: List of AttributeData instances
            
        Returns:
            Dict[str, Any]: Summary information
        """
        if not attributes_list:
            return {"object_count": 0, "category_stats": {}}
        
        category_stats = {}
        
        for category in self.attribute_categories:
            total_attrs = 0
            unique_attrs = set()
            
            for attr_data in attributes_list:
                attrs = attr_data.attributes.get(category, [])
                total_attrs += len(attrs)
                unique_attrs.update(attrs)
            
            category_stats[category] = {
                "total_count": total_attrs,
                "unique_count": len(unique_attrs),
                "unique_values": list(unique_attrs)
            }
        
        return {
            "object_count": len(attributes_list),
            "category_stats": category_stats,
            "avg_probability": round(sum(attr.probability for attr in attributes_list) / len(attributes_list), 3)
        }
    
    def _get_florence_description_for_object(self, image: Image.Image, obj: ObjectDetection, image_path: str) -> str:
        """
        Get Florence-2 description focused on a specific object.
        
        Args:
            image: PIL Image object
            obj: ObjectDetection instance
            image_path: Path to image for caching
            
        Returns:
            str: Florence description for the object region
        """
        try:
            # Get Florence-2 client from ModelManager
            florence2 = self.model_manager.get_florence2()
            
            # Crop object region for focused description
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
            
            # Get dense caption for the cropped object
            description = florence2.generate_caption(object_crop, task="<DETAILED_CAPTION>")
            
            return description
            
        except Exception as e:
            print(f"Warning: Failed to get Florence description for {obj.label}: {e}")
            return f"A {obj.label} object in the image"
    
    def _generate_contextual_attribute_candidates(self, ultimate_question: str, object_label: str, 
                                                 florence_desc: str) -> Dict[str, List[str]]:
        """
        Generate contextual attribute candidates using LLM reasoning.
        
        Args:
            ultimate_question: Main comparative question for context
            object_label: Label of the object
            florence_desc: Florence-2 description of the object
            
        Returns:
            Dict[str, List[str]]: Mapping from attribute category to candidate values
        """
        try:
            llm_client = self.model_manager.get_llm_client()
            
            prompt = f"""Given the ultimate question "{ultimate_question}", suggest relevant attribute values for this object:

Object: {object_label}
Description: {florence_desc}

Based on the question context and object description, suggest specific attribute values that would be relevant for comparison. Focus on visually observable and discriminating features.

For each of the 10 attribute categories, suggest 2-4 candidate values that could apply to this object. If a category doesn't apply, leave it empty.

Return JSON with this exact format:
{{
  "color": ["candidate1", "candidate2"],
  "material": ["candidate1", "candidate2"],
  "texture": ["candidate1", "candidate2"],
  "shape": ["candidate1", "candidate2"],
  "size": ["candidate1", "candidate2"],
  "state": ["candidate1", "candidate2"],
  "pattern": ["candidate1", "candidate2"],
  "style": ["candidate1", "candidate2"],
  "condition": ["candidate1", "candidate2"],
  "function": ["candidate1", "candidate2"]
}}

Examples:
- color: ["brown", "black", "multicolored"]
- size: ["large", "medium", "small"]
- state: ["alive", "dead", "moving", "stationary"]
- condition: ["healthy", "injured", "old", "new"]

Keep values short (1-2 words) and visually verifiable."""

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at suggesting relevant visual attribute candidates for objects based on context. Generate contextually appropriate attribute lists that help with comparative analysis. Return strict JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = llm_client.chat(
                messages,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            candidates = json.loads(response)
            
            # Validate and clean candidates
            cleaned_candidates = {}
            for category in self.attribute_categories:
                values = candidates.get(category, [])
                if isinstance(values, list):
                    cleaned_values = [v.strip().lower() for v in values if isinstance(v, str) and v.strip()]
                    cleaned_candidates[category] = cleaned_values[:4]  # Limit to 4 candidates per category
                else:
                    cleaned_candidates[category] = []
            
            return cleaned_candidates
            
        except Exception as e:
            print(f"Warning: Failed to generate attribute candidates for {object_label}: {e}")
            # Return generic fallback candidates
            return {category: [] for category in self.attribute_categories}
    
    def _verify_and_extract_attribute_values(self, image: Image.Image, obj: ObjectDetection, 
                                           candidates: Dict[str, List[str]], save_crops: bool, 
                                           crop_dir: str) -> AttributeData:
        """
        Use VLM to verify and extract attribute values from LLM candidates.
        
        Args:
            image: PIL Image object
            obj: ObjectDetection instance
            candidates: LLM-generated attribute candidates
            save_crops: Whether to save crops
            crop_dir: Directory for crops
            
        Returns:
            AttributeData: Extracted attributes with individual confidences
        """
        try:
            # Get VLM client for verification
            vlm_client = self.model_manager.get_vlm()
            
            # Crop object for VLM analysis
            x1, y1, x2, y2 = obj.bbox
            width, height = image.size
            
            # Add padding
            padding = 20
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(width, x2 + padding)
            crop_y2 = min(height, y2 + padding)
            
            object_crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            
            # Save crop if requested
            if save_crops:
                os.makedirs(crop_dir, exist_ok=True)
                crop_filename = f"object_{obj.object_id}_{obj.label.replace(' ', '_')}.jpg"
                crop_path = os.path.join(crop_dir, crop_filename)
                object_crop.save(crop_path)
            
            # Extract values for each category with candidates
            final_attributes = {}
            
            for category in self.attribute_categories:
                candidate_values = candidates.get(category, [])
                
                if not candidate_values:
                    final_attributes[category] = []
                    continue
                
                # Use VLM to select and extract the most appropriate value
                verified_values = self._verify_attribute_candidates(
                    vlm_client, object_crop, obj.label, category, candidate_values
                )
                
                final_attributes[category] = verified_values
            
            return AttributeData(
                object_id=obj.object_id,
                attributes=final_attributes
            )
            
        except Exception as e:
            print(f"Warning: Failed to verify attributes for {obj.label}: {e}")
            # Return empty attributes for all categories
            empty_attrs = {category: [] for category in self.attribute_categories}
            return AttributeData(object_id=obj.object_id, attributes=empty_attrs)
    
    def _verify_attribute_candidates(self, vlm_client, object_crop: Image.Image, 
                                   object_label: str, category: str, 
                                   candidates: List[str]) -> List[AttributeValue]:
        """
        Use VLM to verify and select the best attribute values from LLM candidates.
        
        Args:
            vlm_client: VLM client instance
            object_crop: Cropped object image
            object_label: Label of the object
            category: Attribute category (color, size, etc.)
            candidates: List of candidate values
            
        Returns:
            List[AttributeValue]: Verified attribute values with confidences
        """
        try:
            if not candidates:
                return []
            
            # Create verification prompt
            candidates_text = ", ".join(candidates)
            prompt = f"""Look at this {object_label} and determine its {category}.

Based on what you observe, which of these {category} values best describe this {object_label}?
Candidates: {candidates_text}

Select the 1-2 most accurate values from the candidates. If none fit well, you can describe the {category} in your own words (1-2 words maximum).

Answer with just the {category} value(s), separated by commas if multiple."""

            response = vlm_client.run_inference(object_crop, prompt)
            
            # Parse response to extract values
            response_clean = response.strip().lower()
            
            # Extract values from response
            extracted_values = []
            
            # Check if response contains any candidates
            for candidate in candidates:
                if candidate.lower() in response_clean:
                    extracted_values.append(AttributeValue(value=candidate, confidence=1.0))
            
            # If no candidates found, use the response itself (cleaned)
            if not extracted_values:
                # Extract first 1-2 words from response
                words = response_clean.replace(',', ' ').split()[:2]
                if words:
                    value = ' '.join(words)
                    extracted_values.append(AttributeValue(value=value, confidence=1.0))
            
            return extracted_values[:2]  # Limit to 2 values per category
            
        except Exception as e:
            print(f"Warning: Failed to verify {category} candidates for {object_label}: {e}")
            # Return first candidate as fallback
            if candidates:
                return [AttributeValue(value=candidates[0], confidence=1.0)]
            return []


