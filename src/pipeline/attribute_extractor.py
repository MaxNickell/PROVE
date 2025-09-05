from __future__ import annotations
from typing import List, Dict, Any
from PIL import Image
import json
from src.vision.florence2 import Florence2
from src.language.llm_client import LLMClient


class AttributeExtractor:
    def __init__(self):
        self.florence2 = Florence2()
        self.llm_client = LLMClient()
        
        # Define common attribute categories we want to extract
        self.attribute_categories = [
            "color", "material", "texture", "shape", "size", 
            "state", "pattern", "style", "condition", "function"
        ]
    
    def extract_attributes(self, image_path: str, objects: List[Dict[str, Any]], 
                         save_crops: bool = False) -> Dict[int, Dict[str, List[str]]]:
        """Extract normalized attributes for each object in the image.
        
        Args:
            image_path: Path to the image file
            objects: List of detected objects with bboxes
            save_crops: Whether to save cropped objects
            
        Returns:
            Dictionary mapping object ID to categorized attributes
        """
        try:
            image = Image.open(image_path).convert("RGB")
            attributes = {}
            
            # Process each object
            for obj in objects:
                obj_id = obj['id']
                bbox = obj['bbox']
                label = obj['label']
                
                # Crop the object
                cropped = self.florence2.crop_object(image, bbox)
                
                # Get detailed description using Florence-2
                description = self.florence2.describe_region(cropped)
                
                # Extract normalized attributes using LLM
                normalized_attrs = self._normalize_attributes(label, description)
                attributes[obj_id] = normalized_attrs
                
                # Add raw description for reference
                attributes[obj_id]['raw_description'] = [description]
                
            return attributes
            
        except Exception as err:
            raise AttributeExtractorError(f"Attribute extraction failed: {err}")
    
    def _normalize_attributes(self, object_label: str, description: str) -> Dict[str, List[str]]:
        """Use LLM to extract normalized attributes from a description.
        
        Args:
            object_label: The detected object label
            description: Detailed description from Florence-2
            
        Returns:
            Dictionary of categorized attributes
        """
        prompt = f"""Given an object labeled as "{object_label}" with the following description:
"{description}"

Extract and categorize attributes into the following categories:
- color: specific colors mentioned (e.g., red, blue, multicolored)
- material: what it's made of (e.g., metal, wood, plastic, fabric)
- texture: surface qualities (e.g., smooth, rough, glossy, matte)
- shape: geometric or form descriptions (e.g., round, square, curved)
- size: relative size descriptors (e.g., large, small, medium, tall, wide)
- state: current condition or state (e.g., open, closed, moving, stationary)
- pattern: visual patterns (e.g., striped, dotted, plaid, solid)
- style: design or aesthetic style (e.g., modern, vintage, casual, formal)
- condition: physical condition (e.g., new, worn, damaged, clean)
- function: what it's used for or doing (e.g., carrying, supporting, decorative)

Return a JSON object where each key is a category and the value is a list of relevant attributes.
Only include categories that have relevant attributes. Keep attributes concise (1-2 words each).

Example output:
{{
    "color": ["red", "white"],
    "material": ["metal"],
    "shape": ["rectangular"],
    "size": ["large"],
    "state": ["parked"],
    "condition": ["clean"]
}}"""

        messages = [
            {"role": "system", "content": "You are an expert at analyzing objects and extracting structured attributes from descriptions."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.llm_client.chat(messages, temperature=0.1, response_format={"type": "json_object"})
            attributes = json.loads(response)
            
            # Ensure all values are lists
            for key, value in attributes.items():
                if not isinstance(value, list):
                    attributes[key] = [value] if value else []
                    
            return attributes
            
        except Exception as e:
            print(f"Error normalizing attributes: {e}")
            # Return a basic set of attributes from the description
            return {
                "description": [description],
                "label": [object_label]
            }


class AttributeExtractorError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return self.message