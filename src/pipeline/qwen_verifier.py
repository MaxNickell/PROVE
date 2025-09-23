"""
Qwen binary verification component for PROVE pipeline.
Simplified binary verification using Qwen 2.5-VL-7B with bounding box support.
"""

from typing import Tuple
from PIL import Image

from src.core.model_manager import ModelManager
from src.core.types import ObjectDetection
from src.vision.qwen_vl import QwenVL


class QwenVerificationError(RuntimeError):
    """Custom exception for Qwen verification failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return f"Qwen Verification Error: {self.message}"


class QwenVerifier:
    """
    Simplified Qwen 2.5-VL-7B binary verification for attributes and relationships.
    Focuses on Yes/No questions with direct logit probability extraction.
    """
    
    def __init__(self):
        """Initialize verifier with ModelManager singleton."""
        self.model_manager = ModelManager()
    
    def verify_attribute(
        self,
        image: Image.Image,
        obj: ObjectDetection,
        attribute_class: str,
        attribute_value: str
    ) -> Tuple[bool, float]:
        """
        Verify if an object has a specific attribute value.
        
        Args:
            image: PIL Image
            obj: Object detection instance
            attribute_class: Attribute class (e.g., "muscle_mass")
            attribute_value: Candidate value (e.g., "high")
            
        Returns:
            Tuple of (is_positive, confidence)
            
        Raises:
            QwenVerificationError: If verification fails
        """
        try:
            qwen_client = self.model_manager.get_qwen_vl()
            
            # Create binary verification question with bounding box
            question = f"Look at this object: <box>({int(obj.bbox[0])},{int(obj.bbox[1])}),({int(obj.bbox[2])},{int(obj.bbox[3])})</box>{obj.label}\n\nDoes this {obj.label} have {attribute_value} {attribute_class}?\n\nAnswer: Yes or No"
            
            # Get VLM response with logits
            response, logits = qwen_client.run_inference_with_logits(image, question)
            
            # Extract confidence from logits
            confidence = qwen_client.extract_response_probability(logits)
            
            # Determine if positive
            is_positive = response.lower().strip().startswith('yes')
            
            # Adjust confidence based on response
            final_confidence = confidence if is_positive else (1.0 - confidence)
            
            return is_positive, final_confidence
            
        except Exception as err:
            raise QwenVerificationError(f"Attribute verification failed: {err}")
    
    def verify_relationship(
        self,
        image: Image.Image,
        subject_obj: ObjectDetection,
        object_obj: ObjectDetection,
        relation: str
    ) -> Tuple[bool, float]:
        """
        Verify if a relationship exists between two objects.
        
        Args:
            image: PIL Image
            subject_obj: Subject object
            object_obj: Object being acted upon
            relation: Relationship to verify (e.g., "lifting")
            
        Returns:
            Tuple of (is_positive, confidence)
            
        Raises:
            QwenVerificationError: If verification fails
        """
        try:
            qwen_client = self.model_manager.get_qwen_vl()
            
            # Create binary verification question with both bounding boxes
            question = f"""Look at these objects:
Subject: <box>({int(subject_obj.bbox[0])},{int(subject_obj.bbox[1])}),({int(subject_obj.bbox[2])},{int(subject_obj.bbox[3])})</box>{subject_obj.label}
Object: <box>({int(object_obj.bbox[0])},{int(object_obj.bbox[1])}),({int(object_obj.bbox[2])},{int(object_obj.bbox[3])})</box>{object_obj.label}

Is the {subject_obj.label} {relation} the {object_obj.label}?

Answer: Yes or No"""
            
            # Get VLM response with logits
            response, logits = qwen_client.run_inference_with_logits(image, question)
            
            # Extract confidence from logits
            confidence = qwen_client.extract_response_probability(logits)
            
            # Determine if positive
            is_positive = response.lower().strip().startswith('yes')
            
            # Adjust confidence based on response
            final_confidence = confidence if is_positive else (1.0 - confidence)
            
            return is_positive, final_confidence
            
        except Exception as err:
            raise QwenVerificationError(f"Relationship verification failed: {err}")
    
    def verify_binary_question(
        self,
        image: Image.Image,
        question: str
    ) -> Tuple[bool, float]:
        """
        Verify any binary question with the image.
        
        Args:
            image: PIL Image
            question: Binary question (should end with "Answer: Yes or No")
            
        Returns:
            Tuple of (is_positive, confidence)
            
        Raises:
            QwenVerificationError: If verification fails
        """
        try:
            qwen_client = self.model_manager.get_qwen_vl()
            
            # Ensure question has proper format
            if not question.endswith("Answer: Yes or No"):
                question = question + "\n\nAnswer: Yes or No"
            
            # Get VLM response with logits
            response, logits = qwen_client.run_inference_with_logits(image, question)
            
            # Extract confidence from logits
            confidence = qwen_client.extract_response_probability(logits)
            
            # Determine if positive
            is_positive = response.lower().strip().startswith('yes')
            
            # Adjust confidence based on response
            final_confidence = confidence if is_positive else (1.0 - confidence)
            
            return is_positive, final_confidence
            
        except Exception as err:
            raise QwenVerificationError(f"Binary question verification failed: {err}")


# Example usage and testing
if __name__ == "__main__":
    # Test Qwen verifier
    verifier = QwenVerifier()
    
    # Sample data
    from src.core.types import ObjectDetection
    
    sample_obj = ObjectDetection(0, "person", [10.0, 20.0, 100.0, 200.0], 0.95)
    
    try:
        # Note: This test requires actual model clients and image to be available
        print("✓ QwenVerifier component created")
        print("✓ Ready for binary verification with simplified interface")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")