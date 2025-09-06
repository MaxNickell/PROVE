"""
Inter-comparison question generator for PROVE pipeline.
Generates cross-image comparison questions for object pairs between images.
"""

from typing import List, Dict, Any
import json

from src.core.model_manager import ModelManager
from src.core.types import ObjectDetection, InterQuestion


class InterQuestionGeneratorError(RuntimeError):
    """Custom exception for inter-question generation failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return self.message


class InterQuestionGenerator:
    """
    Generate inter-comparison questions for object pairs between images.
    Uses LLM to create cross-image comparison questions that help answer comparative questions.
    """
    
    def __init__(self):
        """Initialize generator with ModelManager singleton."""
        self.model_manager = ModelManager()
    
    def generate_attribute_candidates(self, ultimate_question: str, 
                                     objects_a: List[ObjectDetection], 
                                     objects_b: List[ObjectDetection]) -> Dict[tuple, List[str]]:
        """
        Generate contextual attribute candidates for cross-image comparisons using LLM reasoning.
        
        Args:
            ultimate_question: Main comparative question to answer
            objects_a: List of ObjectDetection instances for image A
            objects_b: List of ObjectDetection instances for image B
            
        Returns:
            Dict[tuple, List[str]]: Mapping from (obj_a_id, obj_b_id) to list of relevant attributes
            
        Raises:
            InterQuestionGeneratorError: If generation fails
        """
        try:
            if not objects_a or not objects_b:
                return {}
            
            # Get LLM client from ModelManager
            llm_client = self.model_manager.get_llm_client()
            
            attribute_candidates = {}
            
            # Generate candidates for each cross-image object pair
            for obj_a in objects_a:
                for obj_b in objects_b:
                    # Generate attribute candidates for this specific pair
                    candidates = self._generate_attributes_for_pair(
                        llm_client, ultimate_question, obj_a, obj_b
                    )
                    attribute_candidates[(obj_a.object_id, obj_b.object_id)] = candidates
            
            return attribute_candidates
            
        except Exception as err:
            raise InterQuestionGeneratorError(f"Attribute candidate generation failed: {err}")
    
    def _generate_attributes_for_pair(self, llm_client, ultimate_question: str, 
                                     obj_a: ObjectDetection, obj_b: ObjectDetection) -> List[str]:
        """
        Generate relevant attribute candidates for a specific cross-image object pair.
        
        Args:
            llm_client: LLM client instance
            ultimate_question: Main comparative question
            obj_a: Object from image A
            obj_b: Object from image B
            
        Returns:
            List[str]: List of relevant attributes to compare between the objects
        """
        # Determine if objects are same class for enhanced discrimination
        same_class = obj_a.label.lower() == obj_b.label.lower()
        
        prompt = f"""Given the ultimate question "{ultimate_question}", determine which attributes are most relevant for comparing these two objects:

Object A: {obj_a.label} (from first image)
Object B: {obj_b.label} (from second image)

Objects are {"the same type" if same_class else "different types"}.

Consider what attributes would help answer the ultimate question by comparing these objects. Focus on:
- Visual attributes: size, color, pattern, shape, texture
- State attributes: condition, state, function, material
- Discriminating features: {"subtle differences that distinguish between similar objects" if same_class else "key contrasts between different object types"}

Generate 3-5 most relevant attributes for this comparison.

Return JSON with this exact format:
{{
  "attributes": ["attribute1", "attribute2", "attribute3", "attribute4"]
}}

Examples of good attributes: "size", "color", "state", "condition", "pattern", "material", "texture"
Keep attributes general (single words) that can be extracted visually."""

        messages = [
            {
                "role": "system",
                "content": "You are an expert at determining which visual attributes are most relevant for comparing objects in images. Generate contextually appropriate attribute lists based on object types and comparative questions. Return strict JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            response = llm_client.chat(
                messages,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            response_data = json.loads(response)
            attributes = response_data.get("attributes", [])
            
            # Validate and clean attributes
            cleaned_attributes = []
            for attr in attributes:
                if isinstance(attr, str) and attr.strip():
                    cleaned_attributes.append(attr.strip().lower())
            
            return cleaned_attributes[:5]  # Limit to 5 attributes max
            
        except Exception as e:
            print(f"Warning: Failed to generate attribute candidates for {obj_a.label}-{obj_b.label}: {e}")
            # Fallback to generic attributes based on object similarity
            if same_class:
                return ["size", "condition", "color", "state"]
            else:
                return ["size", "color", "state", "material"]
    
    def generate_questions(self, ultimate_question: str, objects_a: List[ObjectDetection], 
                          objects_b: List[ObjectDetection]) -> List[InterQuestion]:
        """
        Generate inter-comparison questions for object pairs between images.
        
        Args:
            ultimate_question: The main comparative question to answer
            objects_a: List of ObjectDetection instances for image A
            objects_b: List of ObjectDetection instances for image B
            
        Returns:
            List[InterQuestion]: Generated questions with exact schema compliance
            
        Raises:
            InterQuestionGeneratorError: If generation fails
        """
        try:
            if not objects_a or not objects_b:
                return []
            
            # Get LLM client from ModelManager
            llm_client = self.model_manager.get_llm_client()
            
            # Format objects for LLM prompt
            objects_a_desc = []
            for obj in objects_a:
                obj_desc = f"ID {obj.object_id}: {obj.label} (confidence: {obj.confidence:.2f})"
                objects_a_desc.append(obj_desc)
            
            objects_b_desc = []
            for obj in objects_b:
                obj_desc = f"ID {obj.object_id}: {obj.label} (confidence: {obj.confidence:.2f})"
                objects_b_desc.append(obj_desc)
            
            # Generate questions using LLM
            prompt = self._create_inter_question_prompt(ultimate_question, objects_a_desc, objects_b_desc)
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at generating cross-image comparison questions between objects. Return strict JSON only, no markdown or extra text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Call LLM with JSON response format
            response = llm_client.chat(
                messages,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Parse and validate response
            response_data = json.loads(response)
            questions = self._parse_and_validate_questions(response_data, objects_a, objects_b)
            
            return questions
            
        except Exception as err:
            raise InterQuestionGeneratorError(f"Inter-question generation failed: {err}")
    
    def _create_inter_question_prompt(self, ultimate_question: str, objects_a_desc: List[str], 
                                    objects_b_desc: List[str]) -> str:
        """
        Create prompt for inter-comparison question generation.
        
        Args:
            ultimate_question: Main comparative question
            objects_a_desc: Formatted object descriptions for image A
            objects_b_desc: Formatted object descriptions for image B
            
        Returns:
            str: Formatted prompt for LLM
        """
        objects_a_text = "\n".join(objects_a_desc)
        objects_b_text = "\n".join(objects_b_desc)
        
        prompt = f"""Given the ultimate question "{ultimate_question}", generate attribute extraction questions for objects across Image A and Image B.

Image A Objects:
{objects_a_text}

Image B Objects:
{objects_b_text}

For each meaningful object pair across images, generate attribute-specific questions that extract comparable values. Focus on attributes that help answer the ultimate question.

Return JSON with this exact format:
{{
  "inter_questions": [
    {{
      "image_a_object_id": int,
      "image_b_object_id": int, 
      "question": "What is the [attribute] of this [object_type]?"
    }}
  ]
}}

Rules:
- Only use object IDs from the provided lists
- Generate attribute extraction questions (NOT comparison questions)
- Focus on key attributes: size (large/medium/small), color (red/blue/etc), state (alive/dead/moving/etc), condition (new/old/damaged/etc)
- Examples: "What is the size of this carnivore?", "What is the color of this vehicle?", "What is the state of this animal?"
- Each question should extract a specific attribute value from one object
- Generate questions for both objects in each pair (same attribute, different objects)
- Use exact object IDs from the lists above"""

        return prompt
    
    def _parse_and_validate_questions(self, response_data: Dict[str, Any], 
                                    objects_a: List[ObjectDetection], 
                                    objects_b: List[ObjectDetection]) -> List[InterQuestion]:
        """
        Parse and validate LLM response for inter-questions.
        
        Args:
            response_data: Parsed JSON response from LLM
            objects_a: Image A object list for validation
            objects_b: Image B object list for validation
            
        Returns:
            List[InterQuestion]: Validated question instances
        """
        questions = []
        
        # Get valid object IDs
        valid_object_ids_a = {obj.object_id for obj in objects_a}
        valid_object_ids_b = {obj.object_id for obj in objects_b}
        
        if "inter_questions" not in response_data:
            return questions
        
        for question_data in response_data["inter_questions"]:
            try:
                # Validate required fields
                required_fields = ["image_a_object_id", "image_b_object_id", "question"]
                if not all(field in question_data for field in required_fields):
                    continue
                
                image_a_object_id = question_data["image_a_object_id"]
                image_b_object_id = question_data["image_b_object_id"] 
                question_text = question_data["question"]
                
                # Validate object IDs exist
                if image_a_object_id not in valid_object_ids_a:
                    continue
                if image_b_object_id not in valid_object_ids_b:
                    continue
                
                # Validate question is not empty
                if not question_text or not question_text.strip():
                    continue
                
                # Create InterQuestion instance
                inter_question = InterQuestion(
                    image_a_object_id=int(image_a_object_id),
                    image_b_object_id=int(image_b_object_id),
                    question=question_text.strip()
                )
                
                questions.append(inter_question)
                
            except Exception as e:
                print(f"Warning: Failed to parse inter-question: {e}")
                continue
        
        return questions
    
    def validate_questions(self, questions: List[InterQuestion], objects_a: List[ObjectDetection], 
                          objects_b: List[ObjectDetection]) -> bool:
        """
        Validate that generated questions are well-formed.
        
        Args:
            questions: List of InterQuestion instances
            objects_a: Image A object list
            objects_b: Image B object list
            
        Returns:
            bool: True if all questions are valid
        """
        try:
            valid_object_ids_a = {obj.object_id for obj in objects_a}
            valid_object_ids_b = {obj.object_id for obj in objects_b}
            
            for question in questions:
                # Check required attributes
                assert hasattr(question, 'image_a_object_id')
                assert hasattr(question, 'image_b_object_id')
                assert hasattr(question, 'question')
                
                # Validate types
                assert isinstance(question.image_a_object_id, int)
                assert isinstance(question.image_b_object_id, int)
                assert isinstance(question.question, str)
                
                # Validate object IDs exist
                assert question.image_a_object_id in valid_object_ids_a
                assert question.image_b_object_id in valid_object_ids_b
                
                # Validate question is not empty
                assert question.question.strip()
            
            return True
            
        except AssertionError:
            return False
    
    def get_questions_summary(self, questions: List[InterQuestion]) -> Dict[str, Any]:
        """
        Get summary statistics for generated questions.
        
        Args:
            questions: List of InterQuestion instances
            
        Returns:
            Dict[str, Any]: Summary information
        """
        if not questions:
            return {"count": 0, "unique_a_objects": 0, "unique_b_objects": 0, "avg_question_length": 0}
        
        # Count unique objects referenced
        unique_a_objects = set(q.image_a_object_id for q in questions)
        unique_b_objects = set(q.image_b_object_id for q in questions)
        question_lengths = [len(q.question.split()) for q in questions]
        
        return {
            "count": len(questions),
            "unique_a_objects": len(unique_a_objects),
            "unique_b_objects": len(unique_b_objects),
            "avg_question_length": sum(question_lengths) / len(question_lengths),
            "questions": [q.question for q in questions[:5]]  # Sample questions
        }
    
    def get_attribute_distribution(self, questions: List[InterQuestion]) -> Dict[str, int]:
        """
        Analyze which attributes are most commonly compared.
        
        Args:
            questions: List of InterQuestion instances
            
        Returns:
            Dict[str, int]: Count of questions for each attribute type
        """
        attribute_keywords = {
            "size": ["size", "large", "small", "big", "tiny", "huge"],
            "color": ["color", "red", "blue", "green", "yellow", "black", "white"],
            "shape": ["shape", "round", "square", "rectangular", "circular"],
            "position": ["position", "location", "above", "below", "near", "far"],
            "state": ["state", "open", "closed", "moving", "stationary"],
            "material": ["material", "metal", "wood", "plastic", "fabric"],
            "condition": ["condition", "new", "old", "clean", "dirty", "damaged"]
        }
        
        distribution = {attr: 0 for attr in attribute_keywords.keys()}
        distribution["other"] = 0
        
        for question in questions:
            question_lower = question.question.lower()
            matched = False
            
            for attr_type, keywords in attribute_keywords.items():
                if any(keyword in question_lower for keyword in keywords):
                    distribution[attr_type] += 1
                    matched = True
                    break
            
            if not matched:
                distribution["other"] += 1
        
        return distribution


# Example usage and testing
if __name__ == "__main__":
    # Test inter-question generator
    generator = InterQuestionGenerator()
    
    # Sample objects for two images
    objects_a = [
        ObjectDetection(0, "person", [10.0, 20.0, 100.0, 200.0], 0.95),
        ObjectDetection(1, "car", [150.0, 50.0, 300.0, 250.0], 0.88)
    ]
    
    objects_b = [
        ObjectDetection(0, "person", [20.0, 30.0, 110.0, 210.0], 0.92),
        ObjectDetection(1, "bicycle", [160.0, 60.0, 250.0, 180.0], 0.85)
    ]
    
    # Test validation
    sample_questions = [
        InterQuestion(0, 0, "Compare the size of these people"),
        InterQuestion(1, 1, "Compare the color of these vehicles")
    ]
    
    is_valid = generator.validate_questions(sample_questions, objects_a, objects_b)
    summary = generator.get_questions_summary(sample_questions)
    distribution = generator.get_attribute_distribution(sample_questions)
    
    print(f"✓ Question validation: {is_valid}")
    print(f"✓ Questions summary: {summary}")
    print(f"✓ Attribute distribution: {distribution}")
    print("✓ Inter-question generator ready!")