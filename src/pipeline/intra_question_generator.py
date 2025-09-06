"""
Intra-relationship question generator for PROVE pipeline.
Generates spatial/interaction questions for object pairs within an image.
"""

from typing import List, Dict, Any
import json
import itertools

from src.core.model_manager import ModelManager
from src.core.types import ObjectDetection, IntraQuestion


class IntraQuestionGeneratorError(RuntimeError):
    """Custom exception for intra-question generation failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return self.message


class IntraQuestionGenerator:
    """
    Generate intra-relationship questions for object pairs within an image.
    Uses LLM to create spatial/interaction questions that help answer comparative questions.
    """
    
    def __init__(self):
        """Initialize generator with ModelManager singleton."""
        self.model_manager = ModelManager()
    
    def generate_relation_candidates(self, ultimate_question: str, objects: List[ObjectDetection]) -> Dict[tuple, List[str]]:
        """
        Generate relation candidates for object pairs within an image using LLM reasoning.
        
        Args:
            ultimate_question: The main comparative question to answer
            objects: List of ObjectDetection instances for one image
            
        Returns:
            Dict[tuple, List[str]]: Mapping from (obj1_id, obj2_id) to list of relation candidates
            
        Raises:
            IntraQuestionGeneratorError: If generation fails
        """
        try:
            if not objects or len(objects) < 2:
                return {}
            
            # Get LLM client from ModelManager
            llm_client = self.model_manager.get_llm_client()
            
            # Create object pairs for relationship analysis
            object_pairs = list(itertools.combinations(objects, 2))
            relation_candidates = {}
            
            for obj1, obj2 in object_pairs:
                # Generate relation candidates for this specific pair
                candidates = self._generate_candidates_for_pair(llm_client, ultimate_question, obj1, obj2)
                relation_candidates[(obj1.object_id, obj2.object_id)] = candidates
            
            return relation_candidates
            
        except Exception as err:
            raise IntraQuestionGeneratorError(f"Relation candidate generation failed: {err}")
    
    def generate_questions(self, ultimate_question: str, objects: List[ObjectDetection]) -> List[IntraQuestion]:
        """
        Generate intra-relationship questions for object pairs within an image.
        (Maintained for backward compatibility during transition)
        
        Args:
            ultimate_question: The main comparative question to answer
            objects: List of ObjectDetection instances for one image
            
        Returns:
            List[IntraQuestion]: Generated questions with exact schema compliance
            
        Raises:
            IntraQuestionGeneratorError: If generation fails
        """
        try:
            if not objects or len(objects) < 2:
                return []
            
            # Get LLM client from ModelManager
            llm_client = self.model_manager.get_llm_client()
            
            # Create object pairs for relationship analysis
            object_pairs = list(itertools.combinations(objects, 2))
            
            # Format objects for LLM prompt
            object_descriptions = []
            for obj in objects:
                obj_desc = f"ID {obj.object_id}: {obj.label} (confidence: {obj.confidence:.2f})"
                object_descriptions.append(obj_desc)
            
            # Generate questions using LLM
            prompt = self._create_intra_question_prompt(ultimate_question, object_descriptions, object_pairs)
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at generating spatial and interaction questions between objects in images. Return strict JSON only, no markdown or extra text."
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
            questions = self._parse_and_validate_questions(response_data, objects)
            
            return questions
            
        except Exception as err:
            raise IntraQuestionGeneratorError(f"Intra-question generation failed: {err}")
    
    def _generate_candidates_for_pair(self, llm_client, ultimate_question: str, obj1: ObjectDetection, obj2: ObjectDetection) -> List[str]:
        """
        Generate relation candidates for a specific object pair using LLM reasoning.
        
        Args:
            llm_client: LLM client instance
            ultimate_question: Main comparative question
            obj1: First object
            obj2: Second object
            
        Returns:
            List[str]: List of potential relations between the objects
        """
        prompt = f"""Given the ultimate question "{ultimate_question}", generate potential spatial and interaction relationships between these two objects:

Object 1: {obj1.label} (ID: {obj1.object_id})
Object 2: {obj2.label} (ID: {obj2.object_id})

Consider what relationships might exist between a {obj1.label} and a {obj2.label} that could help answer the ultimate question.

Generate 4-6 potential relationships that can be verified visually. Focus on:
- Spatial relationships (near, far, above, below, left, right, touching)
- Action relationships (eating, chasing, hunting, hiding from, following) 
- State relationships (looking at, facing, turned away from)

Return JSON with this exact format:
{{
  "relations": ["relation1", "relation2", "relation3", "relation4"]
}}

Examples of good relations: "near", "eating", "chasing", "above", "looking at", "touching"
Keep relations short (1-2 words) and visually verifiable."""

        messages = [
            {
                "role": "system",
                "content": "You are an expert at determining potential visual relationships between objects. Generate contextually relevant relation candidates that can be verified through visual analysis. Return strict JSON only."
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
            relations = response_data.get("relations", [])
            
            # Validate and clean relations
            cleaned_relations = []
            for relation in relations:
                if isinstance(relation, str) and relation.strip():
                    cleaned_relations.append(relation.strip().lower())
            
            return cleaned_relations[:6]  # Limit to 6 relations max
            
        except Exception as e:
            print(f"Warning: Failed to generate relation candidates for {obj1.label}-{obj2.label}: {e}")
            # Fallback to generic spatial relations
            return ["near", "far", "above", "below", "touching"]
    
    def _create_intra_question_prompt(self, ultimate_question: str, object_descriptions: List[str], 
                                    object_pairs: List[tuple]) -> str:
        """
        Create prompt for intra-relationship question generation.
        
        Args:
            ultimate_question: Main comparative question
            object_descriptions: Formatted object descriptions
            object_pairs: List of object pair tuples
            
        Returns:
            str: Formatted prompt for LLM
        """
        objects_text = "\n".join(object_descriptions)
        
        # Show example pairs to help LLM understand what relationships to explore
        pairs_text = []
        for i, (obj1, obj2) in enumerate(object_pairs[:10]):  # Limit to first 10 pairs
            pairs_text.append(f"Pair {i+1}: Object {obj1.object_id} ({obj1.label}) and Object {obj2.object_id} ({obj2.label})")
        
        pairs_example = "\n".join(pairs_text)
        
        prompt = f"""Given the ultimate question "{ultimate_question}" and these objects in one image:

{objects_text}

Generate useful pairwise relationship questions that would help answer the ultimate question. Focus on spatial relationships, interactions, and comparative properties between object pairs.

Available object pairs to consider:
{pairs_example}

Return JSON with this exact format:
{{
  "intra_questions": [
    {{
      "object_ids": [subject_id, object_id],
      "question": "short relation question about that pair"
    }}
  ]
}}

Rules:
- Only use object IDs from the provided list
- Keep questions short and pair-focused
- Focus on spatial relations (near/far, above/below, inside/outside)
- Focus on interactions (touching, supporting, facing, etc.)
- Generate questions that help answer the ultimate question
- No limits on count, but prioritize most useful relationships
- Use only the exact object IDs provided"""

        return prompt
    
    def _parse_and_validate_questions(self, response_data: Dict[str, Any], 
                                    objects: List[ObjectDetection]) -> List[IntraQuestion]:
        """
        Parse and validate LLM response for intra-questions.
        
        Args:
            response_data: Parsed JSON response from LLM
            objects: Original object list for validation
            
        Returns:
            List[IntraQuestion]: Validated question instances
        """
        questions = []
        
        # Get valid object IDs
        valid_object_ids = {obj.object_id for obj in objects}
        
        if "intra_questions" not in response_data:
            return questions
        
        for question_data in response_data["intra_questions"]:
            try:
                # Validate required fields
                if "object_ids" not in question_data or "question" not in question_data:
                    continue
                
                object_ids = question_data["object_ids"]
                question_text = question_data["question"]
                
                # Validate object_ids format
                if not isinstance(object_ids, list) or len(object_ids) != 2:
                    continue
                
                # Validate object IDs exist
                if not all(oid in valid_object_ids for oid in object_ids):
                    continue
                
                # Validate question is not empty
                if not question_text or not question_text.strip():
                    continue
                
                # Create IntraQuestion instance
                intra_question = IntraQuestion(
                    object_ids=object_ids,
                    question=question_text.strip()
                )
                
                questions.append(intra_question)
                
            except Exception as e:
                print(f"Warning: Failed to parse intra-question: {e}")
                continue
        
        return questions
    
    def validate_questions(self, questions: List[IntraQuestion], objects: List[ObjectDetection]) -> bool:
        """
        Validate that generated questions are well-formed.
        
        Args:
            questions: List of IntraQuestion instances
            objects: Original object list
            
        Returns:
            bool: True if all questions are valid
        """
        try:
            valid_object_ids = {obj.object_id for obj in objects}
            
            for question in questions:
                # Check required attributes
                assert hasattr(question, 'object_ids')
                assert hasattr(question, 'question')
                
                # Validate types
                assert isinstance(question.object_ids, list)
                assert len(question.object_ids) == 2
                assert isinstance(question.question, str)
                
                # Validate object IDs exist
                assert all(oid in valid_object_ids for oid in question.object_ids)
                
                # Validate question is not empty
                assert question.question.strip()
            
            return True
            
        except AssertionError:
            return False
    
    def get_questions_summary(self, questions: List[IntraQuestion]) -> Dict[str, Any]:
        """
        Get summary statistics for generated questions.
        
        Args:
            questions: List of IntraQuestion instances
            
        Returns:
            Dict[str, Any]: Summary information
        """
        if not questions:
            return {"count": 0, "unique_pairs": 0, "avg_question_length": 0}
        
        # Count unique object pairs
        unique_pairs = set()
        question_lengths = []
        
        for q in questions:
            pair = tuple(sorted(q.object_ids))
            unique_pairs.add(pair)
            question_lengths.append(len(q.question.split()))
        
        return {
            "count": len(questions),
            "unique_pairs": len(unique_pairs),
            "avg_question_length": sum(question_lengths) / len(question_lengths),
            "questions": [q.question for q in questions[:5]]  # Sample questions
        }


# Example usage and testing
if __name__ == "__main__":
    # Test intra-question generator
    generator = IntraQuestionGenerator()
    
    # Sample objects
    sample_objects = [
        ObjectDetection(0, "person", [10.0, 20.0, 100.0, 200.0], 0.95),
        ObjectDetection(1, "car", [150.0, 50.0, 300.0, 250.0], 0.88),
        ObjectDetection(2, "dog", [80.0, 180.0, 120.0, 220.0], 0.82)
    ]
    
    # Test validation
    sample_questions = [
        IntraQuestion([0, 1], "Is the person near the car?"),
        IntraQuestion([0, 2], "Is the person walking the dog?"),
        IntraQuestion([1, 2], "Is the dog in front of the car?")
    ]
    
    is_valid = generator.validate_questions(sample_questions, sample_objects)
    summary = generator.get_questions_summary(sample_questions)
    
    print(f"✓ Question validation: {is_valid}")
    print(f"✓ Questions summary: {summary}")
    print("✓ Intra-question generator ready!")