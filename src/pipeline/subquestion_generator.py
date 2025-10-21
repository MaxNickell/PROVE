"""
Contextual subquestion generator for PROVE pipeline.
Generates binary subquestions using visual context and detected objects to resolve ultimate question ambiguity.
"""

from typing import List, Dict, Any

from src.core.model_manager import ModelManager
from src.core.types import ObjectDetection, BinarySubquestion, ImageData


class SubquestionGeneratorError(RuntimeError):
    """Custom exception for subquestion generation failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return self.message


class SubquestionGenerator:
    """
    Generate contextual binary subquestions that resolve ultimate question ambiguity.
    Uses detailed image captions + detected objects + ultimate question to create specific reasoning questions.
    """

    def __init__(self):
        """Initialize generator with ModelManager singleton."""
        self.model_manager = ModelManager()

    def generate_binary_subquestions(
        self,
        ultimate_question: str,
        images: Dict[str, ImageData]  # Clean ImageData structure
    ) -> List[BinarySubquestion]:
        """
        Generate object-aware binary subquestions that collectively answer the ultimate question.

        Args:
            ultimate_question: Main comparative question to answer
            images: ImageData structure containing objects, captions, and context per image

        Returns:
            List[BinarySubquestion]: Binary questions with object references and types

        Raises:
            SubquestionGeneratorError: If generation fails
        """
        try:
            # Get LLM client from ModelManager
            llm_client = self.model_manager.get_llm_client()
            
            # Create structured context for LLM from ImageData
            context = self._build_structured_context_from_images(images)
            
            # Generate binary subqueries using LLM
            prompt = self._create_subquery_prompt(ultimate_question, images)

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at breaking down an ambiguous comparative question about an image pair into piecewise binary (Yes or No) subquestions using the provided visual context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Use Pydantic validation for robust JSON parsing
            response = llm_client.generate_subquestions(
                messages,
                temperature=0.3
            )

            # Extract objects from ImageData for conversion
            all_objects = {image_id: image_data.objects for image_id, image_data in images.items()}

            # Convert Pydantic response to BinarySubquestion objects
            subquestions = self._convert_to_binary_subquestions(response.subquestions, all_objects)

            return subquestions

        except Exception as err:
            raise SubquestionGeneratorError(f"Binary subquestion generation failed: {err}")
    
    def _build_structured_context_from_images(
        self,
        images: Dict[str, ImageData]
    ) -> str:
        """
        Build structured context string from ImageData structure.

        Args:
            images: ImageData structure containing objects, captions, and context per image

        Returns:
            str: Formatted context for LLM
        """
        context_parts = []

        for image_id in sorted(images.keys()):
            image_data = images[image_id]
            caption = image_data.scene_context.get("caption", "No caption available")
            objects = image_data.objects

            # Format objects with IDs
            object_list = []
            for obj in objects:
                # Create object ID in format: label_imageid_index (using simple image key)
                # Convert "image_a" to "a", "image_b" to "b" for simpler parsing
                simple_image_id = image_id.replace("image_", "")
                obj_id = f"{obj.label}_{simple_image_id}_{obj.object_id}"
                object_list.append(f"{obj_id} ({obj.label}, conf={obj.confidence:.2f})")

            objects_str = ", ".join(object_list)

            context_parts.append(f"""
Image {image_id.upper()}:
Context: {caption}
Objects: {objects_str}""")

        return "\n".join(context_parts)
    
    def _create_subquery_prompt(self, ultimate_question: str, images: Dict[str, ImageData]) -> str:
        """
        Create official prompt for binary subquery generation using template.

        Args:
            ultimate_question: Main comparative question
            images: ImageData structure containing objects and captions

        Returns:
            str: Formatted prompt for LLM with template variables substituted
        """
        # Extract captions and format object lists
        image_ids = sorted(images.keys())
        if len(image_ids) < 2:
            raise ValueError("Expected at least 2 images for subquery generation")

        # Get captions
        caption_a = images[image_ids[0]].scene_context.get("caption", "No caption available")
        caption_b = images[image_ids[1]].scene_context.get("caption", "No caption available")

        # Format object lists in the template format
        def format_object_list(image_data: ImageData, image_key: str) -> str:
            """Format objects as JSON dict for template."""
            simple_image_id = image_key.replace("image_", "")
            objects_dict = {}
            for obj in image_data.objects:
                obj_id = f"{obj.label}_{simple_image_id}_{obj.object_id}"
                objects_dict[obj_id] = obj.label
            # Format as JSON-like structure with proper indentation
            items = [f'  "{k}": "{v}"' for k, v in objects_dict.items()]
            return "{\n" + ",\n".join(items) + "\n}"

        objects_a = format_object_list(images[image_ids[0]], image_ids[0])
        objects_b = format_object_list(images[image_ids[1]], image_ids[1])

        # Use official prompt template
        prompt = f"""TASK
You will be given:
- An ultimate question about two images
- A caption for each image
- An object list for each image
Given the visual context, you must reason through the ultimate question. Break down the ultimate question into a set of binary subquestions that, when answered, can collectively resolve the ultimate question.

SUBQUESTION CATEGORIES
- **attribute**: Specific visual attributes of objects
- **relationship**: Spatial relationships or interactions
- **scene_attribute**: Visually observable property of the entire scene
- **count**: Which object categories' counts must be determined

RULES
- Each subquestion must be answerable with Yes or No.
- Attribute questions should specify which attribute class or value must be verified.
- Relationship questions should ask one explicit visual relation.
- Count questions must explicitly ask about the number of objects of a certain class.
- Scene attribute questions must ask an observable, image-level visual property.
- Only reference objects from the object list using their exact IDs in "referenced_objects".
- The combined subquestions must collectively contain all the information needed to answer the ultimate question.
- Output strict JSON, nothing else.

---

### EXAMPLES

**Example 1**
Ultimate Question: Which scene depicts more power?

IMAGE A
Caption: A king sits on a golden throne in a grand hall surrounded by four guards holding spears. Red carpets line the floor and tall stained glass windows cast colorful light over the crown resting beside him. Three subjects bow before the throne while two musicians stand by holding trumpets.
Objects:
{{
  "king_a_0": "king",
  "throne_a_1": "throne",
  "guard_a_2": "guard",
  "guard_a_3": "guard",
  "guard_a_4": "guard",
  "guard_a_5": "guard",
  "spear_a_6": "spear",
  "spear_a_7": "spear",
  "spear_a_8": "spear",
  "spear_a_9": "spear",
  "crown_a_10": "crown",
  "subject_a_11": "subject",
  "subject_a_12": "subject",
  "subject_a_13": "subject"
}}

IMAGE B
Caption: A man sits cross-legged on the sidewalk with torn clothes and an empty cup beside him. Two people walk past without looking as a gust of wind scatters some coins near his feet. Behind him, a cracked wall with faded posters leans into shadow.
Objects:
{{
  "man_b_0": "man",
  "sidewalk_b_1": "sidewalk",
  "clothing_b_2": "clothing",
  "cup_b_3": "cup",
  "coin_b_4": "coin",
  "coin_b_5": "coin",
  "wall_b_6": "wall",
  "poster_b_7": "poster"
}}

Output:
{{
  "subquestions": [
    {{
      "question": "Is the king sitting on the throne?",
      "subquery_type": "relationship",
      "referenced_objects": ["king_a_0", "throne_a_1"]
    }},
    {{
      "question": "Is the king wearing the crown?",
      "subquery_type": "relationship",
      "referenced_objects": ["king_a_0", "crown_a_10"]
    }},
    {{
      "question": "Do the guards appear to be facing or serving the king?",
      "subquery_type": "relationship",
      "referenced_objects": ["guard_a_2", "guard_a_3", "guard_a_4", "guard_a_5", "king_a_0"]
    }},
    {{
      "question": "Are the subjects bowing toward the king?",
      "subquery_type": "relationship",
      "referenced_objects": ["subject_a_11", "subject_a_12", "subject_a_13", "king_a_0"]
    }},
    {{
      "question": "How many subjects are there?",
      "subquery_type": "count",
      "referenced_objects": ["subject_a_11", "subject_a_12", "subject_a_13"]
    }},
    {{
      "question": "How many guards are there?",
      "subquery_type": "count",
      "referenced_objects": ["guard_a_2", "guard_a_3", "guard_a_4", "guard_a_5"]
    }},
    {{
      "question": "Is the man sitting on the sidewalk?",
      "subquery_type": "relationship",
      "referenced_objects": ["man_b_0", "sidewalk_b_1"]
    }},
    {{
      "question": "Is the man wearing clothing?",
      "subquery_type": "relationship",
      "referenced_objects": ["man_b_0", "clothing_b_2"]
    }},
    {{
      "question": "Does the clothing appear torn or worn out?",
      "subquery_type": "attribute",
      "referenced_objects": ["clothing_b_2"]
    }},
    {{
      "question": "Does the man appear to be poor?",
      "subquery_type": "attribute",
      "referenced_objects": ["man_b_0"]
    }},
    {{
      "question": "Is the man holding or sitting beside a cup for donations (begging)?",
      "subquery_type": "relationship",
      "referenced_objects": ["man_b_0", "cup_b_3"]
    }},
    {{
      "question": "Is the environment of image A bright and ornate?",
      "subquery_type": "scene_attribute",
      "referenced_objects": []
    }},
    {{
      "question": "Is the environment of image B dimly lit and worn down?",
      "subquery_type": "scene_attribute",
      "referenced_objects": []
    }}
  ]
}}

---

**Example 2**
Ultimate Question: What is the difference between the two images?

IMAGE A
Caption: Several dogs of different breeds run freely through a sunny dog park. Two chase tennis balls, one leaps through a sprinkler, and three others roll in the grass while two owners watch from benches. Water bowls and toys are scattered across the open field.
Objects:
{{
  "dog_a_0": "dog",
  "dog_a_1": "dog",
  "dog_a_2": "dog",
  "dog_a_3": "dog",
  "ball_a_4": "ball",
  "ball_a_5": "ball",
  "sprinkler_a_6": "sprinkler",
  "owner_a_7": "owner",
  "owner_a_8": "owner"
}}

IMAGE B
Caption: A crowd of dogs races down a marked track during a dog competition. Four trainers stand at the sidelines holding leashes and stopwatches. A banner with the competition logo waves in the background as spectators cheer from bleachers.
Objects:
{{
  "dog_b_0": "dog",
  "dog_b_1": "dog",
  "dog_b_2": "dog",
  "trainer_b_3": "trainer",
  "trainer_b_4": "trainer",
  "track_b_5": "track",
  "leash_b_6": "leash"
}}

Output:
{{
  "subquestions": [
    {{
      "question": "Do any of the dogs appear relaxed?",
      "subquery_type": "attribute",
      "referenced_objects": ["dog_a_0", "dog_a_1", "dog_a_2", "dog_a_3"]
    }},
    {{
      "question": "Do the dogs appear to be playing with each other?",
      "subquery_type": "relationship",
      "referenced_objects": ["dog_a_0", "dog_a_1", "dog_a_2", "dog_a_3"]
    }},
    {{
      "question": "Do the dogs appear to be playing with balls?",
      "subquery_type": "relationship",
      "referenced_objects": ["dog_a_0", "dog_a_1", "dog_a_2", "dog_a_3", "ball_a_4", "ball_a_5"]
    }},
    {{
      "question": "How many dogs are there?",
      "subquery_type": "count",
      "referenced_objects": ["dog_a_0", "dog_a_1", "dog_a_2", "dog_a_3", "dog_b_0", "dog_b_1", "dog_b_2"]
    }},
    {{
      "question": "Are the dogs in image B running on the track?",
      "subquery_type": "relationship",
      "referenced_objects": ["dog_b_0", "dog_b_1", "dog_b_2", "track_b_5"]
    }},
    {{
      "question": "Are the trainers holding leashes?",
      "subquery_type": "relationship",
      "referenced_objects": ["trainer_b_3", "trainer_b_4", "leash_b_6"]
    }},
    {{
      "question": "Are the dogs in image B competing or racing with each other?",
      "subquery_type": "relationship",
      "referenced_objects": ["dog_b_0", "dog_b_1", "dog_b_2"]
    }},
    {{
      "question": "Is the environment in image A open and natural?",
      "subquery_type": "scene_attribute",
      "referenced_objects": []
    }},
    {{
      "question": "Is the environment in image B structured and man-made?",
      "subquery_type": "scene_attribute",
      "referenced_objects": []
    }}
  ]
}}

---

**Example 3**
Ultimate Question: Which image appears more fair?

IMAGE A
Caption: Five children sit in a circle dividing colorful candies evenly among themselves. Each child smiles and places pieces into small cups. The table is neatly arranged, and everyone receives the same amount. A teacher stands nearby supervising.
Objects:
{{
  "child_a_0": "child",
  "child_a_1": "child",
  "child_a_2": "child",
  "child_a_3": "child",
  "child_a_4": "child",
  "candy_a_5": "candy",
  "candy_a_6": "candy",
  "cup_a_7": "cup",
  "cup_a_8": "cup"
}}

IMAGE B
Caption: Several animals gather around two water troughs under the sun. Three horses drink from a full container while two goats stand beside an empty trough. A farmer watches from the distance without intervening.
Objects:
{{
  "horse_b_0": "horse",
  "horse_b_1": "horse",
  "horse_b_2": "horse",
  "goat_b_3": "goat",
  "goat_b_4": "goat",
  "trough_b_5": "trough",
  "trough_b_6": "trough"
}}

Output:
{{
  "subquestions": [
    {{
      "question": "Do the children each have candies in front of them?",
      "subquery_type": "relationship",
      "referenced_objects": ["child_a_0", "child_a_1", "child_a_2", "child_a_3", "child_a_4", "candy_a_5", "candy_a_6"]
    }},
    {{
      "question": "How many candies are there?",
      "subquery_type": "count",
      "referenced_objects": ["candy_a_5", "candy_a_6"]
    }},
    {{
      "question": "How many children are there?",
      "subquery_type": "count",
      "referenced_objects": ["child_a_0", "child_a_1", "child_a_2", "child_a_3", "child_a_4"]
    }},
    {{
      "question": "Are the horses drinking from the full trough?",
      "subquery_type": "relationship",
      "referenced_objects": ["horse_b_0", "horse_b_1", "horse_b_2", "trough_b_5"]
    }},
    {{
      "question": "Are the goats standing beside the empty trough?",
      "subquery_type": "relationship",
      "referenced_objects": ["goat_b_3", "goat_b_4", "trough_b_6"]
    }},
    {{
      "question": "Do the goats appear thirsty or waiting for water?",
      "subquery_type": "attribute",
      "referenced_objects": ["goat_b_3", "goat_b_4"]
    }},
    {{
      "question": "Is the environment in image A organized?",
      "subquery_type": "scene_attribute",
      "referenced_objects": []
    }},
    {{
      "question": "Is the environment in image B dry?",
      "subquery_type": "scene_attribute",
      "referenced_objects": []
    }}
  ]
}}

---

### NOW BEGIN TASK

IMAGE A
Image Caption: {caption_a}
Object List: {objects_a}

IMAGE B
Image Caption: {caption_b}
Object List: {objects_b}

Ultimate Question: {ultimate_question}"""

        return prompt
    
    def _convert_to_binary_subquestions(
        self,
        subquestions: List,
        all_objects: Dict[str, List[ObjectDetection]]
    ) -> List[BinarySubquestion]:
        """
        Convert Pydantic-validated subquestions to BinarySubquestion objects.
        Trust LLM-provided object references and types, only validate object ID existence.

        Args:
            subquestions: List of SubquestionItem objects from Pydantic validation
            all_objects: Original objects for validation

        Returns:
            List[BinarySubquestion]: BinarySubquestion instances
        """
        binary_subquestions = []

        # Get valid object IDs for validation (use same format as context building)
        valid_object_ids = set()
        for image_id, objects in all_objects.items():
            for obj in objects:
                # Use same format as _build_structured_context: strip "image_" prefix
                simple_image_id = image_id.replace("image_", "")
                obj_id = f"{obj.label}_{simple_image_id}_{obj.object_id}"
                valid_object_ids.add(obj_id)

        for subquestion_item in subquestions:
            try:
                # Extract data from SubquestionItem (Pydantic already validated types)
                question = subquestion_item.question.strip()
                subquery_type = subquestion_item.subquery_type.strip()

                # Trust LLM-provided referenced_objects, but validate they exist
                referenced_objects = getattr(subquestion_item, 'referenced_objects', [])

                # Validate that all referenced objects exist in our valid set
                invalid_objects = [obj_id for obj_id in referenced_objects if obj_id not in valid_object_ids]
                if invalid_objects:
                    print(f"Warning: Skipping subquestion with invalid object IDs: {invalid_objects}")
                    continue

                # Create BinarySubquestion instance with validated data
                binary_subquestion = BinarySubquestion(
                    question=question,
                    referenced_objects=referenced_objects,
                    subquery_type=subquery_type
                )

                binary_subquestions.append(binary_subquestion)

            except Exception as e:
                print(f"Warning: Failed to parse subquestion: {e}")
                continue

        return binary_subquestions
    
    def validate_subquestions(self, subquestions: List[BinarySubquestion]) -> bool:
        """
        Validate that generated subquestions have basic required structure.
        Pydantic handles type validation, we just check basic content.

        Args:
            subquestions: List of BinarySubquestion instances

        Returns:
            bool: True if all subquestions are valid
        """
        try:
            for subquestion in subquestions:
                # Check required attributes exist
                assert hasattr(subquestion, 'question')
                assert hasattr(subquestion, 'referenced_objects')
                assert hasattr(subquestion, 'subquery_type')

                # Validate basic content (non-empty)
                assert subquestion.question.strip()
                assert subquestion.subquery_type.strip()
                assert isinstance(subquestion.referenced_objects, list)

            return True

        except (AssertionError, AttributeError):
            return False
    
    def get_subquestions_summary(
        self,
        subquestions: List[BinarySubquestion]
    ) -> Dict[str, Any]:
        """
        Get summary statistics for generated subquestions.

        Args:
            subquestions: List of BinarySubquestion instances

        Returns:
            Dict[str, Any]: Summary information
        """
        if not subquestions:
            return {"count": 0, "types": {}, "avg_question_length": 0}

        # Count by type
        type_counts = {}
        question_lengths = []
        unique_objects = set()

        for subquestion in subquestions:
            # Count types
            subquery_type = subquestion.subquery_type
            type_counts[subquery_type] = type_counts.get(subquery_type, 0) + 1

            # Track question lengths
            question_lengths.append(len(subquestion.question.split()))

            # Track unique objects
            unique_objects.update(subquestion.referenced_objects)

        return {
            "count": len(subquestions),
            "types": type_counts,
            "avg_question_length": sum(question_lengths) / len(question_lengths) if question_lengths else 0,
            "unique_objects_referenced": len(unique_objects),
            "sample_questions": [sq.question for sq in subquestions[:3]]
        }
    
    


# Example usage and testing
if __name__ == "__main__":
    # Test subquestion generator
    generator = SubquestionGenerator()

    # Sample data
    ultimate_question = "Who is more powerful between these two people?"

    # Create test ImageData structure
    from src.core.types import ObjectDetection, ImageData
    images = {
        "image_a": ImageData(
            objects=[
                ObjectDetection(0, "person", [10.0, 20.0, 100.0, 200.0], 0.95),
                ObjectDetection(1, "weight", [150.0, 50.0, 300.0, 250.0], 0.88)
            ],
            attributes={},
            relationships=[],
            scene_context={"caption": "Muscular man lifting heavy weights in gym setting"}
        ),
        "image_b": ImageData(
            objects=[
                ObjectDetection(0, "person", [20.0, 30.0, 110.0, 210.0], 0.92),
                ObjectDetection(1, "equipment", [160.0, 60.0, 250.0, 180.0], 0.85)
            ],
            attributes={},
            relationships=[],
            scene_context={"caption": "Athletic woman doing pull-ups with defined muscle tone"}
        )
    }

    # Test generation
    try:
        subquestions = generator.generate_binary_subquestions(
            ultimate_question, images
        )

        # Validate subquestions
        is_valid = generator.validate_subquestions(subquestions)
        summary = generator.get_subquestions_summary(subquestions)

        print(f"✓ Generated {len(subquestions)} binary subquestions")
        print(f"✓ Validation: {is_valid}")
        print(f"✓ Summary: {summary}")

        for i, subquestion in enumerate(subquestions):
            print(f"  {i+1}. {subquestion.question}")

        print("✓ Subquestion generator ready!")

    except Exception as e:
        print(f"✗ Test failed: {e}")