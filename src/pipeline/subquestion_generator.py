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
            
            # Generate binary subquestions using LLM
            prompt = self._create_subquestion_prompt(ultimate_question, images)

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
        Build structured context with ONLY object classes (no IDs).
        Classes shown so LLM knows what entities exist in each image.

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

            # Get unique object classes with counts
            class_counts = {}
            for obj in objects:
                class_counts[obj.label] = class_counts.get(obj.label, 0) + 1

            # Format: "label (count)" if multiple instances, else just "label"
            objects_str = ", ".join([
                f"{label} ({count})" if count > 1 else label
                for label, count in sorted(class_counts.items())
            ])

            context_parts.append(f"""
Image {image_id.upper()}:
Image Caption: {caption}
Object List: {objects_str}""")

        return "\n".join(context_parts)
    
    def _create_subquestion_prompt(self, ultimate_question: str, images: Dict[str, ImageData]) -> str:
        """
        Create official prompt for binary subquestion generation using template.

        Args:
            ultimate_question: Main comparative question
            images: ImageData structure containing objects and captions

        Returns:
            str: Formatted prompt for LLM with template variables substituted
        """
        # Extract captions and format object lists
        image_ids = sorted(images.keys())
        if len(image_ids) < 2:
            raise ValueError("Expected at least 2 images for subquestion generation")

        # Get captions
        caption_a = images[image_ids[0]].scene_context.get("caption", "No caption available")
        caption_b = images[image_ids[1]].scene_context.get("caption", "No caption available")

        # Format object lists as class names with counts
        def format_object_list(image_data: ImageData) -> str:
            """Format objects as class names with counts."""
            class_counts = {}
            for obj in image_data.objects:
                class_counts[obj.label] = class_counts.get(obj.label, 0) + 1
            # Format as comma-separated list
            return ", ".join([
                f"{label} ({count})" if count > 1 else label
                for label, count in sorted(class_counts.items())
            ])

        objects_a = format_object_list(images[image_ids[0]])
        objects_b = format_object_list(images[image_ids[1]])

        # Use official prompt template
        prompt = f"""TASK
You will be given:
- An ultimate question about two images
- A caption for each image
- An object list for each image
Given the visual context, you must reason through the ultimate question. Break down the ultimate question into a set of binary subquestions that, when answered, can collectively resolve the ultimate question.

RULES
- Each subquestion must be answerable with Yes or No.
- Write questions in natural language only.
- Mention object classes generically (e.g., "the mailbox", "a rabbit", "the forks").
- The object list shows what entities are detected - use this to inform your questions.
- Together, the subquestions must provide all information needed to answer the ultimate question.
- Output strict JSON.

---

### EXAMPLES

**Example 1**
Ultimate Question: Are there more people wearing blue shirts sitting at tables in image A than in image B?

IMAGE A
Image Caption: A busy outdoor café lines a cobblestone street. Several small round tables with metal chairs are arranged under large umbrellas. Groups of people sit and talk while a waiter carries a tray of drinks between the tables. A couple of customers waiting in line stand near the entrance, and a cyclist passes by in the background. A few people in blue shirts are scattered among the seated and standing customers, some with bags or laptops on the tables.
Object List: person (9), table (6), chair (12), shirt-blue (4), umbrella (3), tray, drink (8), bag (3), laptop (2), bicycle, building (2), menu board

IMAGE B
Image Caption: Inside a modern co-working space, long wooden tables are arranged in rows with office chairs on both sides. Several people work on laptops while others stand near a whiteboard covered in notes. A person in a blue shirt gestures toward the board, and another in a blue hoodie leans against a pillar near the back. Coffee mugs, notebooks, and headphones are scattered across the tables. Large windows look out onto a street with passing cars.
Object List: person (10), table (4), chair (10), shirt-blue (2), hoodie-blue, laptop (7), mug (5), notebook (6), headphone (3), whiteboard, pillar, window (3), car (4)

Output:
[
  "In image A, how many people sitting at a table are wearing a blue shirt?",
  "In Image B, how many people sitting at a table are wearing a blue shirt?"
]

---

**Example 2**
Ultimate Question: Do both images show at least two cats resting on surfaces higher than any dog?

IMAGE A
Image Caption: In a cozy living room, a large dog lies stretched out on a rug near a coffee table. A sofa sits against the wall with a colorful blanket draped over the back. One cat perches on the top of the sofa, and another cat is somewhere on a tall bookshelf beside a potted plant and several stacked books. A TV stands on a low stand opposite the sofa, and a floor lamp glows softly in the corner.
Object List: dog, cat (2), sofa, rug, coffee table, bookshelf, book (10), plant (2), TV, TV stand, lamp, blanket, pillow (3)

IMAGE B
Image Caption: In a small sunroom, sunlight pours through large windows onto a tiled floor. Two dogs rest near a food bowl placed on the ground. A narrow windowsill runs along the back wall, and one cat stretches out along it while another cat curls up on the top of a cushioned chair near a side table. A scratching post stands in one corner, and a watering can sits beside several potted plants.
Object List: dog (2), cat (2), chair, cushion, side table, windowsill, food bowl, scratching post, plant (4), watering can, window (3), tile floor

Output:
[
  "In image A, how many cats are resting on a surface above all dogs?",
  "In image B, how many cats are resting on a surface above all dogs?",
]

---

**Example 3**
Ultimate Question: Do both images contain exactly one red traffic light directly above a painted crosswalk?

IMAGE A
Image Caption: At a downtown intersection at dusk, cars wait at a stop line while pedestrians gather on the sidewalk. Two traffic lights hang over the street from metal poles; one faces the main road while another faces a smaller side street. A wide painted crosswalk with white stripes stretches across the main road in front of the waiting cars. Storefronts with bright signs and a bus stop shelter line the sidewalk, and a bicyclist rides past the corner.
Object List: car (5), traffic light (2), crosswalk, pole (2), bus shelter, bicycle, store (4), sign (5), sidewalk, building (3)

IMAGE B
Image Caption: In a quieter residential area during the day, a single intersection joins two narrow roads. One traffic light is mounted on a horizontal bar extending from a pole at the corner. A faded crosswalk with worn white lines crosses one of the streets, and a school zone sign stands nearby. A parked car, a mailbox, and a row of trees line the sidewalk. A pedestrian walks a dog along the opposite side of the street.
Object List: traffic light, crosswalk, pole, bar, sign-school, car, mailbox, tree (6), sidewalk, pedestrian, dog, house (3)

Output:
[
    "In image A, how many red traffic lights are directly above a painted crosswalk?",
    "In image B, how many red traffic lights are directly above a painted crosswalk?",
]

---

**Example 4**
Ultimate Question: Is there exactly one image where a child holding a red ball stands closer to a leashed dog than to any adult?

IMAGE A
Image Caption: In a grassy park, a child in a striped shirt holds a bright red ball near a paved path. An adult stands a few steps away talking to another grown-up near a picnic table covered with food containers and drinks. A medium-sized dog on a leash is somewhere between the child and the picnic table, with the leash leading back toward the first adult. Another dog without a leash sniffs near a trash can by a tree. A playground and several benches are visible in the distance.
Object List: child, adult (2), dog (2), leash, ball-red, picnic table, container (5), drink (3), trash can, tree (4), bench (3), playground, path

IMAGE B
Image Caption: On a neighborhood sidewalk, a child carries a red ball while walking beside an adult who holds two leashes attached to small dogs trotting ahead. The dogs are slightly farther from the child than the adult is. Another adult walks behind them pushing a stroller. Parked cars line the street, and houses with front yards and mailboxes extend down the block. A streetlamp stands at the corner.
Object List: child, adult (2), dog (2), leash (2), ball-red, stroller, car (4), house (5), mailbox (4), streetlamp, sidewalk, yard (5)

Output:
[
    "In image A, is there a child holding a red ball and the child is closer to a leashed dog than to any adult?",
    "In image B, is there a child holding a red ball and the child is closer to a leashed dog than to any adult?",
]

---

**Example 4**
Ultimate Question: Which image depicts a man on the left of a woman carrying a red umbrella?

IMAGE A
Image Caption: Three people walk along a rainy city street. Two men and one woman are visible. The woman in a yellow raincoat holds an umbrella, while one man in a blue jacket stands nearby carrying a shopping bag. The other man, wearing a gray coat, walks farther behind them. Cars and streetlights line the wet sidewalk.
Object List: umbrella, woman, man (2), shopping bag, car (2), streetlight

IMAGE B
Image Caption: Four people stroll through a sunny park. A woman in a coat holds an umbrella while two men walk nearby — one pushing a stroller and another talking on a phone. Another woman sits on a bench under a tree.
Object List: umbrella, woman (2), man (2), stroller, bench, tree (3)

Output:
[
  "In image A, is there a man to the left of a woman carrying a red umbrella?",
  "In image B, is there a man to the left of a woman carrying a red umbrella?"
]

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
        subquestions: List[str],
        all_objects: Dict[str, List[ObjectDetection]]
    ) -> List[BinarySubquestion]:
        """
        Convert list of question strings to BinarySubquestion objects.

        Args:
            subquestions: List of question strings from Pydantic validation
            all_objects: Original objects (unused now, but kept for compatibility)

        Returns:
            List[BinarySubquestion]: BinarySubquestion instances
        """
        binary_subquestions = []

        for question in subquestions:
            try:
                if not isinstance(question, str):
                    continue

                # Create BinarySubquestion instance (no type field anymore)
                binary_subquestion = BinarySubquestion(question=question.strip())
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

                # Validate basic content (non-empty)
                assert subquestion.question.strip()

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
            subquestion_type = subquestion.subquestion_type
            type_counts[subquestion_type] = type_counts.get(subquestion_type, 0) + 1

            # Track question lengths
            question_lengths.append(len(subquestion.question.split()))

        return {
            "count": len(subquestions),
            "types": type_counts,
            "avg_question_length": sum(question_lengths) / len(question_lengths) if question_lengths else 0,
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