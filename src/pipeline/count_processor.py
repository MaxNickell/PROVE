"""
Count processor for PROVE pipeline.
Handles count subqueries using LLM class determination + Poisson-Binomial probabilistic counting.
Uses detected object probabilities to compute probabilistic count distributions.
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from src.core.model_manager import ModelManager
from src.core.types import BinarySubquestion, ObjectDetection, ImageData
from src.language.output_models import CountRequirementResponse


@dataclass
class CountRequirement:
    """Count requirement for a specific object class in an image."""
    image_id: str
    object_class: str
    required_for_subquestions: List[str]  # Which subqueries need this count


@dataclass
class CountResult:
    """Probabilistic count result for an object class."""
    image_id: str
    object_class: str
    distribution: Dict[int, float]  # Full distribution P(C=k) for k=0,1,2,...

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "object_class": self.object_class,
            "distribution": self.distribution
        }


class CountProcessorError(RuntimeError):
    """Custom exception for count processing failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


class CountProcessor:
    """
    Process count subqueries using LLM class determination + Poisson-Binomial counting.
    Each subquestion: LLM determines required classes → Poisson-Binomial counting → Store probabilistic results
    """

    def __init__(self):
        """Initialize processor with ModelManager singleton."""
        self.model_manager = ModelManager()

    def process_count_subquestions(
        self,
        count_subquestions: List[BinarySubquestion],
        images: Dict[str, ImageData]
    ) -> Dict[str, int]:
        """
        Process count subqueries using probabilistic counting.

        Args:
            count_subquestions: List of count binary subqueries
            images: ImageData structure containing detected objects per image

        Returns:
            Dict[str, int]: Count of count classes computed per image

        Raises:
            CountProcessorError: If processing fails
        """
        try:
            if not count_subquestions:
                return {image_id: 0 for image_id in images.keys()}

            # Initialize counts structure for all images
            for image_id, image_data in images.items():
                if not hasattr(image_data, 'counts') or image_data.counts is None:
                    image_data.counts = {}

            # Load LLM client
            llm_client = self.model_manager.get_llm_client()

            # Step 1: Determine required count classes from all count subqueries
            count_requirements = self._determine_count_requirements(
                llm_client, count_subquestions, images
            )

            print(f"  Determined {len(count_requirements)} count requirements from {len(count_subquestions)} subqueries")

            # Step 2: Compute Poisson-Binomial counts for each requirement
            total_counts_computed = 0
            counts_per_image = {}

            for requirement in count_requirements:
                count_result = self._compute_poisson_binomial_count(
                    requirement, images
                )

                if count_result:
                    # Store count result directly in ImageData
                    self._store_count_result(images, count_result)

                    total_counts_computed += 1
                    counts_per_image[requirement.image_id] = counts_per_image.get(requirement.image_id, 0) + 1

            print(f"  Computed {total_counts_computed} probabilistic counts")

            return counts_per_image

        except Exception as e:
            raise CountProcessorError(f"Failed to process count subquestions: {str(e)}")

    def _determine_count_requirements(
        self,
        llm_client,
        count_subquestions: List[BinarySubquestion],
        images: Dict[str, ImageData]
    ) -> List[CountRequirement]:
        """
        Determine which object classes need counting for which images based on subqueries.

        Args:
            llm_client: LLM client for analysis
            count_subquestions: List of count subqueries to analyze
            images: ImageData structure

        Returns:
            List[CountRequirement]: Required count computations
        """
        all_requirements = []

        for subquestion in count_subquestions:
            if subquestion.subquery_type != "count":
                continue

            # Analyze this subquery to determine count requirements
            requirements = self._analyze_single_count_subquery(
                llm_client, subquery, images
            )

            all_requirements.extend(requirements)

        return all_requirements

    def _analyze_single_count_subquery(
        self,
        llm_client,
        subquestion: BinarySubquestion,
        images: Dict[str, ImageData]
    ) -> List[CountRequirement]:
        """
        Analyze a single count subquery to determine required count computations.

        Args:
            llm_client: LLM client
            subquestion: Count subquery to analyze
            images: ImageData structure

        Returns:
            List[CountRequirement]: Count requirements for this subquery
        """
        try:
            # Build context about available images and their objects
            available_images = []
            for image_id, image_data in images.items():
                object_classes = list(set(obj.label for obj in image_data.objects))
                available_images.append(f"{image_id}: {object_classes}")

            images_context = "\\n".join(available_images)

            prompt = f"""Analyze this count subquery to determine what object classes need counting in which images.

Subquery: "{subquestion.question}"
Type: {subquestion.subquery_type}

Available Images and Object Classes:
{images_context}

Task: Determine which object classes need to be counted in which images to answer this specific question.

Examples:
- "Are there more than 2 cattle in IMAGE_A?" → Need cattle count for IMAGE_A
- "Does IMAGE_A have more birds than IMAGE_B?" → Need bird count for both IMAGE_A and IMAGE_B
- "Are there more cattle in IMAGE_A than birds in IMAGE_B?" → Need cattle count for IMAGE_A, bird count for IMAGE_B

Return JSON with this EXACT format:
{{
  "count_requirements": [
    {{
      "image_id": "image_a",
      "object_class": "cattle"
    }},
    {{
      "image_id": "image_b",
      "object_class": "bird"
    }}
  ]
}}

Generate count requirements for: "{subquestion.question}"."""

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at analyzing count questions to determine what needs to be counted. Return strict JSON only in the required format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            # Use Pydantic validation for guaranteed structure
            response = llm_client.analyze_count_requirements(messages, temperature=0.2)

            # Convert Pydantic response to CountRequirement objects
            requirements = []
            for req_item in response.count_requirements:
                if req_item.image_id in images:
                    requirements.append(CountRequirement(
                        image_id=req_item.image_id,
                        object_class=req_item.object_class,
                        required_for_subqueries=[subquestion.question]
                    ))

            return requirements

        except Exception as e:
            print(f"    Warning: Failed to analyze count subquery '{subquestion.question}': {e}")
            return []

    def _compute_poisson_binomial_count(
        self,
        requirement: CountRequirement,
        images: Dict[str, ImageData]
    ) -> CountResult:
        """
        Compute Poisson-Binomial count distribution for an object class in an image.

        Uses your specified DP algorithm:
        1. Filter detections by target class
        2. Extract probabilities [p1, p2, ..., pn]
        3. Dynamic Programming convolution
        4. Return distribution P(C=k) for k=0,1,2,...,n

        Args:
            requirement: Count requirement specifying image_id and object_class
            images: ImageData structure

        Returns:
            CountResult: Probabilistic count result
        """
        try:
            image_id = requirement.image_id
            target_class = requirement.object_class

            if image_id not in images:
                return None

            # Step 1: Filter detections by target class and extract probabilities
            detections = images[image_id].objects
            relevant_probabilities = []

            for detection in detections:
                if detection.label == target_class:
                    relevant_probabilities.append(detection.confidence)

            print(f"    Computing count for {target_class} in {image_id}: {len(relevant_probabilities)} detections")

            # Step 2: Handle edge cases
            if not relevant_probabilities:
                # No detections of this class
                return CountResult(
                    image_id=image_id,
                    object_class=target_class,
                    distribution={0: 1.0}
                )

            # Step 3: Compute Poisson-Binomial distribution using Dynamic Programming
            distribution = self._compute_poisson_binomial_distribution(relevant_probabilities)

            return CountResult(
                image_id=image_id,
                object_class=target_class,
                distribution=distribution
            )

        except Exception as e:
            print(f"    Warning: Failed to compute count for {requirement.object_class} in {requirement.image_id}: {e}")
            return None

    def _compute_poisson_binomial_distribution(self, probabilities: List[float]) -> Dict[int, float]:
        """
        Compute Poisson-Binomial distribution using Dynamic Programming.

        Your specified algorithm:
        - Initialize: P = [1.0] (meaning 0 objects with probability 1)
        - For each probability p: Convolve P with [1-p, p]
        - Result: P[k] = probability of exactly k objects

        Args:
            probabilities: List of detection probabilities [p1, p2, ..., pn]

        Returns:
            Dict[int, float]: Distribution {k: P(C=k)} for k=0,1,2,...,n
        """
        # Initialize distribution: P(0 objects) = 1.0
        P = [1.0]

        # For each detection probability, convolve with [1-p, p]
        for p in probabilities:
            new_P = [0.0] * (len(P) + 1)

            # Convolve current distribution with [1-p, p]
            for k in range(len(P)):
                # P(k objects) contributes to:
                # - P(k objects) with probability (1-p) [object not counted]
                # - P(k+1 objects) with probability p [object counted]
                new_P[k] += P[k] * (1 - p)
                new_P[k + 1] += P[k] * p

            P = new_P

        # Convert to dictionary format
        distribution = {}
        for k in range(len(P)):
            distribution[k] = P[k]

        return distribution

    def _store_count_result(
        self,
        images: Dict[str, ImageData],
        count_result: CountResult
    ) -> None:
        """
        Store count result directly in ImageData counts structure.

        Format: counts[object_class] = {"distribution": {"0": p0, "1": p1, "2": p2, ...}}
        """
        try:
            image_id = count_result.image_id

            if image_id in images:
                # Convert integer keys to strings for JSON compatibility
                distribution_str_keys = {str(k): v for k, v in count_result.distribution.items()}

                images[image_id].counts[count_result.object_class] = {
                    "distribution": distribution_str_keys
                }

        except Exception as e:
            print(f"    Warning: Failed to store count result: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Test count processor
    processor = CountProcessor()

    # Sample data would go here for testing
    print("✓ Count processor ready for probabilistic counting!")