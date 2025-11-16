"""
Base agent class with shared verification methods.
Eliminates code duplication across attribute, relationship, and scene attribute agents.
"""

from typing import Dict, List, TypeVar, Generic
import json
import re

from src.core.model_manager import ModelManager


# Generic type for binary question types
BinaryQuestionT = TypeVar('BinaryQuestionT')


class BaseVerificationAgent(Generic[BinaryQuestionT]):
    """
    Base class for agents that perform subquestion-aware verification.

    Provides shared methods for:
    - Extracting target claims from subquestions
    - Validating binary questions cover target claims
    - Generating fallback questions for missing claims
    """

    def __init__(self):
        """Initialize with ModelManager."""
        self.model_manager = ModelManager()

    def _extract_target_claims_generic(
        self,
        subquestion: str,
        claim_type: str,
        claim_key: str,
        examples: List[str]
    ) -> Dict[str, List[str]]:
        """
        Generic target claim extraction using LLM.

        Args:
            subquestion: Natural language subquestion
            claim_type: Type description (e.g., "attribute values", "relationship types")
            claim_key: JSON key for claims (e.g., "attribute_values", "relations")
            examples: List of example strings for the prompt

        Returns:
            Dict with claim_key containing list of values to verify
        """
        llm_client = self.model_manager.get_llm_client()

        examples_text = "\n".join(f"- {ex}" for ex in examples)

        prompt = f"""Extract the specific {claim_type} that need to be verified from this question.

Question: "{subquestion}"

Instructions:
1. Identify what {claim_type} are being asked about
2. Extract ONLY the specific values that need to be verified
3. Include ALL alternative values if it's an "or" question
4. Use appropriate format (lowercase, underscores for multi-word)

Examples:
{examples_text}

Output ONLY valid JSON in this exact format:
{{"{claim_key}": ["value1", "value2", ...]}}"""

        try:
            response = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.1)

            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return parsed if claim_key in parsed else {claim_key: []}

            return {claim_key: []}
        except Exception as e:
            print(f"⚠ Warning: Could not extract target claims from '{subquestion}': {e}")
            return {claim_key: []}

    def _validate_binary_questions_generic(
        self,
        binary_questions: List[BinaryQuestionT],
        target_claims: Dict[str, List[str]],
        claim_key: str,
        value_extractor,
        subquestion: str
    ) -> bool:
        """
        Generic validation that binary questions cover target claims.

        Args:
            binary_questions: Generated binary questions
            target_claims: Target claims extracted from subquestion
            claim_key: Key to look up in target_claims
            value_extractor: Function to extract value from binary question
            subquestion: Original subquestion

        Returns:
            bool: True if all target claims are covered
        """
        if not target_claims.get(claim_key):
            return True  # No specific targets to validate

        target_values = set(v.lower().replace(" ", "_") for v in target_claims[claim_key])
        generated_values = set(
            value_extractor(q).lower().replace(" ", "_")
            for q in binary_questions
        )

        missing = target_values - generated_values

        if missing:
            print(f"⚠ Warning: Binary questions don't cover all target claims!")
            print(f"  Subquestion: {subquestion}")
            print(f"  Missing values: {missing}")
            print(f"  Generated values: {generated_values}")
            return False

        return True

    def _get_missing_claims(
        self,
        target_claims: Dict[str, List[str]],
        claim_key: str,
        existing_questions: List[BinaryQuestionT],
        value_extractor
    ) -> List[str]:
        """
        Get list of missing claim values that need fallback questions.

        Args:
            target_claims: Target claims to verify
            claim_key: Key in target_claims dict
            existing_questions: Already generated questions
            value_extractor: Function to extract value from binary question

        Returns:
            List of missing claim values
        """
        if not target_claims.get(claim_key):
            return []

        existing_values = set(
            value_extractor(q).lower().replace(" ", "_")
            for q in existing_questions
        )

        missing_values = [
            v for v in target_claims[claim_key]
            if v.lower().replace(" ", "_") not in existing_values
        ]

        return missing_values
