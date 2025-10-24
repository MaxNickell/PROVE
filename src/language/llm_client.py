from __future__ import annotations
import os
import json
from typing import Any, List, Dict, Type, TypeVar
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .output_models import (
    SubquestionResponse,
    SubquestionItem,
    AttributePlanningResponse,
    CandidateResponse,
    CountRequirementResponse,
    SceneAttributeResponse,
    EntityExtractionResponse,
    ObjectDiscoveryResponse,
    ObjectPairDiscoveryResponse
)

T = TypeVar('T', bound=BaseModel)


class LLMClient:
    """GPT-4o client via Forge API (OpenAI-compatible)."""

    def __init__(self, model: str | None = None) -> None:
        """Initialize the GPT-4o client via Forge."""
        load_dotenv()

        # Model configuration from environment or defaults
        self.model_name = model or os.getenv("FORGE_MODEL_NAME", "Azure/gpt-4o")
        self.base_url = os.getenv("FORGE_BASE_URL", "https://api.forge.tensorblock.co/v1")
        self.api_key = os.getenv("FORGE_API_KEY")

        if not self.api_key:
            raise ValueError("FORGE_API_KEY not found in environment variables")

        # Initialize OpenAI client pointing to Forge
        print(f"Initializing GPT-4o via Forge API...")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        print(f"✓ GPT-4o client initialized successfully (model: {self.model_name})")

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Generate a response using GPT-4o via Forge.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

        Returns:
            Generated response text
        """
        try:
            # Extract parameters with defaults
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", 2048))

            # Call Forge API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Extract the generated text
            generated_text = response.choices[0].message.content
            return generated_text

        except Exception as e:
            raise RuntimeError(f"GPT-4o generation via Forge failed: {e}")

    def chat_with_validation(
        self,
        messages: List[Dict[str, str]],
        output_model: Type[T],
        max_retries: int = 3,
        **kwargs: Any
    ) -> T:
        """
        Generate a response with automatic JSON parsing and Pydantic validation.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            output_model: Pydantic model class for validation
            max_retries: Number of retries for malformed JSON
            **kwargs: Additional generation parameters

        Returns:
            Validated Pydantic model instance
        """
        for attempt in range(max_retries):
            try:
                # Generate response
                response = self.chat(messages, **kwargs)

                # Try to extract JSON from response
                json_str = self._extract_json(response)

                # Parse and validate with Pydantic
                parsed_json = json.loads(json_str)
                validated_output = output_model(**parsed_json)

                return validated_output

            except (json.JSONDecodeError, ValidationError) as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to get valid JSON after {max_retries} attempts. Last error: {e}")

                # Add JSON format instruction for retry
                if messages[-1]["role"] == "user":
                    messages[-1]["content"] += f"\n\nPlease respond with valid JSON only. Format: {output_model.schema()}"

        raise RuntimeError("Unexpected error in chat_with_validation")

    def _extract_json(self, response: str) -> str:
        """
        Extract JSON from response text, handling various formats.

        Args:
            response: Raw response from the model

        Returns:
            Extracted JSON string
        """
        # Try to find JSON in the response
        response = response.strip()

        # Check if entire response is JSON
        if response.startswith('{') and response.endswith('}'):
            return response

        # Look for JSON block markers
        json_markers = ['```json', '```JSON', '```']
        for marker in json_markers:
            if marker in response:
                parts = response.split(marker)
                if len(parts) >= 3:
                    # Extract content between markers
                    json_content = parts[1].strip()
                    if json_content.startswith('{') and json_content.endswith('}'):
                        return json_content

        # Try to find JSON-like content between braces
        start = response.find('{')
        if start != -1:
            brace_count = 0
            for i, char in enumerate(response[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return response[start:i+1]

        # If no JSON found, return the response as-is and let JSON parser handle the error
        return response

    # Convenience methods for specific pipeline components
    def generate_subquestions(self, messages: List[Dict[str, str]], **kwargs) -> SubquestionResponse:
        """Generate subquestions with validation."""
        return self.chat_with_validation(messages, SubquestionResponse, **kwargs)

    def plan_attributes(self, messages: List[Dict[str, str]], **kwargs) -> AttributePlanningResponse:
        """Plan attribute extraction requirements with validation."""
        return self.chat_with_validation(messages, AttributePlanningResponse, **kwargs)

    def generate_candidates(self, messages: List[Dict[str, str]], **kwargs) -> CandidateResponse:
        """Generate attribute value candidates with validation."""
        return self.chat_with_validation(messages, CandidateResponse, **kwargs)

    def analyze_count_requirements(self, messages: List[Dict[str, str]], **kwargs) -> CountRequirementResponse:
        """Analyze count requirements with validation."""
        return self.chat_with_validation(messages, CountRequirementResponse, **kwargs)

    def analyze_scene_attributes(self, messages: List[Dict[str, str]], **kwargs) -> SceneAttributeResponse:
        """Analyze scene attributes with validation."""
        return self.chat_with_validation(messages, SceneAttributeResponse, **kwargs)

    def extract_entities(self, messages: List[Dict[str, str]], **kwargs) -> EntityExtractionResponse:
        """Extract entities from image captions with validation."""
        return self.chat_with_validation(messages, EntityExtractionResponse, **kwargs)

    def discover_objects(self, messages: List[Dict[str, str]], **kwargs) -> ObjectDiscoveryResponse:
        """Discover relevant object IDs from natural language question."""
        return self.chat_with_validation(messages, ObjectDiscoveryResponse, temperature=0.2, **kwargs)

    def discover_object_pairs(self, messages: List[Dict[str, str]], **kwargs) -> ObjectPairDiscoveryResponse:
        """Discover relevant object pairs from natural language question."""
        return self.chat_with_validation(messages, ObjectPairDiscoveryResponse, temperature=0.2, **kwargs)
