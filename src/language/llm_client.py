from __future__ import annotations
import os
import json
from typing import Any, List, Dict, Type, TypeVar
from dotenv import load_dotenv
import boto3
from pydantic import BaseModel, ValidationError

from .output_models import (
    EntityExtractionResponse,
    AgentAction,
    PerceiveAction,
    VerifyAttributeAction,
    VerifyRelationshipAction,
    VerifyCountAction,
    DoneAction
)

T = TypeVar('T', bound=BaseModel)


class LLMClient:
    """Llama 3.3 70B Instruct client via AWS Bedrock."""

    def __init__(self, model_id: str | None = None) -> None:
        """Initialize the Llama 3.3 client via AWS Bedrock."""
        load_dotenv()

        # AWS Bedrock configuration
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.model_id = model_id or os.getenv("LLAMA33_MODEL_ID")

        # Initialize Bedrock client
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.region
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Generate a response using Llama 3.3 via AWS Bedrock.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

        Returns:
            Generated response text
        """
        try:
            # Extract parameters with defaults
            temperature = kwargs.get("temperature", 0.0)
            max_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", 2048))

            # Separate system messages from conversation messages
            system_messages = []
            bedrock_messages = []

            for msg in messages:
                role = msg["role"]
                if role == "system":
                    # System messages go in separate parameter
                    system_messages.append({"text": msg["content"]})
                else:
                    # User and assistant messages go in messages array
                    bedrock_messages.append({
                        "role": role,
                        "content": [{"text": msg["content"]}]
                    })

            # Build converse API parameters
            converse_params = {
                "modelId": self.model_id,
                "messages": bedrock_messages,
                "inferenceConfig": {
                    "temperature": temperature,
                    "maxTokens": max_tokens
                }
            }

            # Add system messages if present
            if system_messages:
                converse_params["system"] = system_messages

            # Call AWS Bedrock Converse API
            response = self.client.converse(**converse_params)

            # Extract the generated text
            generated_text = response['output']['message']['content'][0]['text']
            return generated_text

        except Exception as e:
            raise RuntimeError(f"Llama 3.3 generation via AWS Bedrock failed: {e}")

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

            except (json.JSONDecodeError, ValidationError, TypeError) as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to get valid JSON after {max_retries} attempts. Last error: {e}")

                # Add JSON format instruction for retry (Llama 3.3 may need stronger hints)
                if messages[-1]["role"] == "user":
                    messages[-1]["content"] += (
                        "\n\nIMPORTANT: Respond with ONLY valid JSON, no other text. "
                        "Required format: " + str(output_model.model_json_schema())
                    )

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
    def extract_entities(self, messages: List[Dict[str, str]], **kwargs) -> EntityExtractionResponse:
        """Extract entities from questions with validation."""
        return self.chat_with_validation(messages, EntityExtractionResponse, **kwargs)

    def parse_agent_action(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
        **kwargs: Any
    ) -> AgentAction:
        """
        Parse agent action from LLM response using discriminated union.

        Determines action type from 'action' field and validates with appropriate model.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_retries: Number of retries for malformed JSON
            **kwargs: Additional generation parameters

        Returns:
            Validated action (PerceiveAction, VerifyAttributeAction, etc.)

        Raises:
            RuntimeError: If parsing fails after max_retries
        """
        action_models = {
            "perceive": PerceiveAction,
            "verify_attribute": VerifyAttributeAction,
            "verify_relationship": VerifyRelationshipAction,
            "verify_count": VerifyCountAction,
            "done": DoneAction
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                # Generate response
                response = self.chat(messages, **kwargs)

                # Extract JSON from response
                json_str = self._extract_json(response)

                # Parse JSON
                parsed_json = json.loads(json_str)

                # Determine action type
                action_type = parsed_json.get("action")
                if action_type not in action_models:
                    raise ValueError(f"Unknown action type: {action_type}. Must be one of: {list(action_models.keys())}")

                # Validate with appropriate model
                model_class = action_models[action_type]
                validated_action = model_class(**parsed_json)

                return validated_action

            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as e:
                last_error = e
                if attempt == max_retries - 1:
                    break

                # Add format hint for retry
                if messages[-1]["role"] == "user":
                    messages[-1]["content"] += (
                        "\n\nIMPORTANT: Respond with ONLY valid JSON. "
                        "The 'action' field must be one of: perceive, verify_attribute, verify_relationship, verify_count, done"
                    )

        raise RuntimeError(f"Failed to parse agent action after {max_retries} attempts. Last error: {last_error}")
