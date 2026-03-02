from __future__ import annotations
import os
import json
import re
import time
from typing import Any, List, Dict, Type, TypeVar
from dotenv import load_dotenv
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


class OpenAILLMClient:
    """LLM client via OpenAI API (supports GPT-4o, GPT-4o-mini, etc.)."""

    def __init__(self, model_id: str | None = None, thinking_budget: int | None = None,
                 cot_enabled: bool = False) -> None:
        """Initialize LLM client via OpenAI API.

        Args:
            model_id: OpenAI model name (e.g. 'gpt-4o', 'gpt-4o-mini')
            thinking_budget: Ignored (not applicable to OpenAI models)
            cot_enabled: Enable prompt-level chain-of-thought
        """
        load_dotenv()

        from openai import OpenAI

        self.model_id = model_id or os.getenv("LLAMA33_MODEL_ID")
        self.thinking_budget = None  # not supported
        self.cot_enabled = cot_enabled

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=120.0,
            max_retries=0,  # we handle retries ourselves
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Generate a response using OpenAI API.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

        Returns:
            Generated response text
        """
        try:
            temperature = kwargs.get("temperature", 0.0)
            max_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", 2048))

            # OpenAI accepts messages as-is (role + content dicts)
            openai_messages = [{"role": msg["role"], "content": msg["content"]}
                               for msg in messages]

            # Prompt-level CoT: append reasoning instruction to system messages
            if self.cot_enabled:
                cot_instruction = (
                    "\n\nBefore providing your final answer, think step by step about the problem. "
                    "Write your reasoning first, then provide your final answer."
                )
                # Append to last system message, or insert one
                appended = False
                for i in range(len(openai_messages) - 1, -1, -1):
                    if openai_messages[i]["role"] == "system":
                        openai_messages[i]["content"] += cot_instruction
                        appended = True
                        break
                if not appended:
                    openai_messages.insert(0, {"role": "system",
                                               "content": cot_instruction.strip()})

            # Call OpenAI API with retry on transient errors
            _TRANSIENT_KEYWORDS = (
                'rate_limit', 'Rate limit', 'RateLimitError',
                'timeout', 'timed out', 'APITimeoutError',
                'server_error', 'ServiceUnavailable', 'overloaded',
                'APIConnectionError', 'InternalServerError',
            )
            max_api_retries = 5
            for attempt in range(max_api_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_id,
                        messages=openai_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content
                except Exception as api_err:
                    err_str = str(api_err)
                    is_transient = any(kw in err_str for kw in _TRANSIENT_KEYWORDS)
                    if is_transient and attempt < max_api_retries - 1:
                        wait = 2 ** attempt  # 1, 2, 4, 8s
                        print(f"  Warning: OpenAI API transient error (attempt {attempt+1}/{max_api_retries}), "
                              f"retrying in {wait}s: {err_str[:120]}")
                        time.sleep(wait)
                        continue
                    raise  # non-transient or final attempt

            raise RuntimeError("OpenAI API: max retries exceeded")

        except Exception as e:
            raise RuntimeError(f"LLM generation via OpenAI failed: {e}")

    def chat_with_validation(
        self,
        messages: List[Dict[str, str]],
        output_model: Type[T],
        max_retries: int = 3,
        **kwargs: Any
    ) -> T:
        """Generate a response with automatic JSON parsing and Pydantic validation.

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
                response = self.chat(messages, **kwargs)
                json_str = self._extract_json(response)

                try:
                    parsed_json = json.loads(json_str)
                except json.JSONDecodeError:
                    fixed = json_str.replace("'", '"')
                    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
                    parsed_json = json.loads(fixed)

                validated_output = output_model(**parsed_json)
                return validated_output

            except (json.JSONDecodeError, ValidationError, TypeError) as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to get valid JSON after {max_retries} attempts. Last error: {e}")

                if messages[-1]["role"] == "user":
                    messages[-1]["content"] += (
                        '\n\nIMPORTANT: Respond with ONLY a valid JSON object, no other text. '
                        'Do NOT output a schema definition. '
                        'Example: {"entities": ["dog", "cat"]}'
                    )

        raise RuntimeError("Unexpected error in chat_with_validation")

    def _extract_json(self, response: str) -> str:
        """Extract JSON from response text, handling various formats."""
        response = response.strip()

        if response.startswith('{') and response.endswith('}'):
            return response

        json_markers = ['```json', '```JSON', '```']
        for marker in json_markers:
            if marker in response:
                parts = response.split(marker)
                if len(parts) >= 3:
                    json_content = parts[1].strip()
                    if json_content.startswith('{') and json_content.endswith('}'):
                        return json_content

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

        return response

    def extract_entities(self, messages: List[Dict[str, str]], **kwargs) -> EntityExtractionResponse:
        """Extract entities from questions with validation."""
        return self.chat_with_validation(messages, EntityExtractionResponse, **kwargs)

    def parse_agent_action(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
        **kwargs: Any
    ) -> AgentAction:
        """Parse agent action from LLM response using discriminated union."""
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
                response = self.chat(messages, **kwargs)
                json_str = self._extract_json(response)
                parsed_json = json.loads(json_str)

                action_type = parsed_json.get("action")
                if action_type not in action_models:
                    raise ValueError(f"Unknown action type: {action_type}. Must be one of: {list(action_models.keys())}")

                model_class = action_models[action_type]
                validated_action = model_class(**parsed_json)
                return validated_action

            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as e:
                last_error = e
                if attempt == max_retries - 1:
                    break

                if messages[-1]["role"] == "user":
                    messages[-1]["content"] += (
                        "\n\nIMPORTANT: Respond with ONLY valid JSON. "
                        "The 'action' field must be one of: perceive, verify_attribute, verify_relationship, verify_count, done"
                    )

        raise RuntimeError(f"Failed to parse agent action after {max_retries} attempts. Last error: {last_error}")
