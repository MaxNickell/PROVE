from __future__ import annotations
import os
import json
from typing import Any, List, Dict, Type, TypeVar
from dotenv import load_dotenv
import transformers
import torch
from pydantic import BaseModel, ValidationError

from .output_models import (
    SubqueryResponse, 
    SubqueryItem,
    AttributeResponse, 
    VerificationResponse, 
    RelationshipResponse, 
    ContextResponse,
    AttributePlanningResponse,
    CandidateResponse
)

T = TypeVar('T', bound=BaseModel)


class LLMClient:
    """Local Llama-3.3-70B-Instruct client using Transformers pipeline."""
    
    def __init__(self, model: str | None = None, device_map: dict | None = None) -> None:
        """Initialize the Llama model pipeline."""
        load_dotenv()
        
        # Model configuration from environment or defaults
        self.model_id = model or os.getenv("MODEL_ID", "meta-llama/Llama-3.3-70B-Instruct")
        self.cache_dir = os.getenv("MODEL_CACHE_DIR", "./models")
        self.max_tokens = int(os.getenv("MAX_NEW_TOKENS", "2048"))
        self.device_map = device_map or "auto"
        
        # Initialize the pipeline
        self._initialize_pipeline()
    
    def _initialize_pipeline(self) -> None:
        """Initialize the Transformers pipeline with optimized settings."""
        print(f"Loading Llama model: {self.model_id}")

        # Check if 8-bit quantization is enabled
        use_8bit = os.getenv("USE_8BIT_QUANTIZATION", "false").lower() == "true"

        # Prepare model kwargs
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "cache_dir": self.cache_dir
        }

        # Add quantization config if enabled
        if use_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["quantization_config"] = quantization_config
            print("✓ Using 8-bit quantization for memory efficiency")

        try:
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs=model_kwargs,
                device_map=self.device_map,
                token=True  # Use HuggingFace authentication (updated parameter)
            )

            quantization_status = "with 8-bit quantization" if use_8bit else "with 16-bit precision"
            print(f"✓ Llama model loaded successfully {quantization_status}")

        except Exception as e:
            print(f"❌ Failed to load Llama model: {e}")
            print("Falling back to CPU inference...")

            # Fallback to CPU if GPU fails
            fallback_kwargs = {
                "torch_dtype": torch.float32,
                "cache_dir": self.cache_dir
            }

            self.pipeline = transformers.pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs=fallback_kwargs,
                device_map="cpu",
                token=True
            )
            print("✓ Llama model loaded on CPU")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Generate a response using the Llama model.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional generation parameters

        Returns:
            Generated response text
        """
        # Set pipeline-compatible parameters (no temperature for transformers pipeline)
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_tokens),
            "do_sample": kwargs.get("do_sample", True),
            "pad_token_id": self.pipeline.tokenizer.eos_token_id
        }

        # Add temperature if using text-generation-inference or compatible backend
        # Note: Transformers pipeline doesn't support temperature in generate()

        try:
            # Generate response using the pipeline
            outputs = self.pipeline(
                messages,
                **generation_kwargs
            )
            
            # Extract the generated text (last message in conversation)
            generated_text = outputs[0]["generated_text"][-1]["content"]
            return generated_text
            
        except Exception as e:
            raise RuntimeError(f"Llama generation failed: {e}")
    
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
    def generate_subqueries(self, messages: List[Dict[str, str]], **kwargs) -> SubqueryResponse:
        """Generate subqueries with validation."""
        return self.chat_with_validation(messages, SubqueryResponse, **kwargs)
    
    def extract_attributes(self, messages: List[Dict[str, str]], **kwargs) -> AttributeResponse:
        """Extract attributes with validation."""
        return self.chat_with_validation(messages, AttributeResponse, **kwargs)
    
    def verify_binary(self, messages: List[Dict[str, str]], **kwargs) -> VerificationResponse:
        """Perform binary verification with validation."""
        return self.chat_with_validation(messages, VerificationResponse, **kwargs)
    
    def extract_relationships(self, messages: List[Dict[str, str]], **kwargs) -> RelationshipResponse:
        """Extract relationships with validation."""
        return self.chat_with_validation(messages, RelationshipResponse, **kwargs)
    
    def process_context(self, messages: List[Dict[str, str]], **kwargs) -> ContextResponse:
        """Process scene context with validation."""
        return self.chat_with_validation(messages, ContextResponse, **kwargs)
    
    def plan_attributes(self, messages: List[Dict[str, str]], **kwargs) -> AttributePlanningResponse:
        """Plan attribute extraction requirements with validation."""
        return self.chat_with_validation(messages, AttributePlanningResponse, **kwargs)
    
    def generate_candidates(self, messages: List[Dict[str, str]], **kwargs) -> CandidateResponse:
        """Generate attribute value candidates with validation."""
        return self.chat_with_validation(messages, CandidateResponse, **kwargs)