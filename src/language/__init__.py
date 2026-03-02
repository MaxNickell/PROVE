from __future__ import annotations
import os


def create_llm_client(model_id: str | None = None, thinking_budget: int | None = None,
                      cot_enabled: bool = False):
    """Factory: return OpenAILLMClient or BedrockLLMClient based on model_id prefix."""
    _model_id = model_id or os.getenv("LLAMA33_MODEL_ID", "")
    if _model_id.startswith(("gpt-", "o1-", "o3-", "chatgpt-")):
        from .openai_llm_client import OpenAILLMClient
        return OpenAILLMClient(model_id=_model_id, thinking_budget=thinking_budget,
                               cot_enabled=cot_enabled)
    from .bedrock_llm_client import BedrockLLMClient
    return BedrockLLMClient(model_id=model_id, thinking_budget=thinking_budget,
                            cot_enabled=cot_enabled)
