from __future__ import annotations
import os
from typing import Any, List, Dict
from dotenv import load_dotenv
from openai import OpenAI


class LLMClient:
    def __init__(self, model: str | None = None) -> None:
        load_dotenv()
        self.model = os.environ["FORGE_MODEL_NAME"]
        self.client = OpenAI(
            base_url=os.environ["FORGE_BASE_URL"],
            api_key=os.environ["FORGE_KEY"],
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return resp.choices[0].message.content
