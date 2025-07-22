import json, os, re
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI


class ForgeLLM:
    DEFAULT_MODEL = "OpenAI/gpt-4o"

    def __init__(self, model: str | None = None) -> None:
        load_dotenv()
        base_url = os.environ["FORGE_BASE_URL"]
        api_key = os.environ["FORGE_KEY"]

        self.model = model or self.DEFAULT_MODEL
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return resp.choices[0].message.content

    @staticmethod
    def _strip_md_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if m:
                return m.group(1).strip()
        return text

    def _parse_caption(self, caption: str) -> List[Dict[str, Any]]:
        system_prompt = """
            You are an expert at parsing image descriptions that contain object locations. Extract objects and their bounding boxes from descriptions where objects are followed by [[x1,y1,x2,y2]] coordinate tags.
            IMPORTANT RULES:
            1. Extract the main object/noun that the coordinates refer to
            2. Keep compound objects together (like "coffee table", "traffic light", "kitchen sink")
            3. Remove unnecessary adjectives but keep descriptive parts that identify the object
            4. Return clean, specific object names
            5. Return ONLY valid JSON - no markdown, no explanations

            Return format: {"items":[{"object":"name","bbox":[x1,y1,x2,y2]}]}

            EXAMPLES:

            Input: "A red sports car[[120,300,450,600]] is parked next to a tall oak tree[[500,50,700,580]] while a small dog[[300,520,380,600]] runs nearby."
            Output: {"items":[{"object":"sports car","bbox":[120,300,450,600]},{"object":"oak tree","bbox":[500,50,700,580]},{"object":"dog","bbox":[300,520,380,600]}]}

            Input: "The woman is sitting on a wooden bench[[200,400,800,650]] holding a coffee cup[[350,250,420,350]] with her purse[[150,480,250,580]] beside her."
            Output: {"items":[{"object":"wooden bench","bbox":[200,400,800,650]},{"object":"coffee cup","bbox":[350,250,420,350]},{"object":"purse","bbox":[150,480,250,580]}]}

            Input: "A blue bicycle[[100,200,400,700]] leans against a brick wall[[0,0,999,500]] next to a street lamp[[450,80,550,500]] and a trash can[[600,450,700,650]]."
            Output: {"items":[{"object":"bicycle","bbox":[100,200,400,700]},{"object":"brick wall","bbox":[0,0,999,500]},{"object":"street lamp","bbox":[450,80,550,500]},{"object":"trash can","bbox":[600,450,700,650]}]}

            Input: "The chef is preparing food on a stainless steel counter[[0,600,999,999]] using a sharp knife[[300,400,380,450]] while ingredients sit in a glass bowl[[200,350,350,450]]."
            Output: {"items":[{"object":"counter","bbox":[0,600,999,999]},{"object":"knife","bbox":[300,400,380,450]},{"object":"glass bowl","bbox":[200,350,350,450]}]}"""

        user_prompt = f"Parse this image description:\n\n{caption}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            raw_response = self.chat(messages, temperature=0)
            raw_response = self._strip_md_fences(raw_response)
            
            parsed = json.loads(raw_response)
            return parsed.get("items", [])
            
        except Exception as e:
            print(f"Error parsing with LLM: {e}")
            return self._parse_caption_regex_fallback(caption)

    def _strip_md_fences(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return text
    
    def _parse_caption_regex_fallback(self, caption: str) -> List[Dict[str, Any]]:
        pattern = r'([a-zA-Z\s]+)\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]'
        matches = re.findall(pattern, caption)
        
        items = []
        for text, x1, y1, x2, y2 in matches:
            words = text.strip().split()
            if not words:
                object_name = "object"
            elif len(words) == 1:
                object_name = words[0]
            else:
                object_name = " ".join(words[-2:])
            
            items.append({
                "object": object_name,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
        
        return items