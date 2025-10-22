#!/usr/bin/env python3
"""Test script to verify Forge API connection."""

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Get credentials from .env
api_key = os.getenv("FORGE_API_KEY")
base_url = os.getenv("FORGE_BASE_URL")

print(f"Testing Forge API connection...")
print(f"Base URL: {base_url}")
print(f"API Key: {api_key[:20]}..." if api_key else "API Key: NOT FOUND")
print()

# Initialize client
client = OpenAI(
    base_url=base_url,
    api_key=api_key
)

# First, try to list available models
print("Attempting to list available models...")
try:
    models = client.models.list()
    print("Available models:")
    for model in models.data[:10]:  # Show first 10
        print(f"  - {model.id}")
    print()
except Exception as e:
    print(f"Could not list models: {e}")
    print()

# Test models to try
test_models = [
    "Azure/gpt-4o",
    "OpenAI/gpt-4o",
    "gpt-4o",
    "azure/gpt-4o",  # Try lowercase
    "openai/gpt-4o"   # Try lowercase
]

test_messages = [
    {"role": "user", "content": "Say hello in JSON format: {\"message\": \"your_greeting\"}"}
]

for model_name in test_models:
    print(f"Testing model: {model_name}")
    print("-" * 50)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=test_messages,
            max_tokens=50,
            temperature=0
        )

        print(f"✓ SUCCESS!")
        print(f"Response: {response.choices[0].message.content}")
        print()

    except Exception as e:
        print(f"✗ FAILED: {e}")
        print()

print("=" * 50)
print("Testing complete. Use the model that worked above.")
