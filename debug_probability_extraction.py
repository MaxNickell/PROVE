#!/usr/bin/env python3
"""
Debug script for analyzing probability extraction from VLM responses.
Tests verbalizer token mapping and logit extraction to identify why "No" responses
are returning high probabilities instead of low ones.
"""

import torch
import sys
import os
from PIL import Image
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

from src.core.model_manager import ModelManager
from src.core.probability import get_verifier_probability


def test_verbalizer_tokenization():
    """Test how verbalizers are tokenized and what token IDs they get."""
    print("🔍 VERBALIZER TOKENIZATION ANALYSIS")
    print("=" * 80)

    # Load model and tokenizer
    model_manager = ModelManager()
    qwen_client = model_manager.get_qwen_vl()
    tokenizer = qwen_client.processor.tokenizer

    # Test all verbalizers
    yes_verbalizers = ["Yes", "yes", "YES"]
    no_verbalizers = ["No", "no", "NO"]

    print("\n🟢 YES VERBALIZERS:")
    yes_token_ids = set()
    for verbalizer in yes_verbalizers:
        token_ids = tokenizer.encode(verbalizer, add_special_tokens=False)
        print(f"  '{verbalizer}' → token_ids: {token_ids}")
        yes_token_ids.update(token_ids)

    print(f"\n  All Yes token IDs: {sorted(yes_token_ids)}")

    print("\n🔴 NO VERBALIZERS:")
    no_token_ids = set()
    for verbalizer in no_verbalizers:
        token_ids = tokenizer.encode(verbalizer, add_special_tokens=False)
        print(f"  '{verbalizer}' → token_ids: {token_ids}")
        no_token_ids.update(token_ids)

    print(f"\n  All No token IDs: {sorted(no_token_ids)}")

    # Check for overlaps
    overlap = yes_token_ids & no_token_ids
    if overlap:
        print(f"\n⚠️  TOKEN ID OVERLAP DETECTED: {overlap}")
        print("   This could cause probability extraction issues!")
    else:
        print(f"\n✅ No token ID overlaps found")

    print("\n" + "=" * 80)
    return tokenizer, yes_token_ids, no_token_ids


def create_mock_logits(vocab_size, yes_token_ids, no_token_ids, response_type="yes"):
    """Create mock logits favoring yes or no tokens."""
    logits = torch.randn(vocab_size) * 0.1  # Small random noise

    if response_type == "yes":
        # Make yes tokens have higher logits
        for token_id in yes_token_ids:
            if token_id < vocab_size:
                logits[token_id] = 5.0 + torch.randn(1) * 0.1
        for token_id in no_token_ids:
            if token_id < vocab_size:
                logits[token_id] = -2.0 + torch.randn(1) * 0.1
    else:  # response_type == "no"
        # Make no tokens have higher logits
        for token_id in no_token_ids:
            if token_id < vocab_size:
                logits[token_id] = 5.0 + torch.randn(1) * 0.1
        for token_id in yes_token_ids:
            if token_id < vocab_size:
                logits[token_id] = -2.0 + torch.randn(1) * 0.1

    return [logits.unsqueeze(0)]  # Add batch dimension to match expected format


def test_mock_probability_extraction(tokenizer, yes_token_ids, no_token_ids):
    """Test probability extraction with controlled mock logits."""
    print("\n🧪 MOCK PROBABILITY EXTRACTION TESTS")
    print("=" * 80)

    vocab_size = tokenizer.vocab_size

    # Test Case 1: Mock "Yes" response
    print("\n📊 Test Case 1: Mock 'Yes' Response")
    print("-" * 40)
    mock_yes_logits = create_mock_logits(vocab_size, yes_token_ids, no_token_ids, "yes")
    prob = get_verifier_probability(mock_yes_logits, "Yes", tokenizer, debug=True)
    print(f"Final probability: {prob:.4f}")
    print(f"Expected: HIGH (>0.7) for Yes response")
    print(f"Result: {'✅ CORRECT' if prob > 0.7 else '❌ INCORRECT'}")

    # Test Case 2: Mock "No" response
    print("\n📊 Test Case 2: Mock 'No' Response")
    print("-" * 40)
    mock_no_logits = create_mock_logits(vocab_size, yes_token_ids, no_token_ids, "no")
    prob = get_verifier_probability(mock_no_logits, "No", tokenizer, debug=True)
    print(f"Final probability: {prob:.4f}")
    print(f"Expected: LOW (<0.3) for No response")
    print(f"Result: {'✅ CORRECT' if prob < 0.3 else '❌ INCORRECT'}")

    print("\n" + "=" * 80)


def test_real_vl_inference(model_manager):
    """Test with real VLM inference on a simple image."""
    print("\n🖼️  REAL VLM INFERENCE TEST")
    print("=" * 80)

    try:
        # Create a simple test image (solid color)
        test_image = Image.new('RGB', (224, 224), color='red')

        qwen_client = model_manager.get_qwen_vl()
        tokenizer = qwen_client.processor.tokenizer

        # Test questions that should give clear Yes/No answers
        test_cases = [
            ("Is this image red?", "Expected: Yes → HIGH probability"),
            ("Is this image blue?", "Expected: No → LOW probability"),
            ("Is this a solid color?", "Expected: Yes → HIGH probability"),
            ("Does this contain text?", "Expected: No → LOW probability")
        ]

        for question, expectation in test_cases:
            print(f"\n📝 Testing: '{question}'")
            print(f"   {expectation}")

            # Create binary prompt with strict formatting
            prompt = f"""{question}

Respond with ONLY "Yes" or "No". Do not add punctuation or explanation.

Answer:"""

            try:
                # Get response with logits
                response, logits = qwen_client.run_inference_with_logits(test_image, prompt)

                # DEBUG: Show all generated tokens
                generated_tokens = tokenizer.encode(response.strip(), add_special_tokens=False)
                token_texts = [tokenizer.decode([tok]) for tok in generated_tokens]
                print(f"   → Generated tokens: {generated_tokens}")
                print(f"   → Token texts: {token_texts}")
                print(f"   → Number of logit steps: {len(logits)}")

                # Extract probability with debug
                prob = get_verifier_probability(logits, response, tokenizer, debug=True)

                print(f"   → VLM Response: '{response.strip()}'")
                print(f"   → Extracted Probability: {prob:.4f}")

                # Check if result matches expectation
                response_lower = response.strip().lower()
                if "yes" in response_lower:
                    result = "✅ CORRECT" if prob > 0.5 else "❌ INCORRECT - Yes should be >0.5"
                elif "no" in response_lower:
                    result = "✅ CORRECT" if prob < 0.5 else "❌ INCORRECT - No should be <0.5"
                else:
                    result = "⚠️  UNCLEAR RESPONSE"

                print(f"   → Result: {result}")

            except Exception as e:
                print(f"   → Error: {e}")

            print("-" * 60)

    except Exception as e:
        print(f"Real VLM test failed: {e}")
        print("This might be expected if no test image is available")


def main():
    """Main debug script execution."""
    print("🐛 PROBABILITY EXTRACTION DEBUG SCRIPT")
    print("=" * 80)
    print("Investigating why 'No' responses return high probabilities")
    print("=" * 80)

    try:
        # Step 1: Test verbalizer tokenization
        tokenizer, yes_token_ids, no_token_ids = test_verbalizer_tokenization()

        # Step 2: Test with mock controlled logits
        test_mock_probability_extraction(tokenizer, yes_token_ids, no_token_ids)

        # Step 3: Test with real VLM inference (if possible)
        model_manager = ModelManager()
        test_real_vl_inference(model_manager)

        print("\n🎯 DEBUG COMPLETE")
        print("=" * 80)
        print("Review the debug output above to identify:")
        print("1. Token ID overlaps between Yes/No verbalizers")
        print("2. Incorrect logit extraction")
        print("3. Probability calculation errors")
        print("4. Real VLM response patterns")

    except Exception as e:
        print(f"Debug script failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()