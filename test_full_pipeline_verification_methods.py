#!/usr/bin/env python3
"""
Full PROVE Pipeline Test with Different Verification Methods.

This test runs the complete PROVE pipeline (detection, agentic evidence collection,
ProbLog reasoning) with different verification methods for relational/attribute queries:

1. ITM (BLIP-ITM) - Current default
2. VLM Yes/No (Qwen2-VL with Yes/No prompting)
3. VLM True/False (Qwen2-VL with True/False prompting)

Compares PROVE (probabilistic) vs DePROVE (deterministic) accuracy for each method.
"""

import os
import json
import math
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, Any, List, Union
import torch
import gc


class VLMVerifier:
    """VLM-based verifier that can use Yes/No or True/False prompting."""

    def __init__(self, prompt_style: str = "yes_no", device: str = "auto"):
        self.prompt_style = prompt_style
        self._model = None
        self._processor = None

        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self._load_model()

    def _load_model(self):
        if self._model is not None:
            return

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        print(f"Loading Qwen2-VL-2B for {self.prompt_style} verification...")
        model_name = "Qwen/Qwen2-VL-2B-Instruct"

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        )

        if self.device == "cuda":
            self._model = self._model.cuda()
        elif self.device == "mps":
            self._model = self._model.to("mps")

        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model.eval()
        print(f"  VLM loaded on {self.device}")

    def _get_article(self, word: str) -> str:
        vowels = ('a', 'e', 'i', 'o', 'u')
        return "an" if word.lower().startswith(vowels) else "a"

    def _crop_with_padding(self, image: Image.Image, bbox: List[float], padding_ratio: float = 0.15) -> Image.Image:
        x1, y1, x2, y2 = [float(c) for c in bbox]
        width = x2 - x1
        height = y2 - y1
        pad_x = width * padding_ratio
        pad_y = height * padding_ratio

        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(image.width, int(x2 + pad_x))
        y2 = min(image.height, int(y2 + pad_y))

        return image.crop((x1, y1, x2, y2))

    def _get_vlm_probability(self, image: Image.Image, statement: str) -> float:
        """Get probability from VLM using constrained token extraction."""
        if self.prompt_style == "true_false":
            prompt = f'Determine if the following statement about this image is true or false.\n\nStatement: "{statement}"\n\nAnswer with ONLY "True" or "False".'
            pos_tokens = ["True", "true", "TRUE"]
            neg_tokens = ["False", "false", "FALSE"]
        else:  # yes_no
            prompt = f'Is the following statement about this image correct?\n\nStatement: "{statement}"\n\nAnswer with ONLY "Yes" or "No".'
            pos_tokens = ["Yes", "yes", "YES"]
            neg_tokens = ["No", "no", "NO"]

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], padding=True, return_tensors="pt")

        for k, v in inputs.items():
            if hasattr(v, 'to'):
                inputs[k] = v.to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=10, temperature=0.0,
                do_sample=False, return_dict_in_generate=True,
                output_scores=True, pad_token_id=self._processor.tokenizer.eos_token_id
            )

        return self._extract_probability(outputs.scores, pos_tokens, neg_tokens)

    def _extract_probability(self, scores, pos_tokens, neg_tokens) -> float:
        if not scores:
            return 0.5
        try:
            logits = scores[0][0]

            pos_ids = []
            neg_ids = []
            for v in pos_tokens:
                pos_ids.extend(self._processor.tokenizer.encode(v, add_special_tokens=False))
            for v in neg_tokens:
                neg_ids.extend(self._processor.tokenizer.encode(v, add_special_tokens=False))

            pos_logits = [logits[i].item() for i in pos_ids if i < len(logits)]
            neg_logits = [logits[i].item() for i in neg_ids if i < len(logits)]

            if not pos_logits or not neg_logits:
                return 0.5

            pos_max, neg_max = max(pos_logits), max(neg_logits)
            exp_pos = math.exp(pos_max - max(pos_max, neg_max))
            exp_neg = math.exp(neg_max - max(pos_max, neg_max))
            return exp_pos / (exp_pos + exp_neg)
        except:
            return 0.5

    def verify_attribute(
        self,
        image: Union[Image.Image, str],
        bbox: List[float],
        object_class: str,
        attr_value: str
    ) -> float:
        """Verify if an entity has a specific attribute using VLM."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        cropped = self._crop_with_padding(image, bbox)

        article = self._get_article(attr_value)
        statement = f"This is {article} {attr_value} {object_class}"

        return self._get_vlm_probability(cropped, statement)

    def verify_relationship(
        self,
        image: Union[Image.Image, str],
        bbox1: List[float],
        bbox2: List[float],
        obj1_class: str,
        obj2_class: str,
        relation: str
    ) -> float:
        """Verify if two entities have a specific relationship using VLM."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Compute union bounding box
        x1 = min(bbox1[0], bbox2[0])
        y1 = min(bbox1[1], bbox2[1])
        x2 = max(bbox1[2], bbox2[2])
        y2 = max(bbox1[3], bbox2[3])
        union_bbox = [x1, y1, x2, y2]

        cropped = self._crop_with_padding(image, union_bbox)

        relation_text = relation.replace("_", " ")
        article1 = self._get_article(obj1_class)
        article2 = self._get_article(obj2_class)
        statement = f"There is {article1} {obj1_class} {relation_text} {article2} {obj2_class}"

        return self._get_vlm_probability(cropped, statement)

    def is_available(self) -> bool:
        return self._model is not None

    def cleanup(self):
        if self._model:
            del self._model
            del self._processor
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def load_nlvr2_samples(samples_file="nlvr2_images_full/samples.json", img_dir="nlvr2_images_full", max_samples=50):
    """Load NLVR2 samples for testing."""
    with open(samples_file) as f:
        items = json.load(f)

    samples = []
    for item in items:
        identifier = item["identifier"]
        parts = identifier.rsplit("-", 1)
        img_base = parts[0]
        left_img = os.path.join(img_dir, f"{img_base}-img0.png")
        right_img = os.path.join(img_dir, f"{img_base}-img1.png")

        if not os.path.exists(left_img) or not os.path.exists(right_img):
            continue

        # Verify images are valid
        try:
            Image.open(left_img).verify()
            Image.open(right_img).verify()
        except:
            continue

        samples.append({
            "identifier": identifier,
            "sentence": item["sentence"],
            "label": item["label"].lower() == "true" if isinstance(item["label"], str) else item["label"],
            "left_img": left_img,
            "right_img": right_img,
        })

    np.random.seed(42)
    indices = np.random.permutation(len(samples))[:max_samples]
    return [samples[i] for i in indices]


def run_prove_pipeline(sample: Dict, verification_method: str = "itm") -> Dict[str, Any]:
    """
    Run the full PROVE pipeline on a single sample.

    Returns dict with probabilistic and deterministic results.
    """
    from src.prove import PROVE
    from src.core.model_manager import ModelManager

    # Get model manager instance
    manager = ModelManager()

    # Monkey-patch to use different verifier if needed
    original_get_blip_verifier = None
    vlm_verifier = None

    if verification_method in ["vlm_yes_no", "vlm_true_false"]:
        original_get_blip_verifier = manager.get_blip_verifier
        prompt_style = "yes_no" if verification_method == "vlm_yes_no" else "true_false"

        # Create VLM verifier (reuse if already created)
        if f'vlm_verifier_{prompt_style}' not in manager._models:
            manager._models[f'vlm_verifier_{prompt_style}'] = VLMVerifier(prompt_style=prompt_style)
        vlm_verifier = manager._models[f'vlm_verifier_{prompt_style}']

        # Monkey-patch
        manager.get_blip_verifier = lambda: vlm_verifier

    try:
        # Create PROVE instance and run
        prove = PROVE(threshold=0.5)
        result = prove.predict(
            image_a_path=sample["left_img"],
            image_b_path=sample["right_img"],
            question=sample["sentence"]
        )

        return {
            "success": True,
            "probabilistic_answer": result.probabilistic.final_answer,
            "deterministic_answer": result.deterministic.final_answer,
            "probabilistic_prob": result.probabilistic.probability,
            "deterministic_prob": result.deterministic.probability,
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

    finally:
        # Restore original method if patched
        if original_get_blip_verifier:
            manager.get_blip_verifier = original_get_blip_verifier


def cleanup_gpu_memory():
    """Clear GPU memory between methods."""
    from src.core.model_manager import ModelManager
    manager = ModelManager()
    # Clear all cached models
    for key in list(manager._models.keys()):
        model = manager._models.pop(key, None)
        if hasattr(model, 'cleanup'):
            model.cleanup()
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=50, help="Max samples to test")
    parser.add_argument("--methods", nargs="+", default=["itm", "vlm_yes_no", "vlm_true_false"],
                       help="Verification methods to test")
    parser.add_argument("--output-dir", default="eval/full_pipeline_comparison")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Full PROVE Pipeline: Verification Method Comparison")
    print("=" * 80)

    # Load samples
    samples = load_nlvr2_samples(max_samples=args.max_samples)
    print(f"Loaded {len(samples)} samples")

    # Results storage
    all_results = {}
    all_metrics = {}

    for method in args.methods:
        print(f"\n{'=' * 60}")
        print(f"Testing: {method.upper()}")
        print("=" * 60)

        method_results = []
        prove_correct = 0
        deprove_correct = 0
        both_correct = 0
        prove_only = 0
        deprove_only = 0
        n_success = 0

        for i, sample in enumerate(tqdm(samples, desc=method)):
            result = run_prove_pipeline(sample, verification_method=method)

            # Per-sample memory cleanup to prevent MPS memory buildup
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            if not result["success"]:
                print(f"  Sample {i} failed: {result.get('error', 'Unknown')}")
                method_results.append({
                    "identifier": sample["identifier"],
                    "error": result.get("error", "Unknown")
                })
                continue

            n_success += 1
            label = sample["label"]

            # Parse answers
            prob_answer = str(result["probabilistic_answer"]).lower() == "true"
            det_answer = str(result["deterministic_answer"]).lower() == "true"

            pc = (prob_answer == label)
            dc = (det_answer == label)

            if pc:
                prove_correct += 1
            if dc:
                deprove_correct += 1
            if pc and dc:
                both_correct += 1
            if pc and not dc:
                prove_only += 1
            if dc and not pc:
                deprove_only += 1

            method_results.append({
                "identifier": sample["identifier"],
                "label": label,
                "prove_answer": prob_answer,
                "deprove_answer": det_answer,
                "prove_correct": pc,
                "deprove_correct": dc,
                "prob_probability": result["probabilistic_prob"],
                "det_probability": result["deterministic_prob"],
            })

        # Compute metrics
        prove_acc = prove_correct / n_success if n_success > 0 else 0
        deprove_acc = deprove_correct / n_success if n_success > 0 else 0

        all_results[method] = method_results
        all_metrics[method] = {
            "n_total": len(samples),
            "n_success": n_success,
            "prove_accuracy": prove_acc,
            "deprove_accuracy": deprove_acc,
            "prove_correct": prove_correct,
            "deprove_correct": deprove_correct,
            "both_correct": both_correct,
            "prove_only_correct": prove_only,
            "deprove_only_correct": deprove_only,
            "difference": prove_acc - deprove_acc,
        }

        print(f"\n{method} Results:")
        print(f"  Samples: {n_success}/{len(samples)}")
        print(f"  PROVE:   {prove_acc*100:.1f}% ({prove_correct}/{n_success})")
        print(f"  DePROVE: {deprove_acc*100:.1f}% ({deprove_correct}/{n_success})")
        print(f"  Diff:    {(prove_acc - deprove_acc)*100:+.1f}%")
        print(f"  Both correct: {both_correct}, PROVE only: {prove_only}, DePROVE only: {deprove_only}")

        # Cleanup GPU memory before next method
        print("  Cleaning up GPU memory...")
        cleanup_gpu_memory()

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\n{'Method':<20} {'N':>5} {'PROVE':>10} {'DePROVE':>10} {'Diff':>10}")
    print("-" * 60)

    for method in args.methods:
        m = all_metrics[method]
        print(f"{method:<20} {m['n_success']:>5} {m['prove_accuracy']*100:>9.1f}% {m['deprove_accuracy']*100:>9.1f}% {m['difference']*100:>+9.1f}%")

    print("\n" + "-" * 80)
    print("NOTE: This runs the FULL PROVE pipeline including:")
    print("  1. Object detection (Florence-2)")
    print("  2. Agentic evidence collection (LLM + VLM)")
    print("  3. ProbLog reasoning (dual mode)")
    print("-" * 80)

    # Save results
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    with open(os.path.join(args.output_dir, "predictions.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
