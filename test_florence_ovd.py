#!/usr/bin/env python3
"""
Standalone test script for Florence-2 Open Vocabulary Detection (OVD).

Usage:
    python test_florence_ovd.py <image_path> <object_name>
"""

import sys
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw
from src.core.probability import calibrate_detector_confidence


def load_model(model_id="microsoft/Florence-2-large"):
    """Load Florence-2 model + processor on GPU with appropriate dtype."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=dtype
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor, device, dtype


def run_ovd(model, processor, device, dtype, image_path, object_name):
    """Run Florence-2 OVD on an image for a given object name."""
    image = Image.open(image_path).convert("RGB")

    # Prompt
    task = "<OPEN_VOCABULARY_DETECTION>"
    prompt = task + object_name

    # Preprocess
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs["input_ids"] = inputs["input_ids"].to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(device, dtype=dtype)

    # Generate with scores
    generated = model.generate(
        **inputs,
        max_new_tokens=1024,
        num_beams=3,
        return_dict_in_generate=True,
        output_scores=True,
    )

    # Decode text
    generated_text = processor.batch_decode(
        generated.sequences, skip_special_tokens=False
    )[0]

    # Parse detections (boxes + labels only)
    parsed = processor.post_process_generation(
        generated_text,
        task=task,
        image_size=(image.width, image.height),
    )

    # === Manual confidence extraction ===
    scores = []
    if hasattr(generated, "scores") and hasattr(generated, "beam_indices"):
        transition = model.compute_transition_scores(
            sequences=generated.sequences,
            scores=generated.scores,
            beam_indices=generated.beam_indices,
            normalize_logits=True,   # ensures values are probabilities
        )
        # Geometric mean: exp(mean of log-probs) = length-normalized likelihood
        # This is P(sequence)^(1/L), the standard measure in language modeling
        log_probs = transition[0]
        seq_conf = torch.exp(log_probs.mean()).item()  # Geometric mean
        # Attach same score to all detections (since Florence doesn't split per-box yet)
        num_dets = len(parsed.get(task, {}).get("bboxes", []))
        scores = [seq_conf] * num_dets

    # Inject scores into parsed result
    if task in parsed:
        parsed[task]["scores"] = scores

    return parsed, image



def visualize_results(image, results, output_path):
    """Draw bounding boxes + scores on image and save result."""
    detections = results.get("<OPEN_VOCABULARY_DETECTION>", {})
    bboxes = detections.get("bboxes", [])
    labels = detections.get("bboxes_labels", [])
    scores = detections.get("scores", [])

    draw = ImageDraw.Draw(image)

    print("\nDETECTIONS:")
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        raw_score = scores[i] if scores and i < len(scores) else None
        if raw_score is not None:
            calibrated = calibrate_detector_confidence(raw_score)
            print(f"  {i+1}. {label} {bbox} | raw={raw_score:.4f}, calibrated={calibrated:.4f}")
            text = f"{label} ({calibrated:.2f})"
        else:
            print(f"  {i+1}. {label} {bbox} | (no score)")
            text = label

        draw.rectangle(bbox, outline="red", width=3)
        draw.text((bbox[0], bbox[1] - 10), text, fill="red")

    image.save(output_path)
    print(f"\n✓ Annotated image saved: {output_path}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    image_path, object_name = sys.argv[1], sys.argv[2]
    model, processor, device, dtype = load_model()
    results, image = run_ovd(model, processor, device, dtype, image_path, object_name)

    output_path = image_path.rsplit(".", 1)[0] + "_ovd_test.png"
    visualize_results(image, results, output_path)


if __name__ == "__main__":
    main()
