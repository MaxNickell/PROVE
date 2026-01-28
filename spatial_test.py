#!/usr/bin/env python3
"""
Compare VLM vs Geometric spatial verification on a single subquestion.

Usage:
    python spatial_test.py -i image.png -s "Is the bird on top of the buffalo?"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.unified_agent import EntityCandidate


def parse_subquestion(subquestion: str, llm_client) -> tuple:
    """Extract subject, relation, object from subquestion."""
    import json

    prompt = f"""Parse this spatial subquestion into subject, relation, and object.

Subquestion: "{subquestion}"

Examples:
- "Is the cat to the left of the dog?" → {{"subject": "cat", "relation": "left_of", "object": "dog"}}
- "Is the bird perched on the buffalo?" → {{"subject": "bird", "relation": "on_top_of", "object": "buffalo"}}
- "The chair is under the table." → {{"subject": "chair", "relation": "below", "object": "table"}}
- "The table is in front of the painting." → {{"subject": "table", "relation": "in_front_of", "object": "painting"}}

Return JSON only:"""

    response = llm_client.chat([
        {"role": "system", "content": "Parse spatial questions. Output JSON only."},
        {"role": "user", "content": prompt}
    ], temperature=0)

    response = response.strip()
    if "```" in response:
        response = response.split("```")[1].replace("json", "").strip()

    parsed = json.loads(response)
    return parsed["subject"], parsed["relation"], parsed["object"]


def detection_to_entity(det, image_id: str = "image_a") -> EntityCandidate:
    """Convert ObjectDetection to EntityCandidate."""
    image_letter = image_id.replace("image_", "")
    entity_id = f"{det.label}_{image_letter}_{det.object_id}"

    return EntityCandidate(
        entity_id=entity_id,
        image_id=image_id,
        object_class=det.label,
        bbox=det.bbox,
        confidence=det.confidence
    )


def verify_vlm(image, subject: EntityCandidate, obj: EntityCandidate, relation: str, qwen_client):
    """VLM verification."""
    from PIL import ImageDraw
    from src.core.probability import get_verifier_probability

    # Crop to union with 15% margin
    x1 = min(subject.bbox[0], obj.bbox[0])
    y1 = min(subject.bbox[1], obj.bbox[1])
    x2 = max(subject.bbox[2], obj.bbox[2])
    y2 = max(subject.bbox[3], obj.bbox[3])

    width, height = x2 - x1, y2 - y1
    margin_x, margin_y = width * 0.15, height * 0.15

    img_w, img_h = image.size
    crop_x1 = max(0, x1 - margin_x)
    crop_y1 = max(0, y1 - margin_y)
    crop_x2 = min(img_w, x2 + margin_x)
    crop_y2 = min(img_h, y2 + margin_y)

    cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    # Draw colored boxes (RED=subject, BLUE=object)
    adj_subj = [
        subject.bbox[0] - crop_x1, subject.bbox[1] - crop_y1,
        subject.bbox[2] - crop_x1, subject.bbox[3] - crop_y1
    ]
    adj_obj = [
        obj.bbox[0] - crop_x1, obj.bbox[1] - crop_y1,
        obj.bbox[2] - crop_x1, obj.bbox[3] - crop_y1
    ]

    annotated = cropped.copy()
    draw = ImageDraw.Draw(annotated)
    draw.rectangle(adj_subj, outline="red", width=4)
    draw.rectangle(adj_obj, outline="blue", width=4)

    # Build prompt
    relation_phrase = relation.replace('_', ' ')
    question = f"Is the {subject.object_class} {relation_phrase} the {obj.object_class}?"

    prompt = f"""The {subject.object_class} is marked in RED and the {obj.object_class} is marked in BLUE.

{question}

Respond with ONLY "Yes" or "No". Do not add punctuation or explanation.

Answer:"""

    response, logits = qwen_client.run_inference_with_logits(annotated, prompt)
    probability = get_verifier_probability(logits, response, qwen_client.processor.tokenizer)

    return probability, response.strip()


def compare_methods(image_path: str, subquestion: str):
    """Compare VLM vs Geometric verification."""
    from PIL import Image
    from src.core.model_manager import ModelManager
    from src.pipeline.detector import Detector
    from src.vision.spatial_reasoning import SpatialReasoner

    print("\n" + "=" * 70)
    print("VLM vs GEOMETRIC COMPARISON")
    print("=" * 70)
    print(f"Image: {image_path}")
    print(f"Subquestion: {subquestion}")
    print("=" * 70)

    model_manager = ModelManager()
    llm_client = model_manager.get_llm_client()

    # Parse subquestion
    print("\n[1] Parsing subquestion...")
    try:
        subject_label, relation, object_label = parse_subquestion(subquestion, llm_client)
        print(f"    Subject: {subject_label}")
        print(f"    Relation: {relation}")
        print(f"    Object: {object_label}")
    except Exception as e:
        print(f"    ✗ Parse failed: {e}")
        return

    # Load image
    print("\n[2] Loading image...")
    image = Image.open(image_path).convert("RGB")
    print(f"    Size: {image.size}")

    # Detect objects
    print("\n[3] Detecting objects...")
    detector = Detector()
    query = f"{subject_label}, {object_label}"
    detections = detector.detect_from_question(image_path, query)

    print(f"    Found {len(detections)} objects:")
    for det in detections:
        print(f"      [{det.object_id}] {det.label}")
        print(f"          bbox: {[int(x) for x in det.bbox]}")
        print(f"          conf: {det.confidence:.3f}")

    # Match detections and convert to EntityCandidate
    print("\n[4] Matching detections...")
    subject_det = next((d for d in detections if d.label.lower() == subject_label.lower()), None)
    object_det = next((d for d in detections if d.label.lower() == object_label.lower()), None)

    if not subject_det:
        print(f"    ✗ Subject '{subject_label}' not found")
        return
    if not object_det:
        print(f"    ✗ Object '{object_label}' not found")
        return

    # Convert to EntityCandidate (what unified_agent uses)
    subject = detection_to_entity(subject_det, "image_a")
    obj = detection_to_entity(object_det, "image_a")

    print(f"    Subject: {subject.entity_id} (conf={subject.confidence:.3f})")
    print(f"    Object: {obj.entity_id} (conf={obj.confidence:.3f})")

    # VLM verification
    print("\n[5] VLM Verification...")
    qwen_client = model_manager.get_qwen_vl()
    vlm_prob, vlm_response = verify_vlm(image, subject, obj, relation, qwen_client)
    vlm_answer = vlm_prob >= 0.5
    print(f"    Response: {vlm_response}")
    print(f"    Probability: {vlm_prob:.4f}")
    print(f"    Answer: {'TRUE' if vlm_answer else 'FALSE'}")

    # Geometric verification
    print("\n[6] Geometric Verification...")
    reasoner = SpatialReasoner()

    if not reasoner.can_verify(relation):
        print(f"    ✗ Relation '{relation}' not supported geometrically")
        geo_prob, geo_holds, geo_debug = None, None, None
    else:
        geo_prob, geo_debug = reasoner.verify(image, subject, obj, relation)
        geo_holds = geo_debug["holds"]

        print(f"    Holds: {geo_holds}")
        print(f"    Probability: {geo_prob:.4f}")
        print(f"    Answer: {'TRUE' if geo_holds else 'FALSE'}")

        # Print debug info
        print(f"\n    Debug info:")
        for key, value in geo_debug.items():
            if key in ["holds", "probability", "subject_id", "object_id", "relation"]:
                continue
            if isinstance(value, float):
                print(f"      {key}: {value:.4f}")
            elif isinstance(value, tuple):
                print(f"      {key}: ({value[0]:.1f}, {value[1]:.1f})")
            else:
                print(f"      {key}: {value}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"  Subquestion: {subquestion}")
    print()
    print(f"  VLM:       p={vlm_prob:.4f}  → {'TRUE' if vlm_answer else 'FALSE'}")

    if geo_prob is not None:
        print(f"  Geometric: p={geo_prob:.4f}  → {'TRUE' if geo_holds else 'FALSE'}")
        print()
        if vlm_answer == geo_holds:
            print("  ✓ Methods AGREE")
        else:
            print("  ✗ Methods DISAGREE")
            print(
                f"    VLM says {'TRUE' if vlm_answer else 'FALSE'}, Geometric says {'TRUE' if geo_holds else 'FALSE'}")
    else:
        print(f"  Geometric: Not supported")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare VLM vs Geometric spatial verification"
    )
    parser.add_argument("--image", "-i", type=str, required=True)
    parser.add_argument("--subquestion", "-s", type=str, required=True)

    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return 1

    compare_methods(args.image, args.subquestion)
    return 0


if __name__ == "__main__":
    sys.exit(main())