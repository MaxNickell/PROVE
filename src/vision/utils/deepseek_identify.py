import argparse, json, sys
from pathlib import Path

from src.vision.deepseek_vl2 import DeepSeekVL2

def classify_many(list_path: Path) -> list[str]:
    """Load model once, classify every crop in list_path (a JSON array)."""
    model  = DeepSeekVL2()
    crops  = json.loads(Path(list_path).read_text())
    labels = []
    for p in crops:
        try:
            label = model.classify_object(p)
        except Exception:
            label = "object"
        labels.append(label)
    return labels

def main():
    p = argparse.ArgumentParser()
    p.add_argument("crop_list_json", help="JSON file with crop image paths")
    args = p.parse_args()

    try:
        labels = classify_many(Path(args.crop_list_json))
        out = json.dumps(labels)
        print(f"[START_DEEPSEEK_VL2]{out}[END_DEEPSEEK_VL2]")
    except Exception as e:
        print("[START_DEEPSEEK_VL2][][END_DEEPSEEK_VL2]", file=sys.stderr)
        raise e

if __name__ == "__main__":
    main()
