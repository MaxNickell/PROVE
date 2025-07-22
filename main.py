import json, argparse
from src.pipeline.detector import ObjectDetector

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Multi-detector demo")
    ap.add_argument("image", help="Path to image file")
    args = ap.parse_args()

    det = ObjectDetector()
    out = det.detect(args.image)

    print("\nFINAL DETECTIONS")
    print(json.dumps(out, indent=2))
