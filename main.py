import json, argparse
from src.pipeline.orchestrator import Orchestrator

if __name__ == "__main__":
    orchestrator = Orchestrator()
    # orchestrator.run_pipeline("images/dev-339-2-img0.png", "images/dev-339-2-img1.png", "")
    # orchestrator.run_pipeline("images/dev-476-3-img0.png", "images/dev-476-3-img1.png", "")
    res, image_1_dets, image_2_dets = orchestrator.run_pipeline("images/dev-505-2-img0.png", "images/dev-505-2-img1.png", "What is similar between these two images?")
    # orchestrator.run_pipeline("images/dev-516-0-img0.png", "images/dev-516-0-img1.png", "")
    # res = orchestrator.run_pipeline("images/dev-518-0-img0.png", "images/dev-518-0-img1.png", "which animal can eat more food?")
    print(json.dumps(res, indent=4))
    print(json.dumps(image_1_dets, indent=4))
    print(json.dumps(image_2_dets, indent=4))
