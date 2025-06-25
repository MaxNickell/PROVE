import argparse
from src.pipeline.orchestrator import Orchestrator

def main():
    parser = argparse.ArgumentParser(description="PROVE Vision Pipeline")
    parser.add_argument("image_path", type=str, help="Path to the image to process")
    
    args = parser.parse_args()
    
    orchestrator = Orchestrator()
    results = orchestrator.process_image(args.image_path)

if __name__ == "__main__":
    main()
