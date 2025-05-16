import sys
from vision.captioner import BLIPCaptioner
from vision.object_detector import GroundingDINODetector
from vision.attribute_extractor import ViLTAttributeExtractor
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    load_dotenv()

    BLIP_MODEL_NAME = os.getenv("BLIP_MODEL_NAME")
    BLIP_CAPTION_QUERY = os.getenv("BLIP_CAPTION_QUERY")
    GROUNDING_DINO_MODEL_NAME = os.getenv("GROUNDING_DINO_MODEL_NAME")
    VILT_MODEL_NAME = os.getenv("VILT_MODEL_NAME")
    
    captioner = BLIPCaptioner(BLIP_MODEL_NAME)
    detector = GroundingDINODetector(GROUNDING_DINO_MODEL_NAME)
    attribute_extractor = ViLTAttributeExtractor(VILT_MODEL_NAME)