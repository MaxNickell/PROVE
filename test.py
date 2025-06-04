# from vision.blip_image_text_to_text import BlipImageTextToText
from vision.grounding_dino_zero_shot_object_detection import GroundingDinoZeroShotObjectDetection
# from vision.sam_mask import SamMask
# from vision.vilt_vqa import ViltVqa
# from vision.yolo_object_detection import YoloObjectDetection
from vision.deep_seek_image_text_to_text import DeepSeekImageTextToText
from dotenv import load_dotenv
from PIL import Image
import requests


def test_deepseek_vl2():
    deepseek = DeepSeekImageTextToText()
    print('Classifying object...')
    deepseek.classify_object()
    print('List objects...')
    deepseek.list_objects()
    print('Listing and bounding objects...')
    deepseek.list_and_bound_objects()
    print('Locating objects...')
    deepseek.locate_objects()
    
def test_grounding_dino():
    image = Image.open(requests.get("http://www.louwphotography.com/Africa/Tanzania/Highlights/i-zVB3Zfj/0/L/IMG_5487-L.jpg", stream=True).raw).convert("RGB")
    grounding_dino = GroundingDinoZeroShotObjectDetection()
    grounding_dino.detect_and_save(image, "a zebra.", "images/grounding_dino_output.png")

if __name__ == "__main__":
    #test_grounding_dino()
    test_deepseek_vl2()
