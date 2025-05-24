from vision.blip_image_text_to_text import BlipImageTextToText
from vision.grounding_dino_zero_shot_object_detection import GroundingDinoZeroShotObjectDetection
from vision.sam_mask import SamMask
from vision.vilt_vqa import ViltVqa
from vision.yolo_object_detection import YoloObjectDetection
from dotenv import load_dotenv

def generate_graph(image_url):
    # Step 0: Initialize models
    detector = YoloObjectDetection()
    zero_shot_detector = GroundingDinoZeroShotObjectDetection()

    # Step 1: Object detection first pass
    boxes = detector.detect_and_save(image_url, "output.jpg")
    print("Boxes: ", boxes)

    # Step 2: Object detection second pass


    # Step 3: Get attributes




if __name__ == "__main__":
    load_dotenv()

    image_url = "https://i.ytimg.com/vi/qEwKCR5JCog/maxresdefault.jpg"
    generate_graph(image_url)



