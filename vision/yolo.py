from ultralytics import YOLO
from PIL import Image
import requests


class YOLO():
    def __init__(self, model_name = "yolo12n.pt"):
        self.model = YOLO(model_name)

    def detect_and_save(self, image_url, save_path):
        image   = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        results = self.model(image) 

        results[0].save(save_path)
        return results
