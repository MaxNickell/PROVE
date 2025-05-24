from PIL import Image
import requests
import torch
from transformers import SamModel, SamProcessor, pipeline
import numpy as np
import matplotlib.pyplot as plt


class SamMask():
    def __init__(self, model_name="facebook/sam-vit-base"):
        self.generator = pipeline("mask-generation", device = 0, points_per_batch = 256)
        self.model = SamModel.from_pretrained(model_name)
        self.processor = SamProcessor.from_pretrained(model_name)

    def mask_all(self, image_url, points_per_batch=256):
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        outputs = self.generator(image, points_per_batch = points_per_batch)
        masks = outputs["masks"]
        scores = outputs["scores"]
        return masks

    def mask_region(self, image_url, input_points):
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        inputs = self.processor(image, input_points=input_points, return_tensors="pt")
        outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        scores = outputs.iou_scores
        return masks
    
    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
