from PIL import Image
import numpy as np

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))  # Resize to match model input
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return image_array.transpose((2, 0, 1))  # Change to (C, H, W) format

def convert_bbox_to_cm(bbox_width_px: float, bbox_height_px: float, generic_width: float, generic_height: float) -> (float, float):
    pixel_ratio_w = generic_width / bbox_width_px
    pixel_ratio_h = generic_height / bbox_height_px
    bbox_width_cm = round(bbox_width_px * pixel_ratio_w, 2)
    bbox_height_cm = round(bbox_height_px * pixel_ratio_h, 2)
    return bbox_width_cm, bbox_height_cm