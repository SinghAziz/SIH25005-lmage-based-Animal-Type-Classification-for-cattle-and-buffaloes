import os
import json
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------- CONFIG -----------------
IMAGE_DIR = "Backend/Model/demo_dataset"  # folder with images directly
JSON_DIR = "Backend/Cattle_Labelme"  # Labelme JSONs
MODEL_PATH = "Backend/best.pt"
OUTPUT_DIR = "Backend/Processed_json"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ------------------------------------------

# Load YOLO model
model = YOLO(MODEL_PATH)

# ----------------- FUNCTIONS -----------------
def create_mask(image_shape, json_path):
    """Create a mask from Labelme JSON."""
    with open(json_path) as f:
        data = json.load(f)
    
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    for shape in data.get('shapes', []):
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
    return mask

def compute_atc_metrics(mask_crop, img_shape):
    """Compute bounding box metrics and normalized ATC score."""
    img_h, img_w = img_shape[:2]
    contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    c = max(contours, key=cv2.contourArea)  # largest contour
    x, y, w, h = cv2.boundingRect(c)
    
    # Extreme points
    left = tuple(c[c[:, :, 0].argmin()][0])
    right = tuple(c[c[:, :, 0].argmax()][0])
    top = tuple(c[c[:, :, 1].argmin()][0])
    bottom = tuple(c[c[:, :, 1].argmax()][0])
    
    # Aspect ratio
    aspect_ratio = w / h if h != 0 else 0
    
    # Masked area
    masked_area = int(cv2.countNonZero(mask_crop))
    
    # Normalize
    h_norm = h / img_h
    w_norm = w / img_w
    area_norm = masked_area / (img_h * img_w)
    
    score = (h_norm * 0.4 + w_norm * 0.3 + area_norm * 0.3) * 100
    
    metrics = {
        "bbox_width": int(w),
        "bbox_height": int(h),
        "extreme_left_x": int(left[0]),
        "extreme_right_x": int(right[0]),
        "extreme_top_y": int(top[1]),
        "extreme_bottom_y": int(bottom[1]),
        "aspect_ratio": round(aspect_ratio, 2),
        "masked_area_pixels": masked_area,
        "atc_score": round(score, 2)
    }
    
    return metrics

# ----------------- PROCESS IMAGES -----------------
results_list = []

for image_file in os.listdir(IMAGE_DIR):
    if not image_file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    
    image_path = os.path.join(IMAGE_DIR, image_file)
    json_file = os.path.splitext(image_file)[0] + ".json"
    json_path = os.path.join(JSON_DIR, json_file)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read {image_file}, skipping.")
        continue
    
    print(f"Processing {image_file}...")
    
    # 1️⃣ Run YOLO
    results = model.predict(img)
    if len(results[0].boxes) == 0:
        print(f"  No bounding box detected, skipping image.")
        continue
    
    box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box
    crop_img = img[y1:y2, x1:x2]
    
    # 2️⃣ Masking
    if os.path.exists(json_path):
        mask = create_mask(img.shape, json_path)
        mask_crop = mask[y1:y2, x1:x2]
        masked_crop = cv2.bitwise_and(crop_img, crop_img, mask=mask_crop)
        mask_source = "JSON"
        print(f"  Using JSON mask.")
    else:
        mask_crop = np.ones((y2-y1, x2-x1), dtype=np.uint8) * 255
        masked_crop = crop_img.copy()
        mask_source = "BBox"
        print(f"  JSON not found, using YOLO bounding box as mask.")
    
    # 3️⃣ Metrics
    metrics = compute_atc_metrics(mask_crop, img.shape)
    if metrics is None:
        print(f"  No contour found, skipping metrics.")
        continue
    
    metrics.update({
        "image": image_file,
        "mask_source": mask_source
    })
    results_list.append(metrics)
    
    # 4️⃣ Save masked image
    out_img_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(image_file)[0]}_masked.png")
    cv2.imwrite(out_img_path, masked_crop)

# 5️⃣ Save metrics to JSON
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(results_list, f, indent=4)

print(f"Processing complete. {len(results_list)} images processed.")
print(f"Masked images and metrics saved in {OUTPUT_DIR}")
