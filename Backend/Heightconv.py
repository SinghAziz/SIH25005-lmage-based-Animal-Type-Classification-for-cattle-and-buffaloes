import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

# ---------------- DEVICE SETUP ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- BREED LIST & MEASUREMENTS ----------------
breed_list = ['Vechur', 'Mehsana', 'Hallikar', 'Amritmahal', 'Kankrej', 'Sahiwal', 
              'Surti', 'Jersey', 'Pulikulam', 'Nagpuri', 'Nagori', 'Malnad_gidda', 
              'Dangi', 'Murrah', 'Jaffrabadi', 'Red_Dane', 'Krishna_Valley', 'Guernsey', 
              'Kherigarh', 'Rathi', 'Khillari', 'Bargur', 'Banni', 'Holstein_Friesian', 
              'Toda', 'Alambadi', 'Deoni', 'Kangayam', 'Kenkatha', 'Kasargod', 'Nimari', 
              'Tharparkar', 'Bhadawari', 'Ongole', 'Red_Sindhi', 'Hariana', 'Umblachery', 
              'Gir', 'Ayrshire', 'Brown_Swiss', 'Nili_Ravi']

BREED_MEASUREMENTS = {
    "Vechur": {"height_cm": 90, "width_cm": 50},
    "Mehsana": {"height_cm": 129, "width_cm": 60},
    "Hallikar": {"height_cm": 130, "width_cm": 60},
    "Amritmahal": {"height_cm": 130, "width_cm": 60},
    "Kankrej": {"height_cm": 140, "width_cm": 65},
    "Sahiwal": {"height_cm": 130, "width_cm": 60},
    "Surti": {"height_cm": 125, "width_cm": 55},
    "Jersey": {"height_cm": 120, "width_cm": 55},
    "Pulikulam": {"height_cm": 120, "width_cm": 55},
    "Nagpuri": {"height_cm": 125, "width_cm": 60},
    "Nagori": {"height_cm": 130, "width_cm": 60},
    "Malnad_gidda": {"height_cm": 95, "width_cm": 45},
    "Dangi": {"height_cm": 110, "width_cm": 50},
    "Murrah": {"height_cm": 135, "width_cm": 70},
    "Jaffrabadi": {"height_cm": 130, "width_cm": 65},
    "Red_Dane": {"height_cm": 140, "width_cm": 65},
    "Krishna_Valley": {"height_cm": 120, "width_cm": 55},
    "Guernsey": {"height_cm": 130, "width_cm": 60},
    "Kherigarh": {"height_cm": 125, "width_cm": 60},
    "Rathi": {"height_cm": 120, "width_cm": 55},
    "Khillari": {"height_cm": 120, "width_cm": 55},
    "Bargur": {"height_cm": 115, "width_cm": 50},
    "Banni": {"height_cm": 110, "width_cm": 50},
    "Holstein_Friesian": {"height_cm": 145, "width_cm": 60},
    "Toda": {"height_cm": 90, "width_cm": 45},
    "Alambadi": {"height_cm": 120, "width_cm": 55},
    "Deoni": {"height_cm": 125, "width_cm": 55},
    "Kangayam": {"height_cm": 110, "width_cm": 50},
    "Kenkatha": {"height_cm": 115, "width_cm": 50},
    "Kasargod": {"height_cm": 95, "width_cm": 45},
    "Nimari": {"height_cm": 120, "width_cm": 55},
    "Tharparkar": {"height_cm": 135, "width_cm": 60},
    "Bhadawari": {"height_cm": 120, "width_cm": 55},
    "Ongole": {"height_cm": 140, "width_cm": 65},
    "Red_Sindhi": {"height_cm": 130, "width_cm": 60},
    "Hariana": {"height_cm": 135, "width_cm": 60},
    "Umblachery": {"height_cm": 115, "width_cm": 50},
    "Gir": {"height_cm": 135, "width_cm": 60},
    "Ayrshire": {"height_cm": 130, "width_cm": 60},
    "Brown_Swiss": {"height_cm": 140, "width_cm": 65},
    "Nili_Ravi": {"height_cm": 140, "width_cm": 70},
}

# ---------------- IMAGE FOLDER ----------------
img_folder = "Backend/Model/demo_dataset"

# ---------------- LOAD MODEL ----------------
num_classes = len(breed_list)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# ---------------- PREPROCESSING ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------- LOAD JSON ----------------
with open("Backend/Processed_json/metrics.json", "r") as f:
    data = json.load(f)

# ---------------- PREDICT & CONVERT ----------------
for animal in data:
    img_path = os.path.join(img_folder, animal["image"])
    
    if not os.path.exists(img_path):
        print(f"⚠️ Image not found: {img_path}")
        continue

    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred_idx = torch.max(outputs, 1)

    breed = breed_list[pred_idx.item()]
    animal["breed"] = breed

    # Generic breed measurements
    generic_height = BREED_MEASUREMENTS[breed]["height_cm"]
    generic_width = BREED_MEASUREMENTS[breed]["width_cm"]
    animal["generic_height_cm"] = generic_height
    animal["generic_width_cm"] = generic_width

    # Convert bbox pixels to cm (scale by breed generic dimensions)
    bbox_width_px = animal.get("bbox_width", None)
    bbox_height_px = animal.get("bbox_height", None)
    if bbox_width_px and bbox_height_px:
        # aspect ratio scaling
        pixel_ratio_w = generic_width / bbox_width_px
        pixel_ratio_h = generic_height / bbox_height_px
        animal["bbox_width_cm"] = round(bbox_width_px * pixel_ratio_w, 2)
        animal["bbox_height_cm"] = round(bbox_height_px * pixel_ratio_h, 2)
    else:
        animal["bbox_width_cm"] = None
        animal["bbox_height_cm"] = None

# ---------------- SAVE UPDATED JSON ----------------
with open("animals_with_breeds.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ All breeds predicted and bbox sizes converted to cm!")
