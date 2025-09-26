# full_pipeline.py
import os, json, re
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import joblib
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- CONFIG -----------------
IMG_FOLDER = "Backend/Model/demo_dataset"
YOLO_JSON  = "Backend/Processed_json/metrics.json"   # contains bbox_width/height + image name
BREED_MODEL = "best_model.pth"
WEIGHT_MODEL = "weight_predictor_model.pth"
MILK_MODEL   = "milk_yield_model.pth"
ENCODER_BREED_WEIGHT = "breed_encoder.pkl"
SCALER_X = "scaler_X.pkl"
SCALER_Y = "scaler_y.pkl"
ENCODER_MILK = "milk_breed_encoder.pkl"

# ----------------- BREED MEASUREMENTS -----------------
BREED_MEASUREMENTS = {
    "Vechur": {"height_cm": 90, "width_cm": 50}, "Mehsana": {"height_cm": 129, "width_cm": 60},
    "Hallikar": {"height_cm": 130, "width_cm": 60}, "Amritmahal": {"height_cm": 130, "width_cm": 60},
    "Kankrej": {"height_cm": 140, "width_cm": 65}, "Sahiwal": {"height_cm": 130, "width_cm": 60},
    "Surti": {"height_cm": 125, "width_cm": 55}, "Jersey": {"height_cm": 120, "width_cm": 55},
    "Pulikulam": {"height_cm": 120, "width_cm": 55}, "Nagpuri": {"height_cm": 125, "width_cm": 60},
    "Nagori": {"height_cm": 130, "width_cm": 60}, "Malnad_gidda": {"height_cm": 95, "width_cm": 45},
    "Dangi": {"height_cm": 110, "width_cm": 50}, "Murrah": {"height_cm": 135, "width_cm": 70},
    "Jaffrabadi": {"height_cm": 130, "width_cm": 65}, "Red_Dane": {"height_cm": 140, "width_cm": 65},
    "Krishna_Valley": {"height_cm": 120, "width_cm": 55}, "Guernsey": {"height_cm": 130, "width_cm": 60},
    "Kherigarh": {"height_cm": 125, "width_cm": 60}, "Rathi": {"height_cm": 120, "width_cm": 55},
    "Khillari": {"height_cm": 120, "width_cm": 55}, "Bargur": {"height_cm": 115, "width_cm": 50},
    "Banni": {"height_cm": 110, "width_cm": 50}, "Holstein_Friesian": {"height_cm": 145, "width_cm": 60},
    "Toda": {"height_cm": 90, "width_cm": 45}, "Alambadi": {"height_cm": 120, "width_cm": 55},
    "Deoni": {"height_cm": 125, "width_cm": 55}, "Kangayam": {"height_cm": 110, "width_cm": 50},
    "Kenkatha": {"height_cm": 115, "width_cm": 50}, "Kasargod": {"height_cm": 95, "width_cm": 45},
    "Nimari": {"height_cm": 120, "width_cm": 55}, "Tharparkar": {"height_cm": 135, "width_cm": 60},
    "Bhadawari": {"height_cm": 120, "width_cm": 55}, "Ongole": {"height_cm": 140, "width_cm": 65},
    "Red_Sindhi": {"height_cm": 130, "width_cm": 60}, "Hariana": {"height_cm": 135, "width_cm": 60},
    "Umblachery": {"height_cm": 115, "width_cm": 50}, "Gir": {"height_cm": 135, "width_cm": 60},
    "Ayrshire": {"height_cm": 130, "width_cm": 60}, "Brown_Swiss": {"height_cm": 140, "width_cm": 65},
    "Nili_Ravi": {"height_cm": 140, "width_cm": 70},
}

# ----------------- MODEL DEFINITIONS -----------------
class WeightPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

class MilkYieldPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

# ----------------- HELPERS -----------------
def load_breed_classifier():
    breed_list = list(BREED_MEASUREMENTS.keys())
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(breed_list))
    model.load_state_dict(torch.load(BREED_MODEL, map_location=device))
    model.to(device).eval()
    return model, breed_list

def predict_weight(height, width, breed_name):
    enc = joblib.load(ENCODER_BREED_WEIGHT)
    scX = joblib.load(SCALER_X)
    scY = joblib.load(SCALER_Y)
    model = WeightPredictor(2 + len(enc.categories_[0]))
    
    # Load state dict with key fix
    state = torch.load(WEIGHT_MODEL, map_location=device)
    new_state = OrderedDict()
    for k, v in state.items():
        new_state[k.replace("model", "net")] = v
    model.load_state_dict(new_state)
    model.to(device).eval()

    breed_vec = np.zeros(len(enc.categories_[0]))
    breed_vec[list(enc.categories_[0]).index(breed_name)] = 1
    feat = np.concatenate((scX.transform([[height, width]])[0], breed_vec))
    feat = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_scaled = model(feat).item()
    return scY.inverse_transform([[pred_scaled]])[0][0]

def predict_milk_yield(breed_name, weight):
    enc = joblib.load(ENCODER_MILK)
    model = MilkYieldPredictor(len(enc.categories_[0]) + 1)

    state = torch.load(MILK_MODEL, map_location=device)
    new_state = OrderedDict()
    for k, v in state.items():
        new_state[k.replace("model", "net")] = v
    model.load_state_dict(new_state)

    model.to(device).eval()

    breed_vec = np.zeros(len(enc.categories_[0]))
    breed_vec[list(enc.categories_[0]).index(breed_name)] = 1
    feat = np.concatenate((breed_vec, [weight]))
    feat = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        return model(feat).item()

# ----------------- PIPELINE -----------------
def run_pipeline():
    with open(YOLO_JSON) as f:
        data = json.load(f)

    breed_model, breed_list = load_breed_classifier()
    tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

    results = []
    for item in data:
        img_path = os.path.join(IMG_FOLDER, item["image"])
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}")
            continue

        # breed prediction
        img = Image.open(img_path).convert("RGB")
        inp = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = breed_model(inp)
            _, idx = torch.max(out, 1)
        breed = breed_list[idx.item()]

        # generic height/width
        generic_h = BREED_MEASUREMENTS[breed]["height_cm"]
        generic_w = BREED_MEASUREMENTS[breed]["width_cm"]
        px_w = item.get("bbox_width")
        px_h = item.get("bbox_height")
        if px_w and px_h:
            height_cm = generic_h
            width_cm  = generic_w
        else:
            height_cm, width_cm = generic_h, generic_w

        weight_kg = predict_weight(height_cm, width_cm, breed)
        milk_l = predict_milk_yield(breed, weight_kg)

        results.append({
            "animal_id": item.get("id", None),
            "image": item["image"],
            "breed": breed,
            "height_cm": round(height_cm,2),
            "width_cm": round(width_cm,2),
            "weight_kg": round(weight_kg,2),
            "milk_yield_L_day": round(milk_l,2)
        })

    # save results
    with open("final_animals_output.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

# ----------------- SINGLE IMAGE PREDICTION -----------------
def predict_everything(image_path=None):
    # pick first image if None
    if image_path is None:
        images = os.listdir(IMG_FOLDER)
        images = [img for img in images if img.lower().endswith(('.png','.jpg','.jpeg'))]
        if len(images) == 0:
            raise FileNotFoundError(f"No images found in {IMG_FOLDER}")
        image_path = os.path.join(IMG_FOLDER, images[0])
    
    breed_model, breed_list = load_breed_classifier()
    tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    
    img = Image.open(image_path).convert("RGB")
    inp = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = breed_model(inp)
        _, idx = torch.max(out, 1)
    breed = breed_list[idx.item()]

    generic_h = BREED_MEASUREMENTS[breed]["height_cm"]
    generic_w = BREED_MEASUREMENTS[breed]["width_cm"]

    weight_kg = predict_weight(generic_h, generic_w, breed)
    milk_l = predict_milk_yield(breed, weight_kg)

    return {
        "image": os.path.basename(image_path),
        "breed": breed,
        "height_cm": round(generic_h,2),
        "width_cm": round(generic_w,2),
        "weight_kg": round(weight_kg,2),
        "milk_yield_L_day": round(milk_l,2)
    }

# ----------------- MAIN -----------------
if __name__ == "__main__":
    # batch processing
    final = run_pipeline()
    print(f"✅ Pipeline finished. {len(final)} animals processed.")

    # single image example
    single_out = predict_everything()  # picks first image automatically
    print("🔹 Single image prediction:")
    print(single_out)
