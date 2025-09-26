from fastapi import UploadFile
import torch
import joblib
import numpy as np
from torchvision import models, transforms
from PIL import Image
from app.utils.model_loader import load_model
from app.schemas.cattle import CattlePredictionResponse

class BreedPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.breed_encoder = joblib.load("saved_models/breed_encoder.pkl")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def load_model(self):
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(self.breed_encoder.categories_[0]))
        model.load_state_dict(torch.load("saved_models/best_model.pth", map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def predict_breed(self, image: UploadFile):
        image = Image.open(image.file).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, pred_idx = torch.max(outputs, 1)

        breed = self.breed_encoder.categories_[0][pred_idx.item()]
        return breed

    def get_breed_parameters(self, breed_name):
        breed_parameters = {
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
        return breed_parameters.get(breed_name, None)