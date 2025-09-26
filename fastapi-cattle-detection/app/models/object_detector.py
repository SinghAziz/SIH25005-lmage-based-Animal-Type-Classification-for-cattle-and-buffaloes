from torchvision import models, transforms
import torch
import numpy as np
import os
from PIL import Image
from app.utils.model_loader import load_model
from app.models.weight_predictor import WeightPredictor
from app.models.milk_predictor import MilkPredictor
from app.models.breed_predictor import BreedPredictor

class ObjectDetector:
    def __init__(self, model_path, breed_encoder_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.breed_encoder = self.load_breed_encoder(breed_encoder_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def load_model(self, model_path):
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self.breed_encoder.categories_[0]))
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def load_breed_encoder(self, breed_encoder_path):
        import joblib
        return joblib.load(breed_encoder_path)

    def detect_and_predict(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, pred_idx = torch.max(outputs, 1)

        breed = self.breed_encoder.categories_[0][pred_idx.item()]
        return breed

    def calculate_weight(self, height, width, breed):
        weight_predictor = WeightPredictor()
        weight = weight_predictor.predict(height, width, breed)
        return weight

    def calculate_milk_yield(self, breed, weight):
        milk_predictor = MilkPredictor()
        milk_yield = milk_predictor.predict(breed, weight)
        return milk_yield