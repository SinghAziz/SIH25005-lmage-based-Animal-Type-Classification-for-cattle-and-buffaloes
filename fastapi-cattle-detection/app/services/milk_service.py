from fastapi import UploadFile, File
from fastapi import HTTPException
from app.models.weight_predictor import WeightPredictor
from app.models.milk_predictor import MilkPredictor
from app.utils.model_loader import load_weight_model, load_milk_model, load_breed_encoder, load_scalers
from app.schemas.cattle import CattlePredictionResponse
import numpy as np
import json

class MilkService:
    def __init__(self):
        self.weight_model = load_weight_model()
        self.milk_model = load_milk_model()
        self.breed_encoder = load_breed_encoder()
        self.scaler_X, self.scaler_y = load_scalers()

    def calculate_weight(self, height: float, width: float, breed: str) -> float:
        breed_vector = np.zeros(len(self.breed_encoder.categories_[0]))
        breed_idx = list(self.breed_encoder.categories_[0]).index(breed)
        breed_vector[breed_idx] = 1

        features_scaled = np.concatenate((self.scaler_X.transform([[height, width]])[0], breed_vector))
        weight_scaled = self.weight_model.predict(features_scaled.reshape(1, -1))
        weight = self.scaler_y.inverse_transform(weight_scaled.reshape(1, -1))[0][0]
        return weight

    def calculate_milk_yield(self, breed: str, weight: float) -> float:
        return self.milk_model.predict_milk_yield(breed, weight)

    def process_file_upload(self, file: UploadFile = File(...)) -> CattlePredictionResponse:
        # Here you would implement the logic to handle the uploaded file,
        # perform object detection, breed prediction, and then calculate weight and milk yield.
        # This is a placeholder for the actual implementation.
        
        # Example values for demonstration purposes
        height = 130  # Replace with actual height from detection
        width = 60    # Replace with actual width from detection
        breed = "Murrah"  # Replace with actual breed prediction

        weight = self.calculate_weight(height, width, breed)
        milk_yield = self.calculate_milk_yield(breed, weight)

        return CattlePredictionResponse(breed=breed, weight=weight, milk_yield=milk_yield)