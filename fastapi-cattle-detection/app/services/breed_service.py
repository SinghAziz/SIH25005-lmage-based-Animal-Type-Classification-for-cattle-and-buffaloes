from fastapi import UploadFile, File
from fastapi import HTTPException
from app.models.breed_predictor import BreedPredictor
from app.models.weight_predictor import WeightPredictor
from app.models.milk_predictor import MilkPredictor
from app.utils.model_loader import load_breed_encoder, load_scalers
from app.utils.image_processing import process_image
import os

class BreedService:
    def __init__(self):
        self.breed_predictor = BreedPredictor()
        self.weight_predictor = WeightPredictor()
        self.milk_predictor = MilkPredictor()
        self.breed_encoder = load_breed_encoder()
        self.scaler_X, self.scaler_y = load_scalers()

    def predict_breed(self, image: UploadFile):
        try:
            image_path = process_image(image)
            breed = self.breed_predictor.predict(image_path)
            return breed
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def calculate_weight(self, height: float, width: float, breed: str):
        try:
            weight = self.weight_predictor.predict(height, width, breed)
            return weight
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def calculate_milk_yield(self, breed: str, weight: float):
        try:
            milk_yield = self.milk_predictor.predict(breed, weight)
            return milk_yield
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))