from fastapi import UploadFile, File
from fastapi import HTTPException
import os
import json
from app.utils.model_loader import load_weight_predictor, load_scalers
from app.models.weight_predictor import WeightPredictor
from app.models.milk_predictor import MilkPredictor
from app.models.breed_predictor import BreedPredictor
from app.models.object_detector import ObjectDetector
from app.schemas.cattle import CattlePredictionResponse

class WeightService:
    def __init__(self):
        self.weight_predictor = load_weight_predictor()
        self.scaler_X, self.scaler_y = load_scalers()
        self.breed_predictor = BreedPredictor()
        self.object_detector = ObjectDetector()

    def calculate_weight(self, height: float, width: float, breed: str) -> float:
        features_scaled = self.scaler_X.transform([[height, width]])
        weight_scaled = self.weight_predictor.predict(features_scaled, breed)
        weight = self.scaler_y.inverse_transform([[weight_scaled]])[0][0]
        return weight

    def calculate_milk_yield(self, breed: str, weight: float) -> float:
        milk_predictor = MilkPredictor()
        milk_yield = milk_predictor.predict(breed, weight)
        return milk_yield

    def process_file_upload(self, file: UploadFile = File(...)) -> CattlePredictionResponse:
        if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Invalid file type. Only images are allowed.")
        
        # Save the uploaded file
        file_path = os.path.join("uploads", file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # Perform object detection and breed prediction
        detected_breed = self.object_detector.detect_and_predict(file_path)
        height = detected_breed["height"]
        width = detected_breed["width"]

        # Calculate weight
        weight = self.calculate_weight(height, width, detected_breed["breed"])

        # Calculate milk yield
        milk_yield = self.calculate_milk_yield(detected_breed["breed"], weight)

        # Prepare response
        response = CattlePredictionResponse(
            breed=detected_breed["breed"],
            weight=weight,
            milk_yield=milk_yield
        )

        return response