from fastapi import UploadFile, File
from fastapi import HTTPException
from app.utils.model_loader import load_object_detector, load_breed_predictor, load_weight_predictor, load_milk_predictor
from app.utils.image_processing import preprocess_image
import os
import json

class DetectionService:
    def __init__(self):
        self.object_detector = load_object_detector()
        self.breed_predictor = load_breed_predictor()
        self.weight_predictor = load_weight_predictor()
        self.milk_predictor = load_milk_predictor()
        self.breed_measurements = self.load_breed_measurements()

    def load_breed_measurements(self):
        with open("Backend/Heightconv.py", "r") as f:
            # Assuming the measurements are defined in a dictionary format
            return json.load(f)

    def detect_and_predict(self, file: UploadFile = File(...)):
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")

        # Save the uploaded file temporarily
        file_location = f"uploads/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(file.file.read())

        # Perform object detection
        detections = self.object_detector.detect(file_location)

        results = []
        for detection in detections:
            breed = self.breed_predictor.predict(detection["image"])
            height = self.breed_measurements[breed]["height_cm"]
            width = self.breed_measurements[breed]["width_cm"]
            weight = self.weight_predictor.predict(height, width, breed)
            milk_yield = self.milk_predictor.predict(breed, weight)

            results.append({
                "breed": breed,
                "height_cm": height,
                "width_cm": width,
                "weight_kg": weight,
                "milk_yield_L_per_day": milk_yield
            })

        # Clean up the uploaded file
        os.remove(file_location)

        return results