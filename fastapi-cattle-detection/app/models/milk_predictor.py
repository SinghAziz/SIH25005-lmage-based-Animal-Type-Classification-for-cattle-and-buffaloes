from pydantic import BaseModel
import joblib
import torch
import numpy as np
from app.utils.model_loader import load_milk_model, load_breed_encoder, load_scalers

class MilkYieldPredictor:
    def __init__(self):
        self.model = load_milk_model()
        self.encoder = load_breed_encoder()
        self.scaler_X = load_scalers('scaler_X.pkl')
        self.scaler_y = load_scalers('scaler_y.pkl')

    def predict(self, breed_name: str, weight: float) -> float:
        breed_vec = np.zeros(len(self.encoder.categories_[0]))
        breed_idx = list(self.encoder.categories_[0]).index(breed_name)
        breed_vec[breed_idx] = 1

        features_scaled = np.concatenate((breed_vec, [weight]))
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_scaled = self.model(features_tensor).item()

        pred_weight = self.scaler_y.inverse_transform([[pred_scaled]])[0][0]
        return pred_weight

class MilkYieldRequest(BaseModel):
    breed: str
    weight: float

class MilkYieldResponse(BaseModel):
    predicted_milk_yield: float