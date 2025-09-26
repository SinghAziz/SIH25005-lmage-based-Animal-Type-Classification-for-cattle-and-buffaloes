from pydantic import BaseModel
import joblib
import torch
import numpy as np

class WeightPredictor:
    def __init__(self):
        self.model = self.load_model()
        self.encoder = joblib.load('saved_models/breed_encoder.pkl')
        self.scaler_X = joblib.load('saved_models/scaler_X.pkl')
        self.scaler_y = joblib.load('saved_models/scaler_y.pkl')

    def load_model(self):
        input_dim = 2 + len(self.encoder.categories_[0])
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        model.load_state_dict(torch.load('saved_models/weight_predictor_model.pth', map_location='cpu'))
        model.eval()
        return model

    def predict_weight(self, height: float, width: float, breed_name: str) -> float:
        breed_vec = np.zeros(len(self.encoder.categories_[0]))
        breed_idx = list(self.encoder.categories_[0]).index(breed_name)
        breed_vec[breed_idx] = 1

        features_scaled = np.concatenate((self.scaler_X.transform([[height, width]])[0], breed_vec))
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_scaled = self.model(features_tensor).item()

        pred_weight = self.scaler_y.inverse_transform([[pred_scaled]])[0][0]
        return pred_weight