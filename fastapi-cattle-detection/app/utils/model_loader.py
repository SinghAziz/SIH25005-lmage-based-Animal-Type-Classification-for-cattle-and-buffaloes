from pathlib import Path
import torch
import joblib

def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model

def load_encoder(encoder_path):
    return joblib.load(encoder_path)

def load_scaler(scaler_path):
    return joblib.load(scaler_path)

# Base path to saved models
BASE_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "saved_models"

# Individual loader functions
def load_milk_model():
    return load_model(BASE_MODEL_PATH / "milk_yield_model.pth")

def load_breed_encoder():
    return load_encoder(BASE_MODEL_PATH / "milk_breed_encoder.pkl")

def load_scalers(scaler_name):
    return load_scaler(BASE_MODEL_PATH / scaler_name)

def load_weight_model():
    return load_model(BASE_MODEL_PATH / "weight_predictor_model.pth")

def load_object_detection_model():
    return load_model(BASE_MODEL_PATH / "best_model.pth")

def load_models_and_encoders():
    models_and_encoders = {
        "object_detector": load_model(BASE_MODEL_PATH / "best_model.pth"),
        "weight_predictor": load_model(BASE_MODEL_PATH / "weight_predictor_model.pth"),
        "milk_yield_predictor": load_model(BASE_MODEL_PATH / "milk_yield_model.pth"),
        "breed_encoder": load_encoder(BASE_MODEL_PATH / "breed_encoder.pkl"),
        "scaler_X": load_scaler(BASE_MODEL_PATH / "scaler_X.pkl"),
        "scaler_y": load_scaler(BASE_MODEL_PATH / "scaler_y.pkl"),
        "milk_breed_encoder": load_encoder(BASE_MODEL_PATH / "milk_breed_encoder.pkl"),
    }
    return models_and_encoders
