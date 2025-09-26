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

def load_models_and_encoders():
    models_and_encoders = {
        "object_detector": load_model(Path(__file__).resolve().parent.parent.parent / "saved_models" / "best_model.pth"),
        "weight_predictor": load_model(Path(__file__).resolve().parent.parent.parent / "saved_models" / "weight_predictor_model.pth"),
        "milk_yield_predictor": load_model(Path(__file__).resolve().parent.parent.parent / "saved_models" / "milk_yield_model.pth"),
        "breed_encoder": load_encoder(Path(__file__).resolve().parent.parent.parent / "saved_models" / "breed_encoder.pkl"),
        "scaler_X": load_scaler(Path(__file__).resolve().parent.parent.parent / "saved_models" / "scaler_X.pkl"),
        "scaler_y": load_scaler(Path(__file__).resolve().parent.parent.parent / "saved_models" / "scaler_y.pkl"),
        "milk_breed_encoder": load_encoder(Path(__file__).resolve().parent.parent.parent / "saved_models" / "milk_breed_encoder.pkl"),
    }
    return models_and_encoders
