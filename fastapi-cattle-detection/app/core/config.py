from pydantic import BaseSettings

class Settings(BaseSettings):
    # Define your application settings here
    model_path: str = "saved_models/best_model.pth"
    weight_model_path: str = "saved_models/weight_predictor_model.pth"
    milk_model_path: str = "saved_models/milk_yield_model.pth"
    breed_encoder_path: str = "saved_models/breed_encoder.pkl"
    scaler_x_path: str = "saved_models/scaler_X.pkl"
    scaler_y_path: str = "saved_models/scaler_y.pkl"
    milk_breed_encoder_path: str = "saved_models/milk_breed_encoder.pkl"
    uploads_dir: str = "uploads"

    class Config:
        env_file = ".env"

settings = Settings()