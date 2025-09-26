import numpy as np
import torch
import joblib

# =========================
# 1. Load model and encoder
# =========================
class MilkYieldPredictor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load encoder
encoder = joblib.load('milk_breed_encoder.pkl')

# Load model
input_dim = len(encoder.categories_[0]) + 1
model = MilkYieldPredictor(input_dim)
model.load_state_dict(torch.load('milk_yield_model.pth', map_location=device))
model.to(device)
model.eval()

# =========================
# 2. Prediction function
# =========================
def predict_milk_yield(breed_name, weight):
    # Encode breed
    breed_vec = np.zeros(len(encoder.categories_[0]))
    breed_idx = list(encoder.categories_[0]).index(breed_name)
    breed_vec[breed_idx] = 1

    # Combine breed + weight
    features = np.concatenate((breed_vec, [weight]))
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        return model(features).item()

# =========================
# 3. Example usage
# =========================
example_breed = "Vechur"
example_weight = 120

predicted_milk = predict_milk_yield(example_breed, example_weight)
print(f"Predicted milk yield for {example_breed} (weight {example_weight} kg): {predicted_milk:.2f} L/day")
