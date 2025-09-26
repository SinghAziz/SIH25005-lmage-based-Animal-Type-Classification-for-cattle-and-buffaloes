import torch
import torch.nn as nn
import numpy as np
import joblib

# =========================
# 1. Load saved encoder, scalers, and model
# =========================
encoder = joblib.load('breed_encoder.pkl')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

class WeightPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

input_dim = 2 + len(encoder.categories_[0])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WeightPredictor(input_dim)
model.load_state_dict(torch.load('weight_predictor_model.pth', map_location=device))
model.to(device)
model.eval()

# =========================
# 2. Prediction function
# =========================
def predict_weight(height, width, breed_name):
    # Encode breed
    breed_vec = np.zeros(len(encoder.categories_[0]))
    try:
        breed_idx = list(encoder.categories_[0]).index(breed_name)
    except ValueError:
        raise ValueError(f"Breed '{breed_name}' not found in encoder categories!")
    breed_vec[breed_idx] = 1

    # Scale height & width
    X_num_scaled = scaler_X.transform([[height, width]])  # shape (1,2)

    # Combine features
    features = np.hstack([X_num_scaled, breed_vec.reshape(1, -1)])
    features = torch.tensor(features, dtype=torch.float32).to(device)

    # Predict & inverse scale
    with torch.no_grad():
        pred_scaled = model(features).cpu().numpy()
    pred_weight = scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0][0]
    return pred_weight

# =========================
# 3. Example predictions
# =========================
example_cow = {
    'breed': 'Mehsana',
    'height': 147,
    'width': 100
}

predicted_weight = predict_weight(example_cow['height'], example_cow['width'], example_cow['breed'])
print(f"Predicted weight for {example_cow['breed']} cow: {predicted_weight:.2f} kg")

# Test multiple
test_cows = [
    {'breed': 'Vechur', 'height': 97, 'width': 57},
    {'breed': 'Murrah', 'height': 150, 'width': 105}
]

for cow in test_cows:
    w = predict_weight(cow['height'], cow['width'], cow['breed'])
    print(f"{cow['breed']}: {w:.2f} kg")
