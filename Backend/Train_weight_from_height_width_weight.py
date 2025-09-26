import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import joblib  # for saving encoder and scalers

# =========================
# 1. Load JSON from file
# =========================
with open('Backend/Model/height_width_breed_weight.json', 'r') as f:
    data = json.load(f)

# Flatten ranges
for d in data:
    d['height'] = np.mean(d['height_cm'])
    d['width'] = np.mean(d['width_cm'])
    d['weight'] = np.mean(d['weight_kg'])

df = pd.DataFrame(data)
df = df[['breed', 'height', 'width', 'weight']]

# =========================
# 2. One-hot encode breed
# =========================
encoder = OneHotEncoder(sparse_output=False)
breed_encoded = encoder.fit_transform(df[['breed']])
breed_df = pd.DataFrame(breed_encoded, columns=encoder.get_feature_names_out(['breed']))

# =========================
# 3. Scale numerical features
# =========================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_num = df[['height', 'width']].values
X_scaled = scaler_X.fit_transform(X_num)
y_scaled = scaler_y.fit_transform(df[['weight']].values)

# Combine scaled features with one-hot breed
X = np.hstack([X_scaled, breed_encoded])
y = y_scaled

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# =========================
# 4. Define Model
# =========================
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

input_dim = X_train.shape[1]
model = WeightPredictor(input_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =========================
# 5. Train Model
# =========================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 200

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss/len(train_loader):.4f}")

# =========================
# 6. Evaluate
# =========================
model.eval()
with torch.no_grad():
    X_test_device, y_test_device = X_test.to(device), y_test.to(device)
    preds_test = model(X_test_device)
    mse = criterion(preds_test, y_test_device).item()
    print(f"\nTest MSE (scaled): {mse:.4f}")

# =========================
# 7. Save Model, Encoder, Scalers
# =========================
torch.save(model.state_dict(), 'weight_predictor_model.pth')
joblib.dump(encoder, 'breed_encoder.pkl')
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("Model, encoder, and scalers saved!")

# =========================
# 8. Prediction function
# =========================
def predict_weight(height, width, breed_name):
    encoder = joblib.load('breed_encoder.pkl')
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')

    input_dim = 2 + len(encoder.categories_[0])
    model = WeightPredictor(input_dim)
    model.load_state_dict(torch.load('weight_predictor_model.pth', map_location=device))
    model.to(device)
    model.eval()

    # Encode breed
    breed_vec = np.zeros(len(encoder.categories_[0]))
    breed_idx = list(encoder.categories_[0]).index(breed_name)
    breed_vec[breed_idx] = 1

    # Scale height & width
    features_scaled = np.concatenate((scaler_X.transform([[height, width]])[0], breed_vec))
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        pred_scaled = model(features_tensor).item()
    
    # Inverse scale weight
    pred_weight = scaler_y.inverse_transform([[pred_scaled]])[0][0]
    return pred_weight

# =========================
# 9. Example prediction
# =========================
example = data[0]
pred_weight = predict_weight(np.mean(example['height_cm']),
                             np.mean(example['width_cm']),
                             example['breed'])
print(f"\nPredicted weight for {example['breed']} cow: {pred_weight:.2f} kg")
