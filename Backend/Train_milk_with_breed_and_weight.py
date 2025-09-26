import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import joblib
from tqdm import tqdm
import re

# =========================
# 1. Load JSON
# =========================
with open('breed_weight_milk.json', 'r') as f:
    data = json.load(f)

# =========================
# 2. Process milk yield to numeric
# =========================
def parse_milk_yield(yield_str):
    # extract first number range in L/day
    if "L/day" in yield_str:
        match = re.search(r"(\d+\.?\d*)\s*–\s*(\d+\.?\d*)", yield_str)
        if match:
            return (float(match.group(1)) + float(match.group(2))) / 2
        else:
            num = re.search(r"(\d+\.?\d*)", yield_str)
            return float(num.group(1)) if num else np.nan
    else:
        # if no L/day, fallback to average of numbers in string
        numbers = re.findall(r"\d+\.?\d*", yield_str)
        numbers = [float(n) for n in numbers]
        return np.mean(numbers) if numbers else np.nan

# convert JSON to DataFrame
df = pd.DataFrame(data)
df['milk_yield_numeric'] = df['milk_yield'].apply(parse_milk_yield)
df['weight_avg'] = (df['weight_male_kg'] + df['weight_female_kg']) / 2

# drop rows with NaN milk yield
df = df.dropna(subset=['milk_yield_numeric'])

# =========================
# 3. One-hot encode breed
# =========================
encoder = OneHotEncoder(sparse_output=False)
breed_encoded = encoder.fit_transform(df[['breed']])
breed_df = pd.DataFrame(breed_encoded, columns=encoder.get_feature_names_out(['breed']))

# Features and target
X = pd.concat([breed_df, df[['weight_avg']]], axis=1).values
y = df['milk_yield_numeric'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# =========================
# 4. Define Model
# =========================
class MilkYieldPredictor(nn.Module):
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
model = MilkYieldPredictor(input_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =========================
# 5. Train Model
# =========================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 150

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
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss/len(train_loader):.2f}")

# =========================
# 6. Evaluate
# =========================
model.eval()
with torch.no_grad():
    X_test, y_test = X_test.to(device), y_test.to(device)
    preds = model(X_test)
    mse = criterion(preds, y_test).item()
    print(f"\nTest MSE: {mse:.2f}")

# =========================
# 7. Save model & encoder
# =========================
torch.save(model.state_dict(), 'milk_yield_model.pth')
joblib.dump(encoder, 'milk_breed_encoder.pkl')
print("Milk yield model and encoder saved!")

# =========================
# 8. Prediction function
# =========================
def predict_milk_yield(breed_name, weight):
    encoder = joblib.load('milk_breed_encoder.pkl')
    model = MilkYieldPredictor(len(encoder.categories_[0])+1)
    model.load_state_dict(torch.load('milk_yield_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    breed_vec = np.zeros(len(encoder.categories_[0]))
    breed_idx = list(encoder.categories_[0]).index(breed_name)
    breed_vec[breed_idx] = 1
    
    features = np.concatenate((breed_vec, [weight]))
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        return model(features).item()

# Example
example_breed = "Murrah"
example_weight = 450
predicted_milk = predict_milk_yield(example_breed, example_weight)
print(f"Predicted milk yield for {example_breed} cow (weight {example_weight} kg): {predicted_milk:.2f} L/day")
