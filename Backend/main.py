from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import io
import os
import numpy as np
import joblib
import tempfile
from pathlib import Path

app = FastAPI(title="Cattle Analysis Pipeline", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Base directory
BASE_DIR = Path(__file__).parent

# Class names (breed classification)
class_names = [
    'Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur',
    'Bhadawari', 'Brown_Swiss', 'Dangi', 'Deoni', 'Gir',
    'Guernsey', 'Hallikar', 'Hariana', 'Holstein_Friesian',
    'Jaffrabadi', 'Jersey', 'Kangayam', 'Kankrej', 'Kasargod',
    'Kenkatha', 'Kherigarh', 'Khillari', 'Krishna_Valley',
    'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori', 'Nagpuri',
    'Nili_Ravi', 'Nimari', 'Ongole', 'Pulikulam', 'Rathi',
    'Red_Dane', 'Red_Sindhi', 'Sahiwal', 'Surti', 'Tharparkar',
    'Toda', 'Umblachery', 'Vechur'
]

# Animal type mapping
animal_map = {
    'Alambadi': 'Cattle', 'Amritmahal': 'Cattle', 'Ayrshire': 'Cattle',
    'Banni': 'Buffalo', 'Bargur': 'Cattle', 'Bhadawari': 'Buffalo',
    'Brown_Swiss': 'Cattle', 'Dangi': 'Cattle', 'Deoni': 'Cattle',
    'Gir': 'Cattle', 'Guernsey': 'Cattle', 'Hallikar': 'Cattle',
    'Hariana': 'Cattle', 'Holstein_Friesian': 'Cattle', 'Jaffrabadi': 'Buffalo',
    'Jersey': 'Cattle', 'Kangayam': 'Cattle', 'Kankrej': 'Cattle',
    'Kasargod': 'Cattle', 'Kenkatha': 'Cattle', 'Kherigarh': 'Cattle',
    'Khillari': 'Cattle', 'Krishna_Valley': 'Cattle', 'Malnad_gidda': 'Cattle',
    'Mehsana': 'Buffalo', 'Murrah': 'Buffalo', 'Nagori': 'Cattle',
    'Nagpuri': 'Buffalo', 'Nili_Ravi': 'Buffalo', 'Nimari': 'Cattle',
    'Ongole': 'Cattle', 'Pulikulam': 'Cattle', 'Rathi': 'Cattle',
    'Red_Dane': 'Cattle', 'Red_Sindhi': 'Cattle', 'Sahiwal': 'Cattle',
    'Surti': 'Buffalo', 'Tharparkar': 'Cattle', 'Toda': 'Cattle',
    'Umblachery': 'Cattle', 'Vechur': 'Cattle'
}

# Breed measurements (height and width in cm)
BREED_MEASUREMENTS = {
    "Vechur": {"height_cm": 90, "width_cm": 50},
    "Mehsana": {"height_cm": 129, "width_cm": 60},
    "Hallikar": {"height_cm": 130, "width_cm": 60},
    "Amritmahal": {"height_cm": 130, "width_cm": 60},
    "Kankrej": {"height_cm": 140, "width_cm": 65},
    "Sahiwal": {"height_cm": 130, "width_cm": 60},
    "Surti": {"height_cm": 125, "width_cm": 55},
    "Jersey": {"height_cm": 120, "width_cm": 55},
    "Pulikulam": {"height_cm": 120, "width_cm": 55},
    "Nagpuri": {"height_cm": 125, "width_cm": 60},
    "Nagori": {"height_cm": 130, "width_cm": 60},
    "Malnad_gidda": {"height_cm": 95, "width_cm": 45},
    "Dangi": {"height_cm": 110, "width_cm": 50},
    "Murrah": {"height_cm": 135, "width_cm": 70},
    "Jaffrabadi": {"height_cm": 130, "width_cm": 65},
    "Red_Dane": {"height_cm": 140, "width_cm": 65},
    "Krishna_Valley": {"height_cm": 120, "width_cm": 55},
    "Guernsey": {"height_cm": 130, "width_cm": 60},
    "Kherigarh": {"height_cm": 125, "width_cm": 60},
    "Rathi": {"height_cm": 120, "width_cm": 55},
    "Khillari": {"height_cm": 120, "width_cm": 55},
    "Bargur": {"height_cm": 115, "width_cm": 50},
    "Banni": {"height_cm": 110, "width_cm": 50},
    "Holstein_Friesian": {"height_cm": 145, "width_cm": 60},
    "Toda": {"height_cm": 90, "width_cm": 45},
    "Alambadi": {"height_cm": 120, "width_cm": 55},
    "Deoni": {"height_cm": 125, "width_cm": 55},
    "Kangayam": {"height_cm": 110, "width_cm": 50},
    "Kenkatha": {"height_cm": 115, "width_cm": 50},
    "Kasargod": {"height_cm": 95, "width_cm": 45},
    "Nimari": {"height_cm": 120, "width_cm": 55},
    "Tharparkar": {"height_cm": 135, "width_cm": 60},
    "Bhadawari": {"height_cm": 120, "width_cm": 55},
    "Ongole": {"height_cm": 140, "width_cm": 65},
    "Red_Sindhi": {"height_cm": 130, "width_cm": 60},
    "Hariana": {"height_cm": 135, "width_cm": 60},
    "Umblachery": {"height_cm": 115, "width_cm": 50},
    "Gir": {"height_cm": 135, "width_cm": 60},
    "Ayrshire": {"height_cm": 130, "width_cm": 60},
    "Brown_Swiss": {"height_cm": 140, "width_cm": 65},
    "Nili_Ravi": {"height_cm": 140, "width_cm": 70},
}

# Global model variables
yolo_model = None
breed_model = None
weight_encoder = None
weight_scaler_X = None
weight_scaler_y = None
weight_model = None
milk_encoder = None
milk_model = None

# =========================
# Weight Prediction Model Classes and Functions (from Test_weight_test.py)
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

# =========================
# Milk Yield Prediction Model Classes (from Test_Milk_from_breed_and_Weight_test.py)
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

def load_models():
    """Load all required models with proper file checking"""
    global yolo_model, breed_model, weight_encoder, weight_scaler_X, weight_scaler_y, weight_model, milk_encoder, milk_model
    
    print("🔄 Loading models...")
    
    try:
        # Load YOLO model for detection
        yolo_candidates = [
            BASE_DIR / "best.pt",
            BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt",
            BASE_DIR / "Model" / "best.pt"
        ]
        
        for yolo_path in yolo_candidates:
            if yolo_path.exists():
                yolo_model = YOLO(str(yolo_path))
                print(f"✅ YOLO model loaded from: {yolo_path}")
                break
        else:
            print("⚠️ YOLO model not found")
        
        # Load breed classification model
        breed_candidates = [
            BASE_DIR / "Model" / "best_model.pth",
            BASE_DIR / "best_model.pth"
        ]
        
        for breed_path in breed_candidates:
            if breed_path.exists():
                breed_model = models.resnet18(weights=None)
                breed_model.fc = nn.Linear(breed_model.fc.in_features, len(class_names))
                breed_model.load_state_dict(torch.load(breed_path, map_location=device))
                breed_model.to(device)
                breed_model.eval()
                print(f"✅ Breed model loaded from: {breed_path}")
                break
        else:
            print("⚠️ Breed classification model not found")
        
        # Load weight prediction components
        try:
            weight_encoder_path = BASE_DIR / "breed_encoder.pkl"
            weight_scaler_X_path = BASE_DIR / "scaler_X.pkl"
            weight_scaler_y_path = BASE_DIR / "scaler_y.pkl"
            weight_model_path = BASE_DIR / "weight_predictor_model.pth"
            
            if all(p.exists() for p in [weight_encoder_path, weight_scaler_X_path, weight_scaler_y_path, weight_model_path]):
                weight_encoder = joblib.load(weight_encoder_path)
                weight_scaler_X = joblib.load(weight_scaler_X_path)
                weight_scaler_y = joblib.load(weight_scaler_y_path)
                
                input_dim = 2 + len(weight_encoder.categories_[0])
                weight_model = WeightPredictor(input_dim)
                weight_model.load_state_dict(torch.load(weight_model_path, map_location=device))
                weight_model.to(device)
                weight_model.eval()
                print("✅ Weight prediction model loaded successfully")
            else:
                print("⚠️ Weight prediction model files not found")
                print(f"   Looking for: {[str(p) for p in [weight_encoder_path, weight_scaler_X_path, weight_scaler_y_path, weight_model_path]]}")
        
        except Exception as e:
            print(f"❌ Error loading weight model: {e}")
        
        # Load milk yield prediction components
        try:
            milk_encoder_path = BASE_DIR / "milk_breed_encoder.pkl"
            milk_model_path = BASE_DIR / "milk_yield_model.pth"
            
            if milk_encoder_path.exists() and milk_model_path.exists():
                milk_encoder = joblib.load(milk_encoder_path)
                
                input_dim = len(milk_encoder.categories_[0]) + 1
                milk_model = MilkYieldPredictor(input_dim)
                milk_model.load_state_dict(torch.load(milk_model_path, map_location=device))
                milk_model.to(device)
                milk_model.eval()
                print("✅ Milk yield prediction model loaded successfully")
            else:
                print("⚠️ Milk yield prediction model files not found")
                print(f"   Looking for: {milk_encoder_path}, {milk_model_path}")
        
        except Exception as e:
            print(f"❌ Error loading milk model: {e}")
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")

def predict_weight_from_models(height, width, breed_name):
    """Weight prediction function using trained models (from Test_weight_test.py)"""
    global weight_encoder, weight_scaler_X, weight_scaler_y, weight_model
    
    if not all([weight_encoder, weight_scaler_X, weight_scaler_y, weight_model]):
        raise ValueError("Weight prediction models not loaded")
    
    try:
        # Encode breed
        breed_vec = np.zeros(len(weight_encoder.categories_[0]))
        try:
            breed_idx = list(weight_encoder.categories_[0]).index(breed_name)
        except ValueError:
            # If breed not found in encoder, use a fallback estimation
            raise ValueError(f"Breed '{breed_name}' not found in weight encoder categories")
        
        breed_vec[breed_idx] = 1

        # Scale height & width
        X_num_scaled = weight_scaler_X.transform([[height, width]])  # shape (1,2)

        # Combine features
        features = np.hstack([X_num_scaled, breed_vec.reshape(1, -1)])
        features = torch.tensor(features, dtype=torch.float32).to(device)

        # Predict & inverse scale
        with torch.no_grad():
            pred_scaled = weight_model(features).cpu().numpy()
        pred_weight = weight_scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0][0]
        
        return pred_weight
    
    except Exception as e:
        print(f"Weight prediction error: {e}")
        raise

def predict_milk_from_models(breed_name, weight):
    """Milk yield prediction function using trained models (from Test_Milk_from_breed_and_Weight_test.py)"""
    global milk_encoder, milk_model
    
    if not all([milk_encoder, milk_model]):
        raise ValueError("Milk prediction models not loaded")
    
    try:
        # Encode breed
        breed_vec = np.zeros(len(milk_encoder.categories_[0]))
        breed_idx = list(milk_encoder.categories_[0]).index(breed_name)
        breed_vec[breed_idx] = 1

        # Combine breed + weight
        features = np.concatenate((breed_vec, [weight]))
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            return milk_model(features).item()
            
    except Exception as e:
        print(f"Milk prediction error: {e}")
        raise

# Load models on startup
load_models()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class CattleAnalysisPipeline:
    def __init__(self):
        self.yolo_model = yolo_model
        self.breed_model = breed_model
    
    def detect_animal(self, image_bytes):
        """Step 1: Detect animal using YOLO"""
        if not self.yolo_model:
            return {"error": "YOLO model not available"}
        
        try:
            # Use temporary file with proper cleanup
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_file.write(image_bytes)
                temp_path = temp_file.name
            
            try:
                # Run YOLO detection
                results = self.yolo_model.predict(temp_path, conf=0.5, verbose=False)
                
                detection_data = {
                    "animals_detected": 0,
                    "bbox_data": None,
                    "confidence": 0
                }
                
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        box = boxes[0]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        detection_data.update({
                            "animals_detected": len(boxes),
                            "bbox_data": {
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                "bbox_width": x2 - x1,
                                "bbox_height": y2 - y1
                            },
                            "confidence": float(box.conf[0])
                        })
                
                return detection_data
                
            finally:
                os.unlink(temp_path)
            
        except Exception as e:
            return {"error": f"Detection failed: {str(e)}"}
    
    def predict_breed(self, image_bytes):
        """Step 2: Predict breed"""
        if not self.breed_model:
            return {"error": "Breed model not available"}
        
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = self.breed_model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probabilities, 1)
            
            breed = class_names[pred_idx.item()]
            animal_type = animal_map.get(breed, "Unknown")
            
            return {
                "breed": breed,
                "animal_type": animal_type,
                "confidence": float(confidence.item())
            }
            
        except Exception as e:
            return {"error": f"Breed prediction failed: {str(e)}"}
    
    def calculate_dimensions(self, breed, bbox_data=None):
        """Step 3: Calculate height and width"""
        try:
            if breed not in BREED_MEASUREMENTS:
                # Use average values for unknown breeds
                generic_height = 125
                generic_width = 60
            else:
                generic_height = BREED_MEASUREMENTS[breed]["height_cm"]
                generic_width = BREED_MEASUREMENTS[breed]["width_cm"]
            
            dimensions = {
                "generic_height_cm": generic_height,
                "generic_width_cm": generic_width,
                "scaled_height_cm": None,
                "scaled_width_cm": None
            }
            
            # If bbox data is available, calculate scaled dimensions
            if bbox_data:
                bbox_width_px = bbox_data.get("bbox_width")
                bbox_height_px = bbox_data.get("bbox_height")
                
                if bbox_width_px and bbox_height_px:
                    # Scale based on aspect ratio
                    pixel_ratio_w = generic_width / bbox_width_px
                    pixel_ratio_h = generic_height / bbox_height_px
                    
                    dimensions.update({
                        "scaled_width_cm": round(bbox_width_px * pixel_ratio_w, 2),
                        "scaled_height_cm": round(bbox_height_px * pixel_ratio_h, 2)
                    })
            
            return dimensions
            
        except Exception as e:
            return {"error": f"Dimension calculation failed: {str(e)}"}
    
    def predict_weight(self, breed, height_cm, width_cm):
        """Step 4: Predict weight using trained models"""
        try:
            # Try using trained model first
            if weight_model and weight_encoder and weight_scaler_X and weight_scaler_y:
                try:
                    weight_kg = predict_weight_from_models(height_cm, width_cm, breed)
                    return {"weight_kg": round(weight_kg, 2), "method": "trained_model"}
                except Exception as e:
                    print(f"Trained model failed: {e}")
                    # Fall back to estimation
            
            # Fallback estimation
            breed_factors = {
                "Holstein_Friesian": 3.2, "Brown_Swiss": 3.0, "Jersey": 2.5,
                "Ayrshire": 2.8, "Guernsey": 2.6, "Gir": 2.2, "Sahiwal": 2.4,
                "Red_Sindhi": 2.0, "Tharparkar": 1.8, "Kankrej": 2.3,
                "Murrah": 3.5, "Mehsana": 3.0, "Jaffrabadi": 3.2,
                "Surti": 2.8, "Nili_Ravi": 3.3, "Vechur": 1.2
            }
            
            factor = breed_factors.get(breed, 2.5)
            weight_kg = (height_cm * width_cm * factor) / 100
            # Reasonable bounds
            weight_kg = max(150, min(800, weight_kg))
            
            return {"weight_kg": round(weight_kg, 2), "method": "estimation"}
            
        except Exception as e:
            return {"error": f"Weight prediction failed: {str(e)}"}
    
    def predict_milk_yield(self, breed, weight_kg, animal_type):
        """Step 5: Predict milk yield using trained models"""
        try:
            if animal_type.lower() not in ['cattle', 'buffalo']:
                return {"error": "Milk yield only available for cattle and buffalo"}
            
            # Try using trained model first
            if milk_model and milk_encoder:
                try:
                    milk_per_day = predict_milk_from_models(breed, weight_kg)
                    return {
                        "avg_milk_per_day_liters": round(milk_per_day, 2),
                        "estimated_yearly_yield_liters": round(milk_per_day * 305, 2),
                        "method": "trained_model"
                    }
                except Exception as e:
                    print(f"Trained milk model failed: {e}")
                    # Fall back to breed averages
            
            # Fallback using breed averages
            breed_averages = {
                "Holstein_Friesian": 25, "Jersey": 20, "Brown_Swiss": 22,
                "Ayrshire": 18, "Guernsey": 16, "Gir": 12, "Sahiwal": 15,
                "Red_Sindhi": 10, "Tharparkar": 8, "Kankrej": 10,
                "Murrah": 15, "Mehsana": 12, "Jaffrabadi": 10,
                "Surti": 8, "Nili_Ravi": 14
            }
            
            base_milk = breed_averages.get(breed, 12)  # Default 12L/day
            
            # Adjust for weight (heavier animals produce more milk, within limits)
            weight_factor = min(1.3, weight_kg / 400)
            milk_per_day = base_milk * weight_factor
            
            return {
                "avg_milk_per_day_liters": round(milk_per_day, 2),
                "estimated_yearly_yield_liters": round(milk_per_day * 305, 2),
                "method": "breed_average"
            }
            
        except Exception as e:
            return {"error": f"Milk yield prediction failed: {str(e)}"}
    
    def analyze_complete(self, image_bytes):
        """Complete analysis pipeline"""
        results = {
            "status": "processing",
            "detection": {},
            "breed_prediction": {},
            "dimensions": {},
            "weight_prediction": {},
            "milk_prediction": {},
            "summary": {}
        }
        
        try:
            # Step 1: Animal detection
            print("Step 1: Detecting animal...")
            detection_result = self.detect_animal(image_bytes)
            results["detection"] = detection_result
            
            if "error" in detection_result:
                results["status"] = "error"
                return results
            
            # Step 2: Breed prediction
            print("Step 2: Predicting breed...")
            breed_result = self.predict_breed(image_bytes)
            results["breed_prediction"] = breed_result
            
            if "error" in breed_result:
                results["status"] = "error"
                return results
            
            breed = breed_result["breed"]
            animal_type = breed_result["animal_type"]
            
            # Step 3: Calculate dimensions
            print("Step 3: Calculating dimensions...")
            dimensions_result = self.calculate_dimensions(
                breed, 
                detection_result.get("bbox_data")
            )
            results["dimensions"] = dimensions_result
            
            if "error" in dimensions_result:
                results["status"] = "error"
                return results
            
            # Use scaled dimensions if available, otherwise use generic
            height_cm = dimensions_result.get("scaled_height_cm") or dimensions_result["generic_height_cm"]
            width_cm = dimensions_result.get("scaled_width_cm") or dimensions_result["generic_width_cm"]
            
            # Step 4: Predict weight
            print("Step 4: Predicting weight...")
            weight_result = self.predict_weight(breed, height_cm, width_cm)
            results["weight_prediction"] = weight_result
            
            if "error" in weight_result:
                results["status"] = "error"
                return results
            
            # Step 5: Predict milk yield
            print("Step 5: Predicting milk yield...")
            milk_result = self.predict_milk_yield(
                breed, 
                weight_result["weight_kg"], 
                animal_type
            )
            results["milk_prediction"] = milk_result
            
            # Create summary
            results["summary"] = {
                "animal_type": animal_type,
                "breed": breed,
                "height_cm": height_cm,
                "width_cm": width_cm,
                "weight_kg": weight_result.get("weight_kg", 0),
                "daily_milk_liters": milk_result.get("avg_milk_per_day_liters", 0),
                "yearly_milk_liters": milk_result.get("estimated_yearly_yield_liters", 0),
                "detection_confidence": detection_result.get("confidence", 0),
                "breed_confidence": breed_result.get("confidence", 0),
                "weight_method": weight_result.get("method", "unknown"),
                "milk_method": milk_result.get("method", "unknown")
            }
            
            results["status"] = "success"
            print("✅ Analysis completed successfully!")
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            print(f"❌ Analysis failed: {e}")
        
        return results

# Initialize pipeline
pipeline = CattleAnalysisPipeline()

@app.get("/")
def root():
    return {
        "message": "Cattle Analysis Pipeline API",
        "endpoints": {
            "complete_analysis": "/analyze/",
            "breed_only": "/predict/",
            "health": "/health"
        },
        "status": "running"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models": {
            "yolo": yolo_model is not None,
            "breed": breed_model is not None,
            "weight": weight_model is not None,
            "milk": milk_model is not None
        }
    }

@app.post("/analyze/")
async def complete_analysis(file: UploadFile = File(...)):
    """Complete analysis: detection + breed + dimensions + weight + milk yield"""
    try:
        if not file.content_type.startswith('image/'):
            return JSONResponse(
                content={"error": "File must be an image"}, 
                status_code=400
            )
        
        contents = await file.read()
        result = pipeline.analyze_complete(contents)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(
            content={"error": f"Analysis failed: {str(e)}"}, 
            status_code=500
        )

@app.post("/predict/")
async def predict_breed_only(file: UploadFile = File(...)):
    """Breed prediction only (for compatibility)"""
    try:
        contents = await file.read()
        result = pipeline.predict_breed(contents)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)