from ultralytics import YOLO
import torch
import os

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_YAML = os.path.join(BASE_DIR, "dataset", "yolo_pseudo_labeled", "dataset.yaml")
OUTPUT_DIR = os.path.join(BASE_DIR, "runs", "train")

print("Checking for GPU...")

# Simple GPU check
use_gpu = torch.cuda.is_available()

if use_gpu:
    print("GPU detected! Using GPU for training")
    device = 0  # Use first GPU (YOLO format)
    batch_size = 16
    img_size = 640
    epochs = 100
else:
    print("No GPU found. Using CPU (will be slower)")
    device = 'cpu'
    batch_size = 8
    img_size = 416
    epochs = 50  # Fewer epochs for CPU

# Check if dataset exists
if not os.path.exists(DATASET_YAML):
    print(f"Dataset file not found: {DATASET_YAML}")
    print("Please run pseudo_labeling.py first!")
    exit(1)

# Load model
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')

# Train the model
print(f"Starting training with {'GPU' if use_gpu else 'CPU'}...")
print(f"Settings: batch_size={batch_size}, img_size={img_size}, epochs={epochs}")

try:
    results = model.train(
        data=DATASET_YAML,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        name='cattle_detection',
        project=OUTPUT_DIR,
        save=True,
        verbose=True
    )
    
    print("Training completed!")
    print(f"Model saved at: {results.save_dir}/weights/best.pt")
    
except Exception as e:
    print(f" Error during training: {e}")
    print("Trying with reduced settings...")
    
    # Fallback with minimal settings
    results = model.train(
        data=DATASET_YAML,
        epochs=20,
        imgsz=320,
        batch=4,
        device='cpu',  # Force CPU as fallback
        name='cattle_detection_fallback',
        project=OUTPUT_DIR,
        save=True,
        verbose=True
    )
    print("Fallback training completed!")