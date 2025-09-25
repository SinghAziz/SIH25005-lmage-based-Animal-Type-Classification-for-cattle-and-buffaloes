from ultralytics import YOLO
import os

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "runs", "train", "cattle_detection", "weights", "best.pt")
DATASET_YAML = os.path.join(BASE_DIR, "dataset", "yolo_pseudo_labeled", "dataset.yaml")

print("🧪 Evaluating YOLO model...")

# Check if trained model exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ Trained model not found: {MODEL_PATH}")
    exit(1)

# Load trained model
model = YOLO(MODEL_PATH)

# Validate the model
print("📊 Running validation...")
metrics = model.val(data=DATASET_YAML)

print("✅ Evaluation Results:")
print(f"📈 mAP50: {metrics.box.map50:.3f}")
print(f"📈 mAP50-95: {metrics.box.map:.3f}")
print(f"📈 Precision: {metrics.box.mp:.3f}")
print(f"📈 Recall: {metrics.box.mr:.3f}")