from ultralytics import YOLO
import os
import shutil
import json
import numpy as np

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Base directory: {BASE_DIR}")

MODEL_PATH = os.path.join(BASE_DIR, "runs", "train", "cattle_detection", "weights", "best.pt")
TEST_IMAGES = os.path.join(BASE_DIR, "dataset", "Indian_bovine_breeds")
OUTPUT_DIR = os.path.join(BASE_DIR, "all_test_results")  # Single folder for all results

# Create/clear output directory
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔍 Testing YOLO model...")

# Check if trained model exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ Trained model not found: {MODEL_PATH}")
    exit(1)

# Load trained model
model = YOLO(MODEL_PATH)

# Collect all results
all_results = []
test_summary = {
    'total_images': 0,
    'total_detections': 0,    'breeds_tested': [],
    'breed_results': {},
    'confidence_scores': []
}

print("📸 Running inference on all test images...")

# Process each breed directory
for breed_dir in os.listdir(TEST_IMAGES):
    breed_path = os.path.join(TEST_IMAGES, breed_dir)
    
    if os.path.isdir(breed_path):
        print(f"🔍 Processing breed: {breed_dir}")
        
        # Get all images in this breed folder
        image_files = [f for f in os.listdir(breed_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]
        
        if image_files:
            test_summary['breeds_tested'].append(breed_dir)
            test_summary['breed_results'][breed_dir] = {
                'images_count': len(image_files),
                'total_detections': 0,
                'images_with_detections': 0,
                'avg_confidence': 0
            }
            
            # Run prediction on this breed folder
            results = model.predict(
                source=breed_path,
                save=False,  # Don't save images automatically
                conf=0.5,
                verbose=False
            )
            
            breed_confidences = []
            
            # Process results for this breed
            for result in results:
                test_summary['total_images'] += 1
                
                # Count detections
                num_detections = len(result.boxes) if result.boxes is not None else 0
                test_summary['total_detections'] += num_detections
                test_summary['breed_results'][breed_dir]['total_detections'] += num_detections
                
                if num_detections > 0:
                    test_summary['breed_results'][breed_dir]['images_with_detections'] += 1
                    
                    # Collect confidence scores
                    for box in result.boxes:
                        conf = float(box.conf)
                        test_summary['confidence_scores'].append(conf)
                        breed_confidences.append(conf)
                
                # Save annotated image to single folder
                import cv2
                from pathlib import Path
                
                # Read original image
                img = cv2.imread(result.path)
                
                # Draw bounding boxes if any detections
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf)
                        
                        # Draw rectangle
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw confidence
                        cv2.putText(img, f'{conf:.3f}', (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save to single results folder
                original_name = Path(result.path).name
                output_name = f"{breed_dir}_{original_name}"
                output_path = os.path.join(OUTPUT_DIR, output_name)
                cv2.imwrite(output_path, img)
            
            # Calculate average confidence for this breed
            if breed_confidences:
                test_summary['breed_results'][breed_dir]['avg_confidence'] = np.mean(breed_confidences)
            
            all_results.extend(results)

# Calculate overall metrics
if test_summary['total_images'] > 0:
    detection_rate = sum(1 for breed_data in test_summary['breed_results'].values() 
                        for _ in range(breed_data['images_with_detections'])) / test_summary['total_images']
    avg_detections_per_image = test_summary['total_detections'] / test_summary['total_images']
    
    if test_summary['confidence_scores']:
        overall_avg_confidence = np.mean(test_summary['confidence_scores'])
        min_confidence = min(test_summary['confidence_scores'])
        max_confidence = max(test_summary['confidence_scores'])
    else:
        overall_avg_confidence = min_confidence = max_confidence = 0

# Print comprehensive results
print("\n" + "="*60)
print("📊 COMPREHENSIVE TEST RESULTS")
print("="*60)
print(f"📁 Total test images: {test_summary['total_images']}")
print(f"🐄 Breeds tested: {len(test_summary['breeds_tested'])}")
print(f"🎯 Total detections: {test_summary['total_detections']}")
print(f"📊 Average detections per image: {avg_detections_per_image:.2f}")
print(f"📈 Detection rate: {detection_rate:.2%}")
print(f"🔥 Overall average confidence: {overall_avg_confidence:.3f}")
print(f"📊 Confidence range: {min_confidence:.3f} - {max_confidence:.3f}")

print(f"\n🐄 PER-BREED RESULTS:")
for breed, data in test_summary['breed_results'].items():
    breed_detection_rate = data['images_with_detections'] / data['images_count'] * 100
    avg_det_per_img = data['total_detections'] / data['images_count']
    print(f"   {breed}:")
    print(f"      Images: {data['images_count']}")
    print(f"      Total detections: {data['total_detections']}")
    print(f"      Detection rate: {breed_detection_rate:.1f}%")
    print(f"      Avg detections/image: {avg_det_per_img:.2f}")
    print(f"      Avg confidence: {data['avg_confidence']:.3f}")

# Save results to JSON
results_file = os.path.join(OUTPUT_DIR, "test_metrics.json")
with open(results_file, 'w') as f:
    json.dump(test_summary, f, indent=2)

print(f"\n✅ All results saved in: {OUTPUT_DIR}")
print(f"📄 Detailed metrics saved: {results_file}")
print(f"📸 Total annotated images: {test_summary['total_images']}")