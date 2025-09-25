import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "all_test_results")
METRICS_FILE = os.path.join(RESULTS_DIR, "test_metrics.json")

if not os.path.exists(METRICS_FILE):
    print("❌ No test metrics found. Please run yolo_test_combined.py first!")
    exit(1)

# Load test results
with open(METRICS_FILE, 'r') as f:
    test_data = json.load(f)

print("DETAILED METRICS ANALYSIS")
print("="*50)

# Overall metrics
total_images = test_data['total_images']
total_detections = test_data['total_detections']
confidence_scores = test_data['confidence_scores']

# Calculate advanced metrics
images_with_detections = sum(breed_data['images_with_detections'] 
                           for breed_data in test_data['breed_results'].values())

detection_rate = images_with_detections / total_images * 100 if total_images > 0 else 0
avg_detections_per_image = total_detections / total_images if total_images > 0 else 0

if confidence_scores:
    conf_mean = np.mean(confidence_scores)
    conf_std = np.std(confidence_scores)
    conf_min = min(confidence_scores)
    conf_max = max(confidence_scores)
    
    # Confidence quartiles
    conf_q25 = np.percentile(confidence_scores, 25)
    conf_q50 = np.percentile(confidence_scores, 50)
    conf_q75 = np.percentile(confidence_scores, 75)
else:
    conf_mean = conf_std = conf_min = conf_max = 0
    conf_q25 = conf_q50 = conf_q75 = 0

print(f"📈 OVERALL PERFORMANCE:")
print(f"   Total Images: {total_images}")
print(f"   Total Detections: {total_detections}")
print(f"   Images with Detections: {images_with_detections}")
print(f"   Detection Rate: {detection_rate:.1f}%")
print(f"   Avg Detections/Image: {avg_detections_per_image:.2f}")

print(f"\n🔥 CONFIDENCE STATISTICS:")
print(f"   Mean: {conf_mean:.3f}")
print(f"   Std Dev: {conf_std:.3f}")
print(f"   Min: {conf_min:.3f}")
print(f"   Max: {conf_max:.3f}")
print(f"   25th Percentile: {conf_q25:.3f}")
print(f"   Median (50th): {conf_q50:.3f}")
print(f"   75th Percentile: {conf_q75:.3f}")

# Confidence distribution analysis
if confidence_scores:
    high_conf_count = sum(1 for c in confidence_scores if c >= 0.8)
    med_conf_count = sum(1 for c in confidence_scores if 0.5 <= c < 0.8)
    low_conf_count = sum(1 for c in confidence_scores if c < 0.5)
    
    print(f"\n📊 CONFIDENCE DISTRIBUTION:")
    print(f"   High Confidence (≥0.8): {high_conf_count} ({high_conf_count/len(confidence_scores)*100:.1f}%)")
    print(f"   Medium Confidence (0.5-0.8): {med_conf_count} ({med_conf_count/len(confidence_scores)*100:.1f}%)")
    print(f"   Low Confidence (<0.5): {low_conf_count} ({low_conf_count/len(confidence_scores)*100:.1f}%)")

# Create visualization
plt.figure(figsize=(15, 10))

# 1. Confidence histogram
plt.subplot(2, 3, 1)
if confidence_scores:
    plt.hist(confidence_scores, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(conf_mean, color='red', linestyle='--', label=f'Mean: {conf_mean:.3f}')
    plt.axvline(conf_q50, color='green', linestyle='--', label=f'Median: {conf_q50:.3f}')
plt.title('Confidence Score Distribution')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Detection rate by breed
plt.subplot(2, 3, 2)
breeds = list(test_data['breed_results'].keys())
detection_rates = [test_data['breed_results'][breed]['images_with_detections'] / 
                  test_data['breed_results'][breed]['images_count'] * 100 
                  for breed in breeds]
plt.bar(range(len(breeds)), detection_rates)
plt.title('Detection Rate by Breed')
plt.xlabel('Breed')
plt.ylabel('Detection Rate (%)')
plt.xticks(range(len(breeds)), breeds, rotation=45)
plt.grid(True, alpha=0.3)

# 3. Average detections per image by breed
plt.subplot(2, 3, 3)
avg_detections = [test_data['breed_results'][breed]['total_detections'] / 
                 test_data['breed_results'][breed]['images_count'] 
                 for breed in breeds]
plt.bar(range(len(breeds)), avg_detections)
plt.title('Avg Detections per Image by Breed')
plt.xlabel('Breed')
plt.ylabel('Avg Detections')
plt.xticks(range(len(breeds)), breeds, rotation=45)
plt.grid(True, alpha=0.3)

# 4. Images count by breed
plt.subplot(2, 3, 4)
image_counts = [test_data['breed_results'][breed]['images_count'] for breed in breeds]
plt.bar(range(len(breeds)), image_counts)
plt.title('Number of Test Images by Breed')
plt.xlabel('Breed')
plt.ylabel('Image Count')
plt.xticks(range(len(breeds)), breeds, rotation=45)
plt.grid(True, alpha=0.3)

# 5. Confidence by breed
plt.subplot(2, 3, 5)
breed_confidences = [test_data['breed_results'][breed]['avg_confidence'] for breed in breeds]
plt.bar(range(len(breeds)), breed_confidences)
plt.title('Average Confidence by Breed')
plt.xlabel('Breed')
plt.ylabel('Avg Confidence')
plt.xticks(range(len(breeds)), breeds, rotation=45)
plt.grid(True, alpha=0.3)

# 6. Overall summary pie chart
plt.subplot(2, 3, 6)
labels = ['Images with Detections', 'Images without Detections']
sizes = [images_with_detections, total_images - images_with_detections]
colors = ['lightgreen', 'lightcoral']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Overall Detection Success Rate')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'test_metrics_analysis.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"\n📈 Visualization saved: {RESULTS_DIR}/test_metrics_analysis.png")
print(f"📁 All results available in: {RESULTS_DIR}")