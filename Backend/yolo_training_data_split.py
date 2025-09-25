import os
from PIL import Image
import shutil

# Path to your dataset
dataset_path = "Backend/dataset/Indian_bovine_breeds"
output_path = "Backend/dataset/yolo_training/"

if os.path.exists(output_path) and os.listdir(output_path):
    print(f"Output directory '{output_path}' already exists and is not empty!")
    print("Please remove or rename the existing directory before running this script.")
    exit(1)


os.makedirs(output_path, exist_ok=True)

# Counter to avoid filename conflicts
image_counter = 1

# Iterate over each breed folder
for breed_folder in os.listdir(dataset_path):
    breed_path = os.path.join(dataset_path, breed_folder)
    if not os.path.isdir(breed_path):
        continue
    
    images_info = []
    
    # Get resolution of each image
    for img_file in os.listdir(breed_path):
        img_path = os.path.join(breed_path, img_file)
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                resolution = width * height
                images_info.append((img_file, resolution))
        except:
            continue  # skip non-image files

    # Sort images by resolution (largest first)
    images_info.sort(key=lambda x: x[1], reverse=True)
    
    # Pick top 10
    top_images = images_info[:10]
    
    # Copy top images to single folder with breed prefix
    for img_file, _ in top_images:
        # Get file extension
        file_ext = os.path.splitext(img_file)[1]
        
        # Create new filename with breed name and counter to avoid conflicts
        new_filename = f"{breed_folder}_{image_counter:03d}{file_ext}"
        
        # Copy with new name
        shutil.copy(
            os.path.join(breed_path, img_file), 
            os.path.join(output_path, new_filename)
        )
        image_counter += 1
    
    print(f"{breed_folder} done - {len(top_images)} images copied")

print(f"Done! All {image_counter-1} images are saved in: {output_path}")
