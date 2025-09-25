    import os
    import random
    import shutil
    from ultralytics import YOLO
    from PIL import Image
    import yaml

    # ---------------------------
    # CONFIGURATION
    # ---------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Backend directory
    PROJECT_ROOT = os.path.dirname(BASE_DIR)  # Main project directory

    # Input: where your single folder of images is located
    IMAGES_FOLDER = os.path.join(BASE_DIR, "dataset", "yolo_training")

    # Output: where to create the YOLO dataset
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "dataset", "yolo_pseudo_labeled")

    TRAIN_SPLIT = 0.8  # 80% train, 20% val
    CONF_THRESHOLD = 0.5  # Only keep detections above this confidence
    MODEL = "yolov8n.pt"  # Pretrained model for pseudo-labeling

    # Custom class mapping for cattle/buffalo classification
    CLASS_MAPPING = {
        0: 0,  # person -> animal (you can adjust this)
        # Add more mappings as needed
    }

    print(f"Looking for images in: {IMAGES_FOLDER}")
    print(f"Output will be saved to: {OUTPUT_FOLDER}")

    # ---------------------------
    # STEP 1: Check if output directory exists
    # ---------------------------
    if os.path.exists(OUTPUT_FOLDER) and os.listdir(OUTPUT_FOLDER):
        print(f"❌ Output directory '{OUTPUT_FOLDER}' already exists and is not empty!")
        print("Please remove or rename the existing directory before running this script.")
        exit(1)

    # ---------------------------
    # STEP 2: Load Pretrained YOLOv8
    # ---------------------------
    print("Loading YOLO model...")
    model = YOLO(MODEL)

    # Check if images folder exists and has images
    if not os.path.exists(IMAGES_FOLDER):
        print(f"❌ Images folder not found: {IMAGES_FOLDER}")
        print("Please run yolo_training_data_split.py first to create the images folder.")
        exit(1)

    # Collect all images
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    all_images = [f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith(image_extensions)]

    if not all_images:
        print(f"❌ No images found in {IMAGES_FOLDER}")
        print("Please add your cattle/buffalo images to this folder.")
        exit(1)

    print(f"Found {len(all_images)} images for processing.")
    random.shuffle(all_images)

    # Split into train/val
    split_idx = int(len(all_images) * TRAIN_SPLIT)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    print(f"Train set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")

    # ---------------------------
    # STEP 3: Create output directories
    # ---------------------------
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_FOLDER, 'images', subset), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_FOLDER, 'labels', subset), exist_ok=True)

    # ---------------------------
    # STEP 4: Helper function to run prediction and save YOLO txt
    # ---------------------------
    def predict_and_save(images, subset):
        img_out_folder = os.path.join(OUTPUT_FOLDER, "images", subset)
        label_out_folder = os.path.join(OUTPUT_FOLDER, "labels", subset)
        
        processed_count = 0
        
        for i, img_file in enumerate(images):
            print(f"Processing {subset} {i+1}/{len(images)}: {img_file}")
            
            img_path = os.path.join(IMAGES_FOLDER, img_file)
            
            try:
                # Run YOLO prediction
                results = model.predict(source=img_path, save=False, verbose=False, conf=CONF_THRESHOLD)
                pred = results[0]
                
                # Get image dimensions
                h, w = pred.orig_shape[:2]
                
                # Copy image to output folder
                dst_img_path = os.path.join(img_out_folder, img_file)
                shutil.copy2(img_path, dst_img_path)
                
                # Create label file (even if empty)
                txt_file = os.path.join(label_out_folder, os.path.splitext(img_file)[0] + ".txt")
                
                # Check if any detections were found
                if pred.boxes is not None and len(pred.boxes) > 0:
                    boxes = pred.boxes.xyxy.cpu().numpy()
                    scores = pred.boxes.conf.cpu().numpy()
                    classes = pred.boxes.cls.cpu().numpy().astype(int)
                    
                    # Filter by confidence
                    keep = scores >= CONF_THRESHOLD
                    if keep.any():
                        boxes, classes, scores = boxes[keep], classes[keep], scores[keep]
                        
                        # Save YOLO format labels
                        with open(txt_file, "w") as f:
                            for cls, (x1, y1, x2, y2), conf in zip(classes, boxes, scores):
                                # Convert to YOLO format (normalized coordinates)
                                xc = ((x1 + x2) / 2) / w
                                yc = ((y1 + y2) / 2) / h
                                bw = (x2 - x1) / w
                                bh = (y2 - y1) / h
                                
                                # Use mapped class or original class
                                mapped_cls = CLASS_MAPPING.get(cls, 0)  # Default to class 0
                                
                                f.write(f"{mapped_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                    else:
                        # Create empty label file
                        open(txt_file, 'w').close()
                else:
                    # Create empty label file if no detections
                    open(txt_file, 'w').close()
                
                processed_count += 1
                
            except Exception as e:
                print(f"❌ Error processing {img_file}: {str(e)}")
                continue
        
        print(f"✅ Successfully processed {processed_count}/{len(images)} {subset} images")
        return processed_count

    # ---------------------------
    # STEP 5: Process images
    # ---------------------------
    print("\n" + "="*50)
    print("Starting pseudo-labeling process...")
    print("="*50)

    train_processed = predict_and_save(train_images, "train")
    val_processed = predict_and_save(val_images, "val")

    # ---------------------------
    # STEP 6: Create dataset.yaml file
    # ---------------------------
    yaml_content = {
        'path': OUTPUT_FOLDER,
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,  # Number of classes (adjust as needed)
        'names': ['animal']  # Class names (adjust for cattle/buffalo)
    }

    yaml_path = os.path.join(OUTPUT_FOLDER, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    # ---------------------------
    # STEP 7: Summary
    # ---------------------------
    print("\n" + "="*50)
    print("✅ Pseudo-labeling complete!")
    print("="*50)
    print(f"📁 Dataset created at: {OUTPUT_FOLDER}")
    print(f"🔧 Dataset config: {yaml_path}")
    print(f"📊 Training images processed: {train_processed}")
    print(f"📊 Validation images processed: {val_processed}")
    print(f"📊 Total images processed: {train_processed + val_processed}")

    print(f"\n📝 Dataset structure:")
    print(f"├── {os.path.relpath(OUTPUT_FOLDER, BASE_DIR)}/")
    print(f"│   ├── images/")
    print(f"│   │   ├── train/ ({train_processed} images)")
    print(f"│   │   └── val/ ({val_processed} images)")
    print(f"│   ├── labels/")
    print(f"│   │   ├── train/ ({train_processed} label files)")
    print(f"│   │   └── val/ ({val_processed} label files)")
    print(f"│   └── dataset.yaml")

    print(f"\n📋 Next steps:")
    print("1. Review and manually correct labels if needed")
    print("2. Update dataset.yaml with correct class names")
    print("3. Train YOLO model using: yolo train data=dataset.yaml model=yolov8n.pt")
    print("4. Adjust CONF_THRESHOLD and CLASS_MAPPING if needed")

    print(f"\n💡 Tips:")
    print("- Empty label files indicate no objects were detected above the confidence threshold")
    print("- Consider lowering CONF_THRESHOLD if too few detections")
    print("- Use labelImg or similar tools to manually review/correct labels")