# augment_images.py
# Fixed version for Linux - removes Windows-specific paths and improves functionality

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from augment_config import augment_config

# ------------------------------
# Configuration - Linux compatible
# ------------------------------
RAW_DATA_DIR = "./apple_dataset/raw"  # relative path for Linux compatibility
OUTPUT_DIR = "./apple_dataset/augmented"  # relative path for Linux compatibility
BATCH_SIZE = 16
AUGS_PER_IMAGE = 3  # how many augmented versions per original image

# ------------------------------
# Create output directories
# ------------------------------
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ------------------------------
# Create ImageDataGenerator
# ------------------------------
datagen = augment_config()

# ------------------------------
# Check if raw data directory exists
# ------------------------------
if not os.path.exists(RAW_DATA_DIR):
    print(f"[ERROR] Raw data directory '{RAW_DATA_DIR}' not found!")
    print("Please ensure your dataset structure is:")
    print("./apple_dataset/raw/")
    print("├── alternaria/")
    print("├── healthy/")
    print("├── rust/")
    print("└── scab/")
    exit(1)

# ------------------------------
# Process each class folder
# ------------------------------
for class_name in os.listdir(RAW_DATA_DIR):
    class_input_dir = os.path.join(RAW_DATA_DIR, class_name)
    class_output_dir = os.path.join(OUTPUT_DIR, class_name)

    if not os.path.isdir(class_input_dir):
        continue

    os.makedirs(class_output_dir, exist_ok=True)

    print(f"[INFO] Augmenting class '{class_name}'...")
    
    # Count original images
    original_count = len([f for f in os.listdir(class_input_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"[INFO] Found {original_count} images in '{class_name}'")

    # Flow from directory (single class)
    flow = datagen.flow_from_directory(
        RAW_DATA_DIR,
        classes=[class_name],   # process only this class
        target_size=(128, 128), 
        batch_size=1,
        class_mode=None,  # we don't need labels for augmentation
        save_to_dir=class_output_dir,
        save_prefix=f'aug_{class_name}',
        save_format='jpg',
        shuffle=False  # to maintain order
    )

    # Generate augmented images
    total_images = len(flow.filenames)
    steps = total_images * AUGS_PER_IMAGE

    print(f"[INFO] Generating {steps} augmented images for '{class_name}'...")
    
    for i in range(steps):
        try:
            next(flow)
            if (i + 1) % 10 == 0:  # Progress indicator
                print(f"[INFO] Progress: {i+1}/{steps} images generated")
        except StopIteration:
            print(f"[WARNING] Generator exhausted at {i+1}/{steps}")
            break

    # Count generated images
    generated_count = len([f for f in os.listdir(class_output_dir) 
                          if f.lower().endswith('.jpg')])
    print(f"[INFO] Generated {generated_count} augmented images for '{class_name}'")

print(f"\n[SUCCESS] Augmentation complete!")
print(f"[INFO] Augmented dataset saved to: {OUTPUT_DIR}")
print(f"[INFO] Directory structure:")
for class_name in os.listdir(OUTPUT_DIR):
    class_path = os.path.join(OUTPUT_DIR, class_name)
    if os.path.isdir(class_path):
        count = len([f for f in os.listdir(class_path) if f.lower().endswith('.jpg')])
        print(f"  {class_name}: {count} images")
