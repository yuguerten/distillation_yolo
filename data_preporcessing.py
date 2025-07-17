import numpy as np
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import zipfile
import shutil
import random
from PIL import Image
import cv2
from tqdm import tqdm

# Define paths
DATA_ROOT = Path("/media/yuguerten/data")
FRAMES_DIR = DATA_ROOT / "frames/"
SEG_DIR = DATA_ROOT / "segmentations"
OUTPUT_DIR = Path("/media/yuguerten/data/yolo_dataset")
IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_DIR = OUTPUT_DIR / "labels"
# VIZ_DIR = OUTPUT_DIR / "visualizations"

# Create output directories
for dir_path in [OUTPUT_DIR, IMAGES_DIR, LABELS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

def extract_all_bboxes_from_mask(mask):
    """Extrait les bounding boxes pour chaque objet détecté dans le masque."""
    bboxes = []
    
    # Trouver les composants connectés
    num_components = np.unique(mask)
    
    # Extraire les bounding boxes
    for component_id in num_components:
        if component_id == 0: 
            continue
        
        component_mask = mask == component_id
        rows = np.any(component_mask, axis=1)
        cols = np.any(component_mask, axis=0)

        if not np.any(rows) or not np.any(cols): 
            continue
        
        # Trouver les indices ymin, ymax, xmin, xmax
        ymin, ymax = map(int, np.where(rows)[0][[0, -1]])
        xmin, xmax = map(int, np.where(cols)[0][[0, -1]])

        # Ajouter la bbox avec conversion explicite en `int`
        bboxes.append((xmin, ymin, xmax, ymax))

    return bboxes

# Function to convert bbox to YOLO format
def convert_to_yolo_format(bbox, img_width, img_height):
    """
    Convert bounding box from (xmin, ymin, xmax, ymax) to YOLO format (x_center, y_center, width, height)
    All values normalized between 0 and 1
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Calculate center coordinates and dimensions
    x_center = ((xmin + xmax) / 2) / img_width
    y_center = ((ymin + ymax) / 2) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    
    # Ensure values are within [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return x_center, y_center, width, height

# Function to extract npy file from zip and process it
def process_segmentation_file(zip_file_path, corresponding_image_path):
    """Process a single segmentation zip file and its corresponding image"""
    # Extract filename without extension for output naming
    base_name = zip_file_path.stem.replace('.npy', '')
    
    # Create temporary directory for extraction
    temp_dir = Path("/tmp/seg_extraction")
    temp_dir.mkdir(exist_ok=True)
    
    # Extract the npy file from the zip
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Find the extracted npy file
    npy_files = list(temp_dir.glob('*.npy'))
    if not npy_files:
        print(f"No .npy files found in {zip_file_path}")
        return None
    
    # Load the segmentation data
    segmentation_data = np.load(npy_files[0])
    print(f"Processing {base_name}: Shape {segmentation_data.shape}, Type {segmentation_data.dtype}")
    
    # Load corresponding image to get dimensions
    if corresponding_image_path.exists():
        img = cv2.imread(str(corresponding_image_path))
        if img is None:
            print(f"Error: Unable to load image {corresponding_image_path}")
        img_height, img_width = img.shape[:2]
    else:
        print(f"Warning: Image file {corresponding_image_path} not found")
        img_height, img_width = segmentation_data.shape[:2]
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    # Extract bounding boxes
    bboxes = extract_all_bboxes_from_mask(segmentation_data)
    # print(f"Detected {num_objects} objects in {base_name}")
    
    # Convert to YOLO format
    yolo_bboxes = [convert_to_yolo_format(bbox, img_width, img_height) for bbox in bboxes]
    
    # Clean up temp directory
    shutil.rmtree(temp_dir)
    
    return {
        'image_path': corresponding_image_path,
        'image': img,
        'bboxes': bboxes,
        'yolo_bboxes': yolo_bboxes,
        'base_name': base_name
    }

# # Function to visualize detected bounding boxes on the mask
# def visualize_bboxes(image, bboxes, output_file):
#     """Visualize the detected bounding boxes on the image and save it"""
#     # Create a figure and axes
#     fig, ax = plt.figure(figsize=(12, 10)), plt.gca()
    
#     # Display the image
#     ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
#     # Draw bounding boxes
#     for i, (xmin, ymin, xmax, ymax) in enumerate(bboxes):
#         # Draw rectangle
#         rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                          linewidth=1, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)
        
#         # Draw object number
#         ax.text(xmin, ymin - 5, f"#{i+1}", color='red', fontsize=8)
        
#         # Draw small circle in center of bbox for better visibility
#         center_x = (xmin + xmax) / 2
#         center_y = (ymin + ymax) / 2
#         ax.plot(center_x, center_y, 'ro', markersize=4)
        
#     # Add title 
#     ax.set_title(f'Detected Objects: {len(bboxes)}')
    
#     # Remove axis
#     plt.axis('off')
    
#     # Save the visualization
#     plt.tight_layout()
#     plt.savefig(output_file, dpi=300)
#     plt.close()

# Function to create train/val split
def create_train_val_split(image_files, train_ratio=0.8):
    """Split the dataset into training and validation sets"""
    # Shuffle the file list
    random.shuffle(image_files)
    
    # Calculate split point
    split_idx = int(len(image_files) * train_ratio)
    
    # Split into train and validation sets
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    return train_files, val_files

# Function to write data.yaml for YOLO
def write_yolo_yaml(output_path, nc=1, names=["TBbacillus"]):
    """Write the data.yaml file for YOLO training"""
    with open(output_path, 'w') as f:
        f.write(f"train: {IMAGES_DIR}/train\n")
        f.write(f"val: {IMAGES_DIR}/val\n\n")
        f.write(f"nc: {nc}\n")
        f.write(f"names: {names}\n")

# Function to copy and organize files for YOLO dataset
def organize_yolo_dataset(processed_data, train_ratio=0.8):
    """Organize the dataset into YOLO format"""
    # Create train and val directories
    train_img_dir = IMAGES_DIR / "train"
    val_img_dir = IMAGES_DIR / "val"
    train_label_dir = LABELS_DIR / "train"
    val_label_dir = LABELS_DIR / "val"
    
    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # Get all image files to create train/val split
    image_files = [data['image_path'] for data in processed_data]
    train_files, val_files = create_train_val_split(image_files, train_ratio)
    
    # Process each item
    for data in processed_data:
        base_name = data['base_name']
        image_path = data['image_path']
        
        # Determine if file is in train or val set
        is_train = image_path in train_files
        
        # Set destination directories
        img_dest_dir = train_img_dir if is_train else val_img_dir
        label_dest_dir = train_label_dir if is_train else val_label_dir
        
        # Copy image file to destination
        dest_img_path = img_dest_dir / f"{base_name}.jpg"
        
        # Convert TIFF to JPG if needed
        if image_path.suffix.lower() in ['.tif', '.tiff']:
            # Convert to JPG using PIL
            img = Image.open(image_path)
            rgb_img = img.convert('RGB')
            rgb_img.save(dest_img_path)
        else:
            # Copy or convert the image as needed
            cv2.imwrite(str(dest_img_path), data['image'])
        
        # Create YOLO format labels file
        label_path = label_dest_dir / f"{base_name}.txt"
        with open(label_path, 'w') as f:
            for bbox in data['yolo_bboxes']:
                # class_id x_center y_center width height
                f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    
    # Write data.yaml
    write_yolo_yaml(OUTPUT_DIR / "data.yaml")
    
    # Write train.txt and val.txt (relative paths)
    with open(OUTPUT_DIR / "train.txt", 'w') as f:
        for img_path in train_files:
            f.write(f"./images/train/{img_path.stem}.jpg\n")
    
    with open(OUTPUT_DIR / "val.txt", 'w') as f:
        for img_path in val_files:
            f.write(f"./images/val/{img_path.stem}.jpg\n")

# Main processing function
def process_all_files():
    """Process all segmentation and frame files in the data directories"""
    # Find all segmentation folders
    processed_data = []
    total_objects = 0
    
    # Folder to process
    target_folder = "PAalive_230615_10"
    seg_folder = SEG_DIR / target_folder
    
    if not seg_folder.exists() or not seg_folder.is_dir():
        print(f"Error: Target folder {target_folder} not found in {SEG_DIR}")
        return processed_data
    
    print(f"Processing folder: {seg_folder.name}")
    
    # Find all zip files in this folder
    zip_files = list(seg_folder.glob("*.npy.zip"))
    
    for zip_file in tqdm(zip_files, desc=f"Processing {seg_folder.name}"):
        # Extract the base name for matching with image file
        # Example: PAalive_230615_1_000.npy.zip -> PAalive_230615_1_000
        base_name = zip_file.name.replace(".npy.zip", "")
        
        # Find the corresponding image file in frames directory
        # First, get folder part (PAalive_230615_1)
        folder_part = '_'.join(base_name.split('_')[:-1])  # Remove the last part (frame number)
        frame_folder = FRAMES_DIR / folder_part
        
        # Look for .tif file with matching name
        image_path = frame_folder / f"{base_name}.tif"
        
        if not image_path.exists():
            print(f"Warning: No matching image found for {base_name}")
            continue
        
        # Process this segmentation file
        result = process_segmentation_file(zip_file, image_path)
        
        if result:
            processed_data.append(result)
            total_objects += len(result['bboxes'])
            
            # Visualization code removed
    
    print(f"Processed {len(processed_data)} files with {total_objects} objects total")
    return processed_data

# Run the entire processing pipeline
if __name__ == "__main__":
    print("Starting YOLO dataset creation from segmentation files...")
    processed_data = process_all_files()
    
    if processed_data:
        print(f"Organizing files into YOLO dataset structure...")
        organize_yolo_dataset(processed_data)
        print(f"YOLO dataset created successfully at: {OUTPUT_DIR}")
    else:
        print("No files were processed successfully.")

