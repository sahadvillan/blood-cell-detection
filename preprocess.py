import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
import random
import subprocess
import tempfile

def get_xml_info(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax - xmin, ymax - ymin]
        })
    return width, height, objects

def voc_to_coco(xml_dir, img_dir, output_file, class_to_id, image_list):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": id, "name": name} for name, id in class_to_id.items()]
    }
    
    ann_id = 1
    for img_id, img_name in enumerate(image_list):
        xml_name = img_name.replace('.jpg', '.xml')
        xml_path = os.path.join(xml_dir, xml_name)
        
        if not os.path.exists(xml_path):
            continue
            
        width, height, objects = get_xml_info(xml_path)
        
        coco_data["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })
        
        for obj in objects:
            if obj['name'] not in class_to_id:
                continue
                
            coco_data["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_to_id[obj['name']],
                "bbox": obj['bbox'],
                "area": obj['bbox'][2] * obj['bbox'][3],
                "iscrowd": 0
            })
            ann_id += 1
            
    with open(output_file, 'w') as f:
        json.dump(coco_data, f)

def setup_dataset(repo_path, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    print(f"Setting up dataset from {repo_path} into {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # If repo_path is a URL, clone it first
    if repo_path.startswith("http://") or repo_path.startswith("https://"):
        clone_dir = os.path.join(output_dir, "_repo_clone")
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir)
        print(f"Cloning {repo_path} into {clone_dir}...")
        subprocess.run(["git", "clone", "--depth", "1", repo_path, clone_dir], check=True)
        repo_path = clone_dir
    
    # Try multiple common paths for BCCD
    possible_img_dirs = [
        os.path.join(repo_path, 'BCCD', 'JPEGImages'),
        os.path.join(repo_path, 'JPEGImages'),
        repo_path
    ]
    possible_xml_dirs = [
        os.path.join(repo_path, 'BCCD', 'Annotations'),
        os.path.join(repo_path, 'Annotations'),
        repo_path
    ]
    
    img_dir = next((d for d in possible_img_dirs if os.path.exists(d) and any(f.endswith('.jpg') for f in os.listdir(d))), None)
    xml_dir = next((d for d in possible_xml_dirs if os.path.exists(d) and any(f.endswith('.xml') for f in os.listdir(d))), None)
    
    if not img_dir or not xml_dir:
        raise FileNotFoundError(f"Could not find JPEGImages or Annotations in {repo_path}")

    print(f"Using Image dir: {img_dir}")
    print(f"Using XML dir: {xml_dir}")
    
    images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    print(f"Found {len(images)} images total.")
    random.shuffle(images)
    
    train_count = int(len(images) * train_ratio)
    val_count = int(len(images) * val_ratio)
    
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]
    
    class_to_id = {"WBC": 1, "RBC": 2, "Platelets": 3}
    
    for split, img_list in [("train", train_images), ("val", val_images), ("test", test_images)]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Copy images
        for img in img_list:
            shutil.copy(os.path.join(img_dir, img), os.path.join(split_dir, img))
            
        # Create COCO annotation
        voc_to_coco(xml_dir, img_dir, os.path.join(output_dir, f"{split}.json"), class_to_id, img_list)

if __name__ == "__main__":
    # Example usage:
    # 1. Clone BCCD Dataset: git clone https://github.com/Shenggan/BCCD_Dataset.git
    # 2. Run this script: python preprocess.py BCCD_Dataset blood_cell_data
    import sys
    if len(sys.argv) > 2:
        setup_dataset(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python preprocess.py <repo_path> <output_dir>")
