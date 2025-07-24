# Data Preprocessing for RF-DETR Project
# - Download dataset (Kaggle/Roboflow)
# - Convert to COCO format if needed
# - Split into train/val/test
# - Resize, normalize, augment images

import os
import shutil
import zipfile
import requests
import json
from sklearn.model_selection import train_test_split
from PIL import Image
import albumentations as A
import numpy as np
import mimetypes

# --- Download and extract BCCD dataset (COCO format) from Roboflow ---
def download_and_extract_bccd(dest_dir):
    """
    Downloads and extracts the BCCD dataset (COCO format) from Roboflow.
    Args:
        dest_dir (str): Directory to extract dataset into (e.g., data/)
    """
    url = "https://public.roboflow.com/ds/GVJCultPuQ?key=0AVhhCEQpy"
    zip_path = os.path.join(dest_dir, "bccd.zip")
    print(f"Downloading BCCD dataset from {url} ...")
    response = requests.get(url, stream=True, allow_redirects=True)
    content_type = response.headers.get('content-type')
    if 'zip' not in content_type:
        print("Download failed: URL did not return a zip file. Please download manually from the link in your browser.")
        with open(zip_path + '.html', 'wb') as f:
            f.write(response.content)
        return
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete. Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    os.remove(zip_path)
    print(f"BCCD dataset extracted to {dest_dir}")

# --- Split COCO-format dataset into train/val/test sets ---
def split_coco_dataset(coco_json_path, images_dir, out_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits a COCO-format dataset into train/val/test sets and copies images/annotations.
    Args:
        coco_json_path (str): Path to COCO annotation file
        images_dir (str): Directory containing all images
        out_dir (str): Output directory (should contain images/ and annotations/)
    """
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)
    images = coco['images']
    # Split images into train, val, test
    train_imgs, temp_imgs = train_test_split(images, train_size=train_ratio, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_ratio/(test_ratio+val_ratio), random_state=42)
    splits = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}
    for split, split_imgs in splits.items():
        split_img_dir = os.path.join(out_dir, 'images', split)
        os.makedirs(split_img_dir, exist_ok=True)
        split_ann_path = os.path.join(out_dir, 'annotations', f'{split}.json')
        # Copy images for this split
        for img in split_imgs:
            src = os.path.join(images_dir, img['file_name'])
            dst = os.path.join(split_img_dir, img['file_name'])
            if os.path.exists(src):
                shutil.copy(src, dst)
        # Filter annotations for this split
        img_ids = set(img['id'] for img in split_imgs)
        anns = [ann for ann in coco['annotations'] if ann['image_id'] in img_ids]
        split_coco = {
            'images': split_imgs,
            'annotations': anns,
            'categories': coco['categories']
        }
        with open(split_ann_path, 'w') as f:
            json.dump(split_coco, f)
        print(f"Saved {split} set: {len(split_imgs)} images, {len(anns)} annotations.")

# --- Resize and normalize images in a directory ---
def resize_and_normalize_images(image_dir, size=(416, 416)):
    """
    Resize and normalize all images in a directory to the given size and overwrite them.
    Args:
        image_dir (str): Directory containing images
        size (tuple): Target size (width, height)
    """
    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(image_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(size, Image.ANTIALIAS)
            img.save(img_path)
    print(f"Resized and normalized images in {image_dir} to {size}")

# --- Data augmentation for training images ---
def augment_images_in_folder(image_dir, n_augments=2, size=(416, 416)):
    """
    Apply augmentations to each image in a folder and save augmented copies.
    Args:
        image_dir (str): Directory containing images
        n_augments (int): Number of augmented copies per image
        size (tuple): Target size (width, height)
    """
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=10, p=0.3),
        A.RandomScale(scale_limit=0.1, p=0.2),
        A.GaussNoise(p=0.1),
        A.Resize(*size)
    ])
    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(image_dir, img_name)
            image = np.array(Image.open(img_path).convert('RGB'))
            for i in range(n_augments):
                augmented = transform(image=image)
                aug_img = Image.fromarray(augmented['image'])
                aug_name = f"aug_{i}_" + img_name
                aug_img.save(os.path.join(image_dir, aug_name))
    print(f"Augmented images in {image_dir} (x{n_augments} per original)")

# --- Fix data layout for RF-DETR pipeline ---
def fix_data_layout():
    """
    Move images and annotation files from Roboflow's default structure to the expected pipeline structure.
    """
    SPLITS = [('train', 'train.json'), ('valid', 'val.json'), ('test', 'test.json')]
    for split, ann_name in SPLITS:
        src_img_dir = os.path.join(DATA_DIR, split)
        dst_img_dir = os.path.join(DATA_DIR, 'images', 'train' if split == 'train' else ('val' if split == 'valid' else 'test'))
        os.makedirs(dst_img_dir, exist_ok=True)
        # Move images
        for fname in os.listdir(src_img_dir):
            if fname.endswith('.jpg') or fname.endswith('.png'):
                shutil.move(os.path.join(src_img_dir, fname), os.path.join(dst_img_dir, fname))
        # Move annotation
        src_ann = os.path.join(src_img_dir, '_annotations.coco.json')
        dst_ann_dir = os.path.join(DATA_DIR, 'annotations')
        os.makedirs(dst_ann_dir, exist_ok=True)
        dst_ann = os.path.join(dst_ann_dir, ann_name)
        if os.path.exists(src_ann):
            shutil.move(src_ann, dst_ann)
        print(f"Moved {split} images and annotation to {dst_img_dir} and {dst_ann}")

# --- Main pipeline: download, split, resize, augment ---
if __name__ == '__main__':
    # Set up data directories
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    # 1. Download and extract dataset
    download_and_extract_bccd(DATA_DIR)
    # 2. Split dataset into train/val/test
    bccd_dir = os.path.join(DATA_DIR, 'BCCD.v4i.coco')
    coco_json = os.path.join(bccd_dir, 'valid', '_annotations.coco.json')
    images_dir = os.path.join(bccd_dir, 'valid')
    if os.path.exists(coco_json):
        split_coco_dataset(coco_json, images_dir, DATA_DIR)
    else:
        print(f"COCO annotation file not found: {coco_json}")
    # 3. Resize and normalize images in each split
    for split in ['train', 'val', 'test']:
        split_img_dir = os.path.join(DATA_DIR, 'images', split)
        if os.path.exists(split_img_dir):
            resize_and_normalize_images(split_img_dir, size=(416, 416))
    # 4. Augment training images
    train_img_dir = os.path.join(DATA_DIR, 'images', 'train')
    if os.path.exists(train_img_dir):
        augment_images_in_folder(train_img_dir, n_augments=2, size=(416, 416))
    # 5. Fix data layout
    fix_data_layout()
