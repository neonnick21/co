# data/coco_dataset.py
"""
PyTorch Dataset for COCO-format object detection data.
- Loads images and annotations from COCO JSON
- Returns image, target dict (boxes, labels)
"""
import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class COCODetectionDataset(Dataset):
    def __init__(self, images_dir, ann_path, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        with open(ann_path, 'r') as f:
            coco = json.load(f)
        self.images = coco['images']
        self.annotations = coco['annotations']
        self.categories = coco['categories']
        # Build image_id to annotations mapping
        self.img_id_to_anns = {}
        for ann in self.annotations:
            self.img_id_to_anns.setdefault(ann['image_id'], []).append(ann)
        self.id_to_img = {img['id']: img for img in self.images}
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        anns = self.img_id_to_anns.get(img_info['id'], [])
        boxes = []
        labels = []
        for ann in anns:
            bbox = ann['bbox']  # COCO: [x, y, width, height]
            # Convert to [x_min, y_min, x_max, y_max]
            x_min, y_min, w, h = bbox
            x_max = x_min + w
            y_max = y_min + h
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['category_id'])
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels, 'image_id': img_info['id']}
        if self.transforms:
            image = self.transforms(image)
        return image, target

def coco_collate_fn(batch):
    return tuple(zip(*batch))

# --- DataLoader utility for COCO-format datasets ---
def get_coco_loaders(data_dir, batch_size=4, num_workers=2):
    """
    Returns PyTorch DataLoaders for train, val, test splits.
    Args:
        data_dir (str): Path to data directory
        batch_size (int): Batch size
        num_workers (int): DataLoader workers
    Returns:
        dict: {'train': train_loader, 'val': val_loader, 'test': test_loader}
    """
    split_loaders = {}
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(data_dir, 'images', split)
        ann_path = os.path.join(data_dir, 'annotations', f'{split}.json')
        if not (os.path.exists(img_dir) and os.path.exists(ann_path)):
            continue
        # Basic normalization; add augmentations as needed
        tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset = COCODetectionDataset(img_dir, ann_path, transforms=tfms)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), num_workers=num_workers, collate_fn=coco_collate_fn)
        split_loaders[split] = loader
    return split_loaders
