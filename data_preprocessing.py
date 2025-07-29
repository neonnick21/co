import json
import random
from pathlib import Path
import requests
import zipfile
import io

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def download_and_extract_dataset(url: str, dest_path: Path):
    """
    Downloads a zip file from a URL and extracts it to a destination path.
    Skips the process if the destination directory already exists.
    """
    if dest_path.exists():
        print(f"Dataset already found at '{dest_path}'. Skipping download.")
        return

    print(f"Downloading dataset from Roboflow...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download file: {e}") from e

    print("Download complete. Extracting files...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(dest_path)
    print(f"Dataset successfully extracted to '{dest_path}'.")


def get_transform(train: bool):
    """
    Defines the transformations to be applied to the images and targets.
    - For training, it includes data augmentation (horizontal flip).
    - For validation/testing, it only includes type conversion.
    """
    transforms = []

    # The v2 transforms expect a PIL image, and convert it to a v2.Image tensor.
    transforms.append(T.ToImage())

    if train:
        # This now operates on the v2.Image tensor and its corresponding bounding boxes.
        # The README mentions the original dataset was augmented with flips and rotations.
        # Adding them here can further improve model generalization.
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
        transforms.append(T.RandomRotation(degrees=90))

    # Sanitize bounding boxes to ensure they are valid after transformations.
    # This can remove boxes that are too small or fall outside the image, which is
    # a good practice when applying geometric augmentations.
    transforms.append(T.SanitizeBoundingBoxes())

    # ToDtype converts the image dtype and scales the values.
    # If scale=True, the output is a plain torch.Tensor with values in [0.0, 1.0].
    transforms.append(T.ToDtype(torch.float32, scale=True)) # Ensure float32

    # Convert bounding box format from XYWH (from COCO) to XYXY for consistency
    # with model's expected input and torchmetrics.
    transforms.append(T.ConvertBoundingBoxFormat(T.BoundingBoxFormat.XYXY))

    return T.Compose(transforms)


class BccdDataset(Dataset):
    """
    A PyTorch Dataset for the BCCD dataset in COCO format.
    This class handles both data preparation for the model and visual exploration.
    """

    def __init__(self, root_dir: Path, annotation_file: Path, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        # Create helpful mappings for quick lookups
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
        self.img_id_to_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_annotations:
                self.img_id_to_annotations[img_id] = []
            self.img_id_to_annotations[img_id].append(ann)

        # We only want to work with images that have annotations
        annotated_image_ids = set(self.img_id_to_annotations.keys())
        self.images = [img for img in coco_data['images'] if img['id'] in annotated_image_ids]
        print(f"Found {len(self.images)} images with annotations.")
        # Ensure category IDs are remapped to be 0-indexed if necessary for contiguous labels
        # Assuming COCO categories are already 0-indexed or will be handled by model.
        # For simplicity, we directly use COCO IDs here.

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = self.root_dir / img_info['file_name']
        image = Image.open(img_path).convert("RGB") # Ensure 3 channels

        image_id = img_info['id']
        annotations = self.img_id_to_annotations.get(image_id, [])

        boxes = []
        labels = []
        for ann in annotations:
            # COCO format: [x, y, width, height] (top-left, width, height)
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Wrap targets using tv_tensors.BoundingBoxes. Crucially, specify format="XYWH"
        # because COCO annotations are XYWH. The transforms will then convert it to XYXY.
        target_boxes = tv_tensors.BoundingBoxes(boxes, format="XYWH", canvas_size=image.size[::-1]) # PIL size is (width, height), canvas_size is (height, width)
        target_labels = tv_tensors.Label(labels)

        target = {
            "boxes": target_boxes,
            "labels": target_labels,
            "image_id": torch.tensor([image_id]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.bool) # DETR expects this
        }

        if self.transforms:
            image, target = self.transforms(image, target)
        
        return image, target


if __name__ == '__main__':
    # --- Configuration & Automated Setup ---
    DATASET_URL = "https://public.roboflow.com/ds/GVJCultPuQ?key=0AVhhCEQpy"
    DATASET_BASE_DIR = Path("BCCD.v3-raw.coco")
    
    download_and_extract_dataset(url=DATASET_URL, dest_path=DATASET_BASE_DIR)

    # --- Path Definitions ---
    TRAIN_DATA_ROOT = DATASET_BASE_DIR / "train"
    TRAIN_ANNOTATION_FILE = TRAIN_DATA_ROOT / "_annotations.coco.json"

    # --- Verification Step ---
    if not TRAIN_ANNOTATION_FILE.exists():
        error_msg = (
            f"ERROR: Annotation file not found at '{TRAIN_ANNOTATION_FILE.resolve()}'\n\n"
            "This can happen if the downloaded zip file has an unexpected structure.\n"
            "Please check the contents of the 'BCCD.v3-raw.coco' directory."
        )
        raise FileNotFoundError(error_msg)

    # Step 2: Test the data preparation pipeline for the model
    print(">>> Testing data preparation for PyTorch...")
    dataset_for_model = BccdDataset(
        root_dir=TRAIN_DATA_ROOT,
        annotation_file=TRAIN_ANNOTATION_FILE,
        transforms=get_transform(train=True)
    )
    print(f"Dataset size: {len(dataset_for_model)}")
    image, target = dataset_for_model[0]
    print("\n--- Processed Sample ---")
    print(f"Image shape: {image.shape}, type: {image.dtype}")
    print("Target dict:", {k: v.shape if isinstance(v, torch.Tensor) else type(v) for k, v in target.items()})
    print(f"Sample target boxes (XYXY format): {target['boxes'][:2]}")
    print(f"Sample target labels: {target['labels'][:2]}")
    print("--------------------------\n")

    # Step 3: Test the visual exploration functionality
    print(">>> Testing visual exploration...")
    # We create a new instance without transforms to see the original image
    raw_dataset = BccdDataset(
        root_dir=TRAIN_DATA_ROOT,
        annotation_file=TRAIN_ANNOTATION_FILE,
        transforms=None # No transforms for raw visualization
    )
    raw_image, raw_target = raw_dataset[0] # Get a raw image and target
    
    # Visualize using raw image and raw target, but convert boxes to XYXY if needed for visualization
    # The visualize_sample function in a separate script could handle this.
    # For now, just demonstrating data loading.
    
    # To display directly:
    # fig, ax = plt.subplots(1)
    # ax.imshow(raw_image)
    # for i, box in enumerate(raw_target['boxes']):
    #     # Convert raw XYWH box to matplotlib-compatible XYWH format for Rectangle patch
    #     # Matplotlib expects [x, y, width, height] for Rectangle.
    #     # Since raw_target['boxes'] is still XYWH, no conversion needed here for drawing.
    #     x, y, w, h = box.tolist()
    #     rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    #     ax.add_patch(rect)
    #     label_name = raw_dataset.cat_id_to_name[raw_target['labels'][i].item()]
    #     plt.text(x, y - 5, label_name, color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    # plt.show()
    print("Visual exploration test complete (requires manual plot display in a live environment).")
