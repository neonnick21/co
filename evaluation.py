import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
from model import RFDETR # Ensure RFDETR is importable from model.py
from data_preprocessing import BccdDataset, get_transform, download_and_extract_dataset
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics import MetricCollection # Import MetricCollection
import matplotlib.pyplot as plt
import numpy as np # Import numpy for image conversion

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized targets within a batch.
    It stacks images and keeps targets as a list of dictionaries.
    """
    images = []
    targets = []
    
    for item in batch:
        images.append(item[0]) # Image tensor
        targets.append(item[1]) # Target dictionary
    
    # Stack images into a single tensor for batch processing
    images = torch.stack(images)
    
    return images, targets

def post_process_predictions(pred_logits, pred_boxes, threshold=0.5):
    """
    Converts raw model outputs (logits and boxes) into a format suitable for
    torchmetrics, filtering predictions based on a confidence threshold.
    Corrects bounding box format from cxcywh (normalized) to xyxy (normalized).

    Args:
        pred_logits (torch.Tensor): Predicted class logits from the model.
                                    Shape: [batch_size, num_queries, num_classes]
        pred_boxes (torch.Tensor): Predicted bounding box coordinates (normalized cxcywh).
                                   Shape: [batch_size, num_queries, 4]
        threshold (float): Confidence threshold to filter predictions.

    Returns:
        list[dict]: A list of dictionaries, one per image in the batch, where
                    each dictionary contains 'boxes', 'scores', and 'labels'
                    for the detected objects in xyxy format.
    """
    results = []
    
    for logits, boxes_cxcywh in zip(pred_logits, pred_boxes):
        # Convert boxes from cxcywh (normalized) to xyxy (normalized)
        # boxes_cxcywh shape: [num_queries, 4] -> [cx, cy, w, h]
        cx, cy, w, h = boxes_cxcywh.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        boxes_xyxy = torch.stack((x1, y1, x2, y2), dim=-1)

        # Apply softmax to get probabilities for each class
        probs = F.softmax(logits, dim=-1)
        # Get the highest score and corresponding label for each query,
        # excluding the 'no-object' class (assumed to be the last class)
        scores, labels = probs[:, :-1].max(-1) 

        # Filter out predictions below the confidence threshold
        keep = scores > threshold
        
        results.append({
            "boxes": boxes_xyxy[keep], # Use the converted xyxy boxes
            "scores": scores[keep],  
            "labels": labels[keep],  
        })
    return results

def evaluate(model, data_loader, num_classes, device):
    """
    Evaluates the model's performance on a given data loader.

    Args:
        model (nn.Module): The RFDETR model to evaluate.
        data_loader (DataLoader): DataLoader for the validation/test dataset.
        num_classes (int): Total number of classes including the 'no-object' class.
        device (torch.device): The device (CPU or CUDA) to run evaluation on.

    Returns:
        dict: A dictionary containing computed metrics (mAP, mAP_50, mAP_75, etc.).
    """
    model.eval() # Set the model to evaluation mode
    
    # Initialize MeanAveragePrecision metric from torchmetrics
    # box_format="xyxy" specifies the format of bounding box coordinates.
    # iou_type="bbox" specifies that we are calculating mAP for bounding boxes.
    # class_metrics=True computes metrics per class as well as overall precision/recall.
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True).to(device)

    with torch.no_grad(): # Disable gradient calculations for inference
        for images, targets in data_loader:
            # Move images and targets to the specified device
            images = images.to(device)
            # Ensure targets are on the correct device for metric calculation
            # Also ensure target boxes are XYXY as expected by torchmetrics
            targets_on_device = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            # Get model predictions
            pred_logits, pred_boxes = model(images)
            
            # Post-process predictions into the format required by torchmetrics
            preds = post_process_predictions(pred_logits, pred_boxes)
            
            # Update the metric with the current batch's predictions and targets
            metric.update(preds, targets_on_device)
            
    # Compute the final metrics across all batches
    metrics_result = metric.compute()
    
    return metrics_result

def visualize_predictions(model, dataset, device, num_images=3, threshold=0.5):
    """
    Visualizes predictions on a few sample images from the dataset.

    Args:
        model (nn.Module): The RFDETR model.
        dataset (Dataset): The dataset to draw images from (e.g., validation set).
        device (torch.device): The device (CPU or CUDA) where the model is.
        num_images (int): Number of random images to visualize.
        threshold (float): Confidence threshold for displaying predicted boxes.
    """
    model.eval() # Set model to evaluation mode
    
    plt.figure(figsize=(15, 5 * num_images))
    with torch.no_grad(): # Disable gradient calculations
        # Select random indices for visualization
        # Ensure we don't pick more images than available in the dataset
        num_images = min(num_images, len(dataset))
        random_indices = torch.randperm(len(dataset))[:num_images].tolist()

        for i, idx in enumerate(random_indices):
            # Get image and target from the dataset (they will be on CPU initially)
            image, target = dataset[idx]
            
            # Move image to device for model input
            image_tensor = image.unsqueeze(0).to(device)

            # Get predictions
            pred_logits, pred_boxes = model(image_tensor)
            # Post-process predictions for the single image (batch size 1)
            # The post_process_predictions handles cxcywh to xyxy conversion
            preds = post_process_predictions(pred_logits, pred_boxes, threshold=threshold)[0]
            
            # Denormalize bounding box coordinates to pixel values
            # image.shape is [C, H, W] after transformations
            _, h, w = image.shape
            
            # predictions from post_process_predictions are already xyxy format (normalized)
            preds_boxes_denorm = preds['boxes'].cpu() * torch.tensor([w, h, w, h], device='cpu')
            
            # target['boxes'] are already xyxy and normalized from data_preprocessing
            target_boxes_denorm = target['boxes'].cpu() * torch.tensor([w, h, w, h], device='cpu')

            # Convert image tensor to PIL Image for drawing
            # Permute from [C, H, W] to [H, W, C] and convert to numpy, then to uint8
            # Ensure image is on CPU before converting to numpy
            original_image = Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            draw = ImageDraw.Draw(original_image)
            
            # Get class names for displaying labels
            category_names = dataset.cat_id_to_name
            # Reverse map for class ID to name if necessary, or ensure direct access by ID.
            # Assuming category_names is {id: 'name'} as loaded from dataset.

            # Draw ground truth bounding boxes in green
            for j in range(len(target['boxes'])):
                box = target_boxes_denorm[j].cpu().numpy().tolist()
                label = target['labels'][j].item()
                
                label_name = category_names.get(label, f"Unknown_{label}")
                
                # Draw rectangle: (x1, y1, x2, y2)
                draw.rectangle(box, outline="green", width=2)
                draw.text(
                    (box[0], box[1] - 10), 
                    f"GT: {label_name}", 
                    fill="green"
                )

            # Draw predicted bounding boxes in red
            for j in range(len(preds['boxes'])):
                box = preds_boxes_denorm[j].cpu().numpy().tolist()
                label = preds['labels'][j].item()
                score = preds['scores'][j].item()

                label_name = category_names.get(label, f"Unknown_{label}")

                # Draw rectangle: (x1, y1, x2, y2)
                draw.rectangle(box, outline="red", width=2)
                draw.text(
                    (box[0], box[1] + 5), 
                    f"Pred: {label_name} ({score:.2f})", 
                    fill="red"
                )
            
            # Display the image
            plt.subplot(num_images, 1, i + 1)
            plt.imshow(original_image)
            plt.title(f"Image {idx} - Predictions (red) vs Ground Truth (green) - Threshold: {threshold}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # --- Configuration & Automated Setup ---
    DATASET_URL = "https://public.roboflow.com/ds/GVJCultPuQ?key=0AVhhCEQpy"
    DATASET_BASE_DIR = Path("BCCD.v3-raw.coco")
    
    download_and_extract_dataset(url=DATASET_URL, dest_path=DATASET_BASE_DIR)

    VAL_DATA_ROOT = DATASET_BASE_DIR / "valid"
    VAL_ANNOTATION_FILE = VAL_DATA_ROOT / "_annotations.coco.json"
    
    MODEL_PATH = Path("rfd_et-r_model.pth")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("A trained model file 'rfd_et-r_model.pth' not found. Please train the model first using train.py or hyperparameter_tuning.py.")

    # --- Dataset and DataLoader ---
    val_dataset = BccdDataset(
        root_dir=VAL_DATA_ROOT,
        annotation_file=VAL_ANNOTATION_FILE,
        transforms=get_transform(train=False)
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)

    # --- Model and Device Setup ---
    num_classes = len(val_dataset.cat_id_to_name) + 1 # +1 for 'no-object' class
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")

    model = RFDETR(num_classes=num_classes, num_queries=100)
    # Load model weights, mapping to the correct device
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device) # Move model to device

    # --- Run Evaluation ---
    print(">>> Starting evaluation on the validation set...")
    metrics = evaluate(model, val_loader, num_classes, device)

    # --- Print Metrics ---
    print("\n--- Evaluation Results ---")
    print(f"Mean Average Precision (mAP): {metrics['map']:.4f}")
    print(f"mAP@0.50 IoU: {metrics['map_50']:.4f}")
    print(f"mAP@0.75 IoU: {metrics['map_75']:.4f}")
    print(f"Mean Average Recall (mAR) @ maxDets=100: {metrics['mar_100']:.4f}") # Corrected key for mAR

    print("\n--- Per-Class mAP ---")
    # Invert the mapping for easier lookup of class name by ID
    # Ensure that metrics['classes'] are 0-indexed and align with your dataset's category IDs
    # `torchmetrics` will return the class IDs for which it computed per-class metrics
    for i, class_map in enumerate(metrics['map_per_class']):
        class_id = metrics['classes'][i].item() # Get the actual class ID
        # Only print for actual object classes (not 'no-object' or unknown IDs if present)
        if class_id in val_dataset.cat_id_to_name:
            class_name = val_dataset.cat_id_to_name[class_id]
            print(f"  {class_name}: {class_map:.4f}")
    
    print("--------------------------")
    
    # --- Visualize Predictions ---
    print("\n>>> Visualizing predictions on sample images...")
    # Consider lowering the threshold here (e.g., to 0.3 or 0.5) if you still see no boxes
    # in the visualization after the fixes, to check for low-confidence predictions.
    visualize_predictions(model, val_dataset, device, num_images=3, threshold=0.7)
