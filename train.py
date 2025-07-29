from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from model import RFDETR, compute_loss  # Import updated compute_loss
from data_preprocessing import BccdDataset, get_transform, download_and_extract_dataset

def custom_collate_fn(batch):
    images = []
    targets = []
    
    for item in batch:
        images.append(item[0])
        targets.append(item[1])
    
    images = torch.stack(images)
    
    return images, targets

if __name__ == '__main__':
    DATASET_BASE_DIR = Path("BCCD.v3-raw.coco")
    TRAIN_DATA_ROOT = DATASET_BASE_DIR / "train"
    TRAIN_ANNOTATION_FILE = TRAIN_DATA_ROOT / "_annotations.coco.json"

    # Ensure dataset is downloaded and extracted
    DATASET_URL = "https://public.roboflow.com/ds/GVJCultPuQ?key=0AVhhCEQpy"
    download_and_extract_dataset(url=DATASET_URL, dest_path=DATASET_BASE_DIR)

    if not TRAIN_ANNOTATION_FILE.exists():
        raise FileNotFoundError(f"Annotation file not found at '{TRAIN_ANNOTATION_FILE.resolve()}'")

    print(">>> Preparing dataset for training...")
    dataset = BccdDataset(
        root_dir=TRAIN_DATA_ROOT,
        annotation_file=TRAIN_ANNOTATION_FILE,
        transforms=get_transform(train=True)
    )
    print(f"Dataset size: {len(dataset)}")

    num_classes = len(dataset.cat_id_to_name) + 1  # +1 for 'no-object' class
    print(f"Number of classes (including no-object): {num_classes}")

    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = RFDETR(num_classes=num_classes, num_queries=100)
    model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    num_epochs = 50
    for epoch in range(num_epochs):
        total_loss = 0
        total_class_loss_epoch = 0
        total_bbox_l1_loss_epoch = 0
        total_giou_loss_epoch = 0
        num_batches = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            
            images = images.to(device)
            targets_processed = []
            img_width, img_height = images.shape[2], images.shape[3]  # Get image dimensions

            for t in targets:
                # Normalize target boxes to [0, 1]
                normalized_boxes = t['boxes'].clone()  # Clone to avoid modifying original
                normalized_boxes[:, 0] /= img_width
                normalized_boxes[:, 1] /= img_height
                normalized_boxes[:, 2] /= img_width
                normalized_boxes[:, 3] /= img_height

                targets_processed.append({
                    "boxes": normalized_boxes.to(device).float(),
                    "labels": t["labels"].to(device).long(),
                    "image_id": t["image_id"].to(device)
                })

            pred_logits, pred_boxes = model(images)
            
            # Commented out debugging prints
            # if batch_idx == 0 and epoch == 0:  # Only print for the first batch of the first epoch
            #     print(f"DEBUG: Epoch {epoch+1}, Batch {batch_idx}, Predicted boxes (first 5): {pred_boxes[0, :5]}")
            #     print(f"DEBUG: Epoch {epoch+1}, Batch {batch_idx}, Predicted boxes min/max: {pred_boxes.min().item():.4f}/{pred_boxes.max().item():.4f}")

            loss, class_loss_val, bbox_l1_loss_val, giou_loss_val = compute_loss(
                pred_logits, pred_boxes, targets_processed, num_classes, device
            )
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss is NaN or Inf. Skipping backward pass.")
                continue 

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_class_loss_epoch += class_loss_val.item()
            total_bbox_l1_loss_epoch += bbox_l1_loss_val.item()
            total_giou_loss_epoch += giou_loss_val.item()
            num_batches += 1

        if num_batches == 0:  # Handle case where all batches were skipped due to NaN/Inf
            print(f"Epoch [{epoch + 1}/{num_epochs}], No valid batches processed. Loss remains high.")
            continue

        avg_loss = total_loss / num_batches
        avg_class_loss = total_class_loss_epoch / num_batches
        avg_bbox_l1_loss = total_bbox_l1_loss_epoch / num_batches
        avg_giou_loss = total_giou_loss_epoch / num_batches

        print(f"Epoch [{epoch + 1}/{num_epochs}], Total Loss: {avg_loss:.4f}, "
              f"Class Loss: {avg_class_loss:.4f}, L1 Loss: {avg_bbox_l1_loss:.4f}, "
              f"GIoU Loss: {avg_giou_loss:.4f}")

    print("Training complete.")
    
    torch.save(model.state_dict(), 'rfd_et-r_model.pth')
    print("Model saved to rfd_et-r_model.pth")
