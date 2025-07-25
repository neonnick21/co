# Training script for RF-DETR
# - Loads data
# - Initializes model
# - Trains with optimizer, scheduler, early stopping, checkpointing

# To be implemented after model and data pipeline are ready.

import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from src.model.detr_rfdetr import RFDETR
from src.model.losses import loss_labels, loss_boxes, loss_giou
from src.model.detr_rfdetr import hungarian_matcher
from src.data.coco_dataset import get_coco_loaders

# --- Training loop for RF-DETR ---
def train_rfdetr(
    data_dir,
    num_classes,
    num_epochs=20,
    batch_size=4,
    lr=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    checkpoint_path='rfdetr_best.pth',
    early_stopping_patience=5
):
    loaders = get_coco_loaders(data_dir, batch_size=batch_size)
    model = RFDETR(num_classes=num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    best_val_loss = float('inf')
    patience = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, targets in loaders['train']:
            images = torch.stack(images).to(device)
            # Prepare targets and predictions
            outputs = model(images)
            pred_logits = outputs['pred_logits']  # (batch, num_queries, num_classes+1)
            pred_boxes = outputs['pred_boxes']    # (batch, num_queries, 4)
            batch_loss = 0.0
            for b in range(images.size(0)):
                tgt_labels = targets[b]['labels'].to(device)
                tgt_boxes = targets[b]['boxes'].to(device)
                # Hungarian matching (indices: list of (pred_idx, tgt_idx))
                indices = hungarian_matcher(pred_logits[b], pred_boxes[b], tgt_labels, tgt_boxes)
                # For each matched pair, compute losses
                if len(indices) == 0:
                    continue
                pred_idx, tgt_idx = zip(*indices)
                matched_logits = pred_logits[b][list(pred_idx)]
                matched_boxes = pred_boxes[b][list(pred_idx)]
                matched_labels = tgt_labels[list(tgt_idx)]
                matched_boxes_gt = tgt_boxes[list(tgt_idx)]
                # Losses
                loss_cls = loss_labels(matched_logits, matched_labels)
                loss_bbox = loss_boxes(matched_boxes, matched_boxes_gt)
                loss_giou_val = loss_giou(matched_boxes, matched_boxes_gt)
                total_loss = loss_cls + loss_bbox + loss_giou_val
                batch_loss += total_loss
            if images.size(0) > 0:
                batch_loss = batch_loss / images.size(0)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
        scheduler.step()
        avg_train_loss = total_loss / len(loaders['train'])
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in loaders['val']:
                images = torch.stack(images).to(device)
                outputs = model(images)
                pred_logits = outputs['pred_logits']
                pred_boxes = outputs['pred_boxes']
                batch_loss = 0.0
                for b in range(images.size(0)):
                    tgt_labels = targets[b]['labels'].to(device)
                    tgt_boxes = targets[b]['boxes'].to(device)
                    indices = hungarian_matcher(pred_logits[b], pred_boxes[b], tgt_labels, tgt_boxes)
                    if len(indices) == 0:
                        continue
                    pred_idx, tgt_idx = zip(*indices)
                    matched_logits = pred_logits[b][list(pred_idx)]
                    matched_boxes = pred_boxes[b][list(pred_idx)]
                    matched_labels = tgt_labels[list(tgt_idx)]
                    matched_boxes_gt = tgt_boxes[list(tgt_idx)]
                    loss_cls = loss_labels(matched_logits, matched_labels)
                    loss_bbox = loss_boxes(matched_boxes, matched_boxes_gt)
                    loss_giou_val = loss_giou(matched_boxes, matched_boxes_gt)
                    total_loss = loss_cls + loss_bbox + loss_giou_val
                    batch_loss += total_loss
                if images.size(0) > 0:
                    batch_loss = batch_loss / images.size(0)
                val_loss += batch_loss.item()
        avg_val_loss = val_loss / len(loaders['val'])
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            torch.save(model.state_dict(), checkpoint_path)
            print("Saved new best model.")
        else:
            patience += 1
            if patience >= early_stopping_patience:
                print("Early stopping triggered.")
                break

if __name__ == '__main__':
    # Example usage
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    NUM_CLASSES = 3  # BCCD: WBC, RBC, Platelets
    train_rfdetr(DATA_DIR, num_classes=NUM_CLASSES)
