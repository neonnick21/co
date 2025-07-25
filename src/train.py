import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from src.model.detr_rfdetr import RFDETR
from src.model.losses import loss_labels, loss_boxes, loss_giou
from src.data.coco_dataset import get_coco_loaders
from scipy.optimize import linear_sum_assignment

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
            batch_losses = []
            for b in range(images.size(0)):
                tgt_labels = targets[b]['labels'].to(device)
                tgt_boxes = targets[b]['boxes'].to(device)
                indices = simple_hungarian_matcher(pred_logits[b], pred_boxes[b], tgt_labels, tgt_boxes)
                if len(indices) == 0:
                    continue
                if tgt_labels.numel() == 0 or tgt_boxes.numel() == 0:
                    print(f"Skipping image {b}: no targets.")
                    continue
                indices = simple_hungarian_matcher(pred_logits[b], pred_boxes[b], tgt_labels, tgt_boxes)
                print(f"Image {b}: {len(indices)} matches")
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
                total_loss_img = loss_cls + loss_bbox + loss_giou_val
                print(f"  Losses: cls={loss_cls.item():.4f}, bbox={loss_bbox.item():.4f}, giou={loss_giou_val.item():.4f}, total={total_loss_img.item():.4f}")
                batch_losses.append(total_loss_img)
            if batch_losses:
                batch_loss = torch.stack(batch_losses).mean()
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
                batch_losses = []
                for b in range(images.size(0)):
                    tgt_labels = targets[b]['labels'].to(device)
                    tgt_boxes = targets[b]['boxes'].to(device)
                    indices = simple_hungarian_matcher(pred_logits[b], pred_boxes[b], tgt_labels, tgt_boxes)
                    if len(indices) == 0:
                        continue
                    if tgt_labels.numel() == 0 or tgt_boxes.numel() == 0:
                        print(f"Skipping val image {b}: no targets.")
                        continue
                    indices = simple_hungarian_matcher(pred_logits[b], pred_boxes[b], tgt_labels, tgt_boxes)
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
                    total_loss_img = loss_cls + loss_bbox + loss_giou_val
                    batch_losses.append(total_loss_img)
                if batch_losses:
                    batch_loss = torch.stack(batch_losses).mean()
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

def simple_hungarian_matcher(pred_logits, pred_boxes, tgt_labels, tgt_boxes):
    """
    Simple Hungarian matcher for DETR-style models.
    Matches predictions to targets based on class and box L1 cost.
    Returns list of (pred_idx, tgt_idx).
    """
    if len(tgt_labels) == 0 or len(pred_logits) == 0:
        return []
    out_prob = F.softmax(pred_logits, -1).detach().cpu().numpy()  # (num_queries, num_classes+1)
    tgt_labels_np = tgt_labels.detach().cpu().numpy()  # (num_targets,)
    tgt_boxes_np = tgt_boxes.detach().cpu().numpy()  # (num_targets, 4)
    
    # Debug: print shapes
    print(f"Matcher: out_prob shape {out_prob.shape}, tgt_labels_np shape {tgt_labels_np.shape}")
    if tgt_labels_np.ndim == 0:
        tgt_labels_np = np.expand_dims(tgt_labels_np, 0)
    if tgt_labels_np.size == 0 or out_prob.ndim != 2:
        print("Matcher: empty targets or invalid out_prob shape, returning []")
        return []
    
    # Compute the classification cost.
    class_cost = -out_prob[:, tgt_labels_np]  # (num_queries, num_targets)
    pred_boxes_np = pred_boxes.detach().cpu().numpy()
    bbox_cost = np.abs(pred_boxes_np[:, None, :] - tgt_boxes_np[None, :, :]).sum(-1)  # (num_queries, num_targets)
    # Final cost matrix
    C = class_cost + bbox_cost
    row_ind, col_ind = linear_sum_assignment(C)
    return list(zip(row_ind, col_ind))

if __name__ == '__main__':
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    NUM_CLASSES = 3  # BCCD: WBC, RBC, Platelets
    train_rfdetr(DATA_DIR, num_classes=NUM_CLASSES)
