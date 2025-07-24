# Evaluation script for RF-DETR
# - Loads trained model
# - Runs predictions on test set
# - Calculates metrics (accuracy, precision, recall, IoU, mAP)
# - Visualizes predictions

# To be implemented after training is complete.

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from src.model.detr_rfdetr import RFDETR
from src.data.coco_dataset import get_coco_loaders
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# --- IoU calculation ---
def box_iou(boxA, boxB):
    # boxA, boxB: [x_min, y_min, x_max, y_max]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# --- Visualization ---
def visualize_predictions(image, pred_boxes, pred_labels, class_names, save_path=None):
    draw = ImageDraw.Draw(image)
    for box, label in zip(pred_boxes, pred_labels):
        box = [int(x) for x in box]
        draw.rectangle(box, outline='red', width=2)
        draw.text((box[0], box[1]), class_names[label], fill='red')
    if save_path:
        image.save(save_path)
    else:
        image.show()

# --- mAP calculation using pycocotools ---
def calculate_map(ann_file, pred_json):
    """
    Calculate mAP using pycocotools given ground truth and predictions in COCO format.
    Args:
        ann_file (str): Path to ground truth annotation file (COCO JSON)
        pred_json (str): Path to predictions in COCO format (list of dicts)
    """
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(pred_json)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]  # mAP@0.5:0.95

# --- Evaluation loop ---
def evaluate_model(model_path, data_dir, num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
    loaders = get_coco_loaders(data_dir, batch_size=1)
    model = RFDETR(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    class_names = ['WBC', 'RBC', 'Platelets']
    all_prec, all_rec, all_iou = [], [], []
    with torch.no_grad():
        for images, targets in loaders['test']:
            image = images[0].to(device).unsqueeze(0)
            target = targets[0]
            outputs = model(image)
            pred_logits = outputs['pred_logits'].cpu().numpy()
            pred_boxes = outputs['pred_boxes'].cpu().numpy()
            # For simplicity, take top predictions
            scores = pred_logits[0].max(axis=1)
            labels = pred_logits[0].argmax(axis=1)
            boxes = pred_boxes[0]
            # Filter by score threshold
            keep = scores > 0.5
            boxes = boxes[keep]
            labels = labels[keep]
            # Calculate metrics (IoU, precision, recall)
            gt_boxes = target['boxes'].numpy()
            gt_labels = target['labels'].numpy()
            ious = [max([box_iou(pred, gt) for gt in gt_boxes]) if len(gt_boxes) else 0 for pred in boxes]
            mean_iou = np.mean(ious) if ious else 0
            all_iou.append(mean_iou)
            # Precision/Recall (simple, not mAP)
            tp = sum(iou > 0.5 for iou in ious)
            fp = len(ious) - tp
            fn = len(gt_boxes) - tp
            prec = tp / (tp + fp + 1e-6) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn + 1e-6) if (tp + fn) > 0 else 0
            all_prec.append(prec)
            all_rec.append(rec)
            # Visualization
            img_pil = Image.open(os.path.join(data_dir, 'images', 'test', target['image_id']+'.jpg')).convert('RGB')
            visualize_predictions(img_pil, boxes, labels, class_names)
    print(f"Mean Precision: {np.mean(all_prec):.3f}")
    print(f"Mean Recall: {np.mean(all_rec):.3f}")
    print(f"Mean IoU: {np.mean(all_iou):.3f}")
    # mAP calculation (requires COCO formatted annotations and predictions)
    # ann_file = os.path.join(data_dir, 'annotations', 'instances_val2017.json')
    # pred_file = 'path_to_predictions.json'
    # mean_ap = calculate_map(ann_file, pred_file)
    # print(f"Mean Average Precision (mAP): {mean_ap:.3f}")

if __name__ == '__main__':
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    NUM_CLASSES = 3
    MODEL_PATH = 'rfdetr_best.pth'
    evaluate_model(MODEL_PATH, DATA_DIR, NUM_CLASSES)
