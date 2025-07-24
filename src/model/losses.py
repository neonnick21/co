# model/losses.py
"""
Loss functions for DETR/RF-DETR:
- Classification loss (CrossEntropy)
- Bounding box L1 loss
- Generalized IoU loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_labels(pred_logits, target_classes):
    """
    Classification loss (CrossEntropy) for object detection.
    pred_logits: (num_queries, num_classes+1)
    target_classes: (num_targets,)
    """
    return F.cross_entropy(pred_logits, target_classes)

def loss_boxes(pred_boxes, target_boxes):
    """
    Bounding box L1 loss.
    pred_boxes: (num_queries, 4)
    target_boxes: (num_targets, 4)
    """
    return F.l1_loss(pred_boxes, target_boxes)

def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def generalized_box_iou(boxes1, boxes2):
    """
    Compute generalized IoU between two sets of boxes.
    boxes1, boxes2: (N, 4)
    """
    # Intersection
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    # Union
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = area1[:, None] + area2 - inter
    iou = inter / union
    # Enclosing box
    x1_c = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    y1_c = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    x2_c = torch.max(boxes1[:, None, 2], boxes2[:, 2])
    y2_c = torch.max(boxes1[:, None, 3], boxes2[:, 3])
    area_c = (x2_c - x1_c) * (y2_c - y1_c)
    giou = iou - (area_c - union) / area_c
    return giou

def loss_giou(pred_boxes, target_boxes):
    """
    Generalized IoU loss.
    pred_boxes: (num_queries, 4)
    target_boxes: (num_targets, 4)
    """
    giou = generalized_box_iou(pred_boxes, target_boxes)
    return 1 - giou.mean()
