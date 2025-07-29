import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou
from torchvision.ops import generalized_box_iou_loss
import math

# --- Corrected 2D Positional Encoding for DETR-like models ---
class PositionEmbeddingSine(nn.Module):
    """
    This is a DETR-specific 2D positional encoding.
    It generates a 2D sine positional encoding for image features.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x is the feature map: [batch_size, C, H, W]
        h, w = x.shape[-2:]
        
        # Create a meshgrid for positions
        y_embed = torch.arange(h, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(w, dtype=torch.float32, device=x.device)

        # Normalize positions to [0, 1] and scale
        eps = 1e-6  # Small epsilon to prevent division by zero if h or w is 0
        if self.normalize:
            y_embed = y_embed / (y_embed[-1] + eps) * self.scale
            x_embed = x_embed / (x_embed[-1] + eps) * self.scale

        # Calculate the division term for sine/cosine
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Apply sine/cosine to x and y components
        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t

        # Stack sin and cos components and flatten
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=-1).flatten(1)  # [W, num_pos_feats]
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=-1).flatten(1)  # [H, num_pos_feats]

        # Expand to a 2D grid and concatenate
        pos_y_grid = pos_y.unsqueeze(1).repeat(1, w, 1)
        pos_x_grid = pos_x.unsqueeze(0).repeat(h, 1, 1)

        # Concatenate along the feature dimension
        pos = torch.cat((pos_y_grid, pos_x_grid), dim=-1)  # [H, W, 2 * num_pos_feats]

        # Permute to [C_pos, H, W] and then unsqueeze for batch dimension
        pos = pos.permute(2, 0, 1).unsqueeze(0)  # [1, 2 * num_pos_feats, H, W]
        
        # Flatten for transformer input: [1, H*W, 2 * num_pos_feats]
        pos = pos.flatten(2).permute(0, 2, 1)  # [1, H*W, 2 * num_pos_feats]

        return pos


class RFDETR(nn.Module):
    def __init__(self, num_classes, num_queries):
        super(RFDETR, self).__init__()
        
        # Backbone: ResNet
        self.backbone = resnet50(weights='DEFAULT')
        self.backbone_layers = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Convolutional layer to reduce channels from 2048 to 256
        self.channel_reduction = nn.Conv2d(2048, 256, kernel_size=1)

        self.position_embedding = PositionEmbeddingSine(num_pos_feats=128, normalize=True)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=6)
        
        # Transformer Decoder
        decoder_layers = nn.TransformerDecoderLayer(d_model=256, nhead=8, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=6)
        
        # Receptive Field Enhancement Module
        self.rfe_module = self.create_rfe_module()
        
        # Prediction Heads
        self.class_head = nn.Linear(256, num_classes)
        self.bbox_head = nn.Linear(256, 4)

        # Query embeddings
        self.query_embeddings = nn.Embedding(num_queries, 256)

        # Initialize weights for new layers
        nn.init.xavier_uniform_(self.channel_reduction.weight)
        nn.init.constant_(self.channel_reduction.bias, 0)
        nn.init.xavier_uniform_(self.class_head.weight)
        nn.init.constant_(self.class_head.bias, 0)
        nn.init.xavier_uniform_(self.bbox_head.weight)
        nn.init.constant_(self.bbox_head.bias, 0)

    def create_rfe_module(self):
        return nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, images):
        features = self.backbone_layers(images) 
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("DEBUG: NaN/Inf in backbone features!")
            return torch.zeros(images.shape[0], self.query_embeddings.num_embeddings, self.class_head.out_features, device=images.device), \
                   torch.zeros(images.shape[0], self.query_embeddings.num_embeddings, 4, device=images.device)

        reduced_features = self.channel_reduction(features)
        if torch.isnan(reduced_features).any() or torch.isinf(reduced_features).any():
            print("DEBUG: NaN/Inf in reduced_features!")
            return torch.zeros(images.shape[0], self.query_embeddings.num_embeddings, self.class_head.out_features, device=images.device), \
                   torch.zeros(images.shape[0], self.query_embeddings.num_embeddings, 4, device=images.device)

        enhanced_features = self.rfe_module(reduced_features)
        if torch.isnan(enhanced_features).any() or torch.isinf(enhanced_features).any():
            print("DEBUG: NaN/Inf in enhanced_features (after RFE)!")
            return torch.zeros(images.shape[0], self.query_embeddings.num_embeddings, self.class_head.out_features, device=images.device), \
                   torch.zeros(images.shape[0], self.query_embeddings.num_embeddings, 4, device=images.device)
        
        pos_embed = self.position_embedding(enhanced_features)  # [1, H*W, 256]
        
        bs, c, h_feat, w_feat = enhanced_features.shape
        flattened_features = enhanced_features.flatten(2).permute(0, 2, 1)  # [bs, H_feat*W_feat, 256]
        
        features_with_pos = flattened_features + pos_embed
        
        if torch.isnan(features_with_pos).any() or torch.isinf(features_with_pos).any():
            print("DEBUG: NaN/Inf in features_with_pos (after adding PE)!")
            return torch.zeros(images.shape[0], self.query_embeddings.num_embeddings, self.class_head.out_features, device=images.device), \
                   torch.zeros(images.shape[0], self.query_embeddings.num_embeddings, 4, device=images.device)
        
        memory = self.transformer_encoder(features_with_pos)
        if torch.isnan(memory).any() or torch.isinf(memory).any():
            print("DEBUG: NaN/Inf in transformer encoder memory!")
            return torch.zeros(images.shape[0], self.query_embeddings.num_embeddings, self.class_head.out_features, device=images.device), \
                   torch.zeros(images.shape[0], self.query_embeddings.num_embeddings, 4, device=images.device)
        
        queries = self.query_embeddings.weight.unsqueeze(0).repeat(bs, 1, 1)
        
        outputs = self.transformer_decoder(queries, memory)
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print("DEBUG: NaN/Inf in transformer decoder outputs!")
            return torch.zeros(images.shape[0], self.query_embeddings.num_embeddings, self.class_head.out_features, device=images.device), \
                   torch.zeros(images.shape[0], self.query_embeddings.num_embeddings, 4, device=images.device)
        
        class_logits = self.class_head(outputs)
        bbox_preds = self.bbox_head(outputs).sigmoid().clamp(min=1e-4, max=1-1e-4) 

        if torch.isnan(class_logits).any() or torch.isinf(class_logits).any():
            print("DEBUG: NaN/Inf in final class_logits!")
            return torch.zeros(images.shape[0], self.query_embeddings.num_embeddings, self.class_head.out_features, device=images.device), \
                   torch.zeros(images.shape[0], self.query_embeddings.num_embeddings, 4, device=images.device)
        if torch.isnan(bbox_preds).any() or torch.isinf(bbox_preds).any():
            print("DEBUG: NaN/Inf in final bbox_preds!")
            return torch.zeros(images.shape[0], self.query_embeddings.num_embeddings, self.class_head.out_features, device=images.device), \
                   torch.zeros(images.shape[0], self.query_embeddings.num_embeddings, 4, device=images.device)
        
        return class_logits, bbox_preds

def compute_loss(pred_logits, pred_boxes, targets, num_classes, device,
                 cost_class_weight=1.0, cost_bbox_weight=5.0, cost_giou_weight=2.0,
                 loss_class_weight=1.0, loss_bbox_weight=5.0, loss_giou_weight=2.0):
    """
    Computes the total loss for a batch of predictions and targets using bipartite matching.
    """
    total_class_loss = torch.tensor(0.0, device=device)
    total_bbox_l1_loss = torch.tensor(0.0, device=device)
    total_giou_loss = torch.tensor(0.0, device=device)

    batch_size, num_queries = pred_logits.shape[:2]
    no_object_class_label = num_classes - 1  # Assuming last class is 'no-object'

    # --- ADJUSTED LOSS WEIGHTS (for debugging) ---
    loss_class_weight_adjusted = 0.1 
    loss_bbox_weight_adjusted = 1.0  # Changed from 5.0
    loss_giou_weight_adjusted = 2.0
    # ---------------------------------------------

    for i in range(batch_size):
        pred_logits_i = pred_logits[i]  # [num_queries, num_classes]
        pred_boxes_i = pred_boxes[i]    # [num_queries, 4]
        target_labels_i = targets[i]['labels']  # [num_targets]
        target_boxes_i = targets[i]['boxes']    # [num_targets, 4]

        num_targets = len(target_labels_i)

        # If there are no ground truth objects in this image,
        # all queries should predict 'no-object'.
        if num_targets == 0:
            class_loss = F.cross_entropy(
                pred_logits_i, 
                torch.full((num_queries,), no_object_class_label, dtype=torch.long, device=device)
            )
            total_class_loss += class_loss
            continue 

        # --- Compute Cost Matrix for Hungarian Matching ---
        log_probs = F.log_softmax(pred_logits_i, dim=-1)  # [num_queries, num_classes]
        cost_class = -log_probs[:, target_labels_i]  # [num_queries, num_targets]

        cost_bbox = torch.cdist(pred_boxes_i, target_boxes_i, p=1)  # [num_queries, num_targets]
        cost_giou = 1 - generalized_box_iou(pred_boxes_i, target_boxes_i)  # [num_queries, num_targets]

        # Combine costs with weights
        C = cost_class_weight * cost_class + \
            cost_bbox_weight * cost_bbox + \
            cost_giou_weight * cost_giou
        
        if torch.isnan(C).any() or torch.isinf(C).any():
            print(f"Warning: NaN or Inf found in cost matrix C for image {i}. Skipping this image.")
            continue 

        C_np = C.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(C_np)

        # --- Calculate Losses based on Matching ---

        # 1. Classification Loss
        matched_pred_logits = pred_logits_i[row_ind]
        matched_target_labels = target_labels_i[col_ind]
        class_loss_matched = F.cross_entropy(matched_pred_logits, matched_target_labels)
        total_class_loss += class_loss_matched

        unmatched_queries_mask = torch.ones(num_queries, dtype=torch.bool, device=device)
        unmatched_queries_mask[row_ind] = False 

        num_unmatched_queries = torch.sum(unmatched_queries_mask).item()
        if num_unmatched_queries > 0:
            class_loss_unmatched = F.cross_entropy(
                pred_logits_i[unmatched_queries_mask], 
                torch.full((num_unmatched_queries,), no_object_class_label, dtype=torch.long, device=device)
            )
            total_class_loss += class_loss_unmatched

        # 2. Bounding Box L1 Loss
        matched_pred_boxes = pred_boxes_i[row_ind]
        matched_target_boxes = target_boxes_i[col_ind]

        # Ensure matched boxes are in the expected range
        # Commented out debugging prints for matched boxes
        # if matched_pred_boxes.numel() > 0:
        #     print(f"DEBUG: Image {i}, Matched Pred Boxes min/max: {matched_pred_boxes.min().item():.4f}/{matched_pred_boxes.max().item():.4f}")
        #     print(f"DEBUG: Image {i}, Matched Target Boxes min/max: {matched_target_boxes.min().item():.4f}/{matched_target_boxes.max().item():.4f}")

        bbox_l1_loss = F.l1_loss(matched_pred_boxes, matched_target_boxes, reduction='mean')
        total_bbox_l1_loss += bbox_l1_loss

        # 3. GIoU Loss
        giou_loss = generalized_box_iou_loss(matched_pred_boxes, matched_target_boxes, reduction='mean')
        total_giou_loss += giou_loss

    # Average losses over the batch
    effective_batch_size = batch_size if batch_size > 0 else 1 
    
    # Calculate individual average losses for printing
    avg_class_loss_val = total_class_loss / effective_batch_size
    avg_bbox_l1_loss_val = total_bbox_l1_loss / effective_batch_size
    avg_giou_loss_val = total_giou_loss / effective_batch_size

    final_loss = (loss_class_weight_adjusted * avg_class_loss_val) + \
                 (loss_bbox_weight_adjusted * avg_bbox_l1_loss_val) + \
                 (loss_giou_weight_adjusted * avg_giou_loss_val)

    # Ensure we always return values
    return final_loss, avg_class_loss_val, avg_bbox_l1_loss_val, avg_giou_loss_val
