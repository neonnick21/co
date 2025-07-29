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
        self.bbox_head = nn.Linear(256, 4) # Outputs cx, cy, w, h (normalized)
        
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
        # A simple block to enhance features
        return nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # Backbone features
        features = self.backbone_layers(x) # e.g., [B, 2048, H/32, W/32]
        
        # Channel reduction
        features = self.channel_reduction(features) # [B, 256, H/32, W/32]
        
        # Apply RFE module
        features = self.rfe_module(features)

        # Positional encoding (assuming features are [B, C, H, W])
        pos_embed = self.position_embedding(features) # [1, H*W, 256*2] if using 128 pos feats

        # Flatten features for transformer input: [B, H*W, C]
        features_flat = features.flatten(2).permute(0, 2, 1) # [B, H*W, 256]
        
        # Expand pos_embed to batch_size
        # pos_embed is [1, H*W, C_pos], we need to broadcast to [B, H*W, C_pos]
        # and match its last dim to features_flat's last dim (256).
        # Assuming num_pos_feats=128, then pos_embed's last dim is 256 (128*2 for sin/cos).
        # We need to ensure that the transformer's d_model (256) matches this.
        # Here, pos_embed should directly align with d_model.
        # My PositionEmbeddingSine returns [1, H*W, 2 * num_pos_feats], and 2*128=256, so it matches.
        pos_embed_broadcast = pos_embed.expand(features_flat.shape[0], -1, -1)

        # Transformer Encoder
        # The transformer expects src, src_key_padding_mask, pos
        # For simplicity, we directly add pos_embed to features for now as a common DETR practice.
        # A more robust DETR encoder also uses a mask and proper positional encoding integration.
        memory = self.transformer_encoder(src=features_flat, pos=pos_embed_broadcast)
        
        # Prepare query embeddings
        query_embed = self.query_embeddings.weight.unsqueeze(0).repeat(x.shape[0], 1, 1) # [B, num_queries, 256]

        # Transformer Decoder
        # tgt (target for decoder) is query_embed. memory is output from encoder.
        # query_pos is query_embed itself as positional encoding for queries.
        hs = self.transformer_decoder(tgt=query_embed, memory=memory, pos=pos_embed_broadcast) # hs: [B, num_queries, 256]

        # Prediction Heads (applied to output of decoder)
        # hs: [B, num_queries, D] where D is d_model (256)
        pred_logits = self.class_head(hs) # [B, num_queries, num_classes]
        pred_boxes = torch.sigmoid(self.bbox_head(hs)) # [B, num_queries, 4] - normalized cx, cy, w, h
        
        return pred_logits, pred_boxes


def compute_loss(pred_logits, pred_boxes, targets, num_classes, device):
    """
    Computes the total loss for DETR, including classification, L1 bounding box, and GIoU loss.
    This function performs Hungarian matching between predictions and ground truth.

    Args:
        pred_logits (torch.Tensor): Predicted class logits from the model.
                                    Shape: [batch_size, num_queries, num_classes]
        pred_boxes (torch.Tensor): Predicted bounding box coordinates (normalized cxcywh).
                                   Shape: [batch_size, num_queries, 4]
        targets (list[dict]): List of dictionaries, each containing 'boxes' and 'labels'
                              for a single image in the batch.
                              'boxes' are expected to be normalized XYXY.
        num_classes (int): Total number of classes including the 'no-object' class.
        device (torch.device): The device (CPU or CUDA) where tensors reside.

    Returns:
        tuple: (total_loss, class_loss, bbox_l1_loss, giou_loss)
    """
    batch_size = pred_logits.shape[0]

    total_class_loss = 0.0
    total_bbox_l1_loss = 0.0
    total_giou_loss = 0.0

    # Loss weights (can be tuned)
    # The last class is 'no-object'
    # --- ADJUSTED LOSS WEIGHTS (for debugging) ---
    loss_class_weight_adjusted = 1.0 # Increased importance
    loss_bbox_weight_adjusted = 5.0  # Common in DETR
    loss_giou_weight_adjusted = 2.0  # Common in DETR
    # ---------------------------------------------

    for i in range(batch_size):
        pred_logits_i = pred_logits[i] # [num_queries, num_classes]
        pred_boxes_i = pred_boxes[i]   # [num_queries, 4] (normalized cxcywh)
        
        target_labels_i = targets[i]['labels'] # [num_targets]
        target_boxes_i = targets[i]['boxes']   # [num_targets, 4] (normalized xyxy from data_preprocessing)

        num_targets = len(target_labels_i)
        num_queries = pred_logits_i.shape[0]

        # 1. Compute Cost Matrix for Hungarian Matching
        # Cost = Classification Cost + Bounding Box L1 Cost + GIoU Cost

        # Classification Cost (Negative Log Likelihood)
        # Assuming last class is 'no-object'
        out_prob = F.softmax(pred_logits_i, dim=-1) # [num_queries, num_classes]
        cost_class = -out_prob[:, target_labels_i] # [num_queries, num_targets]

        # Bounding Box Cost (L1 Distance)
        # Convert pred_boxes_i from cxcywh to xyxy for cost calculation and loss
        cx, cy, w, h = pred_boxes_i.unbind(-1)
        pred_boxes_xyxy = torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1) # [num_queries, 4]

        # L1 Cost
        cost_bbox_l1 = torch.cdist(pred_boxes_xyxy, target_boxes_i, p=1) # [num_queries, num_targets]

        # GIoU Cost
        # generalized_box_iou returns a matrix of IoUs, not cost. Cost is 1 - IoU.
        # Make sure inputs are xyxy format
        giou_matrix = generalized_box_iou(pred_boxes_xyxy, target_boxes_i) # [num_queries, num_targets]
        cost_giou = 1 - giou_matrix

        # Total Cost Matrix (weighted)
        # Weights for costs can be different from loss weights
        C = 0.1 * cost_class + 1.0 * cost_bbox_l1 + 1.0 * cost_giou # These weights are for matching, not loss scale

        # Hungarian Matching
        if num_targets == 0:
            # If no ground truth objects, all queries should predict 'no-object'
            # Only classification loss applies for 'no-object'
            cost_class_no_obj = -F.log_softmax(pred_logits_i, dim=-1)[:, num_classes - 1].mean()
            total_class_loss += cost_class_no_obj
            continue

        # Convert cost matrix to CPU for linear_sum_assignment (SciPy function)
        C = C.cpu()
        row_ind, col_ind = linear_sum_assignment(C) # row_ind: query indices, col_ind: target indices

        # Filter out invalid assignments if any (shouldn't happen with full queries)
        row_ind = torch.tensor(row_ind, dtype=torch.int64, device=device)
        col_ind = torch.tensor(col_ind, dtype=torch.int64, device=device)

        # 2. Classification Loss (for matched predictions)
        # For matched queries, compute cross-entropy loss with their assigned target labels
        matched_pred_logits = pred_logits_i[row_ind] # Logits for matched queries
        matched_target_labels = target_labels_i[col_ind] # Labels for matched targets
        
        class_loss = F.cross_entropy(matched_pred_logits, matched_target_labels, reduction='mean')
        total_class_loss += class_loss

        # 3. Bounding Box L1 Loss and GIoU Loss (for matched predictions)
        # pred_boxes_i are cxcywh from model, need to be converted to xyxy for F.l1_loss and generalized_box_iou_loss
        # target_boxes_i are xyxy from data_preprocessing
        
        # Convert matched_pred_boxes from cxcywh to xyxy for loss calculation
        matched_pred_boxes_cxcywh = pred_boxes_i[row_ind] # [num_matched, 4]
        
        cx, cy, w, h = matched_pred_boxes_cxcywh.unbind(-1)
        matched_pred_boxes_xyxy = torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1) # [num_matched, 4]

        matched_target_boxes = target_boxes_i[col_ind] # [num_matched, 4]

        bbox_l1_loss = F.l1_loss(matched_pred_boxes_xyxy, matched_target_boxes, reduction='mean')
        total_bbox_l1_loss += bbox_l1_loss

        giou_loss = generalized_box_iou_loss(matched_pred_boxes_xyxy, matched_target_boxes, reduction='mean')
        total_giou_loss += giou_loss

    # Average losses over the batch
    effective_batch_size = batch_size if batch_size > 0 else 1 
    
    # Calculate individual average losses for printing
    avg_class_loss_val = total_class_loss / effective_batch_size
    avg_bbox_l1_loss_val = total_bbox_l1_loss / effective_batch_size
    avg_giou_loss_val = total_giou_loss / effective_batch_size

    # Total weighted loss
    final_loss = (
        loss_class_weight_adjusted * avg_class_loss_val +
        loss_bbox_weight_adjusted * avg_bbox_l1_loss_val +
        loss_giou_weight_adjusted * avg_giou_loss_val
    )

    return final_loss, avg_class_loss_val, avg_bbox_l1_loss_val, avg_giou_loss_val
