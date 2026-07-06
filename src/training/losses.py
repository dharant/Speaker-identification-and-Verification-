"""
Loss Functions Module
=====================
Implements loss functions for speaker recognition training:
- ArcFace Loss (Additive Angular Margin Softmax)
- Triplet Loss with hard negative mining
- Standard Cross-Entropy Loss wrapper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss: Additive Angular Margin Softmax.
    
    Combines the ArcFace classifier's margin mechanism with cross-entropy loss.
    The margin is applied by the classifier head; this loss wraps CE over the
    scaled logits.
    
    Args:
        num_classes (int): Number of speaker classes.
        embedding_dim (int): Embedding dimension.
        margin (float): Angular margin.
        scale (float): Feature scale.
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 192,
        margin: float = 0.2,
        scale: float = 30.0,
    ):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.margin = margin
        self.scale = scale
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute ArcFace loss.
        
        Args:
            logits (torch.Tensor): Scaled logits from ArcFaceClassifier.
            labels (torch.Tensor): Ground truth labels.
            
        Returns:
            torch.Tensor: Scalar loss value.
        """
        return self.criterion(logits, labels)


class TripletLoss(nn.Module):
    """
    Triplet Loss with semi-hard negative mining.
    
    Learns embeddings where same-speaker pairs are closer than
    different-speaker pairs by at least a margin.
    
    Args:
        margin (float): Minimum distance margin between positive and negative pairs.
    """
    
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss with online semi-hard negative mining.
        
        Args:
            embeddings (torch.Tensor): Speaker embeddings (batch, embedding_dim).
            labels (torch.Tensor): Speaker labels (batch,).
            
        Returns:
            torch.Tensor: Scalar triplet loss.
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distance matrix
        dist_matrix = self._pairwise_distances(embeddings)
        
        # Get valid triplets
        loss = self._batch_hard_triplet_loss(dist_matrix, labels)
        
        return loss
    
    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distance matrix."""
        dot_product = torch.mm(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = (
            square_norm.unsqueeze(0)
            - 2.0 * dot_product
            + square_norm.unsqueeze(1)
        )
        distances = torch.clamp(distances, min=0.0)
        return torch.sqrt(distances + 1e-8)
    
    def _batch_hard_triplet_loss(
        self, dist_matrix: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute batch-hard triplet loss.
        
        For each anchor, select the hardest positive (farthest same-class)
        and hardest negative (closest different-class).
        """
        batch_size = labels.size(0)
        
        # Create masks for positive and negative pairs
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        
        # Mask out self-distances
        identity_mask = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
        positives_mask = labels_equal & ~identity_mask
        negatives_mask = labels_not_equal
        
        # Hardest positive: max distance among same-class pairs
        positive_dist = dist_matrix * positives_mask.float()
        hardest_positive, _ = positive_dist.max(dim=1)
        
        # Hardest negative: min distance among different-class pairs
        # Set same-class distances to a large value
        large_value = dist_matrix.max() + 1
        negative_dist = dist_matrix + (~negatives_mask).float() * large_value
        hardest_negative, _ = negative_dist.min(dim=1)
        
        # Compute triplet loss
        triplet_loss = F.relu(hardest_positive - hardest_negative + self.margin)
        
        # Only count valid triplets (where both positive and negative exist)
        valid_triplets = (positives_mask.sum(dim=1) > 0) & (negatives_mask.sum(dim=1) > 0)
        
        if valid_triplets.sum() > 0:
            loss = triplet_loss[valid_triplets].mean()
        else:
            loss = torch.tensor(0.0, device=embeddings.device if hasattr(embeddings, 'device') else 'cpu', requires_grad=True)
        
        return loss


class CrossEntropyLoss(nn.Module):
    """
    Standard Cross-Entropy Loss wrapper for speaker identification.
    
    Args:
        label_smoothing (float): Label smoothing factor.
    """
    
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss.
        
        Args:
            logits (torch.Tensor): Model logits (batch, num_classes).
            labels (torch.Tensor): Ground truth labels (batch,).
            
        Returns:
            torch.Tensor: Scalar loss value.
        """
        return self.criterion(logits, labels)
