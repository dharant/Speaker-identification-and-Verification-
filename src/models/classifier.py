"""
Classification Head Module
===========================
Implements ArcFace and Softmax classification heads for speaker identification.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceClassifier(nn.Module):
    """
    Additive Angular Margin Softmax (ArcFace) classification head.
    
    Enforces angular margin between speaker embeddings to learn
    more discriminative representations.
    
    Reference:
        Deng, J., et al. (2019). "ArcFace: Additive Angular Margin Loss
        for Deep Face Recognition." CVPR.
    
    Args:
        embedding_dim (int): Input embedding dimension.
        num_classes (int): Number of speaker classes.
        margin (float): Angular margin in radians.
        scale (float): Feature re-scaling factor.
    """
    
    def __init__(
        self,
        embedding_dim: int = 192,
        num_classes: int = 10,
        margin: float = 0.2,
        scale: float = 30.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        # Class weight matrix (centers on the hypersphere)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute margin values
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass with ArcFace margin.
        
        During training (labels provided), applies angular margin to the
        target class logit. During inference, returns standard cosine logits.
        
        Args:
            embeddings (torch.Tensor): Speaker embeddings of shape (batch, embedding_dim).
            labels (torch.Tensor): Speaker labels of shape (batch,). None during inference.
            
        Returns:
            torch.Tensor: Scaled logits of shape (batch, num_classes).
        """
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(embeddings, weight)
        
        if labels is not None and self.training:
            # Apply ArcFace margin to target class
            sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=1e-8))
            phi = cosine * self.cos_m - sine * self.sin_m  # cos(θ + m)
            
            # Handle edge case where cos(θ + m) < cos(π - m)
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
            # Create one-hot mask for target classes
            one_hot = F.one_hot(labels, self.num_classes).float()
            
            # Apply margin only to target class
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        else:
            output = cosine
        
        # Scale the output
        output = output * self.scale
        
        return output


class SoftmaxClassifier(nn.Module):
    """
    Standard Softmax classification head for speaker identification.
    
    Used primarily during inference when no angular margin is needed.
    
    Args:
        embedding_dim (int): Input embedding dimension.
        num_classes (int): Number of speaker classes.
    """
    
    def __init__(self, embedding_dim: int = 192, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)
    
    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through linear + softmax.
        
        Args:
            embeddings (torch.Tensor): Speaker embeddings (batch, embedding_dim).
            labels (torch.Tensor): Unused, kept for API compatibility.
            
        Returns:
            torch.Tensor: Logits of shape (batch, num_classes).
        """
        return self.fc(embeddings)
