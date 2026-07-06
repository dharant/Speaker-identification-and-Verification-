"""
ECAPA-TDNN Model
================
Emphasized Channel Attention, Propagation and Aggregation in Time Delay
Neural Network (ECAPA-TDNN) for speaker embedding extraction.

Reference:
    Desplanques, B., Thienpondt, J., & Demuynck, K. (2020).
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation
    in TDNN Based Speaker Verification."
    Proc. Interspeech 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    
    Args:
        channels (int): Number of input channels.
        se_channels (int): Number of bottleneck channels.
    """
    
    def __init__(self, channels: int, se_channels: int = 128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, se_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(se_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SE attention: x * sigmoid(SE(x))."""
        return x * self.se(x)


class Res2NetBlock(nn.Module):
    """
    Res2Net block with multi-scale feature extraction.
    
    Splits channels into multiple groups and processes them hierarchically
    to capture multi-scale information.
    
    Args:
        channels (int): Number of input/output channels.
        kernel_size (int): Convolution kernel size.
        dilation (int): Dilation factor.
        scale (int): Res2Net scale factor (number of sub-groups).
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 8,
    ):
        super().__init__()
        assert channels % scale == 0, "channels must be divisible by scale"
        
        self.scale = scale
        self.width = channels // scale
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(scale - 1):
            self.convs.append(
                nn.Conv1d(
                    self.width, self.width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation // 2,
                )
            )
            self.bns.append(nn.BatchNorm1d(self.width))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale processing.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, time).
            
        Returns:
            torch.Tensor: Multi-scale processed tensor.
        """
        # Split into sub-groups
        spx = torch.split(x, self.width, dim=1)
        outputs = [spx[0]]
        
        for i in range(1, self.scale):
            if i == 1:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            
            if i <= len(self.convs):
                sp = self.convs[i - 1](sp)
                sp = F.relu(self.bns[i - 1](sp), inplace=True)
            
            outputs.append(sp)
        
        return torch.cat(outputs, dim=1)


class SERes2NetBlock(nn.Module):
    """
    SE-Res2Net block: combines Res2Net multi-scale processing with
    Squeeze-and-Excitation channel attention.
    
    Args:
        channels (int): Number of channels.
        kernel_size (int): Convolution kernel size.
        dilation (int): Dilation factor.
        scale (int): Res2Net scale factor.
        se_channels (int): SE bottleneck channels.
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 8,
        se_channels: int = 128,
    ):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            Res2NetBlock(channels, kernel_size, dilation, scale),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            SEBlock(channels, se_channels),
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        return self.relu(x + self.block(x))


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling layer.
    
    Computes weighted mean and standard deviation of frame-level features
    using an attention mechanism, producing utterance-level representations.
    
    Args:
        channels (int): Number of input channels.
        attention_channels (int): Number of attention bottleneck channels.
    """
    
    def __init__(self, channels: int, attention_channels: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, attention_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_channels, channels, kernel_size=1),
            nn.Softmax(dim=2),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool frame-level features into utterance-level representation.
        
        Args:
            x (torch.Tensor): Frame-level features of shape (batch, channels, time).
            
        Returns:
            torch.Tensor: Utterance-level features of shape (batch, channels * 2).
        """
        # Compute attention weights
        alpha = self.attention(x)
        
        # Weighted mean
        mean = torch.sum(alpha * x, dim=2)
        
        # Weighted standard deviation
        var = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-8))
        
        # Concatenate mean and std
        return torch.cat([mean, std], dim=1)


class ECAPATDNN(nn.Module):
    """
    ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation
    in Time Delay Neural Network.
    
    Architecture:
    1. Initial TDNN layer
    2. Multiple SE-Res2Net blocks with increasing dilation
    3. Multi-layer Feature Aggregation (MFA)
    4. Attentive Statistics Pooling
    5. Final linear layer for embedding extraction
    
    Args:
        input_size (int): Input feature dimension (e.g., 40 for MFCC).
        channels (list): Channel sizes for each TDNN layer.
        kernel_sizes (list): Kernel sizes for each TDNN layer.
        dilations (list): Dilation factors for each TDNN layer.
        embedding_dim (int): Output speaker embedding dimension.
        attention_channels (int): Attention bottleneck channels.
        res2net_scale (int): Res2Net scale factor.
        se_channels (int): SE bottleneck channels.
    """
    
    def __init__(
        self,
        input_size: int = 40,
        channels: list = None,
        kernel_sizes: list = None,
        dilations: list = None,
        embedding_dim: int = 192,
        attention_channels: int = 128,
        res2net_scale: int = 8,
        se_channels: int = 128,
    ):
        super().__init__()
        
        if channels is None:
            channels = [512, 512, 512, 512, 1536]
        if kernel_sizes is None:
            kernel_sizes = [5, 3, 3, 3, 1]
        if dilations is None:
            dilations = [1, 2, 3, 4, 1]
        
        self.embedding_dim = embedding_dim
        
        # Layer 1: Initial TDNN
        self.layer1 = nn.Sequential(
            nn.Conv1d(
                input_size, channels[0],
                kernel_size=kernel_sizes[0],
                dilation=dilations[0],
                padding=(kernel_sizes[0] - 1) * dilations[0] // 2,
            ),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )
        
        # Layers 2-4: SE-Res2Net blocks
        self.layer2 = SERes2NetBlock(
            channels[1], kernel_sizes[1], dilations[1], res2net_scale, se_channels
        )
        self.layer3 = SERes2NetBlock(
            channels[2], kernel_sizes[2], dilations[2], res2net_scale, se_channels
        )
        self.layer4 = SERes2NetBlock(
            channels[3], kernel_sizes[3], dilations[3], res2net_scale, se_channels
        )
        
        # Multi-layer Feature Aggregation (MFA)
        # Concatenate outputs from layers 2, 3, 4
        cat_channels = channels[1] + channels[2] + channels[3]
        self.mfa = nn.Sequential(
            nn.Conv1d(cat_channels, channels[4], kernel_size=kernel_sizes[4]),
            nn.BatchNorm1d(channels[4]),
            nn.ReLU(inplace=True),
        )
        
        # Attentive Statistics Pooling
        self.asp = AttentiveStatisticsPooling(channels[4], attention_channels)
        
        # Final batch norm and linear layer
        self.bn = nn.BatchNorm1d(channels[4] * 2)
        self.fc = nn.Linear(channels[4] * 2, embedding_dim)
        self.bn_embed = nn.BatchNorm1d(embedding_dim)
    
    def forward(
        self, x: torch.Tensor, return_embedding: bool = True
    ) -> torch.Tensor:
        """
        Forward pass to extract speaker embeddings.
        
        Args:
            x (torch.Tensor): Input features of shape (batch, n_features, n_frames).
            return_embedding (bool): If True, return L2-normalized embeddings.
            
        Returns:
            torch.Tensor: Speaker embeddings of shape (batch, embedding_dim).
        """
        # Layer 1
        out1 = self.layer1(x)
        
        # SE-Res2Net blocks
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        
        # Multi-layer Feature Aggregation
        mfa_input = torch.cat([out2, out3, out4], dim=1)
        out5 = self.mfa(mfa_input)
        
        # Attentive Statistics Pooling
        out6 = self.asp(out5)
        
        # Final embedding
        out7 = self.bn(out6)
        embeddings = self.fc(out7)
        embeddings = self.bn_embed(embeddings)
        
        if return_embedding:
            # L2 normalize for cosine similarity
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    @classmethod
    def from_config(cls, config: dict) -> "ECAPATDNN":
        """
        Create an ECAPA-TDNN model from a configuration dictionary.
        
        Args:
            config (dict): Model configuration dictionary.
            
        Returns:
            ECAPATDNN: Configured model instance.
        """
        return cls(
            input_size=config.get("input_size", 40),
            channels=config.get("channels", [512, 512, 512, 512, 1536]),
            kernel_sizes=config.get("kernel_sizes", [5, 3, 3, 3, 1]),
            dilations=config.get("dilations", [1, 2, 3, 4, 1]),
            embedding_dim=config.get("embedding_dim", 192),
            attention_channels=config.get("attention_channels", 128),
            res2net_scale=config.get("res2net_scale", 8),
            se_channels=config.get("se_channels", 128),
        )
    
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim
    
    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
