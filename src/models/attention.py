import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention2D(nn.Module):
    """
    2D Scaled Dot-Product Self-Attention Module for Convolutional Feature Maps.
    
    Formula:
        Attention(Q, K, V) = Softmax(Q @ K.T / sqrt(d_k)) @ V
    
    Features:
    - 1x1 convolutions for Query, Key, and Value projections.
    - Channel reduction factor (e.g. C // 8 or C // 4) for computational and memory efficiency.
    - Learnable scalar parameter gamma (initialized to 0.0) for smooth residual integration.
    - Ability to return attention maps for visualization and interpretability.
    """
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = max(in_channels // reduction, 8)
        
        # 1x1 Conv projections for Q, K, V
        self.query_conv = nn.Conv2d(in_channels, self.mid_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, self.mid_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Learnable scaling factor (initialized to 0 to preserve baseline initially)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Stored attention map from the most recent forward pass
        self.last_attention_map = None

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: Tensor of shape (B, C, H, W)
            return_attention: If True, returns (out, attention_matrix)
        Returns:
            out: Tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.size()
        N = H * W
        
        # Projections
        # Q: (B, C_mid, N) -> Transpose to (B, N, C_mid) for dot product
        proj_query = self.query_conv(x).view(B, self.mid_channels, N).permute(0, 2, 1)
        
        # K: (B, C_mid, N)
        proj_key = self.key_conv(x).view(B, self.mid_channels, N)
        
        # Scaled Dot-Product Attention energy: (B, N, N)
        # Entry (i, j) indicates how much location i attends to location j
        scale = self.mid_channels ** 0.5
        energy = torch.bmm(proj_query, proj_key) / scale
        attention = F.softmax(energy, dim=-1)
        self.last_attention_map = attention.detach()
        
        # V: (B, C, N)
        proj_value = self.value_conv(x).view(B, C, N)
        
        # Out: (B, C, N) x (B, N, N).T -> (B, C, N)
        # attention is (B, N, N) where sum_j A[i, j] = 1
        # out[i] = sum_j A[i, j] * V[j] -> (B, C, N) = V (B, C, N) @ A.permute(0, 2, 1)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        # Residual connection with learnable scale parameter
        out = x + self.gamma * out
        
        if return_attention:
            return out, attention
        return out
