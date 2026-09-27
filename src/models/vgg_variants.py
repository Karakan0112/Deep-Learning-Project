import torch
import torch.nn as nn
from typing import Optional, Literal
from .attention import SelfAttention2D


class VGGBlock(nn.Module):
    """
    Standard VGG-style Convolutional Block:
    [Conv 3x3 -> BN -> ReLU] x 2 -> MaxPool 2x2
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layer(x)


class VGG3Block(nn.Module):
    """
    3-Block VGG Network for CIFAR-10 with configurable 2D Self-Attention placement.
    
    Attention Placements:
        - None: Baseline VGG without self-attention.
        - 'block1': After Block 1 (narrow receptive field, local textures, 16x16 feature map).
        - 'block2': After Block 2 (intermediate representation, 8x8 feature map).
        - 'block3': After Block 3 (deep semantic features, 4x4 feature map).
    """
    def __init__(
        self,
        num_classes: int = 10,
        attention_position: Optional[Literal['block1', 'block2', 'block3']] = None,
        dropout_rate: float = 0.3,
        attention_reduction: int = 8
    ):
        super().__init__()
        self.attention_position = attention_position
        self.dropout_rate = dropout_rate

        # Block 1: 3x32x32 -> 64x16x16
        self.block1 = VGGBlock(in_channels=3, out_channels=64)
        self.attn1 = SelfAttention2D(64, reduction=attention_reduction) if attention_position == 'block1' else None
        self.drop1 = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Block 2: 64x16x16 -> 128x8x8
        self.block2 = VGGBlock(in_channels=64, out_channels=128)
        self.attn2 = SelfAttention2D(128, reduction=attention_reduction) if attention_position == 'block2' else None
        self.drop2 = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Block 3: 128x8x8 -> 256x4x4
        self.block3 = VGGBlock(in_channels=128, out_channels=256)
        self.attn3 = SelfAttention2D(256, reduction=attention_reduction) if attention_position == 'block3' else None
        self.drop3 = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        attn_map = None

        # Block 1
        x = self.block1(x)
        if self.attn1 is not None:
            if return_attention:
                x, attn_map = self.attn1(x, return_attention=True)
            else:
                x = self.attn1(x)
        x = self.drop1(x)

        # Block 2
        x = self.block2(x)
        if self.attn2 is not None:
            if return_attention:
                x, attn_map = self.attn2(x, return_attention=True)
            else:
                x = self.attn2(x)
        x = self.drop2(x)

        # Block 3
        x = self.block3(x)
        if self.attn3 is not None:
            if return_attention:
                x, attn_map = self.attn3(x, return_attention=True)
            else:
                x = self.attn3(x)
        x = self.drop3(x)

        # Head
        x = self.avgpool(x)
        logits = self.classifier(x)

        if return_attention:
            return logits, attn_map
        return logits


def build_model(
    model_type: Literal['baseline', 'attn_block1', 'attn_block2', 'attn_block3'],
    num_classes: int = 10,
    dropout_rate: float = 0.3
) -> VGG3Block:
    """Helper factory function to create experiment models."""
    pos_map = {
        'baseline': None,
        'attn_block1': 'block1',
        'attn_block2': 'block2',
        'attn_block3': 'block3'
    }
    if model_type not in pos_map:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(pos_map.keys())}")
    
    return VGG3Block(
        num_classes=num_classes,
        attention_position=pos_map[model_type],
        dropout_rate=dropout_rate
    )
