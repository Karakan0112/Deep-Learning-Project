import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from ..data.dataset import CIFAR10_MEAN, CIFAR10_STD, CLASSES


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalizes a normalized CIFAR-10 tensor (3, H, W) to (H, W, 3) in range [0, 1].
    """
    img = tensor.cpu().detach().clone()
    for c in range(3):
        img[c] = img[c] * CIFAR10_STD[c] + CIFAR10_MEAN[c]
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()


def compute_saliency_map(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: Optional[int] = None
) -> np.ndarray:
    """
    Computes gradient-based Saliency Map for a single image tensor of shape (1, 3, 32, 32).
    Saliency = max_c | d(Score) / d(Input_c) |
    """
    model.eval()
    img = image_tensor.clone().detach().requires_grad_(True)
    
    output = model(img)
    if isinstance(output, tuple):
        output = output[0]

    if target_class is None:
        target_class = output.argmax(dim=1).item()

    score = output[0, target_class]
    model.zero_grad()
    score.backward()

    # Gradient: (1, 3, 32, 32) -> take absolute value and max over channels
    gradients = img.grad.data.abs()
    saliency, _ = torch.max(gradients[0], dim=0)  # (32, 32)
    saliency = saliency.cpu().numpy()
    
    # Normalize between 0 and 1
    saliency_min, saliency_max = saliency.min(), saliency.max()
    if saliency_max > saliency_min:
        saliency = (saliency - saliency_min) / (saliency_max - saliency_min)
    return saliency


def extract_attention_heatmap(
    attention_matrix: torch.Tensor,
    query_pos: Optional[tuple] = None,
    target_size: tuple = (32, 32)
) -> np.ndarray:
    """
    Converts attention matrix (1, N, N) into a spatial 2D heatmap.
    
    Args:
        attention_matrix: shape (1, N, N) where N = H * W
        query_pos: (row, col) feature map coordinate to inspect.
                   If None, uses center of the grid.
        target_size: (32, 32) to upsample the heatmap to original image resolution.
    """
    attn = attention_matrix.squeeze(0).cpu().detach()  # (N, N)
    N = attn.size(0)
    feat_dim = int(np.sqrt(N))  # 16, 8, or 4
    
    if query_pos is None:
        # Center index
        q_idx = (feat_dim // 2) * feat_dim + (feat_dim // 2)
    else:
        qy, qx = query_pos
        q_idx = qy * feat_dim + qx

    # Row q_idx is attention weights from query to all keys (N,)
    attn_row = attn[q_idx].view(1, 1, feat_dim, feat_dim)
    
    # Bilinear upsample to 32x32
    attn_upsampled = F.interpolate(
        attn_row, size=target_size, mode='bilinear', align_corners=False
    ).squeeze().numpy()
    
    # Normalize [0, 1]
    a_min, a_max = attn_upsampled.min(), attn_upsampled.max()
    if a_max > a_min:
        attn_upsampled = (attn_upsampled - a_min) / (a_max - a_min)
    return attn_upsampled


def plot_learning_curves(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: str = './results/learning_curves.png'
):
    """
    Plots Train and Validation Loss and Accuracy side-by-side for all models.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'baseline': '#2ca02c', 'attn_block1': '#1f77b4', 'attn_block2': '#ff7f0e', 'attn_block3': '#d62728'}
    labels = {'baseline': 'Baseline (No Attn)', 'attn_block1': 'Attention @ Block 1', 'attn_block2': 'Attention @ Block 2', 'attn_block3': 'Attention @ Block 3'}

    # Loss subplot
    ax1 = axes[0]
    for key, h in histories.items():
        c = colors.get(key, None)
        lbl = labels.get(key, key)
        epochs = range(1, len(h['train_loss']) + 1)
        ax1.plot(epochs, h['train_loss'], linestyle='--', color=c, alpha=0.6, label=f"{lbl} (Train)")
        ax1.plot(epochs, h['val_loss'], linestyle='-', color=c, linewidth=2, label=f"{lbl} (Val)")
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=9)

    # Accuracy subplot
    ax2 = axes[1]
    for key, h in histories.items():
        c = colors.get(key, None)
        lbl = labels.get(key, key)
        epochs = range(1, len(h['train_acc']) + 1)
        ax2.plot(epochs, h['train_acc'], linestyle='--', color=c, alpha=0.6, label=f"{lbl} (Train)")
        ax2.plot(epochs, h['val_acc'], linestyle='-', color=c, linewidth=2, label=f"{lbl} (Val)")
    ax2.set_title("Classification Accuracy vs. Epochs", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved learning curves to {save_path}")


def plot_comparative_interpretability(
    models: Dict[str, torch.nn.Module],
    sample_images: torch.Tensor,
    sample_labels: torch.Tensor,
    device: torch.device,
    save_path: str = './results/attention_vs_saliency.png'
):
    """
    Generates comparison figure: Original Image vs Baseline Saliency vs Attention Saliency & Attention Maps.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_samples = min(len(sample_images), 4)

    # 4 rows (samples), columns: Image | Baseline Saliency | Attn B1 (Saliency + Map) | Attn B2 | Attn B3
    fig, axes = plt.subplots(num_samples, 7, figsize=(18, 2.8 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = [
        "Original Image",
        "Baseline Saliency",
        "Attn B1 Saliency",
        "Attn B1 Map",
        "Attn B2 Saliency",
        "Attn B2 Map",
        "Attn B3 Saliency"
    ]

    for row in range(num_samples):
        img_t = sample_images[row:row+1].to(device)
        label_idx = sample_labels[row].item()
        label_name = CLASSES[label_idx]
        rgb_img = denormalize_image(sample_images[row])

        # Col 0: Image
        axes[row, 0].imshow(rgb_img)
        axes[row, 0].set_title(f"True: {label_name}", fontsize=10)
        axes[row, 0].axis('off')

        # Col 1: Baseline Saliency
        if 'baseline' in models:
            sal_base = compute_saliency_map(models['baseline'], img_t, target_class=label_idx)
            axes[row, 1].imshow(sal_base, cmap='hot')
            axes[row, 1].axis('off')

        # Col 2 & 3: Attn Block 1
        if 'attn_block1' in models:
            m1 = models['attn_block1']
            sal_1 = compute_saliency_map(m1, img_t, target_class=label_idx)
            axes[row, 2].imshow(sal_1, cmap='hot')
            axes[row, 2].axis('off')

            _, attn_1 = m1(img_t, return_attention=True)
            if attn_1 is not None:
                map_1 = extract_attention_heatmap(attn_1)
                axes[row, 3].imshow(rgb_img)
                axes[row, 3].imshow(map_1, cmap='jet', alpha=0.55)
            axes[row, 3].axis('off')

        # Col 4 & 5: Attn Block 2
        if 'attn_block2' in models:
            m2 = models['attn_block2']
            sal_2 = compute_saliency_map(m2, img_t, target_class=label_idx)
            axes[row, 4].imshow(sal_2, cmap='hot')
            axes[row, 4].axis('off')

            _, attn_2 = m2(img_t, return_attention=True)
            if attn_2 is not None:
                map_2 = extract_attention_heatmap(attn_2)
                axes[row, 5].imshow(rgb_img)
                axes[row, 5].imshow(map_2, cmap='jet', alpha=0.55)
            axes[row, 5].axis('off')

        # Col 6: Attn Block 3 Saliency
        if 'attn_block3' in models:
            m3 = models['attn_block3']
            sal_3 = compute_saliency_map(m3, img_t, target_class=label_idx)
            axes[row, 6].imshow(sal_3, cmap='hot')
            axes[row, 6].axis('off')

    # Add header titles on the top row
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(f"{title}\n{axes[0, col].get_title()}", fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved comparative interpretability figure to {save_path}")
