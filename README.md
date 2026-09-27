# From Local Receptive Fields to Global Context: A Systematic Study of 2D Self-Attention Placement in VGG for CIFAR-10 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karakan0112/Deep-Learning-Project/blob/main/notebooks/Project_Walkthrough.ipynb)

---

## 📌 Overview & Research Question
Standard Convolutional Neural Networks (CNNs) build hierarchical representations through small, localized filters (e.g. $3 \times 3$), gradually expanding their receptive fields as depth increases. Conversely, **Scaled Dot-Product Self-Attention** allows every spatial location to dynamically attend to every other location simultaneously, providing instantaneous global context:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

This project investigates:
1. **How does the placement of a 2D Self-Attention module within the convolutional hierarchy impact generalization, convergence, and effective receptive fields on CIFAR-10?**
2. **What is the trade-off between early local feature extraction and late abstract global reasoning?**
3. **How do Attention Maps correlate with Gradient-based Saliency Maps across varying depths?**

---

## 🏗️ Architecture & Ablation Matrix

The baseline model is a 3-Block VGG architecture ($3 \times 3$ Convs, BatchNorm, ReLU, MaxPool):
* **Baseline**: Pure 3-Block VGG without Self-Attention.
* **Model 1 (`attn_block1`)**: Self-Attention placed after Block 1 ($16 \times 16$ spatial grid, $N=256$, narrow receptive field).
* **Model 2 (`attn_block2`)**: Self-Attention placed after Block 2 ($8 \times 8$ spatial grid, $N=64$, intermediate features).
* **Model 3 (`attn_block3`)**: Self-Attention placed after Block 3 ($4 \times 4$ spatial grid, $N=16$, high-level semantic representation).

Each attention block includes a learnable residual gating parameter $\gamma$ initialized to $0$, ensuring training stability and smooth transition from local convolutional priors to global self-attention.

---

## 📂 Project Structure

```text
Deep Learning/
├── src/
│   ├── models/
│   │   ├── attention.py          # 2D Scaled Dot-Product Self-Attention module
│   │   └── vgg_variants.py       # 3-Block VGG with flexible attention injection
│   ├── data/
│   │   └── dataset.py            # CIFAR-10 data pipeline & augmentations
│   ├── training/
│   │   └── trainer.py            # Trainer with Early Stopping & metric logging
│   └── visualization/
│       └── interpretability.py   # Attention Maps & Saliency Maps generation
├── experiments/
│   └── run_experiments.py       # Main CLI experiment runner for all ablation models
├── notebooks/
│   └── Project_Walkthrough.ipynb # Colab / Jupyter interactive walkthrough
├── results/                      # Saved checkpoints, metrics CSV, and plots
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```

---

## 🚀 Getting Started

### 1. Installation
Install dependencies via pip:
```bash
pip install -r requirements.txt
```

### 2. Fast Verification (Smoke Test)
Run a 3-second smoke test using mock data to verify end-to-end execution without downloading data:
```bash
python experiments/run_experiments.py --smoke_test --models baseline attn_block1
```

### 3. Full Ablation Study
Run the complete training pipeline across all 4 configurations:
```bash
python experiments/run_experiments.py --epochs 35 --batch_size 128 --output_dir ./results
```

Key arguments:
* `--models`: `baseline`, `attn_block1`, `attn_block2`, `attn_block3`
* `--epochs`: Default 35 (with Early Stopping patience of 8 epochs)
* `--batch_size`: Default 128
* `--lr`: Default 0.001 (Adam)
* `--weight_decay`: Default 1e-4 (L2 regularization)
* `--dropout`: Default 0.3

---

## 📊 Visualizations & Deliverables
After training, the results directory will contain:
1. `experiment_summary.csv`: Test accuracy, test loss, and overfitting gap for all models.
2. `learning_curves.png`: Training vs. Validation Loss and Accuracy side-by-side.
3. `attention_vs_saliency.png`: Direct comparison between Input Image, Baseline Saliency Map, Attention Saliency Maps, and 2D Attention heatmaps.
4. `checkpoints/`: Best model weights (`.pt`) and epoch histories (`.json`).
