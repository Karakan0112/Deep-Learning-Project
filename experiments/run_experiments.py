import os
import sys

# Set OpenMP workaround for Anaconda/Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUNBUFFERED"] = "1"
import argparse
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import build_model
from src.data import get_cifar10_dataloaders
from src.training import Trainer
from src.visualization import (
    plot_learning_curves,
    plot_comparative_interpretability
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_all(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print(f"[*] Running experiments on device: {device}")

    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Data loaders
    print("[*] Loading CIFAR-10 data...")
    loaders = get_cifar10_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        mock_data=args.smoke_test or args.mock_data
    )

    models_to_run = args.models
    trained_models = {}
    histories = {}
    summary_results = []

    for m_type in models_to_run:
        print(f"\n=======================================================")
        print(f"             MODEL: {m_type.upper()}")
        print(f"=======================================================")
        model = build_model(
            model_type=m_type,
            num_classes=10,
            dropout_rate=args.dropout
        )

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {num_params:,}")

        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        trainer = Trainer(
            model=model,
            device=device,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            checkpoint_dir=checkpoint_dir,
            model_name=m_type,
            patience=args.patience
        )

        if args.smoke_test:
            print("[SMOKE TEST] Running 1 epoch with minimal steps...")
            epochs = 1
            train_sub_loader = [next(iter(loaders['train']))]
            val_sub_loader = [next(iter(loaders['val']))]
            test_sub_loader = [next(iter(loaders['test']))]
        else:
            epochs = args.epochs
            train_sub_loader = loaders['train']
            val_sub_loader = loaders['val']
            test_sub_loader = loaders['test']

        # Fit model
        history = trainer.fit(
            train_loader=train_sub_loader,
            val_loader=val_sub_loader,
            epochs=epochs
        )
        histories[m_type] = history
        trained_models[m_type] = model

        # Evaluate best model on test set
        test_metrics = trainer.evaluate(test_sub_loader)
        best_val_acc = max(history['val_acc'])
        final_train_acc = history['train_acc'][-1]
        overfitting_gap = final_train_acc - best_val_acc

        print(f"\n[{m_type}] Final Test Accuracy: {test_metrics['acc']:.2f}% | Test Loss: {test_metrics['loss']:.4f}")
        print(f"[{m_type}] Overfitting Gap (Train - Val Acc): {overfitting_gap:.2f}%")

        summary_results.append({
            'Model': m_type,
            'Parameters': num_params,
            'Best_Val_Acc': round(best_val_acc, 2),
            'Test_Acc': round(test_metrics['acc'], 2),
            'Test_Loss': round(test_metrics['loss'], 4),
            'Overfitting_Gap': round(overfitting_gap, 2)
        })

    # Save summary dataframe
    df = pd.DataFrame(summary_results)
    summary_csv_path = os.path.join(args.output_dir, 'experiment_summary.csv')
    df.to_csv(summary_csv_path, index=False)
    print("\n================== SUMMARY RESULTS ==================")
    print(df.to_string(index=False))

    # Generate Learning Curves Plot
    curves_path = os.path.join(args.output_dir, 'learning_curves.png')
    plot_learning_curves(histories, save_path=curves_path)

    # Generate Comparative Interpretability Plot (Attention vs. Saliency)
    print("\n[*] Generating Interpretability (Attention & Saliency) visualizations...")
    test_batch, test_labels = next(iter(loaders['test']))
    interpretability_path = os.path.join(args.output_dir, 'attention_vs_saliency.png')
    plot_comparative_interpretability(
        models=trained_models,
        sample_images=test_batch[:4],
        sample_labels=test_labels[:4],
        device=device,
        save_path=interpretability_path
    )

    print("\n[Done] All experiments and figures generated successfully!")
    print(f"Results and plots saved to: {os.path.abspath(args.output_dir)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ablation Study: 2D Self-Attention in VGG for CIFAR-10")
    parser.add_argument('--models', nargs='+', default=['baseline', 'attn_block1', 'attn_block2', 'attn_block3'],
                        help="List of models to run: baseline attn_block1 attn_block2 attn_block3")
    parser.add_argument('--epochs', type=int, default=35, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for Adam")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 weight decay")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout rate")
    parser.add_argument('--patience', type=int, default=8, help="Early stopping patience")
    parser.add_argument('--data_dir', type=str, default='./data', help="Path to data directory")
    parser.add_argument('--output_dir', type=str, default='./results', help="Directory to save results")
    parser.add_argument('--num_workers', type=int, default=0, help="DataLoader num_workers")
    parser.add_argument('--val_split', type=float, default=0.1, help="Validation split ratio")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--smoke_test', action='store_true', help="Run 1 epoch smoke test")
    parser.add_argument('--mock_data', action='store_true', help="Use in-memory fake data without downloading CIFAR-10")
    parser.add_argument('--force_cpu', action='store_true', help="Force CPU even if CUDA available")

    args = parser.parse_args()
    run_all(args)
