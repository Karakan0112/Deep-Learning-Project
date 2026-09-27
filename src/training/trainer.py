import os
import time
import copy
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from tqdm import tqdm


class Trainer:
    """
    Standard Trainer for VGG models on CIFAR-10 with Early Stopping,
    history tracking, and checkpointing.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        scheduler: Optional[Any] = None,
        checkpoint_dir: str = './results/checkpoints',
        model_name: str = 'vgg_model',
        patience: int = 10
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.patience = patience

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_name}_best.pt")

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_times': []
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0
        return {'loss': epoch_loss, 'acc': epoch_acc}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        eval_loss = running_loss / total
        eval_acc = (correct / total) * 100.0
        return {'loss': eval_loss, 'acc': eval_acc}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50) -> Dict[str, Any]:
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_model_weights = copy.deepcopy(self.model.state_dict())
        epochs_without_improvement = 0

        print(f"\n================ Starting Training: {self.model_name} ================")
        print(f"Device: {self.device} | Epochs: {epochs} | Early Stopping Patience: {self.patience}")

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            elapsed = time.time() - start_time

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])
            self.history['epoch_times'].append(elapsed)

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['acc']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['acc']:.2f}% | "
                f"Time: {elapsed:.1f}s",
                end=""
            )

            # Check for best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_val_acc = val_metrics['acc']
                best_model_weights = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
                torch.save(best_model_weights, self.best_checkpoint_path)
                print(" -> Best model saved!")
            else:
                epochs_without_improvement += 1
                print(f" -> No improvement ({epochs_without_improvement}/{self.patience})")

            # Early stopping check
            if epochs_without_improvement >= self.patience:
                print(f"\n[Early Stopping Triggered] No improvement in validation loss for {self.patience} epochs.")
                break

        # Load best weights before returning
        self.model.load_state_dict(best_model_weights)
        print(f"\nTraining completed for {self.model_name}.")
        print(f"Best Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc:.2f}%")

        # Save history to json
        history_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        return self.history
