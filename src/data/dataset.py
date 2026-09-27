import os
import tarfile
import urllib.request
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Dict

# Standard CIFAR-10 normalization constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Multiple mirrors for resilience against Toronto server drops
CIFAR10_URLS = [
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "https://storage.googleapis.com/kaggle-data-peertopeer/cifar-10-python.tar.gz"
]


def download_cifar10_robust(data_dir: str):
    """
    Downloads and extracts CIFAR-10 with custom User-Agent and multiple mirror fallback.
    """
    os.makedirs(data_dir, exist_ok=True)
    extracted_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    if os.path.exists(extracted_dir) and len(os.listdir(extracted_dir)) >= 6:
        return

    tar_path = os.path.join(data_dir, 'cifar-10-python.tar.gz')
    
    for url in CIFAR10_URLS:
        try:
            print(f"[*] Downloading CIFAR-10 from: {url}")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req, timeout=30) as response, open(tar_path, 'wb') as out_file:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end="")
            print("\n[*] Download completed! Extracting archive...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=data_dir)
            print("[*] Extraction finished successfully!")
            return
        except Exception as e:
            print(f"\n[!] Mirror failed ({url}): {e}. Trying fallback mirror...")
            if os.path.exists(tar_path):
                os.remove(tar_path)

    raise RuntimeError("Failed to download CIFAR-10 from all available mirrors.")


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns train and evaluation transforms.
    Train includes Random Crop with padding and Random Horizontal Flip.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    return train_transform, eval_transform


def get_cifar10_dataloaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 0,
    val_split: float = 0.1,
    seed: int = 42,
    mock_data: bool = False
) -> Dict[str, DataLoader]:
    """
    Downloads CIFAR-10 and creates Train, Validation, and Test DataLoaders.
    
    Args:
        data_dir: directory to save CIFAR-10 dataset
        batch_size: batch size for loaders
        num_workers: number of workers for data loading (0 recommended for Windows)
        val_split: proportion of training set to use for validation (e.g. 0.1 = 5,000 images)
        seed: random seed for reproducible train/val split
        mock_data: If True, uses in-memory synthetic CIFAR-10 data for instant local smoke tests.
    
    Returns:
        Dictionary with keys 'train', 'val', 'test' containing DataLoaders.
    """
    train_transform, eval_transform = get_transforms()

    if mock_data:
        train_dataset = torchvision.datasets.FakeData(
            size=256, image_size=(3, 32, 32), num_classes=10, transform=train_transform
        )
        val_dataset = torchvision.datasets.FakeData(
            size=64, image_size=(3, 32, 32), num_classes=10, transform=eval_transform
        )
        test_dataset = torchvision.datasets.FakeData(
            size=64, image_size=(3, 32, 32), num_classes=10, transform=eval_transform
        )
    else:
        # Robust download helper with User-Agent & retry
        download_cifar10_robust(data_dir)

        full_train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=train_transform
        )

        val_base_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=eval_transform
        )

        # Split indices reproducibly
        val_size = int(len(full_train_dataset) * val_split)
        train_size = len(full_train_dataset) - val_size

        generator = torch.Generator().manual_seed(seed)
        train_subset_indices, val_subset_indices = random_split(
            range(len(full_train_dataset)), [train_size, val_size], generator=generator
        )

        train_dataset = torch.utils.data.Subset(full_train_dataset, train_subset_indices.indices)
        val_dataset = torch.utils.data.Subset(val_base_dataset, val_subset_indices.indices)

        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=False, transform=eval_transform
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
