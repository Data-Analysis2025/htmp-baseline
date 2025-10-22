"""Hands-on MNIST exploration script for data science students.

This module provides a command-line interface with several entry points
that highlight the most common day-to-day operations when building
hand-crafted machine learning workflows with PyTorch.  It is intentionally
verbose and heavily commented so that learners can read it alongside the
hands-on exercises documented in the accompanying README.

Example usages
--------------
Inspect a batch of samples (prints shapes, dtypes, min/max statistics)::

    python mnist_hands_on.py --step inspect --batch-size 16

Run a short training loop while frequently logging resource usage::

    python mnist_hands_on.py --step train --epochs 1 --limit-batches 20

Drop into the Python debugger inside the model forward call::

    python mnist_hands_on.py --step train --pdb --limit-batches 1
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import random
import statistics
import time
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclasses.dataclass
class ExperimentConfig:
    """Configuration container used throughout the script.

    Attributes
    ----------
    data_dir:
        Folder that will hold the MNIST dataset. The dataset will be
        downloaded automatically when not present.
    batch_size:
        Number of images per mini-batch.
    epochs:
        How many times to iterate through the training dataset.
    limit_batches:
        Optional cap on the number of batches processed per epoch. This is
        extremely helpful for keeping experiments short while debugging.
    seed:
        Global random seed so that runs are reproducible.
    use_cuda:
        When ``True`` and a CUDA device is available, computations are moved to
        the GPU. Learners can toggle this flag to compare CPU vs GPU runs and
        monitor the difference with ``nvidia-smi``.
    use_pdb:
        Injects debugger breakpoints into the training loop so that learners can
        step through tensors interactively using ``pdb`` commands (``n``, ``s``,
        ``p tensor.shape``...).
    """

    data_dir: str = os.environ.get("MNIST_DATA", os.path.expanduser("~/.mnist"))
    batch_size: int = 64
    epochs: int = 1
    limit_batches: Optional[int] = 10
    seed: int = 42
    use_cuda: bool = torch.cuda.is_available()
    use_pdb: bool = False


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible executions."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    """Create DataLoaders for the MNIST dataset.

    In addition to the training loader we also expose a validation loader so
    learners can experiment with ``model.eval()`` and ``torch.no_grad()``.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root=cfg.data_dir, train=True, download=True, transform=transform
    )
    val_dataset = datasets.MNIST(
        root=cfg.data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=cfg.use_cuda,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=cfg.use_cuda,
    )
    return train_loader, val_loader


class SmallConvNet(nn.Module):
    """Minimal convolutional neural network for MNIST classification."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def inspect_batch(batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Print shape, dtype and statistics of an MNIST batch."""

    images, labels = batch
    np_images = images.numpy()

    print("=== Batch inspection ===")
    print(f"Tensor shape: {images.shape}")
    print(f"Tensor dtype: {images.dtype}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")
    print(f"Min/Max pixel value (torch): {images.min().item():.4f}, {images.max().item():.4f}")
    print(
        "Min/Max pixel value (numpy): "
        f"{np_images.min():.4f}, {np_images.max():.4f} (type: {np_images.dtype})"
    )


def summarize_loader(loader: DataLoader, limit: Optional[int]) -> None:
    """Iterate over a loader and compute simple statistics."""

    batch_sizes: list[int] = []
    label_counts = torch.zeros(10)

    for batch_idx, (_, labels) in enumerate(loader):
        batch_sizes.append(labels.shape[0])
        label_counts += torch.bincount(labels, minlength=10)
        if limit is not None and batch_idx + 1 >= limit:
            break

    print("=== Loader summary ===")
    print(f"Batches iterated: {len(batch_sizes)}")
    if batch_sizes:
        print(f"Mean batch size: {statistics.mean(batch_sizes):.2f}")
        print(f"Stdev batch size: {statistics.pstdev(batch_sizes):.2f}")
    print("Label distribution (first 10 counts):")
    for digit, count in enumerate(label_counts.tolist()):
        print(f"  {digit}: {int(count)}")


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute classification accuracy for a batch."""

    predicted_labels = predictions.argmax(dim=1)
    return (predicted_labels == targets).float().mean()


def maybe_breakpoint(enabled: bool, message: str) -> None:
    """Enter ``pdb`` only when the learner opts in via ``--pdb``."""

    if not enabled:
        return

    print(f"\n[pdb] breakpoint hit: {message}")
    import pdb

    pdb.set_trace()


def train(cfg: ExperimentConfig) -> None:
    """Minimal training loop with detailed logging and optional breakpoints."""

    set_seed(cfg.seed)

    device = torch.device("cuda" if cfg.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(cfg)

    model = SmallConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("Model architecture:\n", model)

    for epoch in range(cfg.epochs):
        print(f"\n=== Epoch {epoch + 1}/{cfg.epochs} ===")
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            maybe_breakpoint(cfg.use_pdb, f"Before moving batch {batch_idx} to {device}")
            images, labels = images.to(device), labels.to(device)
            maybe_breakpoint(cfg.use_pdb, "Before forward pass")

            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += acc.item()

            if (batch_idx + 1) % 10 == 0:
                avg_loss = running_loss / (batch_idx + 1)
                avg_acc = running_acc / (batch_idx + 1)
                print(
                    f"Batch {batch_idx + 1:04d}: loss={avg_loss:.4f} accuracy={avg_acc:.4f}"
                )

            if cfg.limit_batches is not None and batch_idx + 1 >= cfg.limit_batches:
                print("Stopping early because --limit-batches was reached.")
                break

        duration = time.time() - epoch_start
        print(f"Epoch duration: {duration:.2f}s")

        evaluate(model, val_loader, device, cfg)


def evaluate(
    model: nn.Module, val_loader: DataLoader, device: torch.device, cfg: ExperimentConfig
) -> None:
    """Evaluate the model in ``eval`` mode with ``torch.no_grad``."""

    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_acc += accuracy(outputs, labels).item()
            if cfg.limit_batches is not None and batch_idx + 1 >= cfg.limit_batches:
                break

    batches = min(len(val_loader), cfg.limit_batches or len(val_loader))
    print(f"Validation loss: {total_loss / batches:.4f}")
    print(f"Validation accuracy: {total_acc / batches:.4f}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=["inspect", "train"], default="inspect")
    parser.add_argument("--data-dir", default=ExperimentConfig.data_dir)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=ExperimentConfig.epochs)
    parser.add_argument("--limit-batches", type=int, default=ExperimentConfig.limit_batches or -1)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available")
    parser.add_argument("--pdb", action="store_true", help="Enable pdb breakpoints during training")

    args = parser.parse_args(argv)
    return args


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    cfg = ExperimentConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        limit_batches=None if args.limit_batches == -1 else args.limit_batches,
        seed=args.seed,
        use_cuda=not args.cpu and torch.cuda.is_available(),
        use_pdb=args.pdb,
    )

    train_loader, _ = build_dataloaders(cfg)

    if args.step == "inspect":
        first_batch = next(iter(train_loader))
        inspect_batch(first_batch)
        summarize_loader(train_loader, cfg.limit_batches)
        print(
            "Tip: try editing this script and re-running with --pdb to observe\n"
            "how tensors change throughout the forward/backward pass."
        )
    else:
        train(cfg)


if __name__ == "__main__":
    main()
