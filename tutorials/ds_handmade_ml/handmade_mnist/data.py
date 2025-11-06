"""データローダー構築や基本的なデータ調査を担当するモジュール。"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import ExperimentConfig


def build_transforms(augment: bool) -> transforms.Compose:
    """MNIST 画像に適用する前処理を定義する。"""

    base_transforms = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]

    if augment:
        aug_transforms = [
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
        ]
        return transforms.Compose(aug_transforms + base_transforms)
    return transforms.Compose(base_transforms)


def create_dataloaders(cfg: ExperimentConfig) -> Tuple[DataLoader, DataLoader]:
    """訓練・評価用の DataLoader を構築する。"""

    train_transform = build_transforms(cfg.augment)
    eval_transform = build_transforms(False)

    train_dataset = datasets.MNIST(
        root=str(cfg.data_dir),
        train=True,
        download=True,
        transform=train_transform,
    )
    val_dataset = datasets.MNIST(
        root=str(cfg.data_dir),
        train=False,
        download=True,
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.effective_val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
    )
    return train_loader, val_loader


def inspect_batch(batch: tuple[torch.Tensor, torch.Tensor]) -> Dict[str, object]:
    """テンソルの形状や値域など、1バッチ分の情報をまとめて表示する。"""

    images, labels = batch
    numpy_images = images.detach().cpu().numpy()

    import pdb; pdb.set_trace()  # テンソル(images)が作成された直後にブレーク

    stats: Dict[str, object] = {
        "tensor_shape": tuple(images.shape),
        "tensor_dtype": str(images.dtype),
        "label_shape": tuple(labels.shape),
        "label_dtype": str(labels.dtype),
        "min_value": float(images.min().item()),
        "max_value": float(images.max().item()),
        "mean_value": float(images.mean().item()),
        "std_value": float(images.std().item()),
        "numpy_dtype": str(numpy_images.dtype),
    }

    print("=== Batch Inspection ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    return stats


def summarize_loader(
    loader: DataLoader,
    limit: Optional[int],
    *,
    log_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """DataLoader 全体の統計量を集計し、必要に応じて CSV として保存する。"""

    batch_sizes: list[int] = []
    label_counts = torch.zeros(10, dtype=torch.long)

    for batch_idx, (_, labels) in enumerate(loader):
        batch_sizes.append(labels.shape[0])
        label_counts += torch.bincount(labels, minlength=10)
        if limit is not None and batch_idx + 1 >= limit:
            break

    summary: Dict[str, object] = {
        "num_batches": len(batch_sizes),
        "mean_batch_size": statistics.mean(batch_sizes) if batch_sizes else 0.0,
        "stdev_batch_size": statistics.pstdev(batch_sizes) if len(batch_sizes) > 1 else 0.0,
        "label_distribution": label_counts.tolist(),
    }

    print("=== Loader Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = log_dir / "label_distribution.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["digit", "count"])
            for digit, count in enumerate(summary["label_distribution"]):
                writer.writerow([digit, count])
        print(f"Label distribution saved to: {csv_path}")

    return summary
