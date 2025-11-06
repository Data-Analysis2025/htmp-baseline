"""学習ループと評価・可視化を担当するモジュール。"""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.auto import tqdm

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore[assignment]

from .config import ExperimentConfig
from .model import SmallConvNet


@dataclass
class TrainingReport:
    """学習結果のサマリを保持するデータクラス。"""

    history: List[dict[str, float]]
    duration: float
    train_samples: int
    val_samples: int
    model: nn.Module


def set_seed(seed: int) -> None:
    """乱数シードを統一するユーティリティ。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """バッチ精度を計算する。"""

    predicted = predictions.argmax(dim=1)
    return (predicted == targets).float().mean()


def maybe_breakpoint(enabled: bool, message: str) -> None:
    """`--pdb` オプションに連動したブレークポイント。"""

    if not enabled:
        return
    print(f"[pdb] {message}")
    import pdb

    pdb.set_trace()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_pdb: bool = False,
    limit: Optional[int] = None,
) -> tuple[float, float, List[dict[str, float]]]:
    """1エポック分の訓練ループを実行する。"""

    model.train()
    running_loss = 0.0
    running_acc = 0.0
    batch_records: List[dict[str, float]] = []

    progress = tqdm(
        loader,
        desc="train",
        leave=False,
        total=limit,
    )

    # train_one_epoch ループ開始直前にブレーク
    import pdb; pdb.set_trace()

    for batch_idx, (images, labels) in enumerate(progress):
        maybe_breakpoint(use_pdb, f"before to({device}) batch={batch_idx}")
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        maybe_breakpoint(use_pdb, "before forward")
        outputs = model(images)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)
        maybe_breakpoint(use_pdb, "before backward")
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += acc.item()

        record = {
            "batch": batch_idx,
            "loss": loss.item(),
            "accuracy": acc.item(),
        }
        batch_records.append(record)

        avg_loss = running_loss / (batch_idx + 1)
        avg_acc = running_acc / (batch_idx + 1)
        progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

        if limit is not None and batch_idx + 1 >= limit:
            progress.write("Stopping train loop due to --limit-batches.")
            break

    num_batches = max(len(batch_records), 1)
    return running_loss / num_batches, running_acc / num_batches, batch_records


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    limit: Optional[int] = None,
) -> tuple[float, float]:
    """検証データに対する損失と精度を計算する。"""

    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0

    progress = tqdm(
        loader,
        desc="eval",
        leave=False,
        total=limit,
    )

    # evaluate ループ開始直前にブレーク
    import pdb; pdb.set_trace()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(progress):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_acc += accuracy(outputs, labels).item()
            count += 1
            if limit is not None and batch_idx + 1 >= limit:
                progress.write("Stopping eval loop due to --limit-batches.")
                break

    num_batches = max(count, 1)
    return total_loss / num_batches, total_acc / num_batches


def train_model(
    cfg: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> TrainingReport:
    """モデルの学習からレポート作成までを一括で実行する。"""

    set_seed(cfg.seed)
    device = cfg.resolve_device()
    print(f"Using device: {device}")

    model = SmallConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history: List[dict[str, float]] = []
    batch_history: List[dict[str, float]] = []
    start = time.time()

    for epoch in range(cfg.epochs):
        print(f"\n=== Epoch {epoch + 1}/{cfg.epochs} ===")
        train_loss, train_acc, batch_records = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            use_pdb=cfg.use_pdb,
            limit=cfg.limit_batches,
        )
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            limit=cfg.limit_batches,
        )

        print(
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        for record in batch_records:
            record_with_epoch = {"epoch": epoch + 1, **record}
            batch_history.append(record_with_epoch)

    duration = time.time() - start
    print(f"Training finished in {duration:.2f}s")

    if cfg.log_dir is not None:
        cfg.ensure_log_dir()
        save_history_csv(cfg.log_dir / "epoch_metrics.csv", history)
        save_history_csv(cfg.log_dir / "batch_metrics.csv", batch_history)

    return TrainingReport(
        history=history,
        duration=duration,
        train_samples=len(train_loader.dataset),
        val_samples=len(val_loader.dataset),
        model=model,
    )


def save_history_csv(path: Path, records: Iterable[Dict[str, object]]) -> None:
    """学習履歴のリストを CSV として保存する。"""

    records = list(records)
    if not records:
        return

    fieldnames = list(records[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    print(f"Saved metrics to: {path}")


def evaluate_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    limit: Optional[int] = None,
    log_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """検証データに対する混同行列と全体精度を算出する。"""

    model.eval()
    confusion = torch.zeros(10, 10, dtype=torch.long)
    correct = 0
    total = 0
    count = 0

    progress = tqdm(
        loader,
        desc="predict",
        leave=False,
        total=limit,
    )

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(progress):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.numel()
            count += 1

            for target, pred in zip(labels.view(-1), preds.view(-1)):
                confusion[target.long(), pred.long()] += 1

            if limit is not None and batch_idx + 1 >= limit:
                progress.write("Stopping predict loop due to --limit-batches.")
                break

    accuracy_value = correct / max(total, 1)
    print("=== Evaluation Summary ===")
    print(f"accuracy: {accuracy_value:.4f}")
    print("confusion matrix:")
    print(confusion)

    result = {
        "accuracy": accuracy_value,
        "confusion_matrix": confusion,
        "batches": count,
    }

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        conf_path = log_dir / "confusion_matrix.csv"
        with conf_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label"] + list(range(10)))
            for digit, row in enumerate(confusion.tolist()):
                writer.writerow([digit, *row])
        print(f"Confusion matrix saved to: {conf_path}")

    return result


def save_sample_grid(images: torch.Tensor, path: Path, *, nrow: int = 8) -> None:
    """バッチ内の画像をグリッド状に並べて保存する。"""

    grid = make_grid(images[: nrow * nrow], nrow=nrow, normalize=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    if MATPLOTLIB_AVAILABLE:
        assert plt is not None  # for type checkers
        plt.figure(figsize=(6, 6))
        plt.axis("off")
        array = grid.permute(1, 2, 0).cpu().numpy()
        if array.shape[-1] == 1:
            plt.imshow(array.squeeze(-1), cmap="gray")
        else:
            plt.imshow(array)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Sample grid saved to: {path}")
    else:
        from torchvision.utils import save_image

        save_image(grid, path)
        print(
            "matplotlib が見つからないため torchvision の save_image を用いて保存しました:",
            f" {path}",
        )


def plot_learning_curve(history: Iterable[dict[str, float]], path: Path) -> None:
    """エポック単位で記録された学習曲線を描画する。"""

    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib がインストールされていないため学習曲線は保存されませんでした。")
        return

    history = list(history)
    if not history:
        print("学習履歴が空のため学習曲線は描画されませんでした。")
        return

    epochs: List[int] = []
    train_loss: List[float] = []
    val_loss: List[float] = []
    train_acc: List[float] = []
    val_acc: List[float] = []

    for record in history:
        epochs.append(int(record["epoch"]))
        train_loss.append(float(record["train_loss"]))
        val_loss.append(float(record["val_loss"]))
        train_acc.append(float(record["train_accuracy"]))
        val_acc.append(float(record["val_accuracy"]))

    path.parent.mkdir(parents=True, exist_ok=True)
    assert plt is not None
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="train")
    plt.plot(epochs, val_acc, label="val")
    plt.title("Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Learning curve saved to: {path}")