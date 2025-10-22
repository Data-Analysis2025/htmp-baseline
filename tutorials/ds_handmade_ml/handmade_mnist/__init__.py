"""MNIST 手作りハンズオン用のコンポーネント群。"""

from .config import ExperimentConfig
from .data import create_dataloaders, inspect_batch, summarize_loader
from .model import SmallConvNet
from .training import (
    TrainingReport,
    evaluate_predictions,
    plot_learning_curve,
    save_sample_grid,
    train_model,
)

__all__ = [
    "ExperimentConfig",
    "create_dataloaders",
    "SmallConvNet",
    "TrainingReport",
    "train_model",
    "inspect_batch",
    "summarize_loader",
    "evaluate_predictions",
    "save_sample_grid",
    "plot_learning_curve",
]
