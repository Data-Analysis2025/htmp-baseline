"""実験設定をまとめるデータクラスとユーティリティ。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class ExperimentConfig:
    """学習実験やデータ調査で共通して利用する設定値。"""

    data_dir: Path = Path(os.environ.get("MNIST_DATA", Path.home() / ".mnist"))
    batch_size: int = 64
    val_batch_size: Optional[int] = None
    epochs: int = 3
    learning_rate: float = 1e-3
    limit_batches: Optional[int] = 50
    seed: int = 42
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: Optional[Path] = None
    use_pdb: bool = False
    augment: bool = False

    def resolve_device(self) -> torch.device:
        """文字列表現を `torch.device` に変換する。"""

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(self.device)

    def ensure_log_dir(self) -> Optional[Path]:
        """ログ出力用のディレクトリを必要に応じて生成する。"""

        if self.log_dir is None:
            return None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return self.log_dir

    @property
    def effective_val_batch_size(self) -> int:
        """検証用バッチサイズを決定する。"""

        return self.val_batch_size or self.batch_size
