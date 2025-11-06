"""MNIST 用のモデル定義を管理するモジュール。"""

from __future__ import annotations

import torch.nn as nn


class SmallConvNet(nn.Module):
    """コンペ入門向けの軽量な CNN モデル。"""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )
        # モジュール構造が定義された直後にブレーク
        import pdb; pdb.set_trace()
        
    def forward(self, x):  # type: ignore[override]
        """入力テンソルをクラス確率に変換する。"""

        return self.classifier(self.features(x))
