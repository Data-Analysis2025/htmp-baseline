"""MNIST 手作り学習パイプラインの CLI エントリポイント。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import torch

from handmade_mnist import (
    ExperimentConfig,
    create_dataloaders,
    evaluate_predictions,
    inspect_batch,
    plot_learning_curve,
    save_sample_grid,
    summarize_loader,
    train_model,
)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """共通で利用する引数を定義する。"""

    parser.add_argument("--data-dir", type=Path, default=ExperimentConfig().data_dir)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=0)
    parser.add_argument("--limit-batches", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--augment", action="store_true", help="簡易的なデータ拡張を有効にする")
    parser.add_argument("--device", type=str, default="auto", help="使用するデバイス (auto/cpu/cuda:0 など)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="メトリクスや可視化を保存するディレクトリ",
    )


def build_parser() -> argparse.ArgumentParser:
    """サブコマンド付きの引数パーサを構築する。"""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="データローダーの挙動を確認する")
    add_common_arguments(inspect_parser)
    inspect_parser.add_argument(
        "--save-grid",
        action="store_true",
        help="最初のバッチを画像グリッドとして保存する",
    )

    train_parser = subparsers.add_parser("train", help="モデルの学習と評価を実行する")
    add_common_arguments(train_parser)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--pdb", action="store_true", help="学習ループに pdb を差し込む")
    train_parser.add_argument(
        "--plot-learning-curve",
        action="store_true",
        help="学習後に損失と精度の推移を保存する",
    )
    train_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="検証データで混同行列とレポートを算出する",
    )

    return parser


def to_config(args: argparse.Namespace) -> ExperimentConfig:
    """引数から `ExperimentConfig` を生成する。"""

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    limit = None if args.limit_batches < 0 else args.limit_batches
    val_batch = None if getattr(args, "val_batch_size", 0) <= 0 else args.val_batch_size

    return ExperimentConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_batch_size=val_batch,
        epochs=getattr(args, "epochs", ExperimentConfig().epochs),
        learning_rate=getattr(args, "learning_rate", ExperimentConfig().learning_rate),
        limit_batches=limit,
        seed=args.seed,
        num_workers=args.num_workers,
        device=device,
        log_dir=args.log_dir,
        use_pdb=getattr(args, "pdb", False),
        augment=args.augment,
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = to_config(args)
    cfg.ensure_log_dir()
    train_loader, val_loader = create_dataloaders(cfg)

    if args.command == "inspect":
        first_batch = next(iter(train_loader))
        inspect_batch(first_batch)
        summarize_loader(train_loader, cfg.limit_batches, log_dir=cfg.log_dir)
        if args.save_grid and cfg.log_dir is not None:
            save_sample_grid(first_batch[0], cfg.log_dir / "inspect_grid.png")
        print("ヒント: --augment や --batch-size を変えて DataLoader の変化を観察してみましょう。")
        return

    if args.command == "train":
        report = train_model(cfg, train_loader, val_loader)
        print(f"学習時間: {report.duration:.2f}s / train_samples={report.train_samples}")

        if cfg.log_dir is not None and args.plot_learning_curve:
            plot_learning_curve(report.history, cfg.log_dir / "learning_curve.png")

        if args.evaluate:
            device = cfg.resolve_device()
            evaluate_predictions(
                report.model,
                val_loader,
                device,
                limit=cfg.limit_batches,
                log_dir=cfg.log_dir,
            )

        if cfg.log_dir is not None:
            first_batch = next(iter(val_loader))
            save_sample_grid(first_batch[0], cfg.log_dir / "validation_grid.png")

if __name__ == "__main__":
    main()
