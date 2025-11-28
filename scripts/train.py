"""Train a baseline model for the Hull Tactical Market Prediction competition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

# Ensure src modules are importable when running as a script
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.features import FeatureConfig, SimpleFeatureExtractor
from src.model import (
    ModelConfig,
    build_cv,
    create_model,
    get_loss_fn,
    rmse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    return parser.parse_args()


def load_data(paths: Dict[str, str], files: Dict[str, str], target_column: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load training data and extract target."""
    train_df = pd.read_csv(Path(paths["input_dir"]) / files["train"])
    if target_column not in train_df.columns:
        raise KeyError(f"Target column '{target_column}' not found in training data.")
    return train_df, train_df[target_column].values


def build_feature_extractor(feature_cfg: Dict[str, object], cv_cfg: Dict[str, object]) -> FeatureConfig:
    """Construct FeatureConfig with defaults."""
    return FeatureConfig(
        drop_columns=feature_cfg.get("drop_columns", []),
        imputation_strategy=feature_cfg.get("imputation_strategy", "median"),
        scale=feature_cfg.get("scale", True),
        rolling_windows=feature_cfg.get("rolling_windows"),
        enable_interactions=feature_cfg.get("enable_interactions", True),
        time_column=cv_cfg.get("time_column"),
        group_column=cv_cfg.get("group_column"),
        missing_ratio_threshold=feature_cfg.get("missing_ratio_threshold", 40.0),
    )


def build_model_config(model_cfg: Dict[str, object], seed: int | None) -> ModelConfig:
    """Construct ModelConfig with objective and seed."""
    return ModelConfig(
        type=model_cfg.get("type", "ridge"),
        params=model_cfg.get("params", {}),
        fit_intercept=model_cfg.get("fit_intercept", True),
        objective=model_cfg.get("objective", "rmse"),
        random_state=seed,
    )


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    if args.seed is not None:
        config.data["seed"] = args.seed
    config.ensure_dirs()

    paths = config.get("paths")
    files = config.get("files")
    target_cfg = config.get("target")
    cv_cfg = config.get("cv")
    feature_cfg = config.get("features")
    model_cfg = config.get("model")

    train_df, y = load_data(paths, files, target_cfg["column"])
    feature_config = build_feature_extractor(feature_cfg, cv_cfg)
    extractor = SimpleFeatureExtractor(feature_config)
    X = extractor.fit_transform(train_df, target_column=target_cfg["column"])

    cv_strategy = build_cv(
        strategy=cv_cfg.get("strategy", "time_series"),
        n_splits=cv_cfg.get("n_splits", 5),
        shuffle=cv_cfg.get("strategy") != "time_series",
        random_state=config.get("seed"),
    )

    model_config = build_model_config(model_cfg, seed=config.get("seed"))
    loss_fn = get_loss_fn(model_config.objective)

    oof_predictions = np.zeros(len(train_df))
    train_losses: List[float] = []
    val_losses: List[float] = []
    val_rmse_scores: List[float] = []

    for fold, (train_idx, valid_idx) in enumerate(cv_strategy.split(X, y)):
        model = create_model(model_config)
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y[valid_idx]

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        valid_pred = model.predict(X_valid)

        train_loss = loss_fn(y_train, train_pred)
        val_loss = loss_fn(y_valid, valid_pred)
        val_rmse_score = rmse(y_valid, valid_pred)

        oof_predictions[valid_idx] = valid_pred
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_rmse_scores.append(val_rmse_score)

        model_path = Path(paths["models_dir"]) / f"{config.get('run_name')}_fold_{fold}.pkl"
        joblib.dump({
            "model": model,
            "feature_extractor": extractor,   # ★ 完全な状態を丸ごと保存
        },
            model_path,
        )
        print(
            f"Fold {fold}: train_{model_config.objective}={train_loss:.5f} "
            f"val_{model_config.objective}={val_loss:.5f} | val_RMSE={val_rmse_score:.5f}"
        )

    overall_oof_loss = loss_fn(y, oof_predictions)
    overall_oof_rmse = rmse(y, oof_predictions)
    print(
        f"Overall OOF {model_config.objective.upper()}: {overall_oof_loss:.5f} | "
        f"OOF RMSE: {overall_oof_rmse:.5f}"
    )

    metadata = {
        "run_name": config.get("run_name"),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_rmse_scores": val_rmse_scores,
        "oof_loss": overall_oof_loss,
        "oof_rmse": overall_oof_rmse,
        "objective": model_config.objective,
        "feature_columns": list(X.columns),
        "config_path": str(config.path),
        "model_type": model_config.type,
    }
    metadata_path = Path(paths["models_dir"]) / f"{config.get('run_name')}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    oof_path = Path(paths["processed_dir"]) / f"{config.get('run_name')}_oof.csv"
    Path(paths["processed_dir"]).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"oof_pred": oof_predictions}).to_csv(oof_path, index=False)


if __name__ == "__main__":
    main()
