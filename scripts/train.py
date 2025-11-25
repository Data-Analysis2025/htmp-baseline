"""Train a baseline model for the Hull Tactical Market Prediction competition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

# Ensure src modules are importable when running as a script
import sys


# OptunaとLightGBM
import lightgbm as lgb
import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.features import FeatureConfig, SimpleFeatureExtractor
from src.model import ModelConfig, build_cv, create_model, rmse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    return parser.parse_args()


# ヘルパー関数
def parse_optuna_params(trial: optuna.Trial, param_config: dict) -> dict:
    params = {}
    for name, config in param_config.items():
        if isinstance(config, dict):
            low = config["low"]
            high = config["high"]

            if config["type"] == "loguniform":
                params[name] = trial.suggest_loguniform(name, float(low), float(high))
            elif config["type"] == "uniform":
                params[name] = trial.suggest_uniform(name, float(low), float(high))
            elif config["type"] == "int":
                params[name] = trial.suggest_int(name, int(low), int(high))
        
        else:
            params[name] = config
    return params


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
    optuna_cfg = config.get("optuna") # Optuna設定を読み込む

    train_df = pd.read_csv(Path(paths["input_dir"]) / files["train"])
    target_column = target_cfg["column"]
    if target_column not in train_df.columns:
        raise KeyError(f"Target column '{target_column}' not found in training data.")

    y = train_df[target_column].values

    feature_config = FeatureConfig(
        drop_columns=feature_cfg.get("drop_columns", []),
        imputation_strategy=feature_cfg.get("imputation_strategy", "median"),
        scale=feature_cfg.get("scale", True),
        rolling_windows=feature_cfg.get("rolling_windows"),
        enable_interactions=feature_cfg.get("enable_interactions", False),
        time_column=cv_cfg.get("time_column"),
        group_column=cv_cfg.get("group_column"),
    )
    extractor = SimpleFeatureExtractor(feature_config)
    X = extractor.fit_transform(train_df, target_column=target_column)

    cv_strategy = build_cv(
        strategy=cv_cfg.get("strategy", "time_series"),
        n_splits=cv_cfg.get("n_splits", 5),
        shuffle=cv_cfg.get("strategy") != "time_series",
        random_state=config.get("seed"),
    )


    # Optunaによるハイパーパラメータ探索
    def objective(trial: optuna.Trial) -> float:
        # YAMLの設定からOptunaの探索パラメータを動的に生成
        model_params = parse_optuna_params(trial, model_cfg.get("params", {}))
        
        scores = []
        cv_search = build_cv(
            strategy=cv_cfg.get("strategy", "time_series"),
            n_splits=cv_cfg.get("n_splits", 5),
            shuffle=cv_cfg.get("strategy") != "time_series",
            random_state=config.get("seed"),
        )
        
        for fold, (train_idx, valid_idx) in enumerate(cv_search.split(X, y)):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            model = lgb.LGBMRegressor(**model_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
            preds = model.predict(X_valid)
            scores.append(rmse(y_valid, preds))
            
        return np.mean(scores)

    study = optuna.create_study(direction=optuna_cfg.get("direction", "minimize"))
    study.optimize(objective, n_trials=optuna_cfg.get("n_trials", 50))

    best_params = study.best_params
    print(f"--- Optuna Search Finished ---")
    print(f"Best RMSE (avg): {study.best_value:.5f}")
    print(f"Best Hyperparameters: {best_params}")


    # 見つけたBest ParamsでK-Fold訓練 & モデル保存
    # model_configを、Optunaで見つけたbest_paramsで上書きする
    final_model_params = model_cfg.get("params", {}).copy()
    final_model_params.update(best_params) # best_paramsで上書き

    model_config = ModelConfig(
        type=model_cfg.get("type", "lightgbm"),
        params=final_model_params, # Optunaの結果を使う
        fit_intercept=model_cfg.get("fit_intercept", False),
    )

    oof_predictions = np.zeros(len(train_df))
    scores: List[float] = []

    for fold, (train_idx, valid_idx) in enumerate(cv_strategy.split(X, y)):
        model = create_model(model_config)
        # LGBMのfit時には early_stopping を使う
        model.fit(
            X.iloc[train_idx], y[train_idx],
            eval_set=[(X.iloc[valid_idx], y[valid_idx])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        preds = model.predict(X.iloc[valid_idx])
        oof_predictions[valid_idx] = preds
        fold_score = rmse(y[valid_idx], preds)
        scores.append(fold_score)
        model_path = Path(paths["models_dir"]) / f"{config.get('run_name')}_fold_{fold}.pkl"
        joblib.dump(
            {
                "model": model,
                "scaler": extractor.scaler,
                "feature_columns": list(X.columns),
            },
            model_path,
        )
        print(f"Fold {fold}: RMSE={fold_score:.5f}")

    overall_rmse = rmse(y, oof_predictions)
    print(f"Overall OOF RMSE: {overall_rmse:.5f}")

    metadata = {
        "run_name": config.get("run_name"),
        "scores": scores,
        "oof_rmse": overall_rmse,
        "optuna_best_value": study.best_value,
        "optuna_best_params": best_params,
        "feature_columns": list(X.columns),
        "config_path": str(config.path),
    }
    metadata_path = Path(paths["models_dir"]) / f"{config.get('run_name')}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    oof_path = Path(paths["processed_dir"]) / f"{config.get('run_name')}_oof.csv"
    Path(paths["processed_dir"]).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"oof_pred": oof_predictions}).to_csv(oof_path, index=False)


if __name__ == "__main__":
    main()
