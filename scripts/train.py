"""Train a baseline model for the Hull Tactical Market Prediction competition."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# NOTE: In some sandboxed environments OpenMP-backed linear algebra aborts when it cannot
# allocate shared memory. Force a sequential threading layer before importing numpy/scipy/sklearn.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQ")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd

# Ensure src modules are importable when running as a script
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from src.config import Config
from cv_schemes import get_cv_splits
from src.feature_selection import (
    ensemble_rank,
    f_statistic_scores,
    lgbm_importance_scores,
    mutual_information_scores,
    stability_scores,
)
from src.calendar_features import merge_calendar_features
from src.features import FeatureConfig, SimpleFeatureExtractor
from src.model import (
    ModelConfig,
    build_cv,
    create_model,
    get_loss_fn,
    rmse,
)
try:
    import optuna
except ImportError:  # pragma: no cover - optional
    optuna = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    return parser.parse_args()


def load_data(
    paths: Dict[str, str],
    files: Dict[str, str],
    target_column: str,
    date_key: str,
    calendar_path: Path | None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load training data and extract target."""
    train_df = pd.read_csv(Path(paths["input_dir"]) / files["train"])
    if calendar_path is not None:
        train_df = merge_calendar_features(train_df, calendar_path, date_key=date_key)
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
        rolling_feature_naming=feature_cfg.get("rolling_feature_naming", "roll"),
        enable_interactions=feature_cfg.get("enable_interactions", True),
        time_column=cv_cfg.get("time_column"),
        group_column=cv_cfg.get("group_column"),
        missing_ratio_threshold=feature_cfg.get("missing_ratio_threshold", 40.0),
        use_limited_features=feature_cfg.get("use_limited_features", False),
        limited_features=feature_cfg.get("limited_features"),
        manual_features=feature_cfg.get("manual_features"),
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


def build_custom_cv_splits(
    df: pd.DataFrame,
    cv_cfg: Dict[str, object],
) -> List[tuple[np.ndarray, np.ndarray]] | None:
    """Return custom CV splits if strategy is cpcv or monthly_walk."""
    strategy = cv_cfg.get("strategy", "time_series")
    if strategy not in {"cpcv", "monthly_walk"}:
        return None
    return get_cv_splits(df, cv_type=strategy, cv_params=cv_cfg)


def suggest_lgbm_params(trial, base: Dict[str, object]) -> Dict[str, object]:
    params = {**base}
    params.update(
        {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }
    )
    return params


def build_optuna_objective(
    X: pd.DataFrame,
    y: np.ndarray,
    model_config: ModelConfig,
    loss_fn,
    cv_splits: List[tuple[np.ndarray, np.ndarray]],
):
    if optuna is None:
        raise ImportError("optuna is not installed. Install optuna to use hyperparameter search.")

    def objective(trial):
        if model_config.type != "lightgbm":
            raise ValueError("Optuna search is currently supported only for LightGBM.")
        trial_params = suggest_lgbm_params(trial, model_config.params)
        scores: List[float] = []
        for train_idx, valid_idx in cv_splits:
            model = create_model(
                ModelConfig(
                    type=model_config.type,
                    params=trial_params,
                    fit_intercept=model_config.fit_intercept,
                    objective=model_config.objective,
                    random_state=model_config.random_state,
                )
            )
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y[valid_idx]
            if model_config.type == "lightgbm":
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric="rmse",
                )
            else:
                model.fit(X_train, y_train)
            preds = model.predict(X_valid)
            scores.append(loss_fn(y_valid, preds))
        return float(np.mean(scores))

    return objective


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
    optuna_cfg = config.get("optuna", default={})

    date_key = cv_cfg.get("time_column") or "date_id"
    calendar_path = None
    calendar_file = files.get("calendar")
    if calendar_file:
        calendar_path = Path(calendar_file)
        if not calendar_path.is_absolute():
            calendar_path = Path(paths["input_dir"]) / calendar_file

    train_df, y = load_data(paths, files, target_cfg["column"], date_key, calendar_path)
    train_df = train_df.sort_values(date_key).reset_index(drop=True)
    y = y[train_df.index]
    feature_config = build_feature_extractor(feature_cfg, cv_cfg)
    extractor = SimpleFeatureExtractor(feature_config)
    X = extractor.fit_transform(train_df, target_column=target_cfg["column"])

    original_feature_count = X.shape[1]
    selection_method = feature_cfg.get("selection_method", "ensemble")
    n_selected_cfg = feature_cfg.get("n_selected")
    if n_selected_cfg is None:
        n_selected: int | None = None
    else:
        try:
            n_selected = int(n_selected_cfg)
        except (TypeError, ValueError):
            raise ValueError("features.n_selected must be an integer or null.") from None

    selected_features = list(X.columns)
    selection_enabled = (
        selection_method == "ensemble"
        and n_selected is not None
        and 0 < n_selected < original_feature_count
    )

    if selection_enabled:
        score_dict = {
            "mutual_information": mutual_information_scores(X, y),
            "f_statistic": f_statistic_scores(X, y),
            "stability": stability_scores(X, y, n_periods=cv_cfg.get("n_splits", 5)),
            "lgbm_importance": lgbm_importance_scores(X, y),
        }
        selected_features = ensemble_rank(score_dict, top_k=n_selected)
        if selected_features:
            X = X[selected_features]
        else:
            selected_features = list(X.columns)
        print(
            f"Selected {len(selected_features)} features out of {original_feature_count} "
            f"using ensemble ranking."
        )
    else:
        print("Feature selection skipped; using all features.")

    custom_cv_splits = build_custom_cv_splits(train_df, cv_cfg)
    cv_strategy = None
    if custom_cv_splits is None:
        cv_strategy = build_cv(
            strategy=cv_cfg.get("strategy", "time_series"),
            n_splits=cv_cfg.get("n_splits", 5),
            shuffle=cv_cfg.get("strategy") != "time_series",
            random_state=config.get("seed"),
        )

    model_config = build_model_config(model_cfg, seed=config.get("seed"))
    loss_fn = get_loss_fn(model_config.objective)

    if optuna_cfg.get("use_optuna", False):
        if model_config.type != "lightgbm":
            raise ValueError("optuna tuning is only implemented for LightGBM models.")
        if optuna is None:
            raise ImportError("optuna is not installed. Please install optuna or disable optuna tuning.")
        if custom_cv_splits is None and cv_strategy is not None:
            cv_splits = list(cv_strategy.split(X, y))
        else:
            cv_splits = custom_cv_splits or []
        if not cv_splits:
            raise ValueError("No CV splits available for optuna tuning.")
        objective = build_optuna_objective(X, y, model_config, loss_fn, cv_splits)
        study = optuna.create_study(direction="minimize", study_name=f"lgbm_{config.get('run_name')}")
        study.optimize(
            objective,
            n_trials=int(optuna_cfg.get("n_trials", 20)),
            timeout=optuna_cfg.get("timeout"),
            show_progress_bar=optuna_cfg.get("show_progress_bar", False),
        )
        model_config.params = suggest_lgbm_params(study.best_trial, model_config.params)
        print(f"Optuna best score: {study.best_value:.5f}")
        print(f"Optuna best params: {model_config.params}")

    oof_predictions = np.zeros(len(train_df))
    train_losses: List[float] = []
    val_losses: List[float] = []
    val_rmse_scores: List[float] = []

    if custom_cv_splits is not None:
        split_iter = list(custom_cv_splits)
    elif cv_strategy is not None:
        split_iter = list(cv_strategy.split(X, y))
    else:
        split_iter = []

    if not split_iter:
        raise ValueError("No CV splits available. Check cv configuration.")

    for fold, (train_idx, valid_idx) in enumerate(split_iter):
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
            "feature_columns": selected_features,
            "scaler": extractor.scaler,
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
        "selected_features": selected_features,
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
