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

# --- 定数定義 ---
# コンペのルールやベースラインに基づく設定
MAX_INVESTMENT = 2.0
MIN_INVESTMENT = 0.0
SIGNAL_MULTIPLIER = 400.0  # リターン予測をポジションに変換するための係数


class ParticipantVisibleError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    return parser.parse_args()


# --- カスタム評価関数 (提供されたコード) ---
def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Calculates a custom evaluation metric (volatility-adjusted Sharpe ratio).
    """

    if not pd.api.types.is_numeric_dtype(submission['prediction']):
        raise ParticipantVisibleError('Predictions must be numeric')

    # solutionをコピーして副作用を防ぐ
    solution = solution.copy()
    solution['position'] = submission['prediction'].values # valuesで代入してインデックス不一致を防ぐ

    if solution['position'].max() > MAX_INVESTMENT + 1e-5: # 浮動小数点誤差を許容
        print(f"Warning: Position max {solution['position'].max()} exceeds {MAX_INVESTMENT}")
    if solution['position'].min() < MIN_INVESTMENT - 1e-5:
        print(f"Warning: Position min {solution['position'].min()} below {MIN_INVESTMENT}")

    solution['strategy_returns'] = solution['risk_free_rate'] * (1 - solution['position']) + solution['position'] * solution['forward_returns']

    # Calculate strategy's Sharpe ratio
    strategy_excess_returns = solution['strategy_returns'] - solution['risk_free_rate']
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(solution)) - 1
    strategy_std = solution['strategy_returns'].std()

    trading_days_per_yr = 252
    if strategy_std == 0:
        # stdが0の場合はスコア0とする（またはエラー）
        return 0.0
        
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)

    # Calculate market return and volatility
    market_excess_returns = solution['forward_returns'] - solution['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(solution)) - 1
    market_std = solution['forward_returns'].std()

    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)

    if market_volatility == 0:
        return 0.0

    # Calculate the volatility penalty
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol

    # Calculate the return penalty
    return_gap = max(
        0,
        (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr,
    )
    return_penalty = 1 + (return_gap**2) / 100

    # Adjust the Sharpe ratio by the volatility and return penalty
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return min(float(adjusted_sharpe), 1_000_000)


def calculate_metric(y_pred: np.ndarray, df_valid: pd.DataFrame) -> float:
    """予測値(リターン)と検証用データからコンペ指標を計算するラッパー"""
    
    # 1. 予測値(リターン)をポジション(シグナル)に変換
    #    Baselineのロジック: signal = pred * 400 + 1 (0.0~2.0にクリップ)
    signals = y_pred * SIGNAL_MULTIPLIER + 1.0
    signals = np.clip(signals, MIN_INVESTMENT, MAX_INVESTMENT)
    
    # 2. 提出用DF作成
    submission = pd.DataFrame({'prediction': signals})
    
    # 3. 正解データDF作成 (risk_free_rateとforward_returnsが必要)
    solution = df_valid[['risk_free_rate', 'forward_returns']].reset_index(drop=True)
    
    # 4. スコア計算
    try:
        return score(solution, submission, 'dummy_id')
    except Exception as e:
        print(f"Score calc error: {e}")
        return 0.0


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

    # データを読み込む
    train_df = pd.read_csv(Path(paths["input_dir"]) / files["train"])
    
    # スコア計算に必要な列が存在するか確認
    required_cols = ['risk_free_rate', 'forward_returns']
    for col in required_cols:
        if col not in train_df.columns:
            raise KeyError(f"Column '{col}' is required for scoring but not found in training data.")

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
            
            # --- RMSEではなくカスタムスコアを計算 ---
            # 検証期間の生データ(risk_free_rateなど)を取得
            df_valid_fold = train_df.iloc[valid_idx]
            fold_score = calculate_metric(preds, df_valid_fold)
            scores.append(fold_score)
        
        # Optunaはデフォルトで最小化(minimize)を行う設定になっていることが多い。
        # シャープレシオは「最大化」したい指標なので、マイナスをかけて返す。
        # (configのdirectionがminimizeの場合)
        return -np.mean(scores)

    study = optuna.create_study(direction=optuna_cfg.get("direction", "minimize"))
    study.optimize(objective, n_trials=optuna_cfg.get("n_trials", 50))

    best_params = study.best_params
    best_score = -study.best_value # マイナスを戻して本来のスコア表示
    
    print(f"--- Optuna Search Finished ---")
    print(f"Best Score (Sharpe): {best_score:.5f}")
    print(f"Best Hyperparameters: {best_params}")


    # 見つけたBest ParamsでK-Fold訓練 & モデル保存
    final_model_params = model_cfg.get("params", {}).copy()
    final_model_params.update(best_params) 

    model_config = ModelConfig(
        type=model_cfg.get("type", "lightgbm"),
        params=final_model_params, 
        fit_intercept=model_cfg.get("fit_intercept", False),
    )

    oof_predictions = np.zeros(len(train_df))
    scores: List[float] = []

    cv_strategy = build_cv(
        strategy=cv_cfg.get("strategy", "time_series"),
        n_splits=cv_cfg.get("n_splits", 5),
        shuffle=cv_cfg.get("strategy") != "time_series",
        random_state=config.get("seed"),
    )

    for fold, (train_idx, valid_idx) in enumerate(cv_strategy.split(X, y)):
        model = create_model(model_config)
        model.fit(
            X.iloc[train_idx], y[train_idx],
            eval_set=[(X.iloc[valid_idx], y[valid_idx])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        preds = model.predict(X.iloc[valid_idx])
        oof_predictions[valid_idx] = preds
        
        # --- カスタムスコア計算 ---
        df_valid_fold = train_df.iloc[valid_idx]
        fold_score = calculate_metric(preds, df_valid_fold)
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
        print(f"Fold {fold}: Sharpe={fold_score:.5f}")

    # 全体でのOOFスコア計算
    overall_score = calculate_metric(oof_predictions, train_df)
    print(f"Overall OOF Sharpe: {overall_score:.5f}")

    metadata = {
        "run_name": config.get("run_name"),
        "scores": scores,
        "oof_score": overall_score, # RMSEではなくScoreを保存
        "optuna_best_value": best_score,
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