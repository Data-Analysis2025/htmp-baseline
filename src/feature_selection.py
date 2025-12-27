"""Feature selection utilities for the HTMP baseline."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression, mutual_info_regression


def _normalize_scores(columns: List[str], scores: np.ndarray) -> Dict[str, float]:
    """Scale scores to [0, 1] while handling degenerate cases."""
    arr = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return {}
    max_val = float(arr.max())
    if max_val <= 0.0:
        return {col: 0.0 for col in columns}
    normalized = arr / max_val
    return {col: float(score) for col, score in zip(columns, normalized)}


def mutual_information_scores(X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
    """Compute normalized mutual information scores for each feature."""
    scores = mutual_info_regression(X, y, random_state=0)
    return _normalize_scores(list(X.columns), scores)


def f_statistic_scores(X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
    """Compute normalized F-statistic scores for each feature."""
    f_stats, _ = f_regression(X, y)
    return _normalize_scores(list(X.columns), f_stats)


def stability_scores(X: pd.DataFrame, y: np.ndarray, n_periods: int = 5) -> Dict[str, float]:
    """Estimate stability via per-period correlations.

    Splits the time index into ``n_periods`` consecutive chunks and computes the
    Pearson correlation between each feature and ``y`` within each chunk. The
    stability score combines magnitude (average absolute correlation),
    variability (penalized by standard deviation), and sign consistency.
    """
    n_samples = len(X)
    if n_samples == 0:
        return {}

    period_indices = np.array_split(np.arange(n_samples), n_periods)
    y_array = np.asarray(y)
    scores: Dict[str, float] = {}

    for col in X.columns:
        feature = X[col].values
        corrs: List[float] = []
        for idx in period_indices:
            if idx.size == 0:
                continue
            x_slice = feature[idx]
            y_slice = y_array[idx]
            if np.all(np.isnan(x_slice)) or np.all(np.isnan(y_slice)):
                corrs.append(np.nan)
                continue
            if np.nanstd(x_slice) == 0.0 or np.nanstd(y_slice) == 0.0:
                corrs.append(0.0)
                continue
            corr = np.corrcoef(x_slice, y_slice)[0, 1]
            corrs.append(corr)

        corrs_arr = np.asarray(corrs, dtype=float)
        valid = corrs_arr[~np.isnan(corrs_arr)]
        if valid.size == 0:
            scores[col] = 0.0
            continue

        avg_abs = float(np.mean(np.abs(valid)))
        std_corr = float(np.std(valid))
        sign_consistency = float(np.abs(np.mean(np.sign(valid))))  # 1 if all signs match, 0 if mixed

        stability = avg_abs * (1.0 / (1.0 + std_corr)) * sign_consistency
        scores[col] = float(np.clip(stability, 0.0, 1.0))

    return scores


def lgbm_importance_scores(X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
    """Compute normalized feature importances from a small LightGBM model."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError:  # pragma: no cover - optional dependency
        return {col: 0.0 for col in X.columns}

    model = LGBMRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
    )
    model.fit(X, y)
    importances = getattr(model, "feature_importances_", np.zeros(X.shape[1]))
    return _normalize_scores(list(X.columns), importances)


def ensemble_rank(scores: Dict[str, Dict[str, float]], top_k: int | None = None) -> List[str]:
    """Aggregate scores across methods and return ranked feature names."""
    if not scores:
        return []

    aggregated: Dict[str, List[float]] = {}
    for method_scores in scores.values():
        for feature, value in method_scores.items():
            aggregated.setdefault(feature, []).append(float(value))

    averaged = {feature: float(np.mean(values)) for feature, values in aggregated.items()}
    ranked = [name for name, _ in sorted(averaged.items(), key=lambda kv: kv[1], reverse=True)]
    if top_k is not None:
        return ranked[:top_k]
    return ranked


__all__ = [
    "mutual_information_scores",
    "f_statistic_scores",
    "stability_scores",
    "lgbm_importance_scores",
    "ensemble_rank",
]
