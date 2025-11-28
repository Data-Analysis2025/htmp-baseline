"""Model factory for baseline training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, KFold

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover - optional dependency
    LGBMRegressor = None


@dataclass
class ModelConfig:
    type: str
    params: Dict[str, Any]
    fit_intercept: bool = True
    objective: str = "rmse"
    random_state: int | None = None


def create_model(config: ModelConfig):
    """Instantiate a model according to the config."""
    if config.type == "ridge":
        return Ridge(**config.params)
    if config.type == "linear_regression":
        return LinearRegression(fit_intercept=config.fit_intercept)
    if config.type == "elasticnet":
        return ElasticNet(random_state=config.random_state, **config.params)
    if config.type == "lightgbm":
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not installed. Please install lightgbm to use this model.")
        lgbm_params = {**config.params}
        if config.objective == "rmse":
            lgbm_params.setdefault("objective", "regression")
            lgbm_params.setdefault("metric", "rmse")
        elif config.objective == "rmae":
            lgbm_params.setdefault("objective", "mae")
            lgbm_params.setdefault("metric", "mae")
        lgbm_params.setdefault("random_state", config.random_state)
        return LGBMRegressor(**lgbm_params)
    raise ValueError(f"Unsupported model type: {config.type}")


def build_cv(strategy: str, n_splits: int, shuffle: bool = False, random_state: int | None = None):
    if strategy == "time_series":
        return TimeSeriesSplit(n_splits=n_splits)
    if strategy == "kfold":
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    raise ValueError(f"Unsupported CV strategy: {strategy}")


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def rmae(y_true, y_pred):
    """Robust MAE (sqrt of MAE to keep scale similar to RMSE)."""
    return mean_absolute_error(y_true, y_pred) ** 0.5


def get_loss_fn(objective: str) -> Callable:
    """Return loss callable for training/validation reporting."""
    name = objective.lower()
    if name == "rmse":
        return rmse
    if name == "rmae":
        return rmae
    raise ValueError(f"Unsupported objective: {objective}")


__all__ = ["ModelConfig", "create_model", "build_cv", "rmse", "rmae", "get_loss_fn"]
