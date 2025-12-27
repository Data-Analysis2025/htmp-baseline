"""Granger causality utilities (F-test) without statsmodels dependency."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GrangerResult:
    lag: int
    f_stat: float
    p_value: float


def _ols_rss(y: np.ndarray, X: np.ndarray) -> float:
    """Return residual sum of squares for OLS fit y ~ X (via least squares)."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return float(np.dot(resid, resid))


def granger_causality_f_test(
    y: np.ndarray,
    x: np.ndarray,
    maxlag: int = 10,
) -> pd.DataFrame:
    """Compute Granger causality F-test for x -> y for lags 1..maxlag.

    Uses the classic nested OLS comparison:
      restricted:   y_t ~ 1 + sum_{i=1..L} y_{t-i}
      unrestricted: y_t ~ 1 + sum_{i=1..L} y_{t-i} + sum_{i=1..L} x_{t-i}

    Returns a dataframe with columns: lag, f_stat, p_value.
    """
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    if y_arr.shape[0] != x_arr.shape[0]:
        raise ValueError("y and x must have the same length.")
    n = int(y_arr.shape[0])
    if n == 0:
        return pd.DataFrame(columns=["lag", "f_stat", "p_value"])

    if np.any(~np.isfinite(y_arr)) or np.any(~np.isfinite(x_arr)):
        raise ValueError("y/x must be finite (no NaN/inf). Impute before Granger test.")

    try:
        from scipy.stats import f as f_dist
    except Exception as exc:  # pragma: no cover
        raise ImportError("scipy is required for Granger F-test p-values.") from exc

    results: list[GrangerResult] = []
    for lag in range(1, int(maxlag) + 1):
        n_obs = n - lag
        k_u = 1 + 2 * lag
        df2 = n_obs - k_u
        if df2 <= 0:
            break

        y_target = y_arr[lag:]
        y_lags = np.column_stack([y_arr[lag - i - 1 : n - i - 1] for i in range(lag)])
        x_lags = np.column_stack([x_arr[lag - i - 1 : n - i - 1] for i in range(lag)])

        X_r = np.column_stack([np.ones(n_obs), y_lags])
        X_u = np.column_stack([np.ones(n_obs), y_lags, x_lags])

        rss_r = _ols_rss(y_target, X_r)
        rss_u = _ols_rss(y_target, X_u)

        df1 = lag
        numer = max(rss_r - rss_u, 0.0) / float(df1)
        denom = (rss_u / float(df2)) if rss_u > 0.0 else np.inf
        f_stat = float(numer / denom) if denom > 0.0 and np.isfinite(denom) else 0.0
        p_value = float(f_dist.sf(f_stat, df1, df2)) if f_stat > 0.0 else 1.0
        results.append(GrangerResult(lag=lag, f_stat=f_stat, p_value=p_value))

    return pd.DataFrame([r.__dict__ for r in results])


__all__ = ["GrangerResult", "granger_causality_f_test"]

