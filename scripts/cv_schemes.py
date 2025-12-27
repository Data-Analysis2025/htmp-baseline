"""Custom cross-validation schemes for time-series style validation."""
from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

CVSplit = Tuple[np.ndarray, np.ndarray]


def _split_into_blocks(indices: np.ndarray, n_splits: int) -> List[np.ndarray]:
    """Split indices into n_splits contiguous blocks (last block may be larger)."""
    return np.array_split(indices, n_splits)


def generate_cpcv_splits(
    df: pd.DataFrame,
    n_splits: int,
    n_test_blocks: int,
    embargo_blocks: int = 0,
    time_column: str | None = None,
) -> List[CVSplit]:
    """Generate combinatorial purged CV splits.

    Args:
        df: DataFrame sorted by time_column (or index order if None).
        n_splits: Number of contiguous blocks.
        n_test_blocks: Number of blocks to use as validation in each split (combinations).
        embargo_blocks: Number of blocks to embargo on each side of validation blocks.
        time_column: Optional column to sort by before blocking.
    """
    if n_test_blocks <= 0 or n_test_blocks >= n_splits:
        raise ValueError("n_test_blocks must be >=1 and < n_splits.")

    work_df = df
    if time_column and time_column in df.columns:
        work_df = df.sort_values(time_column).reset_index(drop=True)
    indices = work_df.index.to_numpy()
    blocks = _split_into_blocks(indices, n_splits)

    splits: List[CVSplit] = []
    for test_combo in itertools.combinations(range(n_splits), n_test_blocks):
        test_blocks = set(test_combo)
        embargoed = set()
        if embargo_blocks > 0:
            for b in test_blocks:
                for emb in range(b - embargo_blocks, b + embargo_blocks + 1):
                    if 0 <= emb < n_splits:
                        embargoed.add(emb)
        forbidden = test_blocks | embargoed
        train_blocks = [i for i in range(n_splits) if i not in forbidden]

        test_idx = np.concatenate([blocks[b] for b in test_blocks]) if test_blocks else np.array([], dtype=int)
        train_idx = np.concatenate([blocks[b] for b in train_blocks]) if train_blocks else np.array([], dtype=int)
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits


def generate_monthly_walk_splits(
    df: pd.DataFrame,
    month_col: str,
    train_months: int,
    valid_months: int,
    step_months: int = 1,
    min_train_months: int = 12,
    time_column: str | None = None,
    approx_month_length: int = 30,
) -> List[CVSplit]:
    """Generate walk-forward splits using monthly windows.

    Args:
        df: DataFrame containing month_col.
        month_col: Column name representing month index (e.g., month_id or YYYYMM).
        train_months: Number of months used for training window.
        valid_months: Number of months used for validation window.
        step_months: Step size to slide the window.
        min_train_months: Minimum months required to keep a split.
        time_column: Optional column to sort by first (then month ordering is taken from month_col unique).
        approx_month_length: If month_col is missing, create a pseudo month_id by dividing
            a time column (or index) by this length (e.g., 30 days ~ 1 month).
    """
    work_df = df
    if time_column and time_column in df.columns:
        work_df = df.sort_values(time_column).reset_index(drop=True)

    if month_col not in work_df.columns:
        # Fallback: approximate month buckets from a time/index column
        if time_column and time_column in work_df.columns:
            base_series = work_df[time_column]
        else:
            base_series = work_df.index.to_series()
        pseudo_month = ((base_series - base_series.min()) // approx_month_length).astype(int)
        work_df = work_df.copy()
        work_df[month_col] = pseudo_month

    months = sorted(work_df[month_col].unique())
    total_months = len(months)
    splits: List[CVSplit] = []

    max_start = total_months - (train_months + valid_months)
    for start in range(0, max_start + 1, step_months):
        train_months_range = months[start : start + train_months]
        valid_months_range = months[start + train_months : start + train_months + valid_months]
        if len(train_months_range) < min_train_months or len(valid_months_range) == 0:
            continue

        train_idx = work_df.index[work_df[month_col].isin(train_months_range)].to_numpy()
        valid_idx = work_df.index[work_df[month_col].isin(valid_months_range)].to_numpy()
        if len(train_idx) == 0 or len(valid_idx) == 0:
            continue
        splits.append((train_idx, valid_idx))
    return splits


def get_cv_splits(df: pd.DataFrame, cv_type: str, cv_params: Dict) -> List[CVSplit]:
    """Dispatcher to build CV splits."""
    params = dict(cv_params)
    params.pop("strategy", None)

    if cv_type == "cpcv":
        return generate_cpcv_splits(
            df,
            n_splits=int(params.get("n_splits", 5)),
            n_test_blocks=int(params.get("n_test_blocks", 2)),
            embargo_blocks=int(params.get("embargo_blocks", 0)),
            time_column=params.get("time_column"),
        )
    if cv_type == "monthly_walk":
        return generate_monthly_walk_splits(
            df,
            month_col=params.get("month_col", "month_id"),
            train_months=int(params.get("train_months", 24)),
            valid_months=int(params.get("valid_months", 1)),
            step_months=int(params.get("step_months", 1)),
            min_train_months=int(params.get("min_train_months", 12)),
            time_column=params.get("time_column"),
            approx_month_length=int(params.get("approx_month_length", 30)),
        )
    raise ValueError(f"Unsupported cv_type: {cv_type}")


__all__ = [
    "CVSplit",
    "get_cv_splits",
    "generate_cpcv_splits",
    "generate_monthly_walk_splits",
]
