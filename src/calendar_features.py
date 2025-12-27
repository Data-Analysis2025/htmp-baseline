"""Utilities for enriching HTMP data with calendar-based cyclic features."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return first column whose lowercase name matches any of the candidates."""
    lowered = {c.lower() for c in candidates}
    for col in df.columns:
        if col.lower() in lowered:
            return col
    return None


def _build_cyclic_features(day_of_week: pd.Series, month: pd.Series) -> pd.DataFrame:
    """Compute sin/cos encodings for day-of-week and month."""
    dow_zero_based = day_of_week.astype(float)
    if dow_zero_based.dropna().between(1, 7).all():
        dow_zero_based = dow_zero_based - 1  # Align 1-7 data to 0-6 for stable encodings.

    month_aligned = month.astype(float)
    if month_aligned.dropna().between(1, 12).all():
        month_aligned = month_aligned - 1  # Keep December close to January on the unit circle.

    features = pd.DataFrame(index=day_of_week.index)
    features["cal_dow_sin"] = np.sin(2 * np.pi * dow_zero_based / 7.0)
    features["cal_dow_cos"] = np.cos(2 * np.pi * dow_zero_based / 7.0)
    features["cal_month_sin"] = np.sin(2 * np.pi * month_aligned / 12.0)
    features["cal_month_cos"] = np.cos(2 * np.pi * month_aligned / 12.0)
    return features


def _prepare_calendar_frame(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure calendar dataframe has day_of_week and month columns."""
    df = calendar_df.copy()
    date_col = _find_column(df, ["date", "datetime", "timestamp"])
    dow_col = _find_column(df, ["day_of_week", "dayofweek", "weekday", "week_day", "dow"])
    month_col = _find_column(df, ["month", "month_num", "month_number"])

    if dow_col is None or month_col is None:
        if date_col is None:
            raise ValueError(
                "Calendar data must include either a date column or explicit day-of-week/month columns."
            )
        df[date_col] = pd.to_datetime(df[date_col])
        if dow_col is None:
            df["day_of_week"] = df[date_col].dt.dayofweek
            dow_col = "day_of_week"
        if month_col is None:
            df["month"] = df[date_col].dt.month
            month_col = "month"

    dow_series = pd.to_numeric(df[dow_col], errors="coerce")
    month_series = pd.to_numeric(df[month_col], errors="coerce")
    cyclic = _build_cyclic_features(dow_series, month_series)
    df = pd.concat([df, cyclic], axis=1)
    return df


def merge_calendar_features(
    df: pd.DataFrame,
    calendar_path: str | Path,
    date_key: str = "date_id",
) -> pd.DataFrame:
    """Merge cyclic calendar features into the provided dataframe."""
    calendar_file = Path(calendar_path)
    if not calendar_file.exists():
        raise FileNotFoundError(f"Calendar file not found at {calendar_file}")

    calendar_df = pd.read_csv(calendar_file)
    calendar_df = _prepare_calendar_frame(calendar_df)

    # Build/normalize merge key:
    # - If a date_id-like column is absent, generate it from the row position to align with competition rows.
    # - If the found key is non-numeric (e.g., actual dates), use the row index instead of coercing datetime to ints.
    calendar_key = _find_column(calendar_df, [date_key, "Date"])
    if calendar_key is None:
        calendar_df[date_key] = range(len(calendar_df))
    else:
        if calendar_key != date_key:
            calendar_df = calendar_df.rename(columns={calendar_key: date_key})
        if pd.api.types.is_numeric_dtype(calendar_df[date_key]):
            calendar_df[date_key] = calendar_df[date_key].astype(int)
        else:
            calendar_df[date_key] = range(len(calendar_df))

    calendar_df = calendar_df.drop_duplicates(subset=[date_key])
    merge_cols = [date_key, "cal_dow_sin", "cal_dow_cos", "cal_month_sin", "cal_month_cos"]
    missing_cols = [c for c in merge_cols if c not in calendar_df.columns]
    if missing_cols:
        raise KeyError(f"Missing calendar feature columns: {', '.join(missing_cols)}")

    merged = df.merge(calendar_df[merge_cols], on=date_key, how="left")
    return merged


__all__ = ["merge_calendar_features"]
