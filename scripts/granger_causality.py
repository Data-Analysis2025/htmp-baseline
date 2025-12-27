"""Run Granger causality tests (x -> target) for engineered features."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

# NOTE: In some sandboxed environments OpenMP-backed linear algebra aborts when it cannot
# allocate shared memory. Force a sequential threading layer before importing numpy/scipy.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQ")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calendar_features import merge_calendar_features
from src.config import Config
from src.features import FeatureConfig, SimpleFeatureExtractor
from src.granger import granger_causality_f_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--maxlag", type=int, default=10, help="Maximum lag to test (>=1).")
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional number of rows to read from train.csv (for quick smoke tests).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Print top-K features by minimum p-value.",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Disable feature scaling for this run (regardless of config).",
    )
    return parser.parse_args()


def load_train(
    paths: Dict[str, str],
    files: Dict[str, str],
    target_column: str,
    date_key: str,
    calendar_path: Path | None,
    nrows: int | None = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    train_df = pd.read_csv(Path(paths["input_dir"]) / files["train"], nrows=nrows)
    if calendar_path is not None:
        train_df = merge_calendar_features(train_df, calendar_path, date_key=date_key)
    if target_column not in train_df.columns:
        raise KeyError(f"Target column '{target_column}' not found in training data.")
    train_df = train_df.sort_values(date_key).reset_index(drop=True)
    y = train_df[target_column].values
    return train_df, y


def build_feature_config(
    feature_cfg: Dict[str, object],
    cv_cfg: Dict[str, object],
    scale_override: bool | None = None,
) -> FeatureConfig:
    scale = bool(feature_cfg.get("scale", True)) if scale_override is None else bool(scale_override)
    return FeatureConfig(
        drop_columns=feature_cfg.get("drop_columns", []),
        imputation_strategy=feature_cfg.get("imputation_strategy", "median"),
        scale=scale,
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


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    config.ensure_dirs()

    paths = config.get("paths")
    files = config.get("files")
    target_cfg = config.get("target")
    cv_cfg = config.get("cv")
    feature_cfg = config.get("features")

    date_key = cv_cfg.get("time_column") or "date_id"
    calendar_path = None
    calendar_file = files.get("calendar")
    if calendar_file:
        calendar_path = Path(calendar_file)
        if not calendar_path.is_absolute():
            calendar_path = Path(paths["input_dir"]) / calendar_file

    train_df, y = load_train(
        paths=paths,
        files=files,
        target_column=target_cfg["column"],
        date_key=date_key,
        calendar_path=calendar_path,
        nrows=args.nrows,
    )

    feature_config = build_feature_config(feature_cfg, cv_cfg, scale_override=False if args.no_scale else None)
    extractor = SimpleFeatureExtractor(feature_config)
    X = extractor.fit_transform(train_df, target_column=target_cfg["column"])

    by_lag_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for col in X.columns:
        test_df = granger_causality_f_test(y=y, x=X[col].values, maxlag=args.maxlag)
        if test_df.empty:
            continue
        test_df.insert(0, "feature", col)
        by_lag_rows.append(test_df)
        best_idx = int(test_df["p_value"].idxmin())
        best_row = test_df.loc[best_idx]
        summary_rows.append(
            {
                "feature": col,
                "best_lag": int(best_row["lag"]),
                "best_f_stat": float(best_row["f_stat"]),
                "best_p_value": float(best_row["p_value"]),
            }
        )

    by_lag = pd.concat(by_lag_rows, axis=0, ignore_index=True) if by_lag_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows).sort_values("best_p_value", ascending=True)

    reports_dir = Path(paths.get("reports_dir", PROJECT_ROOT / "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    by_lag_path = reports_dir / "granger_by_lag.csv"
    summary_path = reports_dir / "granger_summary.csv"

    by_lag.to_csv(by_lag_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved Granger results:\n- {by_lag_path}\n- {summary_path}")
    if not summary.empty and args.top_k > 0:
        print("\nTop features (min p-value):")
        top = summary.head(int(args.top_k)).copy()
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(top.to_string(index=False))


if __name__ == "__main__":
    main()
