"""Feature engineering for the Hull Tactical Market Prediction baseline."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Sequence, Set

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureConfig:
    drop_columns: Sequence[str]
    imputation_strategy: str = "median"
    scale: bool = True
    rolling_windows: Sequence[int] | None = None
    rolling_feature_naming: str = "roll"  # "roll" -> *_roll{w}, "rolling_mean" -> *_rolling_mean_{w}
    enable_interactions: bool = True
    time_column: str | None = None
    group_column: str | None = None
    missing_ratio_threshold: float = 40.0  # percent; keep cols with missing <= threshold
    use_limited_features: bool = False
    limited_features: Sequence[str] | None = None
    manual_features: Sequence[str] | None = None  # raw + derived (e.g., P10_inv, P7_x_V13_rolling_mean_3)


class SimpleFeatureExtractor:
    """Extracts features for HTMP with ordered missing handling and domain crosses."""

    _ROLLING_MEAN_RE = re.compile(r"^(?P<base>.+)_rolling_mean_(?P<window>[0-9]+)$")
    _ROLL_RE = re.compile(r"^(?P<base>.+)_roll(?P<window>[0-9]+)$")

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.scaler: StandardScaler | None = None
        self.base_columns: List[str] = []
        self.filtered_columns: List[str] = []
        self.impute_values: Dict[str, float] = {}

    def _prepare_columns(self, df: pd.DataFrame, target_column: str | None) -> List[str]:
        drop_cols = set(self.config.drop_columns)
        if target_column is not None:
            drop_cols.add(target_column)
        numeric_columns = [
            col
            for col in df.columns
            if col not in drop_cols and pd.api.types.is_numeric_dtype(df[col])
        ]
        if self.config.use_limited_features:
            if not self.config.limited_features:
                raise ValueError("use_limited_features is True but limited_features is empty.")
            allowed = set(self.config.limited_features)
            numeric_columns = [col for col in numeric_columns if col in allowed]
        if not numeric_columns:
            raise ValueError("No numeric columns available for feature extraction.")
        return numeric_columns

    def _filter_by_missing_ratio(self, df: pd.DataFrame, columns: List[str]) -> List[str]:
        """Keep columns whose missing ratio is <= threshold%."""
        threshold = self.config.missing_ratio_threshold / 100.0
        ratios = df[columns].isna().mean()
        kept = [col for col in columns if ratios[col] <= threshold]
        if not kept:
            raise ValueError("No columns left after missing ratio filtering.")
        return kept

    def fit_transform(self, df: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
        return self._create_features(df.copy(), target_column=target_column, fit=True)

    def transform(self, df: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
        return self._create_features(df.copy(), target_column=target_column, fit=False)

    def _create_features(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
        fit: bool = False,
    ) -> pd.DataFrame:
        if self.config.manual_features:
            return self._create_manual_features(df, target_column=target_column, fit=fit)

        if fit:
            self.base_columns = self._prepare_columns(df, target_column=target_column)
            self.filtered_columns = self._filter_by_missing_ratio(df, self.base_columns)
        if not self.filtered_columns:
            raise RuntimeError("FeatureExtractor is not fitted or no columns available.")

        df_numeric = df[self.filtered_columns].copy()

        df_numeric = self._apply_imputation(df_numeric, fit=fit)
        df_numeric = self._add_rolling_statistics(df_numeric, df)
        df_numeric = self._add_domain_cross_features(df_numeric)

        if self.config.scale:
            if fit:
                self.scaler = StandardScaler()
                scaled = self.scaler.fit_transform(df_numeric)
            else:
                if self.scaler is None:
                    raise RuntimeError("Scaler has not been fitted. Call fit_transform first.")
                scaled = self.scaler.transform(df_numeric)
            df_numeric = pd.DataFrame(scaled, columns=df_numeric.columns, index=df_numeric.index)

        return df_numeric

    def _apply_imputation(self, df_numeric: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if fit:
            if self.config.imputation_strategy == "mean":
                self.impute_values = df_numeric.mean(numeric_only=True).to_dict()
            elif self.config.imputation_strategy == "median":
                self.impute_values = df_numeric.median(numeric_only=True).to_dict()
            else:
                self.impute_values = {col: 0.0 for col in df_numeric.columns}
            # If a column is entirely missing, mean/median can be NaN; replace with 0.0.
            self.impute_values = {
                col: (float(val) if np.isfinite(val) else 0.0)
                for col, val in self.impute_values.items()
            }
        if not self.impute_values:
            raise RuntimeError("Imputation values not prepared. Call fit_transform first.")
        missing_imputes = [col for col in df_numeric.columns if col not in self.impute_values]
        if missing_imputes:
            raise RuntimeError(
                "Imputation values missing for columns: " + ", ".join(sorted(missing_imputes))
            )
        filled = df_numeric.fillna(self.impute_values)
        return filled

    def _add_rolling_statistics(self, df_numeric: pd.DataFrame, df_original: pd.DataFrame) -> pd.DataFrame:
        windows = self.config.rolling_windows or []
        time_col = self.config.time_column
        group_col = self.config.group_column
        if not windows or time_col is None or time_col not in df_original.columns:
            return df_numeric

        df_with_time = df_original[[time_col]].copy()
        if group_col and group_col in df_original.columns:
            df_with_time[group_col] = df_original[group_col]

        augmented = df_numeric.copy()
        for window in windows:
            suffix = (
                f"_rolling_mean_{window}"
                if self.config.rolling_feature_naming == "rolling_mean"
                else f"_roll{window}"
            )
            if group_col and group_col in df_with_time.columns:
                rolled_parts = []
                for col in df_numeric.columns:
                    rolled = (
                        df_numeric[col]
                        .groupby(df_with_time[group_col])
                        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
                    )
                    rolled_parts.append(rolled.rename(f"{col}{suffix}"))
                augmented = pd.concat([augmented] + rolled_parts, axis=1)
            else:
                sorter = df_with_time[time_col].argsort()
                df_sorted = df_numeric.iloc[sorter]
                for col in df_numeric.columns:
                    rolled_values = df_sorted[col].rolling(window=window, min_periods=1).mean()
                    augmented.loc[:, f"{col}{suffix}"] = rolled_values.sort_index()
        return augmented

    def _safe_divide(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Avoid blow-ups when volatility/rates are near zero."""
        eps = 1e-6
        return numerator / (denominator.replace(0, np.nan).fillna(eps))

    def _safe_inverse(self, series: pd.Series) -> pd.Series:
        eps = 1e-6
        denom = series.replace(0, np.nan).fillna(eps)
        return 1.0 / denom

    def _rolling_mean(
        self,
        series: pd.Series,
        df_original: pd.DataFrame,
        window: int,
    ) -> pd.Series:
        time_col = self.config.time_column
        if time_col is None or time_col not in df_original.columns:
            raise KeyError(
                "time_column is required for rolling_mean features but was not provided in config."
            )
        time_values = df_original[time_col]
        group_col = self.config.group_column
        if group_col and group_col in df_original.columns:
            group_values = df_original[group_col]
            out = pd.Series(index=series.index, dtype=float)
            groups = group_values.groupby(group_values, sort=False).groups
            for _, idx in groups.items():
                idx = pd.Index(idx)
                sorted_idx = time_values.loc[idx].sort_values().index
                rolled = series.loc[sorted_idx].rolling(window=window, min_periods=1).mean()
                out.loc[sorted_idx] = rolled.values
            return out

        sorter = time_values.argsort()
        sorted_idx = series.index[sorter]
        rolled = series.loc[sorted_idx].rolling(window=window, min_periods=1).mean()
        return rolled.reindex(series.index)

    def _required_raw_columns(self, feature_name: str) -> Set[str]:
        name = feature_name
        if name.endswith("_raw"):
            return self._required_raw_columns(name[: -len("_raw")])
        if "_x_" in name:
            left, right = name.split("_x_", 1)
            return self._required_raw_columns(left) | self._required_raw_columns(right)
        if "_div_" in name:
            left, right = name.split("_div_", 1)
            return self._required_raw_columns(left) | self._required_raw_columns(right)
        if name.endswith("_inv"):
            return self._required_raw_columns(name[: -len("_inv")])

        match = self._ROLLING_MEAN_RE.match(name)
        if match is not None:
            return self._required_raw_columns(match.group("base"))
        match = self._ROLL_RE.match(name)
        if match is not None:
            return self._required_raw_columns(match.group("base"))
        return {name}

    def _create_manual_features(
        self,
        df_original: pd.DataFrame,
        target_column: str | None,
        fit: bool,
    ) -> pd.DataFrame:
        requested = [str(f) for f in (self.config.manual_features or []) if str(f).strip()]
        if not requested:
            raise ValueError("manual_features is enabled but empty.")

        drop_cols = set(self.config.drop_columns)
        if target_column is not None:
            drop_cols.add(target_column)

        raw_cols: Set[str] = set()
        for feat in requested:
            raw_cols |= self._required_raw_columns(feat)
        raw_cols = {c for c in raw_cols if c not in drop_cols}

        missing = [c for c in sorted(raw_cols) if c not in df_original.columns]
        if missing:
            raise KeyError("Missing raw columns required by manual_features: " + ", ".join(missing))

        non_numeric = [
            c for c in sorted(raw_cols) if not pd.api.types.is_numeric_dtype(df_original[c])
        ]
        if non_numeric:
            raise TypeError(
                "manual_features requires numeric raw columns only; non-numeric: "
                + ", ".join(non_numeric)
            )

        base_df = df_original[list(sorted(raw_cols))].copy()
        base_df = self._apply_imputation(base_df, fit=fit)

        cache: Dict[str, pd.Series] = {}

        def eval_feature(name: str) -> pd.Series:
            if name in cache:
                return cache[name]

            if name in base_df.columns:
                series = base_df[name]
            elif name.endswith("_raw"):
                series = eval_feature(name[: -len("_raw")])
            elif "_x_" in name:
                left, right = name.split("_x_", 1)
                series = eval_feature(left) * eval_feature(right)
            elif "_div_" in name:
                left, right = name.split("_div_", 1)
                series = self._safe_divide(eval_feature(left), eval_feature(right))
            elif name.endswith("_inv"):
                series = self._safe_inverse(eval_feature(name[: -len("_inv")]))
            else:
                match = self._ROLLING_MEAN_RE.match(name)
                if match is not None:
                    base = match.group("base")
                    window = int(match.group("window"))
                    series = self._rolling_mean(eval_feature(base), df_original=df_original, window=window)
                else:
                    match = self._ROLL_RE.match(name)
                    if match is not None:
                        base = match.group("base")
                        window = int(match.group("window"))
                        series = self._rolling_mean(eval_feature(base), df_original=df_original, window=window)
                    else:
                        raise KeyError(f"Unknown feature expression: {name}")

            cache[name] = series
            return series

        features: Dict[str, pd.Series] = {}
        for feat in requested:
            features[feat] = eval_feature(feat)
        out = pd.DataFrame(features, index=df_original.index)

        if self.config.scale:
            if fit:
                self.scaler = StandardScaler()
                scaled = self.scaler.fit_transform(out)
            else:
                if self.scaler is None:
                    raise RuntimeError("Scaler has not been fitted. Call fit_transform first.")
                scaled = self.scaler.transform(out)
            out = pd.DataFrame(scaled, columns=out.columns, index=out.index)

        return out

    def _add_domain_cross_features(self, df_numeric: pd.DataFrame) -> pd.DataFrame:
        """Generate domain-inspired crosses. Applied after missing-filter and imputation."""
        if not self.config.enable_interactions:
            return df_numeric

        def cols_with_prefix(prefix: str) -> List[str]:
            return [c for c in df_numeric.columns if c.startswith(prefix)]

        crosses: Dict[str, pd.Series] = {}

        # (1) P × I: Valuation × rates captures discount-rate sensitivity in pricing.
        for p_col in cols_with_prefix("P"):
            for i_col in cols_with_prefix("I"):
                crosses[f"{p_col}_x_{i_col}"] = (
                    df_numeric[p_col] * df_numeric[i_col]
                )

        # (2) MOM / V: Momentum scaled by volatility highlights risk-adjusted trend strength.
        for mom_col in cols_with_prefix("MOM"):
            for v_col in cols_with_prefix("V"):
                crosses[f"{mom_col}_div_{v_col}"] = self._safe_divide(
                    df_numeric[mom_col], df_numeric[v_col]
                )

        # (3) M × E: Technical market state interacting with macro regime signals.
        for m_col in cols_with_prefix("M"):
            for e_col in cols_with_prefix("E"):
                crosses[f"{m_col}_x_{e_col}"] = df_numeric[m_col] * df_numeric[e_col]

        # (4) S × MOM: Sentiment-weighted momentum to capture behavioral accelerations/reversals.
        for s_col in cols_with_prefix("S"):
            for mom_col in cols_with_prefix("MOM"):
                crosses[f"{s_col}_x_{mom_col}"] = df_numeric[s_col] * df_numeric[mom_col]

        # (5) X × lagged_Y: Current signals with lagged fundamentals/targets for mean-reversion carry-over.
        lagged_cols = [c for c in df_numeric.columns if c.startswith("lagged_")]
        base_cols = [c for c in df_numeric.columns if not c.startswith("lagged_") and not c.startswith("D")]
        for base in base_cols:
            for lag_col in lagged_cols:
                crosses[f"{base}_x_{lag_col}"] = df_numeric[base] * df_numeric[lag_col]

        # (6) X × D: Condition features on binary regime flags (policy/earnings events).
        dummy_cols = cols_with_prefix("D")
        for base in base_cols:
            for d_col in dummy_cols:
                crosses[f"{base}_x_{d_col}"] = df_numeric[base] * df_numeric[d_col]

        # (7) (MOM × V) / I: Risk-adjusted momentum moderated by rates (carry/discount pressure).
        for mom_col in cols_with_prefix("MOM"):
            for v_col in cols_with_prefix("V"):
                mom_v = df_numeric[mom_col] * df_numeric[v_col]
                for i_col in cols_with_prefix("I"):
                    crosses[f"{mom_col}_x_{v_col}_div_{i_col}"] = self._safe_divide(mom_v, df_numeric[i_col])

        if crosses:
            df_numeric = pd.concat(
                [df_numeric, pd.DataFrame(crosses, index=df_numeric.index)], axis=1
            )
        return df_numeric


__all__ = ["FeatureConfig", "SimpleFeatureExtractor"]
