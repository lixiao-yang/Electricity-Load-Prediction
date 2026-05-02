from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import MonthEnd

from tft.src.pipeline_utils import ensure_dir, load_yaml_config, require_columns, write_json, write_yaml_config


OBSERVED_FEATURE_COLUMNS = ["lag_24", "lag_48", "lag_168", "roll_mean_24", "roll_std_24", "roll_mean_168"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe TFT datasets directly from notebook-produced train/test parquet files."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to tft/configs/data.yaml.")
    parser.add_argument("--train-wide-path", type=str, default=None, help="Override train wide parquet path.")
    parser.add_argument("--test-wide-path", type=str, default=None, help="Override test wide parquet path.")
    parser.add_argument("--clusters-path", type=str, default=None, help="Override clusters parquet path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override tft/artifacts/data directory.")
    return parser.parse_args()


def load_wide_panel(path: Path, panel_name: str) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_index()
    df.index = pd.to_datetime(df.index)
    df.index.name = "timestamp"
    df.columns = df.columns.astype(str)
    if df.empty:
        raise ValueError(f"{panel_name} parquet is empty: {path}")
    return df


def normalize_test_end(test_end: pd.Timestamp) -> pd.Timestamp:
    test_end = pd.to_datetime(test_end)
    if test_end == test_end.to_period("M").to_timestamp():
        return test_end - pd.Timedelta(hours=1)
    return test_end


def month_last_hour(timestamp: pd.Timestamp) -> pd.Timestamp:
    month_start = pd.to_datetime(timestamp).to_period("M").to_timestamp()
    return month_start + MonthEnd(1) + pd.Timedelta(hours=23)


def validate_test_window(test_wide: pd.DataFrame, expected_test_months: int) -> tuple[pd.Timestamp, pd.Timestamp, list[str], list[str]]:
    if test_wide.empty:
        return pd.NaT, pd.NaT, ["Test parquet is empty."], []

    test_start = pd.to_datetime(test_wide.index.min())
    test_end = normalize_test_end(test_wide.index.max())
    trimmed = test_wide.loc[test_start:test_end]
    test_month_periods = pd.PeriodIndex(trimmed.index.to_period("M").unique()).sort_values()
    test_month_labels = [str(period) for period in test_month_periods]

    violations: list[str] = []
    if len(test_month_periods) != expected_test_months:
        violations.append(
            f"Configured test_months={expected_test_months}, but test parquet covers {len(test_month_periods)} months: {test_month_labels}."
        )

    expected_start = test_end.to_period("M").to_timestamp() - pd.DateOffset(months=expected_test_months - 1)
    if test_start != expected_start:
        violations.append(
            f"Test parquet must start at {expected_start} for the final {expected_test_months} calendar months, got {test_start}."
        )

    if test_start != test_start.to_period("M").to_timestamp():
        violations.append(f"Test start must be at month start 00:00, got {test_start}.")

    expected_end = month_last_hour(test_end)
    if test_end != expected_end:
        violations.append(f"Test end must be the final hour of its month, expected {expected_end}, got {test_end}.")

    return test_start, test_end, violations, test_month_labels


def load_cluster_map(path: Path, cluster_label_col: str, cluster_ids: Iterable[int]) -> pd.DataFrame:
    cluster_df = pd.read_parquet(path).reset_index()
    if "user_id" not in cluster_df.columns:
        if "meter_id" in cluster_df.columns:
            cluster_df = cluster_df.rename(columns={"meter_id": "user_id"})
        elif "index" in cluster_df.columns:
            cluster_df = cluster_df.rename(columns={"index": "user_id"})
        else:
            first_col = cluster_df.columns[0]
            cluster_df = cluster_df.rename(columns={first_col: "user_id"})
    require_columns(cluster_df, ["user_id", cluster_label_col], "cluster_labels")
    cluster_df["user_id"] = cluster_df["user_id"].astype(str)
    cluster_df["cluster_id"] = pd.to_numeric(cluster_df[cluster_label_col], errors="raise").astype(int)
    cluster_df = cluster_df.loc[cluster_df["cluster_id"].isin(cluster_ids), ["user_id", "cluster_id"]].copy()
    if cluster_df.empty:
        raise ValueError(f"No users found for cluster ids {list(cluster_ids)} in {path}.")
    return cluster_df.sort_values(["cluster_id", "user_id"]).reset_index(drop=True)


def trim_train_wide_to_18m_plus_3m_val(train_wide: pd.DataFrame, test_start: pd.Timestamp, train_months: int, val_months: int) -> pd.DataFrame:
    val_start = test_start - pd.DateOffset(months=val_months)
    train_start = val_start - pd.DateOffset(months=train_months)
    trimmed = train_wide.loc[train_start : test_start - pd.Timedelta(hours=1)].copy()
    if trimmed.empty:
        raise ValueError(
            "Trimmed train parquet is empty. Upstream train split does not cover the required 18m train + 3m val history."
        )
    return trimmed


def wide_to_long(wide_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    index_name = wide_df.index.name or "timestamp"
    long_df = (
        wide_df.rename_axis(index_name)
        .reset_index()
        .melt(id_vars=index_name, var_name="user_id", value_name=target_col)
        .rename(columns={index_name: "timestamp"})
    )
    long_df["timestamp"] = pd.to_datetime(long_df["timestamp"])
    long_df["user_id"] = long_df["user_id"].astype(str)
    return long_df


def add_known_features(df: pd.DataFrame, calendar_name: str) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour.astype(int)
    df["dayofweek"] = df["timestamp"].dt.dayofweek.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["month"] = df["timestamp"].dt.month.astype(int)
    if calendar_name == "us_federal":
        calendar = USFederalHolidayCalendar()
        holidays = calendar.holidays(start=df["timestamp"].min(), end=df["timestamp"].max()).normalize()
        df["holiday_flag"] = df["timestamp"].dt.normalize().isin(holidays).astype(int)
    else:
        df["holiday_flag"] = 0
    return df


def add_observed_features(df: pd.DataFrame, target_col: str, lag_hours: Iterable[int], roll_windows: Iterable[int]) -> pd.DataFrame:
    df = df.sort_values(["user_id", "timestamp"]).copy()
    grouped = df.groupby("user_id", sort=False)[target_col]
    for lag in lag_hours:
        df[f"lag_{lag}"] = grouped.shift(lag)
    for window in roll_windows:
        df[f"roll_mean_{window}"] = grouped.transform(lambda series, w=window: series.shift(1).rolling(w).mean())
        if window == 24:
            df["roll_std_24"] = grouped.transform(lambda series: series.shift(1).rolling(24).std())
    return df


def add_time_index(df: pd.DataFrame, base_timestamp: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    df["time_idx"] = ((df["timestamp"] - base_timestamp) / pd.Timedelta(hours=1)).astype(int)
    return df


def build_future_known(user_cluster_df: pd.DataFrame, future_start: pd.Timestamp, future_end: pd.Timestamp, target_col: str, calendar_name: str) -> pd.DataFrame:
    future_index = pd.date_range(future_start, future_end, freq="h")
    frames = []
    for _, row in user_cluster_df.iterrows():
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": future_index,
                    "user_id": row["user_id"],
                    "cluster_id": int(row["cluster_id"]),
                    target_col: np.nan,
                }
            )
        )
    future_df = pd.concat(frames, ignore_index=True)
    future_df = add_known_features(future_df, calendar_name=calendar_name)
    future_df["split"] = "future"
    for column in ["lag_24", "lag_48", "lag_168", "roll_mean_24", "roll_std_24", "roll_mean_168"]:
        future_df[column] = np.nan
    return future_df


def build_shared_panel(observed_df: pd.DataFrame, future_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    panel = pd.concat([observed_df, future_df], ignore_index=True, sort=False)
    panel["cluster_id_cat"] = panel["cluster_id"].astype(str)
    panel["hour_cat"] = panel["hour"].astype(str)
    panel["day_of_week_cat"] = panel["dayofweek"].astype(str)
    panel["month_cat"] = panel["month"].astype(str)
    panel["is_weekend_cat"] = panel["is_weekend"].astype(str)
    panel["holiday_flag_cat"] = panel["holiday_flag"].astype(str)
    panel["day_of_month"] = panel["timestamp"].dt.day.astype(int)
    return panel.sort_values(["cluster_id", "user_id", "timestamp"]).reset_index(drop=True)


def validate_time_order(boundaries: Dict[str, pd.Timestamp]) -> tuple[bool, list[str]]:
    violations: list[str] = []
    if not (
        boundaries["train_start"] <= boundaries["train_end"] < boundaries["val_start"] <= boundaries["val_end"]
        < boundaries["test_start"] <= boundaries["test_end"] < boundaries["future_start"] <= boundaries["future_end"]
    ):
        violations.append("Split boundaries are not strictly ordered train -> val -> test -> future.")
    return len(violations) == 0, violations


def validate_feature_shift(
    observed_df: pd.DataFrame,
    target_col: str,
    lag_hours: Iterable[int],
    roll_windows: Iterable[int],
) -> tuple[bool, list[str]]:
    violations: list[str] = []
    reference = add_observed_features(
        observed_df.loc[:, ["timestamp", "user_id", target_col]].copy(),
        target_col=target_col,
        lag_hours=lag_hours,
        roll_windows=roll_windows,
    )
    feature_columns: list[str] = [f"lag_{lag}" for lag in lag_hours]
    feature_columns.extend(f"roll_mean_{window}" for window in roll_windows)
    if 24 in set(int(window) for window in roll_windows):
        feature_columns.append("roll_std_24")

    for feature_name in feature_columns:
        actual = pd.to_numeric(observed_df[feature_name], errors="coerce")
        expected = pd.to_numeric(reference[feature_name], errors="coerce")
        matches = (actual.isna() & expected.isna()) | np.isclose(
            actual.fillna(0.0).to_numpy(),
            expected.fillna(0.0).to_numpy(),
            atol=1e-9,
            rtol=1e-9,
        )
        if not bool(np.all(matches)):
            mismatch_index = np.flatnonzero(~matches)[:3]
            examples = []
            for row_index in mismatch_index:
                examples.append(
                    {
                        "user_id": str(observed_df.iloc[row_index]["user_id"]),
                        "timestamp": str(observed_df.iloc[row_index]["timestamp"]),
                        "actual": None if pd.isna(actual.iloc[row_index]) else float(actual.iloc[row_index]),
                        "expected": None if pd.isna(expected.iloc[row_index]) else float(expected.iloc[row_index]),
                    }
                )
            violations.append(f"{feature_name} failed causal recomputation audit: {examples}.")
    return len(violations) == 0, violations


def validate_train_fit_only(observed_df: pd.DataFrame, future_known_df: pd.DataFrame, target_col: str) -> tuple[bool, list[str]]:
    violations: list[str] = []
    disallowed_fit_columns = [
        column
        for column in observed_df.columns
        if column.endswith("_scaled") or column.endswith("_normalized") or column.endswith("_encoded")
    ]
    if disallowed_fit_columns:
        violations.append(f"Prompt A outputs must not contain fitted transform columns, found {sorted(disallowed_fit_columns)}.")
    if future_known_df[target_col].notna().any():
        violations.append("Future known dataframe must keep target_load as NaN to avoid leaking future truth.")
    return len(violations) == 0, violations


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    train_wide_path = Path(args.train_wide_path or config["paths"]["train_wide_path"])
    test_wide_path = Path(args.test_wide_path or config["paths"]["test_wide_path"])
    clusters_path = Path(args.clusters_path or config["paths"]["clusters_path"])
    output_dir = ensure_dir(args.output_dir or config["paths"]["data_output_dir"])

    train_wide = load_wide_panel(train_wide_path, "train")
    test_wide = load_wide_panel(test_wide_path, "test")
    cluster_map = load_cluster_map(
        clusters_path,
        cluster_label_col=config["data"]["cluster_label_col"],
        cluster_ids=config["data"]["cluster_ids"],
    )

    available_users = set(train_wide.columns) & set(test_wide.columns)
    selected_users = [user_id for user_id in cluster_map["user_id"].tolist() if user_id in available_users]
    if not selected_users:
        raise ValueError("No overlapping cluster users were found in both train and test parquet files.")

    missing_users = sorted(set(cluster_map["user_id"]) - set(selected_users))
    if missing_users:
        cluster_map = cluster_map.loc[cluster_map["user_id"].isin(selected_users)].copy()

    test_start, test_end, test_window_violations, test_month_labels = validate_test_window(
        test_wide=test_wide,
        expected_test_months=int(config["data"]["test_months"]),
    )
    if test_window_violations:
        raise ValueError(" ".join(test_window_violations))
    trimmed_train_wide = trim_train_wide_to_18m_plus_3m_val(
        train_wide=train_wide.loc[:, selected_users].copy(),
        test_start=test_start,
        train_months=int(config["data"]["train_months"]),
        val_months=int(config["data"]["val_months"]),
    )
    test_wide = test_wide.loc[:test_end, selected_users].copy()

    train_long = wide_to_long(trimmed_train_wide, target_col=config["data"]["target_col"])
    test_long = wide_to_long(test_wide, target_col=config["data"]["target_col"])
    observed_df = pd.concat([train_long, test_long], ignore_index=True, sort=False)
    observed_df = observed_df.merge(cluster_map, on="user_id", how="left", validate="many_to_one")
    require_columns(observed_df, ["timestamp", "user_id", "cluster_id", config["data"]["target_col"]], "observed_df")

    train_start = pd.to_datetime(trimmed_train_wide.index.min())
    train_end = test_start - pd.DateOffset(months=int(config["data"]["val_months"])) - pd.Timedelta(hours=1)
    val_start = train_end + pd.Timedelta(hours=1)
    val_end = test_start - pd.Timedelta(hours=1)
    future_start = test_end + pd.Timedelta(hours=1)
    future_end = future_start + pd.Timedelta(days=int(config["data"]["future_horizon_days"])) - pd.Timedelta(hours=1)

    boundaries = {
        "train_start": train_start,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": test_end,
        "future_start": future_start,
        "future_end": future_end,
    }

    observed_df = add_known_features(observed_df, calendar_name=config["data"]["holiday_calendar"])
    observed_df = add_observed_features(
        observed_df,
        target_col=config["data"]["target_col"],
        lag_hours=config["data"]["target_lag_hours"],
        roll_windows=config["data"]["target_roll_windows"],
    )
    observed_df["split"] = np.select(
        [
            (observed_df["timestamp"] >= train_start) & (observed_df["timestamp"] <= train_end),
            (observed_df["timestamp"] >= val_start) & (observed_df["timestamp"] <= val_end),
            (observed_df["timestamp"] >= test_start) & (observed_df["timestamp"] <= test_end),
        ],
        ["train", "val", "test"],
        default="out_of_scope",
    )
    observed_df = add_time_index(observed_df, base_timestamp=train_start)

    future_known_df = build_future_known(
        user_cluster_df=cluster_map,
        future_start=future_start,
        future_end=future_end,
        target_col=config["data"]["target_col"],
        calendar_name=config["data"]["holiday_calendar"],
    )
    future_known_df = add_time_index(future_known_df, base_timestamp=train_start)

    train_df = observed_df.loc[observed_df["split"] == "train"].copy()
    val_df = observed_df.loc[observed_df["split"] == "val"].copy()
    test_df = observed_df.loc[observed_df["split"] == "test"].copy()

    train_df.to_parquet(config["paths"]["features_train_path"], index=False)
    val_df.to_parquet(config["paths"]["features_val_path"], index=False)
    test_df.to_parquet(config["paths"]["features_test_path"], index=False)
    future_known_df.to_parquet(config["paths"]["features_future_known_path"], index=False)

    shared_panel = build_shared_panel(observed_df, future_known_df, target_col=config["data"]["target_col"])
    shared_panel.to_parquet(config["paths"]["shared_panel_path"], index=False)
    shared_panel.loc[shared_panel["cluster_id"] == 10].copy().to_parquet(config["paths"]["cluster_10_panel_path"], index=False)
    shared_panel.loc[shared_panel["cluster_id"] == 12].copy().to_parquet(config["paths"]["cluster_12_panel_path"], index=False)

    split_boundaries = {
        "train_wide_path": str(train_wide_path),
        "test_wide_path": str(test_wide_path),
        "clusters_path": str(clusters_path),
        "cluster_ids": config["data"]["cluster_ids"],
        "test_months_detected": test_month_labels,
        "train_start": str(train_start),
        "train_end": str(train_end),
        "val_start": str(val_start),
        "val_end": str(val_end),
        "test_start": str(test_start),
        "test_end": str(test_end),
        "future_start": str(future_start),
        "future_end": str(future_end),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "future_rows": int(len(future_known_df)),
        "user_count": int(observed_df["user_id"].nunique()),
        "cluster_user_counts": observed_df.groupby("cluster_id")["user_id"].nunique().astype(int).to_dict(),
        "dropped_cluster_users_missing_in_panels": missing_users,
    }
    write_json(split_boundaries, config["paths"]["split_boundaries_path"])

    time_order_valid, time_order_violations = validate_time_order(boundaries)
    feature_shift_valid, feature_shift_violations = validate_feature_shift(
        observed_df=observed_df,
        target_col=config["data"]["target_col"],
        lag_hours=config["data"]["target_lag_hours"],
        roll_windows=config["data"]["target_roll_windows"],
    )
    train_fit_only_valid, train_fit_only_violations = validate_train_fit_only(
        observed_df=observed_df,
        future_known_df=future_known_df,
        target_col=config["data"]["target_col"],
    )
    violations = test_window_violations + time_order_violations + feature_shift_violations + train_fit_only_violations
    leakage_audit = {
        "test_month_window_valid": len(test_window_violations) == 0,
        "time_order_valid": time_order_valid,
        "feature_shift_valid": feature_shift_valid,
        "train_fit_only_valid": train_fit_only_valid,
        "passed": (len(test_window_violations) == 0) and time_order_valid and feature_shift_valid and train_fit_only_valid,
        "violations": violations,
    }
    write_json(leakage_audit, config["paths"]["leakage_audit_path"])
    write_yaml_config(config, output_dir / "resolved_data_config.yaml")


if __name__ == "__main__":
    main()

# 如何运行本脚本：
# python -m tft.src.data.build_tft_dataset --config tft/configs/data.yaml
