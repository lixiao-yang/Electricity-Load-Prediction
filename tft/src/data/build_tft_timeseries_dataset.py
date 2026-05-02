from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Missing torch. Install PyTorch before building TFT datasets.") from exc

try:
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing pytorch-forecasting. Install pytorch-forecasting and lightning before building TFT datasets."
    ) from exc

from tft.src.pipeline_utils import load_yaml_config, require_columns, write_json, write_yaml_config


KNOWN_FUTURE_COLUMNS = ["hour", "dayofweek", "is_weekend", "month", "holiday_flag"]
OBSERVED_COLUMNS = ["lag_24", "lag_48", "lag_168", "roll_mean_24", "roll_std_24", "roll_mean_168"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PyTorch Forecasting TimeSeriesDataSet objects from Prompt A feature parquet outputs."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to tft/configs/data.yaml.")
    parser.add_argument("--train-path", type=str, default=None, help="Override features_train parquet path.")
    parser.add_argument("--val-path", type=str, default=None, help="Override features_val parquet path.")
    parser.add_argument("--test-path", type=str, default=None, help="Override features_test parquet path.")
    parser.add_argument("--future-path", type=str, default=None, help="Override features_future_known parquet path.")
    parser.add_argument("--dataset-meta-path", type=str, default=None, help="Override TFT dataset metadata json path.")
    parser.add_argument(
        "--dataset-parameters-path",
        type=str,
        default=None,
        help="Override train-fitted TimeSeriesDataSet parameters .pt path.",
    )
    parser.add_argument(
        "--dataloader-config-path",
        type=str,
        default=None,
        help="Override reusable dataloader yaml path.",
    )
    return parser.parse_args()


def load_feature_frame(path: Path, split_name: str, required_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    require_columns(frame, required_columns, f"{split_name}_features")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["user_id"] = frame["user_id"].astype(str)
    frame["cluster_id"] = pd.to_numeric(frame["cluster_id"], errors="raise").astype(int)
    if "split" not in frame.columns:
        frame["split"] = split_name
    else:
        frame["split"] = frame["split"].astype(str)
    frame = frame.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return frame


def hourly_time_delta(frequency: str) -> pd.Timedelta:
    if str(frequency).lower() != "h":
        raise ValueError(
            f"build_tft_timeseries_dataset currently expects hourly Prompt A outputs. Received frequency={frequency!r}."
        )
    return pd.Timedelta(hours=1)


def assign_continuous_time_idx(split_frames: Dict[str, pd.DataFrame], frequency: str) -> Dict[str, pd.DataFrame]:
    delta = hourly_time_delta(frequency)
    base_timestamp = min(frame["timestamp"].min() for frame in split_frames.values())
    updated: Dict[str, pd.DataFrame] = {}
    for split_name, frame in split_frames.items():
        result = frame.copy()
        offsets = (result["timestamp"] - base_timestamp) / delta
        rounded = np.round(offsets.to_numpy(dtype=float))
        if not np.allclose(offsets.to_numpy(dtype=float), rounded):
            raise ValueError(f"{split_name} contains timestamps that do not align to the configured hourly grid.")
        result["time_idx"] = rounded.astype(int)
        updated[split_name] = result
    return updated


def validate_split_boundaries(split_frames: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    ordered_names = ["train", "val", "test", "future"]
    summary: Dict[str, Dict[str, Any]] = {}
    for split_name in ordered_names:
        frame = split_frames[split_name]
        summary[split_name] = {
            "timestamp_start": pd.Timestamp(frame["timestamp"].min()),
            "timestamp_end": pd.Timestamp(frame["timestamp"].max()),
            "time_idx_start": int(frame["time_idx"].min()),
            "time_idx_end": int(frame["time_idx"].max()),
            "rows": int(len(frame)),
            "users": int(frame["user_id"].nunique()),
        }

    for left_name, right_name in zip(ordered_names, ordered_names[1:]):
        left = summary[left_name]
        right = summary[right_name]
        if left["timestamp_end"] >= right["timestamp_start"]:
            raise ValueError(f"{left_name} and {right_name} overlap in timestamp space.")
        if left["time_idx_end"] >= right["time_idx_start"]:
            raise ValueError(f"{left_name} and {right_name} overlap in time_idx space.")

    return summary


def validate_tft_only_paths(config: Dict[str, Any]) -> None:
    path_config = config.get("paths", {})
    compatibility_keys = sorted(key for key in path_config if key.startswith("compatibility_"))
    if compatibility_keys:
        raise ValueError(
            f"Remove root compatibility paths from tft/configs/data.yaml. Found unsupported keys: {compatibility_keys}."
        )

    required_tft_paths = [
        "artifacts_root",
        "data_output_dir",
        "tft_dataset_meta_path",
        "tft_dataset_parameters_path",
        "shared_panel_path",
        "cluster_10_panel_path",
        "cluster_12_panel_path",
        "models_dir",
        "eval_dir",
        "infer_dir",
        "agent_bridge_dir",
        "dataloader_config_path",
    ]
    invalid_paths = []
    for key in required_tft_paths:
        value = str(path_config.get(key, "")).replace("\\", "/")
        if not value.startswith("tft/"):
            invalid_paths.append(f"{key}={value}")
    if invalid_paths:
        raise ValueError(f"All TFT model/data outputs must live under tft/. Invalid paths: {invalid_paths}.")


def validate_calendar_windows(split_frames: Dict[str, pd.DataFrame], expected_test_months: int, expected_future_days: int) -> None:
    test_months = sorted(split_frames["test"]["timestamp"].dt.to_period("M").astype(str).unique().tolist())
    if len(test_months) != expected_test_months:
        raise ValueError(
            f"Expected exactly {expected_test_months} calendar months in features_test.parquet, got {test_months}."
        )

    expected_future_hours = int(expected_future_days) * 24
    future_lengths = split_frames["future"].groupby("user_id", sort=False).size()
    invalid_users = future_lengths.loc[future_lengths != expected_future_hours]
    if not invalid_users.empty:
        details = invalid_users.astype(int).to_dict()
        raise ValueError(
            f"Each user must have exactly {expected_future_hours} future-known hours. Invalid counts: {details}."
        )


def add_dataset_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["cluster_id_cat"] = result["cluster_id"].astype(str)
    result["hour_cat"] = result["hour"].astype(int).astype(str)
    result["day_of_week_cat"] = result["dayofweek"].astype(int).astype(str)
    result["month_cat"] = result["month"].astype(int).astype(str)
    result["is_weekend_cat"] = result["is_weekend"].astype(int).astype(str)
    result["holiday_flag_cat"] = result["holiday_flag"].astype(int).astype(str)
    return result


def fit_future_target_placeholder(split_frames: Dict[str, pd.DataFrame], target_col: str) -> Dict[str, float]:
    observed = pd.concat(
        [split_frames["train"], split_frames["val"], split_frames["test"]],
        ignore_index=True,
        sort=False,
    )
    observed = observed.sort_values(["user_id", "timestamp"]).copy()
    observed[target_col] = pd.to_numeric(observed[target_col], errors="coerce")
    last_target = observed.groupby("user_id", sort=False)[target_col].last()
    if last_target.isna().any():
        global_fallback = float(observed[target_col].median()) if observed[target_col].notna().any() else 0.0
        last_target = last_target.fillna(global_fallback)
    return {str(user_id): float(value) for user_id, value in last_target.items()}


def apply_future_target_placeholder(
    split_frames: Dict[str, pd.DataFrame],
    target_col: str,
    placeholder_by_user: Dict[str, float],
) -> Dict[str, pd.DataFrame]:
    updated = {name: frame.copy() for name, frame in split_frames.items()}
    future_frame = updated["future"].copy()
    future_frame[target_col] = pd.to_numeric(future_frame[target_col], errors="coerce")
    future_frame[target_col] = future_frame[target_col].fillna(future_frame["user_id"].map(placeholder_by_user)).astype(float)
    updated["future"] = future_frame
    return updated


def dataset_feature_spec(target_col: str, use_target_derived_covariates: bool) -> Dict[str, list[str]]:
    unknown_reals = [target_col]
    if use_target_derived_covariates:
        unknown_reals.extend(OBSERVED_COLUMNS)
    return {
        "group_ids": ["user_id"],
        "static_categoricals": ["user_id", "cluster_id_cat"],
        "time_varying_known_categoricals": [
            "hour_cat",
            "day_of_week_cat",
            "month_cat",
            "is_weekend_cat",
            "holiday_flag_cat",
        ],
        "time_varying_known_reals": [],
        "time_varying_unknown_categoricals": [],
        "time_varying_unknown_reals": unknown_reals,
    }


def build_train_dataset(train_frame: pd.DataFrame, config: Dict[str, Any], spec: Dict[str, list[str]]) -> TimeSeriesDataSet:
    max_encoder_length = int(config["model"]["max_encoder_length"])
    max_prediction_length = int(config["model"]["max_prediction_length"])
    min_encoder_length = int(config["model"].get("min_encoder_length", max_encoder_length))
    target_col = config["data"]["target_col"]

    return TimeSeriesDataSet(
        train_frame.sort_values(["user_id", "time_idx"]).reset_index(drop=True),
        time_idx="time_idx",
        target=target_col,
        group_ids=spec["group_ids"],
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=spec["static_categoricals"],
        time_varying_known_categoricals=spec["time_varying_known_categoricals"],
        time_varying_known_reals=spec["time_varying_known_reals"],
        time_varying_unknown_categoricals=spec["time_varying_unknown_categoricals"],
        time_varying_unknown_reals=spec["time_varying_unknown_reals"],
        target_normalizer=GroupNormalizer(groups=["user_id"]),
        categorical_encoders={
            "user_id": NaNLabelEncoder(add_nan=False),
            "cluster_id_cat": NaNLabelEncoder(add_nan=False),
            "hour_cat": NaNLabelEncoder(add_nan=False),
            "day_of_week_cat": NaNLabelEncoder(add_nan=False),
            "month_cat": NaNLabelEncoder(add_nan=False),
            "is_weekend_cat": NaNLabelEncoder(add_nan=False),
            "holiday_flag_cat": NaNLabelEncoder(add_nan=False),
        },
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )


def build_derived_datasets(
    train_dataset: TimeSeriesDataSet,
    split_frames: Dict[str, pd.DataFrame],
) -> Dict[str, TimeSeriesDataSet]:
    val_source = pd.concat([split_frames["train"], split_frames["val"]], ignore_index=True, sort=False)
    test_source = pd.concat([val_source, split_frames["test"]], ignore_index=True, sort=False)
    future_source = pd.concat([test_source, split_frames["future"]], ignore_index=True, sort=False)

    datasets = {
        "train": train_dataset,
        "val": TimeSeriesDataSet.from_dataset(
            train_dataset,
            val_source,
            min_prediction_idx=int(split_frames["val"]["time_idx"].min()),
            predict=False,
            stop_randomization=True,
        ),
        "test": TimeSeriesDataSet.from_dataset(
            train_dataset,
            test_source,
            min_prediction_idx=int(split_frames["test"]["time_idx"].min()),
            predict=False,
            stop_randomization=True,
        ),
        "future": TimeSeriesDataSet.from_dataset(
            train_dataset,
            future_source,
            min_prediction_idx=int(split_frames["future"]["time_idx"].min()),
            predict=True,
            stop_randomization=True,
        ),
    }
    return datasets


def dataset_source_ranges(split_frames: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    val_source = pd.concat([split_frames["train"], split_frames["val"]], ignore_index=True, sort=False)
    test_source = pd.concat([val_source, split_frames["test"]], ignore_index=True, sort=False)
    future_source = pd.concat([test_source, split_frames["future"]], ignore_index=True, sort=False)
    sources = {
        "train": split_frames["train"],
        "val": val_source,
        "test": test_source,
        "future": future_source,
    }
    summary: Dict[str, Dict[str, Any]] = {}
    for name, frame in sources.items():
        summary[name] = {
            "timestamp_start": pd.Timestamp(frame["timestamp"].min()),
            "timestamp_end": pd.Timestamp(frame["timestamp"].max()),
            "time_idx_start": int(frame["time_idx"].min()),
            "time_idx_end": int(frame["time_idx"].max()),
            "rows": int(len(frame)),
        }
    return summary


def json_ready_records(frame: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    preview = frame.head(limit).copy()
    for column in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[column]):
            preview[column] = preview[column].astype(str)
        else:
            preview[column] = preview[column].astype(object).where(preview[column].notna(), None)
    return preview.to_dict(orient="records")


def try_extract_decoded_index(dataset: TimeSeriesDataSet) -> pd.DataFrame | None:
    decoded_index = getattr(dataset, "decoded_index", None)
    if decoded_index is None:
        return None
    if callable(decoded_index):
        decoded_index = decoded_index()
    if isinstance(decoded_index, pd.DataFrame):
        return decoded_index.copy()
    return None


def encoder_mapping(dataset: TimeSeriesDataSet, name: str) -> Dict[str, int] | None:
    encoders = getattr(dataset, "categorical_encoders", {}) or {}
    encoder = encoders.get(name)
    if encoder is None:
        return None
    classes = getattr(encoder, "classes_", None)
    if classes is None:
        return None
    if isinstance(classes, dict):
        return {str(key): int(value) for key, value in classes.items()}
    if hasattr(classes, "tolist"):
        class_values = classes.tolist()
    else:
        class_values = list(classes)
    return {str(value): int(index) for index, value in enumerate(class_values)}


def dataset_entry(
    name: str,
    dataset: TimeSeriesDataSet,
    source_range: Dict[str, Any],
    prediction_range: Dict[str, Any],
) -> Dict[str, Any]:
    decoded_index = try_extract_decoded_index(dataset)
    entry: Dict[str, Any] = {
        "dataset_name": name,
        "sequence_count": int(len(dataset)),
        "predict_mode": bool(name == "future"),
        "source_range": source_range,
        "prediction_range": prediction_range,
    }
    if decoded_index is not None and not decoded_index.empty:
        entry["decoded_index_columns"] = decoded_index.columns.tolist()
        entry["decoded_index_preview"] = json_ready_records(decoded_index)
        if "user_id" in decoded_index.columns:
            entry["sequence_count_by_user"] = decoded_index.groupby("user_id").size().astype(int).to_dict()
    return entry


def dataloader_config(config: Dict[str, Any], spec: Dict[str, list[str]], meta_path: Path, params_path: Path) -> Dict[str, Any]:
    batch_size = int(config["model"]["batch_size"])
    num_workers = int(config["model"]["num_workers"])
    max_encoder_length = int(config["model"]["max_encoder_length"])
    max_prediction_length = int(config["model"]["max_prediction_length"])
    min_encoder_length = int(config["model"].get("min_encoder_length", max_encoder_length))
    return {
        "dataset": {
            "group_ids": spec["group_ids"],
            "target": config["data"]["target_col"],
            "time_idx": "time_idx",
            "min_encoder_length": min_encoder_length,
            "max_encoder_length": max_encoder_length,
            "max_prediction_length": max_prediction_length,
            "static_categoricals": spec["static_categoricals"],
            "time_varying_known_categoricals": spec["time_varying_known_categoricals"],
            "time_varying_known_reals": spec["time_varying_known_reals"],
            "time_varying_unknown_categoricals": spec["time_varying_unknown_categoricals"],
            "time_varying_unknown_reals": spec["time_varying_unknown_reals"],
            "dataset_meta_path": str(meta_path),
            "train_dataset_parameters_path": str(params_path),
        },
        "dataloader": {
            "train": {
                "train": True,
                "batch_size": batch_size,
                "num_workers": num_workers,
            },
            "eval": {
                "train": False,
                "batch_size": batch_size,
                "num_workers": num_workers,
            },
            "predict": {
                "train": False,
                "batch_size": batch_size,
                "num_workers": num_workers,
            },
        },
    }


def save_dataset_parameters(dataset: TimeSeriesDataSet, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset.get_parameters(), output_path)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    validate_tft_only_paths(config)

    target_col = config["data"]["target_col"]
    use_target_derived_covariates = bool(config["data"].get("use_target_derived_covariates", False))
    required_columns = ["timestamp", "user_id", "cluster_id", target_col, *KNOWN_FUTURE_COLUMNS]
    if use_target_derived_covariates:
        required_columns.extend(OBSERVED_COLUMNS)

    train_path = Path(args.train_path or config["paths"]["features_train_path"])
    val_path = Path(args.val_path or config["paths"]["features_val_path"])
    test_path = Path(args.test_path or config["paths"]["features_test_path"])
    future_path = Path(args.future_path or config["paths"]["features_future_known_path"])
    dataset_meta_path = Path(args.dataset_meta_path or config["paths"]["tft_dataset_meta_path"])
    dataset_parameters_path = Path(args.dataset_parameters_path or config["paths"]["tft_dataset_parameters_path"])
    dataloader_config_path = Path(args.dataloader_config_path or config["paths"]["dataloader_config_path"])

    split_frames = {
        "train": load_feature_frame(train_path, "train", required_columns),
        "val": load_feature_frame(val_path, "val", required_columns),
        "test": load_feature_frame(test_path, "test", required_columns),
        "future": load_feature_frame(future_path, "future", required_columns),
    }
    split_frames = assign_continuous_time_idx(split_frames, config["data"]["frequency"])
    future_target_placeholder_by_user = fit_future_target_placeholder(split_frames, target_col)
    split_frames = apply_future_target_placeholder(split_frames, target_col, future_target_placeholder_by_user)
    split_frames = {name: add_dataset_features(frame) for name, frame in split_frames.items()}

    split_summary = validate_split_boundaries(split_frames)
    validate_calendar_windows(
        split_frames,
        expected_test_months=int(config["data"]["test_months"]),
        expected_future_days=int(config["data"]["future_horizon_days"]),
    )
    spec = dataset_feature_spec(target_col, use_target_derived_covariates)
    train_dataset = build_train_dataset(split_frames["train"], config, spec)
    datasets = build_derived_datasets(train_dataset, split_frames)
    source_summary = dataset_source_ranges(split_frames)

    save_dataset_parameters(train_dataset, dataset_parameters_path)

    meta_payload = {
        "builder": "tft.src.data.build_tft_timeseries_dataset",
        "input_paths": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
            "future": str(future_path),
        },
        "feature_spec": {
            "prompt_a_known_future_columns": KNOWN_FUTURE_COLUMNS,
            "prompt_a_target_derived_columns": OBSERVED_COLUMNS,
            "use_target_derived_covariates": use_target_derived_covariates,
            "prompt_a_target_derived_columns_excluded_from_tft_inputs": (not use_target_derived_covariates),
            **spec,
        },
        "time_index_mapping": {
            "base_timestamp": str(split_summary["train"]["timestamp_start"]),
            "frequency": config["data"]["frequency"],
            "formula": "time_idx = (timestamp - base_timestamp) / 1 hour",
        },
        "split_ranges": split_summary,
        "dataset_ranges": {
            name: dataset_entry(name, dataset, source_summary[name], split_summary[name])
            for name, dataset in datasets.items()
        },
        "categorical_encoder_mappings": {
            "user_id": encoder_mapping(train_dataset, "user_id"),
            "cluster_id_cat": encoder_mapping(train_dataset, "cluster_id_cat"),
            "hour_cat": encoder_mapping(train_dataset, "hour_cat"),
            "day_of_week_cat": encoder_mapping(train_dataset, "day_of_week_cat"),
            "month_cat": encoder_mapping(train_dataset, "month_cat"),
            "is_weekend_cat": encoder_mapping(train_dataset, "is_weekend_cat"),
            "holiday_flag_cat": encoder_mapping(train_dataset, "holiday_flag_cat"),
        },
        "paths": {
            "dataset_meta_path": str(dataset_meta_path),
            "train_dataset_parameters_path": str(dataset_parameters_path),
            "dataloader_config_path": str(dataloader_config_path),
        },
        "future_target_placeholder": {
            "applied_split": "future",
            "strategy": "per-user last observed target from train+val+test",
            "target_col": target_col,
            "user_count": len(future_target_placeholder_by_user),
        },
        "leakage_controls": {
            "train_normalizer_fit_only": True,
            "target_derived_covariates_excluded_from_tft_inputs": (not use_target_derived_covariates),
            "future_target_placeholder_uses_observed_history_only": True,
            "val_dataset_from_train_parameters_only": True,
            "test_dataset_from_train_parameters_only": True,
            "future_dataset_from_train_parameters_only": True,
            "prediction_ranges_are_split_isolated": True,
        },
    }
    write_json(meta_payload, dataset_meta_path)

    loader_payload = dataloader_config(config, spec, dataset_meta_path, dataset_parameters_path)
    write_yaml_config(loader_payload, dataloader_config_path)


if __name__ == "__main__":
    main()

# How to run:
# python -m tft.src.data.build_tft_timeseries_dataset --config tft/configs/data.yaml
