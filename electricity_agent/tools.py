from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from electricity_agent.config import (
    CLUSTER6_FUTURE_PREDICTIONS_PATH,
    CLUSTER6_METRICS_PATH,
    CLUSTER6_TEST_PREDICTIONS_PATH,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    FORECAST_BUNDLE_PATH,
    MANIFEST_PATH,
    METRICS_BUNDLE_PATH,
    REGISTRY_PATH,
    TFT_FUTURE_PREDICTIONS_PATH,
    TFT_METRICS_PATH,
    TFT_TEST_PREDICTIONS_PATH,
)


def normalize_meter_id(value: object) -> str:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if text.startswith("MT_"):
        prefix, _, digits = text.partition("_")
        if digits.isdigit():
            return f"{prefix}_{int(digits):03d}"
    if text.isdigit():
        return f"MT_{int(text):03d}"
    return text


@lru_cache(maxsize=2)
def load_registry(registry_path: str | Path = REGISTRY_PATH) -> pd.DataFrame:
    path = Path(registry_path)
    if not path.exists():
        return pd.DataFrame(
            columns=["meter_id", "meter_id_norm", "cluster", "model_name", "status", "notes"]
        )
    df = pd.read_csv(path)
    if "meter_id_norm" not in df.columns:
        df["meter_id_norm"] = df["meter_id"].map(normalize_meter_id)
    return df


def resolve_meter_strict(query: str, registry_path: str | Path = REGISTRY_PATH) -> Dict[str, Any]:
    registry_df = load_registry(registry_path)
    meter_id = normalize_meter_id(query)
    if not meter_id:
        return {"status": "not_found", "query": query, "matches": []}

    exact = registry_df[registry_df["meter_id_norm"] == meter_id].copy()
    if len(exact) == 1:
        return {"status": "resolved", "query": query, "match": _row_to_payload(exact.iloc[0])}
    if len(exact) > 1:
        return {"status": "ambiguous", "query": query, "matches": _rows_to_payload(exact)}

    return {"status": "not_found", "query": query, "matches": []}


def get_meter_metadata(meter_id: str, registry_path: str | Path = REGISTRY_PATH) -> Dict[str, Any]:
    registry_df = load_registry(registry_path)
    meter_norm = normalize_meter_id(meter_id)
    match = registry_df[registry_df["meter_id_norm"] == meter_norm]
    if match.empty:
        raise KeyError(f"Unknown meter_id: {meter_id}")
    return _row_to_payload(match.iloc[0])


def get_meter_forecast(
    meter_id: str,
    horizon_hours: int,
    mode: str = "future",
    registry_path: str | Path = REGISTRY_PATH,
) -> Dict[str, Any]:
    metadata = get_meter_metadata(meter_id, registry_path=registry_path)
    meter_norm = normalize_meter_id(meter_id)
    normalized_mode = "evaluation" if mode == "evaluation" else "future"
    forecast_df = load_forecast_bundle()
    if forecast_df.empty:
        metadata = {
            **metadata,
            "mode": normalized_mode,
            "forecast_interface_status": "missing_bundle",
        }
        return {
            "metadata": metadata,
            "forecast": pd.DataFrame(columns=["forecast_timestamp", "forecast_value", "model_name"]),
            "message": "Forecast bundle is not available. Rebuild `electricity_agent/artifacts/forecast_bundle.csv` from model outputs.",
        }

    rows = forecast_df[
        (forecast_df["meter_id_norm"] == meter_norm) & (forecast_df["mode"] == normalized_mode)
    ].copy()
    if rows.empty:
        metadata = {
            **metadata,
            "mode": normalized_mode,
            "forecast_interface_status": "no_rows_for_mode",
        }
        return {
            "metadata": metadata,
            "forecast": pd.DataFrame(columns=["forecast_timestamp", "forecast_value", "model_name"]),
            "message": f"No {normalized_mode} forecast rows are available for {meter_norm}.",
        }

    rows["forecast_timestamp"] = pd.to_datetime(rows["forecast_timestamp"], errors="coerce")
    rows = rows.dropna(subset=["forecast_timestamp"]).sort_values("forecast_timestamp")
    if normalized_mode == "future":
        rows = rows.head(max(1, int(horizon_hours)))
    else:
        rows = rows.tail(max(1, int(horizon_hours)))

    metadata = _merge_metric_metadata(
        {
            **metadata,
            "mode": normalized_mode,
            "forecast_interface_status": "connected",
        }
    )
    forecast = rows[
        [
            "forecast_timestamp",
            "forecast_value",
            "model_name",
            "actual_value",
            "prediction_lower",
            "prediction_upper",
            "phase",
        ]
    ].reset_index(drop=True)
    return {
        "metadata": metadata,
        "forecast": forecast,
        "message": "",
    }


@lru_cache(maxsize=2)
def load_forecast_bundle(bundle_path: str | Path = FORECAST_BUNDLE_PATH) -> pd.DataFrame:
    path = Path(bundle_path)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = _build_forecast_bundle_from_sources()
    if df.empty:
        return _empty_forecast_bundle()
    df["meter_id"] = df["meter_id"].astype(str)
    df["meter_id_norm"] = df["meter_id"].map(normalize_meter_id)
    df["mode"] = df["mode"].astype(str)
    df["forecast_timestamp"] = pd.to_datetime(df["forecast_timestamp"], errors="coerce")
    numeric_cols = ["forecast_value", "actual_value", "prediction_lower", "prediction_upper"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["forecast_timestamp"]).sort_values(
        ["meter_id_norm", "mode", "forecast_timestamp"]
    )


@lru_cache(maxsize=2)
def load_metrics_bundle(bundle_path: str | Path = METRICS_BUNDLE_PATH) -> pd.DataFrame:
    path = Path(bundle_path)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = _build_metrics_bundle_from_sources()
    if df.empty:
        return pd.DataFrame(columns=["cluster", "model_name"])
    df["cluster"] = df["cluster"].astype(str)
    return df


def _merge_metric_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    metrics_df = load_metrics_bundle()
    cluster = str(metadata.get("cluster", ""))
    match = metrics_df[metrics_df["cluster"] == cluster]
    if match.empty:
        return metadata
    metric_row = match.iloc[0].dropna().to_dict()
    return {**metadata, **{f"metric_{key}": value for key, value in metric_row.items()}}


def _build_forecast_bundle_from_sources() -> pd.DataFrame:
    frames = [
        _standardize_tft_future(TFT_FUTURE_PREDICTIONS_PATH),
        _standardize_tft_test(TFT_TEST_PREDICTIONS_PATH),
        _standardize_cluster6_future(CLUSTER6_FUTURE_PREDICTIONS_PATH),
        _standardize_cluster6_test(CLUSTER6_TEST_PREDICTIONS_PATH),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return _empty_forecast_bundle()
    return pd.concat(frames, ignore_index=True, sort=False)


def _build_metrics_bundle_from_sources() -> pd.DataFrame:
    frames = [_standardize_tft_metrics(TFT_METRICS_PATH), _standardize_cluster6_metrics(CLUSTER6_METRICS_PATH)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["cluster", "model_name"])
    return pd.concat(frames, ignore_index=True, sort=False)


def _standardize_tft_future(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return _empty_forecast_bundle()
    df = _read_parquet_compat(path)
    if df.empty:
        return _empty_forecast_bundle()
    return pd.DataFrame(
        {
            "meter_id": df["user_id"].astype(str),
            "cluster": "C" + df["cluster_id"].astype(str),
            "model_name": df["model_name"].astype(str),
            "mode": "future",
            "forecast_timestamp": pd.to_datetime(df["timestamp"], errors="coerce"),
            "forecast_value": pd.to_numeric(df["y_pred_p50"], errors="coerce"),
            "actual_value": pd.NA,
            "prediction_lower": pd.to_numeric(df["y_pred_p10"], errors="coerce"),
            "prediction_upper": pd.to_numeric(df["y_pred_p90"], errors="coerce"),
            "phase": df.get("phase", "FUTURE"),
        }
    ).assign(meter_id_norm=lambda rows: rows["meter_id"].map(normalize_meter_id))


def _standardize_tft_test(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return _empty_forecast_bundle()
    df = _read_parquet_compat(path)
    if df.empty:
        return _empty_forecast_bundle()
    return pd.DataFrame(
        {
            "meter_id": df["user_id"].astype(str),
            "cluster": "C" + df["cluster_id"].astype(str),
            "model_name": df["model_name"].astype(str),
            "mode": "evaluation",
            "forecast_timestamp": pd.to_datetime(df["timestamp"], errors="coerce"),
            "forecast_value": pd.to_numeric(df["prediction"], errors="coerce"),
            "actual_value": pd.to_numeric(df["actual"], errors="coerce"),
            "prediction_lower": pd.NA,
            "prediction_upper": pd.NA,
            "phase": df.get("phase", "TEST"),
        }
    ).assign(meter_id_norm=lambda rows: rows["meter_id"].map(normalize_meter_id))


def _standardize_cluster6_future(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return _empty_forecast_bundle()
    df = _read_parquet_compat(path)
    if df.empty:
        return _empty_forecast_bundle()
    model = df["model"].astype(str) if "model" in df.columns else "direct_xgboost"
    return pd.DataFrame(
        {
            "meter_id": "MT_362",
            "cluster": "C6",
            "model_name": model,
            "mode": "future",
            "forecast_timestamp": pd.to_datetime(df["timestamp"], errors="coerce"),
            "forecast_value": pd.to_numeric(df["prediction"], errors="coerce"),
            "actual_value": pd.NA,
            "prediction_lower": pd.NA,
            "prediction_upper": pd.NA,
            "phase": "FUTURE",
        }
    ).assign(meter_id_norm="MT_362")


def _standardize_cluster6_test(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return _empty_forecast_bundle()
    df = _read_parquet_compat(path)
    if df.empty:
        return _empty_forecast_bundle()
    return pd.DataFrame(
        {
            "meter_id": "MT_362",
            "cluster": "C6",
            "model_name": "direct_xgboost",
            "mode": "evaluation",
            "forecast_timestamp": pd.to_datetime(df["timestamp"], errors="coerce"),
            "forecast_value": pd.to_numeric(df["prediction"], errors="coerce"),
            "actual_value": pd.to_numeric(df["actual"], errors="coerce"),
            "prediction_lower": pd.NA,
            "prediction_upper": pd.NA,
            "phase": "TEST",
        }
    ).assign(meter_id_norm="MT_362")


def _standardize_tft_metrics(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["cluster", "model_name"])
    df = _read_parquet_compat(path)
    if df.empty:
        return pd.DataFrame(columns=["cluster", "model_name"])
    return pd.DataFrame(
        {
            "cluster": "C" + df["cluster_id"].astype(str),
            "model_name": df["model_name"].astype(str),
            "test_mape_0_100": pd.to_numeric(df.get("MAPE_0_100"), errors="coerce"),
            "test_wmape_0_100": pd.to_numeric(df.get("WMAPE_0_100"), errors="coerce"),
            "test_n_obs": pd.to_numeric(df.get("n_obs"), errors="coerce"),
        }
    )


def _standardize_cluster6_metrics(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["cluster", "model_name"])
    df = _read_parquet_compat(path)
    if df.empty:
        return pd.DataFrame(columns=["cluster", "model_name"])
    row = df.iloc[0]
    return pd.DataFrame(
        [
            {
                "cluster": "C6",
                "model_name": row.get("selected_model", "direct_xgboost"),
                "test_mape_0_100": row.get("test_mape_0_100"),
                "test_wmape_0_100": row.get("test_wmape_0_100"),
                "test_n_obs": row.get("test_n_obs"),
            }
        ]
    )


def _empty_forecast_bundle() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "meter_id",
            "meter_id_norm",
            "cluster",
            "model_name",
            "mode",
            "forecast_timestamp",
            "forecast_value",
            "actual_value",
            "prediction_lower",
            "prediction_upper",
            "phase",
        ]
    )


def _legacy_placeholder_forecast(
    meter_id: str,
    horizon_hours: int,
    mode: str = "future",
    registry_path: str | Path = REGISTRY_PATH,
) -> Dict[str, Any]:
    metadata = get_meter_metadata(meter_id, registry_path=registry_path)
    metadata = {
        **metadata,
        "mode": mode,
        "forecast_interface_status": "placeholder",
    }
    forecast = pd.DataFrame(columns=["forecast_timestamp", "forecast_value", "model_name"])
    return {
        "metadata": metadata,
        "forecast": forecast,
        "message": (
            "Forecast data interface is intentionally left empty. "
            "Connect the model output table here when final forecasting artifacts are ready."
        ),
    }


@lru_cache(maxsize=2)
def load_observed_history(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for path in [Path(train_path), Path(test_path)]:
        if not path.exists():
            continue
        try:
            df = _read_parquet_compat(path)
        except RuntimeError:
            continue
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index(pd.to_datetime(df["timestamp"]))
            else:
                continue
        df = df.sort_index()
        long = df.reset_index(names="timestamp").melt(
            id_vars=["timestamp"],
            var_name="meter_id",
            value_name="actual_value",
        )
        parts.append(long)
    if not parts:
        return pd.DataFrame(columns=["timestamp", "meter_id", "actual_value"])
    combined = pd.concat(parts, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    combined["meter_id"] = combined["meter_id"].astype(str)
    combined["actual_value"] = pd.to_numeric(combined["actual_value"], errors="coerce")
    return combined.dropna(subset=["timestamp"]).sort_values(["meter_id", "timestamp"]).reset_index(drop=True)


def get_meter_history(meter_id: str, limit: int = 24 * 30) -> pd.DataFrame:
    history_df = load_observed_history()
    meter_norm = normalize_meter_id(meter_id)
    match = history_df[history_df["meter_id"].map(normalize_meter_id) == meter_norm].copy()
    if match.empty:
        return pd.DataFrame(columns=["timestamp", "actual_value"])
    match = match[["timestamp", "actual_value"]].sort_values("timestamp")
    return match.tail(int(limit)).reset_index(drop=True)


def load_manifest(manifest_path: str | Path = MANIFEST_PATH) -> Optional[Dict[str, Any]]:
    path = Path(manifest_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clear_caches() -> None:
    load_registry.cache_clear()
    load_forecast_bundle.cache_clear()
    load_metrics_bundle.cache_clear()
    load_observed_history.cache_clear()


def _row_to_payload(row: pd.Series) -> Dict[str, Any]:
    return row.to_dict()


def _rows_to_payload(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [_row_to_payload(row) for _, row in df.iterrows()]


def _read_parquet_compat(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for engine in ("fastparquet", "pyarrow"):
        try:
            return pd.read_parquet(path, engine=engine)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not read {path}: {last_error}")
