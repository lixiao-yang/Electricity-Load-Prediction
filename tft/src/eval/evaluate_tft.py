from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tft.src.model.tft_utils import (
    build_dataset_from_parameters,
    build_predict_trainer,
    load_tft_checkpoint,
    predict_with_trainer,
    load_dataset_parameters,
    observed_only,
    split_time_index_bounds,
)
from tft.src.pipeline_utils import ensure_dir, load_yaml_config, write_json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a finetuned TFT checkpoint on cluster-level test data.")
    parser.add_argument("--config", type=str, required=True, help="Path to base data config.")
    parser.add_argument("--panel-path", type=str, required=True, help="Prepared cluster panel parquet path.")
    parser.add_argument("--dataset-params-path", type=str, required=True, help="Shared dataset params path.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Finetuned checkpoint path.")
    parser.add_argument("--cluster-id", type=int, required=True, help="Cluster id to evaluate.")
    parser.add_argument("--model-name", type=str, required=True, help="Logical model name, e.g. tft_c10_ft.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for parquet metrics and predictions.")
    return parser.parse_args()


def prediction_to_numpy(prediction_output) -> np.ndarray:
    if isinstance(prediction_output, dict) and "prediction" in prediction_output:
        prediction_output = prediction_output["prediction"]
    if hasattr(prediction_output, "output"):
        prediction_output = prediction_output.output
    if isinstance(prediction_output, (list, tuple)):
        prediction_output = prediction_output[0]
    if hasattr(prediction_output, "detach"):
        prediction_output = prediction_output.detach().cpu().numpy()
    return np.asarray(prediction_output).reshape(-1)


def fill_decoder_target_placeholder(source_df: pd.DataFrame, target_col: str, origin_idx: int) -> pd.DataFrame:
    source_df = source_df.sort_values("time_idx").copy()
    history = source_df.loc[source_df["time_idx"] < origin_idx, target_col]
    if history.empty:
        raise ValueError(f"Cannot build target placeholder because no history exists before origin_idx={origin_idx}.")
    placeholder_value = float(pd.to_numeric(history, errors="coerce").iloc[-1])
    source_df.loc[source_df["time_idx"] >= origin_idx, target_col] = placeholder_value
    return source_df


def safe_mape(actual: pd.Series, predicted: pd.Series, epsilon: float) -> float:
    denom = actual.abs().clip(lower=epsilon)
    return float(((actual - predicted).abs() / denom).mean() * 100.0)


def safe_mape_0_100(actual: pd.Series, predicted: pd.Series, epsilon: float) -> float:
    denom = actual.abs().clip(lower=epsilon)
    values = ((actual - predicted).abs() / denom) * 100.0
    return float(values.clip(lower=0.0, upper=100.0).mean())


def safe_epsilon_mape_pct(actual: pd.Series, predicted: pd.Series, epsilon: float) -> float:
    denom = actual.abs().clip(lower=epsilon)
    return float((((actual - predicted).abs() / denom) * 100.0).mean())


def safe_wmape(actual: pd.Series, predicted: pd.Series, epsilon: float) -> float:
    denom = max(float(actual.abs().sum()), float(epsilon))
    return float(100.0 * (actual - predicted).abs().sum() / denom)


def assign_test_periods(test_df: pd.DataFrame) -> pd.DataFrame:
    test_df = test_df.copy()
    test_df["test_month"] = test_df["timestamp"].dt.to_period("M").astype(str)
    ordered_months = sorted(test_df["test_month"].dropna().unique().tolist())
    if len(ordered_months) != 3:
        raise ValueError(
            f"Expected exactly 3 calendar months in test predictions for P1/P2/P3 mapping, got {ordered_months}."
        )
    period_map = {month: f"P{index}" for index, month in enumerate(ordered_months, start=1)}
    test_df["period"] = test_df["test_month"].map(period_map)
    if test_df["period"].isna().any():
        raise ValueError("Failed to assign P1/P2/P3 periods for all test prediction rows.")
    return test_df


def n_positive(actual: pd.Series, epsilon: float) -> int:
    return int((actual.abs() > epsilon).sum())


def compute_overall_metrics(predictions: pd.DataFrame, epsilon: float, cluster_id: int, model_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_name": model_name,
                "cluster_id": cluster_id,
                "period": "OVERALL",
                "MAPE_0_100": safe_mape_0_100(predictions["actual"], predictions["prediction"], epsilon),
                "EPSILON_MAPE_PCT": safe_epsilon_mape_pct(predictions["actual"], predictions["prediction"], epsilon),
                "WMAPE_0_100": safe_wmape(predictions["actual"], predictions["prediction"], epsilon),
                "n_obs": int(len(predictions)),
                "n_positive": n_positive(predictions["actual"], epsilon),
            }
        ]
    )


def compute_metrics_by_period(predictions: pd.DataFrame, epsilon: float, cluster_id: int, model_name: str) -> pd.DataFrame:
    rows = []
    for period in ["P1", "P2", "P3"]:
        period_df = predictions.loc[predictions["period"] == period]
        if period_df.empty:
            raise ValueError(f"Missing test rows for required period={period}.")
        rows.append(
            {
                "model_name": model_name,
                "cluster_id": cluster_id,
                "period": period,
                "MAPE_0_100": safe_mape_0_100(period_df["actual"], period_df["prediction"], epsilon),
                "EPSILON_MAPE_PCT": safe_epsilon_mape_pct(period_df["actual"], period_df["prediction"], epsilon),
                "WMAPE_0_100": safe_wmape(period_df["actual"], period_df["prediction"], epsilon),
                "n_obs": int(len(period_df)),
                "n_positive": n_positive(period_df["actual"], epsilon),
            }
        )
    return pd.DataFrame(rows)


def compute_metrics_by_user_period(
    predictions: pd.DataFrame,
    epsilon: float,
    cluster_id: int,
    model_name: str,
) -> pd.DataFrame:
    rows = []
    for (user_id, period), group_df in predictions.groupby(["user_id", "period"], sort=True):
        rows.append(
            {
                "model_name": model_name,
                "cluster_id": cluster_id,
                "user_id": str(user_id),
                "period": str(period),
                "MAPE_0_100": safe_mape_0_100(group_df["actual"], group_df["prediction"], epsilon),
                "EPSILON_MAPE_PCT": safe_epsilon_mape_pct(group_df["actual"], group_df["prediction"], epsilon),
                "WMAPE_0_100": safe_wmape(group_df["actual"], group_df["prediction"], epsilon),
                "n_obs": int(len(group_df)),
                "n_positive": n_positive(group_df["actual"], epsilon),
            }
        )
    result = pd.DataFrame(rows)
    expected_periods = {"P1", "P2", "P3"}
    observed_periods = set(result["period"].unique().tolist())
    if observed_periods != expected_periods:
        raise ValueError(f"Expected user-period metrics for periods {sorted(expected_periods)}, got {sorted(observed_periods)}.")
    return result


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    output_dir = ensure_dir(args.output_dir)
    panel_df = pd.read_parquet(args.panel_path)
    panel_df = panel_df.loc[panel_df["cluster_id"] == args.cluster_id].copy()
    observed_df = observed_only(panel_df)
    split_bounds = split_time_index_bounds(panel_df)
    dataset_parameters = load_dataset_parameters(args.dataset_params_path)
    model = load_tft_checkpoint(args.checkpoint_path)
    predict_trainer = build_predict_trainer(config)

    prediction_length = int(config["model"]["max_prediction_length"])
    stride_hours = int(config["model"]["rolling_stride_hours"])
    epsilon = float(config["data"]["mape_epsilon"])

    test_origins = range(
        split_bounds["test_start_idx"],
        split_bounds["test_end_idx"] - prediction_length + 2,
        stride_hours,
    )

    records = []
    for user_id in sorted(observed_df["user_id"].unique()):
        user_df = observed_df.loc[observed_df["user_id"] == user_id].sort_values("time_idx").copy()
        cluster_id = int(user_df["cluster_id"].iloc[0])
        for origin_idx in test_origins:
            horizon_end_idx = origin_idx + prediction_length - 1
            source_df = user_df.loc[user_df["time_idx"] <= horizon_end_idx].copy()
            actual_df = user_df.loc[(user_df["time_idx"] >= origin_idx) & (user_df["time_idx"] <= horizon_end_idx)].copy()
            if len(actual_df) != prediction_length:
                continue
            origin_timestamp = actual_df["timestamp"].min()
            source_df.loc[source_df["time_idx"] >= origin_idx, config["data"]["target_col"]] = np.nan
            source_df = fill_decoder_target_placeholder(source_df, config["data"]["target_col"], origin_idx)

            predict_dataset = build_dataset_from_parameters(
                dataset_parameters,
                source_df,
                predict=True,
                stop_randomization=True,
            )
            predict_loader = predict_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
            predictions = prediction_to_numpy(
                predict_with_trainer(
                    predict_trainer,
                    model,
                    predict_loader,
                    mode="prediction",
                )
            )

            for step, (_, actual_row) in enumerate(actual_df.iterrows(), start=1):
                records.append(
                    {
                        "model_name": args.model_name,
                        "cluster_id": cluster_id,
                        "user_id": user_id,
                        "forecast_origin": origin_timestamp,
                        "timestamp": actual_row["timestamp"],
                        "time_idx": int(actual_row["time_idx"]),
                        "horizon_step": step,
                        "prediction": float(predictions[step - 1]),
                        "actual": float(actual_row[config["data"]["target_col"]]),
                    }
                )

    predictions_df = pd.DataFrame(records)
    if predictions_df.empty:
        raise ValueError("No test predictions were generated. Check panel boundaries and checkpoint compatibility.")
    predictions_df = assign_test_periods(predictions_df).sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    overall_metrics_df = compute_overall_metrics(
        predictions_df,
        epsilon=epsilon,
        cluster_id=args.cluster_id,
        model_name=args.model_name,
    )
    by_period_metrics_df = compute_metrics_by_period(
        predictions_df,
        epsilon=epsilon,
        cluster_id=args.cluster_id,
        model_name=args.model_name,
    )
    by_user_period_metrics_df = compute_metrics_by_user_period(
        predictions_df,
        epsilon=epsilon,
        cluster_id=args.cluster_id,
        model_name=args.model_name,
    )

    predictions_path = output_dir / f"{args.model_name}_cluster_{args.cluster_id}_test_predictions.parquet"
    metrics_overall_path = output_dir / f"{args.model_name}_cluster_{args.cluster_id}_test_metrics_overall.parquet"
    metrics_by_period_path = output_dir / f"{args.model_name}_cluster_{args.cluster_id}_test_metrics_by_period.parquet"
    metrics_by_user_period_path = output_dir / f"{args.model_name}_cluster_{args.cluster_id}_test_metrics_by_user_period.parquet"
    predictions_df.to_parquet(predictions_path, index=False)
    overall_metrics_df.to_parquet(metrics_overall_path, index=False)
    by_period_metrics_df.to_parquet(metrics_by_period_path, index=False)
    by_user_period_metrics_df.to_parquet(metrics_by_user_period_path, index=False)
    write_json(
        {
            "model_name": args.model_name,
            "cluster_id": args.cluster_id,
            "predictions_path": str(predictions_path),
            "metrics_overall_path": str(metrics_overall_path),
            "metrics_by_period_path": str(metrics_by_period_path),
            "metrics_by_user_period_path": str(metrics_by_user_period_path),
            "overall_mape_0_100": float(overall_metrics_df["MAPE_0_100"].iloc[0]),
            "overall_epsilon_mape_pct": float(overall_metrics_df["EPSILON_MAPE_PCT"].iloc[0]),
        },
        output_dir / f"{args.model_name}_cluster_{args.cluster_id}_eval_manifest.json",
    )


if __name__ == "__main__":
    main()
