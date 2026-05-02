from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tft.src.model.tft_utils import (
    build_dataset_from_parameters,
    build_predict_trainer,
    load_dataset_parameters,
    load_tft_checkpoint,
    observed_only,
    predict_with_trainer,
)
from tft.src.pipeline_utils import ensure_dir, load_yaml_config, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate recursive 14-day user-level TFT forecasts.")
    parser.add_argument("--config", type=str, required=True, help="Path to base data config.")
    parser.add_argument("--panel-path", type=str, required=True, help="Prepared cluster panel parquet path.")
    parser.add_argument("--dataset-params-path", type=str, required=True, help="Shared dataset params path.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Finetuned checkpoint path.")
    parser.add_argument("--cluster-id", type=int, required=True, help="Cluster id for the finetuned model.")
    parser.add_argument("--model-name", type=str, required=True, help="Logical model name, e.g. tft_c10_ft.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for parquet future predictions.")
    return parser.parse_args()


def quantile_prediction_to_frame(prediction_output, quantiles: list[float]) -> pd.DataFrame:
    if isinstance(prediction_output, dict) and "prediction" in prediction_output:
        prediction_output = prediction_output["prediction"]
    if hasattr(prediction_output, "output"):
        prediction_output = prediction_output.output
    if isinstance(prediction_output, (list, tuple)):
        prediction_output = prediction_output[0]
    if hasattr(prediction_output, "detach"):
        prediction_output = prediction_output.detach().cpu().numpy()

    array = np.asarray(prediction_output)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for quantile predictions, got shape {array.shape}.")
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected quantile prediction array with 2 dimensions after squeeze, got shape {array.shape}.")
    if array.shape[1] != len(quantiles) and array.shape[0] == len(quantiles):
        array = array.T
    if array.shape[1] != len(quantiles):
        raise ValueError(f"Quantile prediction shape {array.shape} does not match configured quantiles {quantiles}.")

    quantile_index = {round(float(quantile), 4): index for index, quantile in enumerate(quantiles)}
    required = {
        0.1: "y_pred_p10",
        0.5: "y_pred_p50",
        0.9: "y_pred_p90",
    }
    missing = [quantile for quantile in required if round(quantile, 4) not in quantile_index]
    if missing:
        raise ValueError(f"Configured quantiles must include 0.1, 0.5, 0.9. Missing {missing} from {quantiles}.")

    return pd.DataFrame(
        {
            output_name: array[:, quantile_index[round(quantile, 4)]].astype(float)
            for quantile, output_name in required.items()
        }
    )


def fill_future_target_placeholder(future_block: pd.DataFrame, user_history: pd.DataFrame, target_col: str) -> pd.DataFrame:
    future_block = future_block.copy()
    history = user_history.sort_values("timestamp")[target_col]
    if history.empty:
        raise ValueError("Cannot build future target placeholder because user history is empty.")
    placeholder_value = float(pd.to_numeric(history, errors="coerce").iloc[-1])
    future_block[target_col] = pd.to_numeric(future_block[target_col], errors="coerce")
    future_block[target_col] = future_block[target_col].fillna(placeholder_value).astype(float)
    return future_block


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_dir(args.output_dir)

    panel_df = pd.read_parquet(args.panel_path)
    panel_df = panel_df.loc[panel_df["cluster_id"] == args.cluster_id].copy()
    observed_df = observed_only(panel_df)
    future_known_df = panel_df.loc[panel_df["split"] == "future"].copy()
    dataset_parameters = load_dataset_parameters(args.dataset_params_path)
    model = load_tft_checkpoint(args.checkpoint_path)
    predict_trainer = build_predict_trainer(config)

    target_col = config["data"]["target_col"]
    prediction_length = int(config["model"]["max_prediction_length"])
    future_days = int(config["data"]["future_horizon_days"])
    quantiles = [float(quantile) for quantile in config["model"]["quantiles"]]
    future_horizon_hours = future_days * 24
    if future_horizon_hours % prediction_length != 0:
        raise ValueError(
            "Future horizon hours must be an integer multiple of model.max_prediction_length for recursive inference. "
            f"Got future_horizon_hours={future_horizon_hours}, prediction_length={prediction_length}."
        )
    num_blocks = future_horizon_hours // prediction_length

    working_df = observed_df.sort_values(["user_id", "timestamp"]).copy()
    future_records = []

    for block_number in range(1, num_blocks + 1):
        block_predictions = []
        for user_id in sorted(working_df["user_id"].unique()):
            user_history = working_df.loc[working_df["user_id"] == user_id].sort_values("timestamp").copy()
            cluster_id = int(user_history["cluster_id"].iloc[0])
            block_start = user_history["timestamp"].max() + pd.Timedelta(hours=1)
            block_end = block_start + pd.Timedelta(hours=prediction_length - 1)
            future_block = future_known_df.loc[
                (future_known_df["user_id"] == user_id)
                & (future_known_df["timestamp"] >= block_start)
                & (future_known_df["timestamp"] <= block_end)
            ].copy()
            if len(future_block) != prediction_length:
                raise ValueError(f"Future block length mismatch for user_id={user_id}, block_number={block_number}.")
            future_block = fill_future_target_placeholder(future_block, user_history, target_col)

            source_df = pd.concat([user_history, future_block], ignore_index=True, sort=False)
            predict_dataset = build_dataset_from_parameters(
                dataset_parameters,
                source_df,
                predict=True,
                stop_randomization=True,
            )
            predict_loader = predict_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
            quantile_predictions = quantile_prediction_to_frame(
                predict_with_trainer(
                    predict_trainer,
                    model,
                    predict_loader,
                    mode="quantiles",
                ),
                quantiles=quantiles,
            )
            if len(quantile_predictions) != prediction_length:
                raise ValueError(
                    f"Quantile prediction length mismatch for user_id={user_id}, block_number={block_number}. "
                    f"Expected {prediction_length}, got {len(quantile_predictions)}."
                )

            for step, ((_, future_row), (_, quantile_row)) in enumerate(zip(future_block.iterrows(), quantile_predictions.iterrows()), start=1):
                block_predictions.append(
                    {
                        "model_name": args.model_name,
                        "cluster_id": cluster_id,
                        "user_id": user_id,
                        "forecast_origin": block_start,
                        "timestamp": future_row["timestamp"],
                        "horizon_step": step,
                        "y_pred_p10": float(quantile_row["y_pred_p10"]),
                        "y_pred_p50": float(quantile_row["y_pred_p50"]),
                        "y_pred_p90": float(quantile_row["y_pred_p90"]),
                    }
                )

        block_predictions_df = pd.DataFrame(block_predictions)
        future_records.append(block_predictions_df)

        appended_rows = block_predictions_df.loc[:, ["cluster_id", "user_id", "forecast_origin", "timestamp", "horizon_step", "y_pred_p50"]].copy()
        appended_rows = appended_rows.rename(columns={"y_pred_p50": target_col})
        appended_rows["split"] = "future_pred"
        known_lookup = future_known_df.loc[:, ["user_id", "timestamp", "hour", "dayofweek", "is_weekend", "month", "holiday_flag"]].copy()
        appended_rows = appended_rows.merge(known_lookup, on=["user_id", "timestamp"], how="left", validate="one_to_one")
        appended_rows["time_idx"] = ((appended_rows["timestamp"] - panel_df["timestamp"].min()) / pd.Timedelta(hours=1)).astype(int)
        appended_rows["cluster_id_cat"] = appended_rows["cluster_id"].astype(str)
        appended_rows["hour_cat"] = appended_rows["timestamp"].dt.hour.astype(str)
        appended_rows["day_of_week_cat"] = appended_rows["timestamp"].dt.dayofweek.astype(str)
        appended_rows["month_cat"] = appended_rows["timestamp"].dt.month.astype(str)
        appended_rows["is_weekend_cat"] = (appended_rows["timestamp"].dt.dayofweek >= 5).astype(int).astype(str)
        appended_rows["holiday_flag_cat"] = appended_rows["holiday_flag"].fillna(0).astype(int).astype(str)
        appended_rows["day_of_month"] = appended_rows["timestamp"].dt.day.astype(int)
        for column in ["hour", "dayofweek", "is_weekend", "month", "holiday_flag", "lag_24", "lag_48", "lag_168", "roll_mean_24", "roll_std_24", "roll_mean_168"]:
            if column not in appended_rows.columns:
                appended_rows[column] = np.nan
        appended_rows = appended_rows.reindex(columns=working_df.columns)
        working_df = pd.concat([working_df, appended_rows], ignore_index=True, sort=False)
        working_df = working_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    future_predictions_df = pd.concat(future_records, ignore_index=True).sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    predictions_path = output_dir / f"{args.model_name}_cluster_{args.cluster_id}_future_14d_predictions.parquet"
    future_predictions_df.to_parquet(predictions_path, index=False)
    write_json(
        {
            "model_name": args.model_name,
            "cluster_id": args.cluster_id,
            "future_days": future_days,
            "prediction_length_hours": prediction_length,
            "predictions_path": str(predictions_path),
        },
        output_dir / f"{args.model_name}_cluster_{args.cluster_id}_future_manifest.json",
    )


if __name__ == "__main__":
    main()
