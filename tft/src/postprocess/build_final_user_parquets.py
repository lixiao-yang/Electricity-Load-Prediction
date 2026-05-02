from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from tft.src.pipeline_utils import ensure_dir, load_yaml_config, require_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final user-level parquet deliverables from TFT eval and infer outputs.")
    parser.add_argument("--eval-config", type=str, default="tft/configs/eval.yaml", help="Path to eval yaml config.")
    parser.add_argument("--infer-config", type=str, default="tft/configs/infer.yaml", help="Path to infer yaml config.")
    parser.add_argument("--output-dir", type=str, default="tft/artifacts/final", help="Output directory for final parquet files.")
    return parser.parse_args()


def cluster_specs(eval_config: Dict[str, Any], infer_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    eval_clusters = eval_config.get("clusters", {})
    infer_clusters = infer_config.get("clusters", {})
    if not eval_clusters or not infer_clusters:
        raise ValueError("Both eval and infer configs must define non-empty clusters sections.")

    specs: List[Dict[str, Any]] = []
    for cluster_key, eval_spec in sorted(eval_clusters.items(), key=lambda item: int(item[0])):
        if cluster_key not in infer_clusters:
            raise ValueError(f"Cluster {cluster_key} exists in eval config but not in infer config.")
        infer_spec = infer_clusters[cluster_key]
        cluster_id = int(eval_spec["cluster_id"])
        model_name = str(eval_spec["model_name"])
        if int(infer_spec["cluster_id"]) != cluster_id:
            raise ValueError(f"Cluster id mismatch between eval/infer configs for cluster key {cluster_key}.")
        if str(infer_spec["model_name"]) != model_name:
            raise ValueError(f"Model name mismatch between eval/infer configs for cluster key {cluster_key}.")
        specs.append(
            {
                "cluster_id": cluster_id,
                "model_name": model_name,
                "eval_output_dir": Path(eval_spec["output_dir"]),
                "infer_output_dir": Path(infer_spec["output_dir"]),
            }
        )
    return specs


def load_test_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    require_columns(df, ["model_name", "cluster_id", "user_id", "timestamp", "period", "horizon_step", "actual", "prediction"], "test_predictions")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["user_id"] = df["user_id"].astype(str)
    df["split"] = "test"
    df["phase"] = df["period"].astype(str)
    return df.loc[:, ["model_name", "cluster_id", "user_id", "timestamp", "split", "phase", "horizon_step", "actual", "prediction"]]


def load_future_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    require_columns(
        df,
        ["model_name", "cluster_id", "user_id", "forecast_origin", "timestamp", "horizon_step", "y_pred_p10", "y_pred_p50", "y_pred_p90"],
        "future_predictions",
    )
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["user_id"] = df["user_id"].astype(str)
    df["split"] = "future"
    df["phase"] = "FUTURE"
    return df.loc[:, ["model_name", "cluster_id", "user_id", "timestamp", "split", "phase", "horizon_step", "y_pred_p10", "y_pred_p50", "y_pred_p90"]]


def load_metrics_by_period(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    require_columns(
        df,
        ["model_name", "cluster_id", "period", "MAPE_0_100", "EPSILON_MAPE_PCT", "WMAPE_0_100", "n_obs", "n_positive"],
        "metrics_by_period",
    )
    df = df.copy()
    periods = sorted(df["period"].astype(str).unique().tolist())
    if periods != ["P1", "P2", "P3"]:
        raise ValueError(f"metrics_by_period must contain exactly P1/P2/P3, got {periods} from {path}.")
    return df


def load_metrics_overall(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    require_columns(
        df,
        ["model_name", "cluster_id", "period", "MAPE_0_100", "EPSILON_MAPE_PCT", "WMAPE_0_100", "n_obs", "n_positive"],
        "metrics_overall",
    )
    df = df.copy()
    periods = sorted(df["period"].astype(str).unique().tolist())
    if periods != ["OVERALL"]:
        raise ValueError(f"metrics_overall must contain exactly OVERALL, got {periods} from {path}.")
    return df


def main() -> None:
    args = parse_args()
    eval_config = load_yaml_config(args.eval_config)
    infer_config = load_yaml_config(args.infer_config)
    output_dir = ensure_dir(args.output_dir)

    test_frames = []
    future_frames = []
    metrics_by_period_frames = []
    metrics_overall_frames = []

    for spec in cluster_specs(eval_config, infer_config):
        cluster_id = spec["cluster_id"]
        model_name = spec["model_name"]
        eval_output_dir = spec["eval_output_dir"]
        infer_output_dir = spec["infer_output_dir"]

        test_predictions_path = eval_output_dir / f"{model_name}_cluster_{cluster_id}_test_predictions.parquet"
        metrics_overall_path = eval_output_dir / f"{model_name}_cluster_{cluster_id}_test_metrics_overall.parquet"
        metrics_by_period_path = eval_output_dir / f"{model_name}_cluster_{cluster_id}_test_metrics_by_period.parquet"
        future_predictions_path = infer_output_dir / f"{model_name}_cluster_{cluster_id}_future_14d_predictions.parquet"

        test_frames.append(load_test_predictions(test_predictions_path))
        metrics_overall_frames.append(load_metrics_overall(metrics_overall_path))
        metrics_by_period_frames.append(load_metrics_by_period(metrics_by_period_path))
        future_frames.append(load_future_predictions(future_predictions_path))

    user_level_test_predictions = pd.concat(test_frames, ignore_index=True).sort_values(
        ["cluster_id", "user_id", "timestamp", "horizon_step"]
    ).reset_index(drop=True)
    user_level_future_predictions = pd.concat(future_frames, ignore_index=True).sort_values(
        ["cluster_id", "user_id", "timestamp", "horizon_step"]
    ).reset_index(drop=True)

    test_all = user_level_test_predictions.copy()
    test_all["y_pred_p10"] = pd.NA
    test_all["y_pred_p50"] = pd.NA
    test_all["y_pred_p90"] = pd.NA
    future_all = user_level_future_predictions.copy()
    future_all["actual"] = pd.NA
    future_all["prediction"] = pd.NA
    user_level_predictions_all = pd.concat([test_all, future_all], ignore_index=True, sort=False)
    user_level_predictions_all = user_level_predictions_all.loc[
        :,
        [
            "model_name",
            "cluster_id",
            "user_id",
            "timestamp",
            "split",
            "phase",
            "horizon_step",
            "actual",
            "prediction",
            "y_pred_p10",
            "y_pred_p50",
            "y_pred_p90",
        ],
    ].sort_values(["split", "cluster_id", "user_id", "timestamp", "horizon_step"]).reset_index(drop=True)

    multiphase_metrics_by_period = pd.concat(metrics_by_period_frames, ignore_index=True).sort_values(
        ["cluster_id", "period"]
    ).reset_index(drop=True)
    multiphase_metrics_overall = pd.concat(metrics_overall_frames, ignore_index=True).sort_values(
        ["cluster_id", "period"]
    ).reset_index(drop=True)

    user_level_test_predictions.to_parquet(output_dir / "user_level_test_predictions.parquet", index=False)
    user_level_future_predictions.to_parquet(output_dir / "user_level_future_predictions_14d.parquet", index=False)
    user_level_predictions_all.to_parquet(output_dir / "user_level_predictions_all.parquet", index=False)
    multiphase_metrics_by_period.to_parquet(output_dir / "multiphase_metrics_by_period.parquet", index=False)
    multiphase_metrics_overall.to_parquet(output_dir / "multiphase_metrics_overall.parquet", index=False)


if __name__ == "__main__":
    main()

# How to run:
# python -m tft.src.postprocess.build_final_user_parquets --eval-config tft/configs/eval.yaml --infer-config tft/configs/infer.yaml
