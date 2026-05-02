from __future__ import annotations

import argparse

import pandas as pd

from tft.src.pipeline_utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundle parquet outputs for downstream agent analysis.")
    parser.add_argument("--test-prediction-paths", type=str, nargs="+", required=True, help="Test prediction parquet paths.")
    parser.add_argument("--future-prediction-paths", type=str, nargs="+", required=True, help="Future 14-day prediction parquet paths.")
    parser.add_argument("--metric-paths", type=str, nargs="+", required=True, help="Cluster-level metric parquet paths.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for the combined agent bundle.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    test_frames = [pd.read_parquet(path).assign(dataset_type="test_eval") for path in args.test_prediction_paths]
    future_frames = [pd.read_parquet(path).assign(dataset_type="future_14d", actual=pd.NA, segment="future") for path in args.future_prediction_paths]
    metric_frames = [pd.read_parquet(path) for path in args.metric_paths]

    combined_predictions = pd.concat(test_frames + future_frames, ignore_index=True, sort=False)
    combined_metrics = pd.concat(metric_frames, ignore_index=True, sort=False)

    prediction_bundle_path = output_dir / "agent_predictions_bundle.parquet"
    metric_bundle_path = output_dir / "agent_metrics_bundle.parquet"
    combined_predictions.to_parquet(prediction_bundle_path, index=False)
    combined_metrics.to_parquet(metric_bundle_path, index=False)

    write_json(
        {
            "prediction_bundle_path": str(prediction_bundle_path),
            "metric_bundle_path": str(metric_bundle_path),
            "test_prediction_count": int(len(pd.concat(test_frames, ignore_index=True))),
            "future_prediction_count": int(len(pd.concat(future_frames, ignore_index=True))),
            "metric_row_count": int(len(combined_metrics)),
        },
        output_dir / "agent_bundle_manifest.json",
    )


if __name__ == "__main__":
    main()

