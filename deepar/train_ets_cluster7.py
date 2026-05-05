from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


DEFAULT_TRAIN_PATH = Path("data/train_hourly_preprocessed.parquet")
DEFAULT_TEST_PATH = Path("data/test_hourly_preprocessed.parquet")
DEFAULT_CLUSTER_PATH = Path("data/extended-clustering-high-cov/clusters_3models.parquet")
DEFAULT_OUTPUT_ROOT = Path("outputs")


@dataclass
class ETSConfig:
    train_path: str = str(DEFAULT_TRAIN_PATH)
    test_path: str = str(DEFAULT_TEST_PATH)
    cluster_path: str = str(DEFAULT_CLUSTER_PATH)
    output_dir: str | None = None
    cluster_id: int = 7
    cluster_col: str = "cluster_kmeans"
    validation_months: int = 3
    prediction_length: int = 24
    seasonal_periods: int = 24
    trend: str | None = "add"
    seasonal: str | None = "add"
    damped_trend: bool = False
    use_boxcox: bool = False
    initialization_method: str = "estimated"
    strategy: str = "direct"


def resolve_output_dir(config: ETSConfig) -> Path:
    if config.output_dir:
        return Path(config.output_dir)
    return DEFAULT_OUTPUT_ROOT / f"ets_cluster_{config.cluster_id}"


def load_cluster_panels(config: ETSConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_parquet(config.train_path).sort_index()
    test_df = pd.read_parquet(config.test_path).sort_index()
    labels = pd.read_parquet(config.cluster_path)

    train_df.index = pd.to_datetime(train_df.index)
    test_df.index = pd.to_datetime(test_df.index)

    if config.cluster_col not in labels.columns:
        raise KeyError(f"Missing cluster column: {config.cluster_col}")

    common_meters = train_df.columns.intersection(test_df.columns).intersection(labels.index)
    labels = labels.loc[common_meters].copy()
    cluster_meters = labels.index[labels[config.cluster_col] == config.cluster_id].tolist()
    if not cluster_meters:
        raise ValueError(f"No meters found for cluster {config.cluster_id}")

    return train_df.loc[:, cluster_meters].copy(), test_df.loc[:, cluster_meters].copy()


def split_train_validation(
    train_panel: pd.DataFrame,
    validation_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_start = train_panel.index.max().to_period("M").to_timestamp() - pd.DateOffset(months=validation_months - 1)
    train_sub = train_panel.loc[train_panel.index < val_start].copy()
    val_panel = train_panel.loc[train_panel.index >= val_start].copy()
    if train_sub.empty or val_panel.empty:
        raise ValueError("Validation split failed; empty train_sub or val_panel.")
    return train_sub, val_panel


def fit_ets(history: pd.Series, config: ETSConfig):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    history = history.astype(float)
    history = history.clip(lower=0)

    model = ExponentialSmoothing(
        history,
        trend=config.trend,
        seasonal=config.seasonal,
        seasonal_periods=config.seasonal_periods,
        damped_trend=config.damped_trend,
        initialization_method=config.initialization_method,
        use_boxcox=config.use_boxcox,
    )
    return model.fit(optimized=True, remove_bias=False)


def rolling_predict_ets(
    history_panel: pd.DataFrame,
    eval_panel: pd.DataFrame,
    config: ETSConfig,
    progress_desc: str = "ETS rolling forecast",
) -> pd.DataFrame:
    pred_frames: list[pd.DataFrame] = []
    step = config.prediction_length
    total_windows = sum((len(eval_panel[col]) + step - 1) // step for col in history_panel.columns)
    progress = tqdm(total=total_windows, desc=progress_desc, unit="window")

    for meter_id in history_panel.columns:
        history = history_panel[meter_id].astype(float).copy()
        eval_series = eval_panel[meter_id].astype(float)
        meter_preds: list[pd.DataFrame] = []

        for start_idx in range(0, len(eval_series), step):
            end_idx = min(start_idx + step, len(eval_series))
            horizon = end_idx - start_idx
            forecast_index = eval_series.index[start_idx:end_idx]
            fit_res = fit_ets(history, config)
            pred_mean = np.asarray(fit_res.forecast(horizon), dtype=float)
            truth = eval_series.iloc[start_idx:end_idx].to_numpy(dtype=float)

            meter_preds.append(
                pd.DataFrame(
                    {
                        "timestamp": forecast_index,
                        "meter_id": meter_id,
                        "cluster_id": config.cluster_id,
                        "y_true": truth,
                        "y_pred": pred_mean,
                    }
                )
            )

            history = pd.concat([history, eval_series.iloc[start_idx:end_idx]])
            progress.update(1)

        pred_frames.append(pd.concat(meter_preds, ignore_index=True))

    progress.close()
    pred_df = pd.concat(pred_frames, ignore_index=True)
    pred_df["ape"] = np.where(
        np.abs(pred_df["y_true"]) > 1e-8,
        np.abs((pred_df["y_true"] - pred_df["y_pred"]) / pred_df["y_true"]) * 100.0,
        np.nan,
    )
    pred_df["epsilon_ape"] = (
        np.abs(pred_df["y_true"] - pred_df["y_pred"]) / (np.abs(pred_df["y_true"]) + 1.0) * 100.0
    )
    return pred_df


def direct_predict_ets(
    history_panel: pd.DataFrame,
    eval_panel: pd.DataFrame,
    config: ETSConfig,
    progress_desc: str = "ETS direct forecast",
) -> pd.DataFrame:
    pred_frames: list[pd.DataFrame] = []

    for meter_id in tqdm(history_panel.columns, desc=progress_desc, unit="meter"):
        history = history_panel[meter_id].astype(float).copy()
        eval_series = eval_panel[meter_id].astype(float)
        fit_res = fit_ets(history, config)
        pred_mean = np.asarray(fit_res.forecast(len(eval_series)), dtype=float)
        pred_frames.append(
            pd.DataFrame(
                {
                    "timestamp": eval_series.index,
                    "meter_id": meter_id,
                    "cluster_id": config.cluster_id,
                    "y_true": eval_series.to_numpy(dtype=float),
                    "y_pred": pred_mean,
                }
            )
        )

    pred_df = pd.concat(pred_frames, ignore_index=True)
    pred_df["ape"] = np.where(
        np.abs(pred_df["y_true"]) > 1e-8,
        np.abs((pred_df["y_true"] - pred_df["y_pred"]) / pred_df["y_true"]) * 100.0,
        np.nan,
    )
    pred_df["epsilon_ape"] = (
        np.abs(pred_df["y_true"] - pred_df["y_pred"]) / (np.abs(pred_df["y_true"]) + 1.0) * 100.0
    )
    return pred_df


def summarize_metrics(pred_df: pd.DataFrame) -> dict:
    valid_df = pred_df.loc[pred_df["ape"].notna()].copy()
    overall_mape = float(valid_df["ape"].mean())
    overall_wmape = float(
        np.abs(valid_df["y_true"] - valid_df["y_pred"]).sum() / np.abs(valid_df["y_true"]).sum() * 100.0
    )
    overall_epsilon_mape = float(pred_df["epsilon_ape"].mean())
    by_meter = (
        valid_df.groupby("meter_id", as_index=True)["ape"]
        .mean()
        .dropna()
        .sort_index()
        .to_dict()
    )

    return {
        "overall_mape": overall_mape,
        "overall_wmape": overall_wmape,
        "overall_epsilon_mape": overall_epsilon_mape,
        "n_predictions": int(len(pred_df)),
        "n_nonzero_targets": int(valid_df.shape[0]),
        "n_meters": int(pred_df["meter_id"].nunique()),
        "meter_mape_head": dict(list((k, float(v)) for k, v in by_meter.items())[:10]),
    }


def save_outputs(pred_df: pd.DataFrame, config: ETSConfig, eval_split: str = "validation") -> Path:
    out_dir = resolve_output_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"ets_cluster_{config.cluster_id}_{eval_split}_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    return pred_path


def run_experiment(config: ETSConfig) -> dict:
    train_panel, test_panel = load_cluster_panels(config)
    train_sub, val_panel = split_train_validation(train_panel, config.validation_months)
    if config.strategy == "rolling":
        pred_df = rolling_predict_ets(train_sub, val_panel, config, progress_desc="ETS validation rolling forecast")
    else:
        pred_df = direct_predict_ets(train_sub, val_panel, config, progress_desc="ETS validation direct forecast")
    metrics = summarize_metrics(pred_df)
    pred_path = save_outputs(pred_df, config, eval_split="validation")

    return {
        "config": asdict(config),
        "metrics": metrics,
        "prediction_path": str(pred_path),
        "evaluation_split": "validation",
        "train_range": [str(train_sub.index.min()), str(train_sub.index.max())],
        "validation_range": [str(val_panel.index.min()), str(val_panel.index.max())],
        "test_range": [str(test_panel.index.min()), str(test_panel.index.max())],
        "strategy": config.strategy,
    }


def parse_args() -> ETSConfig:
    parser = argparse.ArgumentParser(description="Train ETS baseline for clustered hourly loads.")
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--test-path", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--cluster-path", default=str(DEFAULT_CLUSTER_PATH))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cluster-id", type=int, default=7)
    parser.add_argument("--validation-months", type=int, default=3)
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--seasonal-periods", type=int, default=24)
    parser.add_argument("--trend", default="add")
    parser.add_argument("--seasonal", default="add")
    parser.add_argument("--damped-trend", action="store_true")
    parser.add_argument("--strategy", choices=["direct", "rolling"], default="direct")
    args = parser.parse_args()

    return ETSConfig(
        train_path=args.train_path,
        test_path=args.test_path,
        cluster_path=args.cluster_path,
        output_dir=args.output_dir,
        cluster_id=args.cluster_id,
        validation_months=args.validation_months,
        prediction_length=args.prediction_length,
        seasonal_periods=args.seasonal_periods,
        trend=None if args.trend == "none" else args.trend,
        seasonal=None if args.seasonal == "none" else args.seasonal,
        damped_trend=args.damped_trend,
        strategy=args.strategy,
    )


def main() -> None:
    config = parse_args()
    results = run_experiment(config)

    print("ETS finished.")
    print(f"Cluster: {config.cluster_id}")
    print(f"Evaluation split: {results['evaluation_split']}")
    print(f"Strategy: {results['strategy']}")
    print(f"Train range: {results['train_range'][0]} -> {results['train_range'][1]}")
    print(f"Validation range: {results['validation_range'][0]} -> {results['validation_range'][1]}")
    print(f"Test range: {results['test_range'][0]} -> {results['test_range'][1]}")
    print(f"Overall MAPE: {results['metrics']['overall_mape']:.4f}")
    print(f"Overall wMAPE: {results['metrics']['overall_wmape']:.4f}")
    print(f"Overall epsilon-MAPE: {results['metrics']['overall_epsilon_mape']:.4f}")
    print(f"Predictions saved to: {results['prediction_path']}")


if __name__ == "__main__":
    main()
