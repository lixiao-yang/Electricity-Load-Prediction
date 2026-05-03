from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_TRAIN_PATH = Path("data/train_hourly_preprocessed.parquet")
DEFAULT_TEST_PATH = Path("data/test_hourly_preprocessed.parquet")
DEFAULT_CLUSTER_PATH = Path("data/extended-clustering-high-cov/clusters_3models.parquet")
DEFAULT_OUTPUT_ROOT = Path("outputs")


@dataclass
class TrainConfig:
    train_path: str = str(DEFAULT_TRAIN_PATH)
    test_path: str = str(DEFAULT_TEST_PATH)
    cluster_path: str = str(DEFAULT_CLUSTER_PATH)
    output_dir: str | None = None
    cluster_ids: tuple[int, ...] = (2, 3)
    cluster_col: str = "cluster_kmeans"
    freq: str = "h"
    prediction_length: int = 24
    context_length: int = 24 * 14
    epochs: int = 20
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_batches_per_epoch: int = 50
    num_layers: int = 2
    hidden_size: int = 64
    dropout_rate: float = 0.1
    num_parallel_samples: int = 100
    seed: int = 42
    use_hour_of_day: bool = True
    use_day_of_week: bool = True
    use_weekend: bool = True
    use_month: bool = True
    use_holiday: bool = True
    holiday_country: str = "PT"
    validation_months: int = 3


@dataclass
class RandomSearchConfig:
    n_trials: int = 8
    context_length_choices: tuple[int, ...] = (24 * 7, 24 * 14, 24 * 21)
    hidden_size_choices: tuple[int, ...] = (48, 64, 96, 128)
    dropout_rate_choices: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3)
    learning_rate_choices: tuple[float, ...] = (1e-3, 5e-4, 2e-4)
    epochs_choices: tuple[int, ...] = (5, 10)
    num_batches_per_epoch_choices: tuple[int, ...] = (30, 50)
    batch_size_choices: tuple[int, ...] = (64,)
    num_parallel_samples_choices: tuple[int, ...] = (50,)
    output_dir: str | None = None
    seed: int = 123


def cluster_tag(cluster_ids: Iterable[int]) -> str:
    return "_".join(str(int(cluster_id)) for cluster_id in sorted(cluster_ids))


def resolve_output_dir(config: TrainConfig) -> Path:
    if config.output_dir:
        return Path(config.output_dir)
    return DEFAULT_OUTPUT_ROOT / f"deepar_clusters_{cluster_tag(config.cluster_ids)}"


def resolve_search_output_dir(base_config: TrainConfig, search_config: RandomSearchConfig) -> Path:
    if search_config.output_dir:
        return Path(search_config.output_dir)
    return DEFAULT_OUTPUT_ROOT / f"deepar_random_search_{cluster_tag(base_config.cluster_ids)}"


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_cluster_panels(config: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_parquet(config.train_path).sort_index()
    test_df = pd.read_parquet(config.test_path).sort_index()
    labels = pd.read_parquet(config.cluster_path)

    train_df.index = pd.to_datetime(train_df.index)
    test_df.index = pd.to_datetime(test_df.index)

    if config.cluster_col not in labels.columns:
        raise KeyError(f"Missing cluster column: {config.cluster_col}")

    common_meters = train_df.columns.intersection(test_df.columns).intersection(labels.index)
    labels = labels.loc[common_meters].copy()
    selected = labels[labels[config.cluster_col].isin(config.cluster_ids)].copy()
    selected_meters = selected.index.tolist()

    if not selected_meters:
        raise ValueError(f"No meters found for cluster IDs: {config.cluster_ids}")

    train_panel = train_df.loc[:, selected_meters].copy()
    test_panel = test_df.loc[:, selected_meters].copy()

    return train_panel, test_panel, selected


def split_train_validation(
    train_panel: pd.DataFrame,
    config: TrainConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_panel.empty:
        raise ValueError("Train panel is empty.")

    val_start = train_panel.index.max().to_period("M").to_timestamp() - pd.DateOffset(months=config.validation_months - 1)
    train_sub = train_panel.loc[train_panel.index < val_start].copy()
    val_panel = train_panel.loc[train_panel.index >= val_start].copy()

    if train_sub.empty or val_panel.empty:
        raise ValueError(
            "Validation split failed; train_sub or val_panel is empty. "
            f"validation_months={config.validation_months}"
        )

    return train_sub, val_panel


def build_metadata(train_panel: pd.DataFrame, labels: pd.DataFrame, config: TrainConfig) -> pd.DataFrame:
    cluster_vocab = {cluster_id: idx for idx, cluster_id in enumerate(sorted(config.cluster_ids))}
    mean_load = train_panel.mean(axis=0)

    meta = pd.DataFrame(index=train_panel.columns)
    meta["cluster_id"] = labels.loc[train_panel.columns, config.cluster_col].astype(int)
    meta["cluster_code"] = meta["cluster_id"].map(cluster_vocab).astype(int)
    meta["mean_hourly_load"] = mean_load.astype(float)
    meta["log_mean_hourly_load"] = np.log1p(meta["mean_hourly_load"])
    return meta


def build_dynamic_features(
    history_index: pd.DatetimeIndex,
    eval_index: pd.DatetimeIndex,
    config: TrainConfig,
) -> pd.DataFrame:
    full_index = history_index.append(eval_index)
    feat_df = pd.DataFrame(index=full_index)

    if config.use_hour_of_day:
        feat_df["hour_of_day"] = full_index.hour.astype(float) / 23.0
    if config.use_day_of_week:
        feat_df["day_of_week"] = full_index.dayofweek.astype(float) / 6.0
    if config.use_weekend:
        feat_df["weekend"] = full_index.dayofweek.isin([5, 6]).astype(float)
    if config.use_month:
        feat_df["month"] = (full_index.month.astype(float) - 1.0) / 11.0
    if config.use_holiday:
        import holidays

        holiday_calendar = holidays.country_holidays(config.holiday_country)
        feat_df["holiday"] = full_index.normalize().map(lambda ts: float(ts in holiday_calendar))

    if feat_df.empty:
        return feat_df

    feat_df = feat_df.astype(float)
    return feat_df


def build_list_dataset(
    panel: pd.DataFrame,
    meta: pd.DataFrame,
    dynamic_feat_df: pd.DataFrame,
    freq: str,
):
    from gluonts.dataset.common import ListDataset

    records = []
    for meter_id in panel.columns:
        record = {
            "item_id": str(meter_id),
            "start": panel.index[0],
            "target": panel[meter_id].astype(float).to_numpy(),
            "feat_static_cat": [int(meta.loc[meter_id, "cluster_code"])],
            "feat_static_real": [float(meta.loc[meter_id, "log_mean_hourly_load"])],
        }
        if not dynamic_feat_df.empty:
            record["feat_dynamic_real"] = dynamic_feat_df.loc[panel.index].to_numpy(dtype=float).T
        records.append(record)
    return ListDataset(records, freq=freq)


def fit_predictor(
    train_panel: pd.DataFrame,
    meta: pd.DataFrame,
    dynamic_feat_df: pd.DataFrame,
    config: TrainConfig,
):
    from gluonts.torch.distributions import StudentTOutput
    from gluonts.torch.model.deepar import DeepAREstimator

    train_ds = build_list_dataset(train_panel, meta, dynamic_feat_df, config.freq)
    num_feat_dynamic_real = 0 if dynamic_feat_df.empty else dynamic_feat_df.shape[1]
    estimator = DeepAREstimator(
        freq=config.freq,
        prediction_length=config.prediction_length,
        context_length=config.context_length,
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        dropout_rate=config.dropout_rate,
        distr_output=StudentTOutput(),
        time_features=[],
        num_feat_dynamic_real=num_feat_dynamic_real,
        num_feat_static_cat=1,
        num_feat_static_real=1,
        cardinality=[len(sorted(config.cluster_ids))],
        batch_size=config.batch_size,
        num_batches_per_epoch=config.num_batches_per_epoch,
        trainer_kwargs={
            "max_epochs": config.epochs,
            "logger": False,
        },
        lr=config.learning_rate,
    )
    predictor = estimator.train(train_ds)
    return predictor


def rolling_predict(
    predictor,
    history_panel: pd.DataFrame,
    eval_panel: pd.DataFrame,
    meta: pd.DataFrame,
    dynamic_feat_df: pd.DataFrame,
    config: TrainConfig,
    progress_desc: str = "Rolling forecast",
) -> pd.DataFrame:
    from gluonts.dataset.common import ListDataset
    from tqdm.auto import tqdm

    pred_frames: list[pd.DataFrame] = []
    step = config.prediction_length

    total_windows = sum((len(eval_panel[meter_id]) + step - 1) // step for meter_id in history_panel.columns)
    progress = tqdm(total=total_windows, desc=progress_desc, unit="window")

    for meter_id in history_panel.columns:
        history = history_panel[meter_id].astype(float).copy()
        eval_series = eval_panel[meter_id].astype(float)
        cluster_id = int(meta.loc[meter_id, "cluster_id"])
        static_cat = [int(meta.loc[meter_id, "cluster_code"])]
        static_real = [float(meta.loc[meter_id, "log_mean_hourly_load"])]

        meter_preds: list[pd.DataFrame] = []
        for start_idx in range(0, len(eval_series), step):
            end_idx = min(start_idx + step, len(eval_series))
            horizon = end_idx - start_idx
            forecast_start = eval_series.index[start_idx]

            ds = ListDataset(
                [
                    {
                        "item_id": str(meter_id),
                        "start": history.index[0],
                        "target": history.to_numpy(),
                        "feat_static_cat": static_cat,
                        "feat_static_real": static_real,
                    }
                ],
                freq=config.freq,
            )
            if not dynamic_feat_df.empty:
                forecast_index = eval_series.index[start_idx:end_idx]
                dynamic_index = history.index.append(forecast_index)
                ds = ListDataset(
                    [
                        {
                            "item_id": str(meter_id),
                            "start": history.index[0],
                            "target": history.to_numpy(),
                            "feat_static_cat": static_cat,
                            "feat_static_real": static_real,
                            "feat_dynamic_real": dynamic_feat_df.loc[dynamic_index].to_numpy(dtype=float).T,
                        }
                    ],
                    freq=config.freq,
                )
            forecast = next(predictor.predict(ds, num_samples=config.num_parallel_samples))
            pred_mean = np.asarray(forecast.mean[:horizon], dtype=float)
            truth = eval_series.iloc[start_idx:end_idx].to_numpy(dtype=float)

            meter_preds.append(
                pd.DataFrame(
                    {
                        "timestamp": eval_series.index[start_idx:end_idx],
                        "meter_id": meter_id,
                        "cluster_id": cluster_id,
                        "y_true": truth,
                        "y_pred": pred_mean,
                        "forecast_start": forecast_start,
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


def summarize_metrics(pred_df: pd.DataFrame) -> dict:
    overall_mape = float(pred_df["ape"].dropna().mean())
    overall_epsilon_mape = float(pred_df["epsilon_ape"].mean())
    valid_df = pred_df.loc[pred_df["ape"].notna()].copy()
    overall_wmape = float(
        np.abs(valid_df["y_true"] - valid_df["y_pred"]).sum() / np.abs(valid_df["y_true"]).sum() * 100.0
    )

    by_cluster = (
        pred_df.groupby("cluster_id", as_index=True)["ape"]
        .mean()
        .dropna()
        .sort_index()
        .to_dict()
    )
    wmape_by_cluster = {}
    for cluster_id, cluster_df in valid_df.groupby("cluster_id"):
        denom = float(np.abs(cluster_df["y_true"]).sum())
        if denom == 0:
            continue
        wmape_by_cluster[str(int(cluster_id))] = float(
            np.abs(cluster_df["y_true"] - cluster_df["y_pred"]).sum() / denom * 100.0
        )
    epsilon_mape_by_cluster = (
        pred_df.groupby("cluster_id", as_index=True)["epsilon_ape"]
        .mean()
        .dropna()
        .sort_index()
        .to_dict()
    )
    by_meter = (
        pred_df.groupby("meter_id", as_index=True)["ape"]
        .mean()
        .dropna()
        .sort_index()
        .to_dict()
    )

    metrics = {
        "overall_mape": overall_mape,
        "overall_wmape": overall_wmape,
        "overall_epsilon_mape": overall_epsilon_mape,
        "cluster_mape": {str(int(k)): float(v) for k, v in by_cluster.items()},
        "cluster_wmape": wmape_by_cluster,
        "cluster_epsilon_mape": {str(int(k)): float(v) for k, v in epsilon_mape_by_cluster.items()},
        "n_predictions": int(len(pred_df)),
        "n_nonzero_targets": int(pred_df["ape"].notna().sum()),
        "n_meters": int(pred_df["meter_id"].nunique()),
        "meter_mape_head": dict(list((k, float(v)) for k, v in by_meter.items())[:10]),
    }
    return metrics


def save_outputs(
    pred_df: pd.DataFrame,
    metrics: dict,
    meta: pd.DataFrame,
    config: TrainConfig,
    eval_split: str = "validation",
) -> tuple[Path, Path]:
    out_dir = resolve_output_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = cluster_tag(config.cluster_ids)

    pred_path = out_dir / f"deepar_clusters_{tag}_{eval_split}_predictions.parquet"
    meta_path = out_dir / f"deepar_clusters_{tag}_{eval_split}_metadata.parquet"

    pred_df.to_parquet(pred_path, index=False)
    meta.to_parquet(meta_path, index=True)

    return pred_path, meta_path


def run_experiment(config: TrainConfig) -> dict:
    set_seed(config.seed)

    train_panel, test_panel, labels = load_cluster_panels(config)
    train_sub, val_panel = split_train_validation(train_panel, config)
    meta = build_metadata(train_panel, labels, config)
    dynamic_feat_df = build_dynamic_features(train_sub.index, val_panel.index, config)
    predictor = fit_predictor(train_sub, meta, dynamic_feat_df, config)
    pred_df = rolling_predict(
        predictor,
        train_sub,
        val_panel,
        meta,
        dynamic_feat_df,
        config,
        progress_desc="Validation rolling forecast",
    )
    metrics = summarize_metrics(pred_df)
    pred_path, meta_path = save_outputs(pred_df, metrics, meta, config, eval_split="validation")

    results = {
        "config": asdict(config),
        "metrics": metrics,
        "dynamic_feature_names": list(dynamic_feat_df.columns),
        "prediction_path": str(pred_path),
        "metadata_path": str(meta_path),
        "evaluation_split": "validation",
        "train_range": [str(train_sub.index.min()), str(train_sub.index.max())],
        "validation_range": [str(val_panel.index.min()), str(val_panel.index.max())],
        "test_range": [str(test_panel.index.min()), str(test_panel.index.max())],
    }
    return results


def sample_random_configs(
    base_config: TrainConfig,
    search_config: RandomSearchConfig,
) -> list[TrainConfig]:
    rng = np.random.default_rng(search_config.seed)
    configs: list[TrainConfig] = []

    for trial_idx in range(search_config.n_trials):
        config = TrainConfig(**asdict(base_config))
        config.context_length = int(rng.choice(search_config.context_length_choices))
        config.hidden_size = int(rng.choice(search_config.hidden_size_choices))
        config.dropout_rate = float(rng.choice(search_config.dropout_rate_choices))
        config.learning_rate = float(rng.choice(search_config.learning_rate_choices))
        config.epochs = int(rng.choice(search_config.epochs_choices))
        config.num_batches_per_epoch = int(rng.choice(search_config.num_batches_per_epoch_choices))
        config.batch_size = int(rng.choice(search_config.batch_size_choices))
        config.num_parallel_samples = int(rng.choice(search_config.num_parallel_samples_choices))
        config.seed = int(base_config.seed + trial_idx)
        configs.append(config)

    return configs


def summarize_search_results(search_results: list[dict]) -> pd.DataFrame:
    rows = []
    for idx, result in enumerate(search_results, start=1):
        cfg = result["config"]
        metrics = result["metrics"]
        row = {
            "trial": idx,
            "overall_mape": metrics["overall_mape"],
            "overall_wmape": metrics["overall_wmape"],
            "overall_epsilon_mape": metrics["overall_epsilon_mape"],
            "context_length": cfg["context_length"],
            "hidden_size": cfg["hidden_size"],
            "dropout_rate": cfg["dropout_rate"],
            "learning_rate": cfg["learning_rate"],
            "epochs": cfg["epochs"],
            "num_batches_per_epoch": cfg["num_batches_per_epoch"],
            "batch_size": cfg["batch_size"],
            "num_parallel_samples": cfg["num_parallel_samples"],
            "prediction_path": result["prediction_path"],
        }
        for cluster_id, value in result["metrics"]["cluster_mape"].items():
            row[f"cluster_{cluster_id}_mape"] = value
        for cluster_id, value in result["metrics"].get("cluster_wmape", {}).items():
            row[f"cluster_{cluster_id}_wmape"] = value
        for cluster_id, value in result["metrics"].get("cluster_epsilon_mape", {}).items():
            row[f"cluster_{cluster_id}_epsilon_mape"] = value
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values("overall_wmape", ascending=True).reset_index(drop=True)
    return summary_df


def run_random_search(
    base_config: TrainConfig,
    search_config: RandomSearchConfig,
) -> dict:
    sampled_configs = sample_random_configs(base_config, search_config)
    out_dir = resolve_search_output_dir(base_config, search_config)
    out_dir.mkdir(parents=True, exist_ok=True)

    search_results: list[dict] = []
    for trial_idx, trial_config in enumerate(sampled_configs, start=1):
        trial_config.output_dir = str(out_dir / f"trial_{trial_idx:02d}")
        print(
            f"[Trial {trial_idx}/{len(sampled_configs)}] "
            f"context={trial_config.context_length}, "
            f"hidden={trial_config.hidden_size}, "
            f"dropout={trial_config.dropout_rate}, "
            f"lr={trial_config.learning_rate}, "
            f"epochs={trial_config.epochs}, "
            f"batches={trial_config.num_batches_per_epoch}"
        )
        result = run_experiment(trial_config)
        search_results.append(result)

    summary_df = summarize_search_results(search_results)
    summary_path = out_dir / "random_search_summary.csv"
    results_path = out_dir / "random_search_results.json"
    summary_df.to_csv(summary_path, index=False)
    results_path.write_text(json.dumps(search_results, indent=2), encoding="utf-8")

    return {
        "summary_df": summary_df,
        "summary_path": str(summary_path),
        "results_path": str(results_path),
        "best_result": search_results[int(summary_df.index[0])] if len(summary_df) > 0 else None,
    }


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train DeepAR on selected hourly-load clusters.")
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--test-path", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--cluster-path", default=str(DEFAULT_CLUSTER_PATH))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cluster-ids", nargs="+", type=int, required=True)
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--context-length", type=int, default=24 * 14)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches-per-epoch", type=int, default=100)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--num-parallel-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holiday-country", default="PT")
    parser.add_argument("--validation-months", type=int, default=3)
    args = parser.parse_args()

    return TrainConfig(
        train_path=args.train_path,
        test_path=args.test_path,
        cluster_path=args.cluster_path,
        output_dir=args.output_dir,
        cluster_ids=tuple(args.cluster_ids),
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_batches_per_epoch=args.num_batches_per_epoch,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        dropout_rate=args.dropout_rate,
        num_parallel_samples=args.num_parallel_samples,
        seed=args.seed,
        holiday_country=args.holiday_country,
        validation_months=args.validation_months,
    )


def main() -> None:
    config = parse_args()
    results = run_experiment(config)
    tag = cluster_tag(config.cluster_ids)

    print("DeepAR training finished.")
    print(f"Cluster tag: {tag}")
    print(f"Evaluation split: {results['evaluation_split']}")
    print(f"Train range: {results['train_range'][0]} -> {results['train_range'][1]}")
    print(f"Validation range: {results['validation_range'][0]} -> {results['validation_range'][1]}")
    print(f"Test range: {results['test_range'][0]} -> {results['test_range'][1]}")
    print(f"Overall MAPE: {results['metrics']['overall_mape']:.4f}")
    print(f"Overall wMAPE: {results['metrics']['overall_wmape']:.4f}")
    print(f"Overall epsilon-MAPE: {results['metrics']['overall_epsilon_mape']:.4f}")
    print(f"Dynamic features: {results['dynamic_feature_names']}")
    print("Cluster MAPE:")
    for cluster_id, value in results["metrics"]["cluster_mape"].items():
        print(f"  Cluster {cluster_id}: {value:.4f}")
    print("Cluster wMAPE:")
    for cluster_id, value in results["metrics"]["cluster_wmape"].items():
        print(f"  Cluster {cluster_id}: {value:.4f}")
    print("Cluster epsilon-MAPE:")
    for cluster_id, value in results["metrics"]["cluster_epsilon_mape"].items():
        print(f"  Cluster {cluster_id}: {value:.4f}")
    print(f"Predictions saved to: {results['prediction_path']}")


if __name__ == "__main__":
    main()
