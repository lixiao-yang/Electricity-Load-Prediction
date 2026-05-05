from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "train_hourly_preprocessed.parquet"
TEST_PATH = PROJECT_ROOT / "data" / "test_hourly_preprocessed.parquet"

PERIOD_ORDER = ["P1", "P2", "P3"]
MODEL_COLOR = "#2ca02c"
BASELINE_COLOR = "#1f77b4"
FUTURE_COLOR = "#d62728"
ETS_BASELINE_LABEL = "ETS"


@dataclass(frozen=True)
class GroupConfig:
    group_key: str
    cluster_ids: tuple[int, ...]
    future_path: Path
    summary_path: Path | None = None
    prediction_path: Path | None = None
    metadata_path: Path | None = None
    eval_label: str = "validation"


GROUP_CONFIGS: dict[str, GroupConfig] = {
    "2_3": GroupConfig(
        group_key="2_3",
        cluster_ids=(2, 3),
        future_path=PROJECT_ROOT / "deepar" / "output" / "future_3months_predictions_2_3.parquet",
        summary_path=PROJECT_ROOT / "deepar" / "output" / "deepar_random_search_2_3" / "random_search_summary.csv",
        eval_label="validation",
    ),
    "7": GroupConfig(
        group_key="7",
        cluster_ids=(7,),
        future_path=PROJECT_ROOT / "deepar" / "output" / "future_3months_predictions_7.parquet",
        summary_path=PROJECT_ROOT / "deepar" / "output" / "deepar_random_search_7" / "random_search_summary.csv",
        eval_label="validation",
    ),
    "1_11": GroupConfig(
        group_key="1_11",
        cluster_ids=(1, 11),
        future_path=PROJECT_ROOT / "deepar" / "output" / "future_3months_predictions_1_11.parquet",
        summary_path=PROJECT_ROOT / "deepar" / "output" / "deepar_random_search_1_11" / "random_search_summary.csv",
        eval_label="validation",
    ),
}


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.titleweight"] = "bold"


def _save_fig(fig: plt.Figure, file_stem: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{file_stem}.png", dpi=180, bbox_inches="tight")


def _finish_fig(fig: plt.Figure, file_stem: str, out_dir: Path, show: bool = True) -> None:
    _save_fig(fig, file_stem=file_stem, out_dir=out_dir)
    if show:
        plt.show()
    plt.close(fig)


def _mape_0_100(y_true: pd.Series, y_pred: pd.Series, eps: float = 1.0) -> float:
    ape_pct = np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps) * 100.0
    return float(np.mean(np.clip(ape_pct, 0.0, 100.0)))


def _wmape_0_100(y_true: pd.Series, y_pred: pd.Series, eps: float = 1.0) -> float:
    return float(100.0 * np.sum(np.abs(y_true - y_pred)) / max(float(np.sum(np.abs(y_true))), eps))


def _cluster_label(cluster_id: int) -> str:
    return f"Cluster {cluster_id}"


def _cluster_palette(cluster_ids: list[int]) -> dict[str, tuple[float, float, float]]:
    colors = sns.color_palette("tab10", n_colors=max(len(cluster_ids), 3))
    return {_cluster_label(cluster_id): colors[idx] for idx, cluster_id in enumerate(cluster_ids)}


def _assign_period(timestamp: pd.Series) -> pd.Categorical:
    months = pd.Series(timestamp.dt.to_period("M").astype(str), index=timestamp.index)
    unique_months = sorted(months.dropna().unique().tolist())
    mapping = {month: PERIOD_ORDER[idx] for idx, month in enumerate(unique_months[: len(PERIOD_ORDER)])}
    phases = months.map(mapping).fillna(PERIOD_ORDER[min(len(unique_months), len(PERIOD_ORDER)) - 1])
    return pd.Categorical(phases, categories=PERIOD_ORDER, ordered=True)


def _resolve_prediction_path(path_str: str) -> Path:
    raw = Path(path_str)
    candidates = [
        PROJECT_ROOT / raw,
        PROJECT_ROOT / "deepar" / "output" / raw.name if raw.parent.name.startswith("trial_") else PROJECT_ROOT / raw,
        PROJECT_ROOT / "deepar" / "output" / raw.relative_to("outputs") if raw.parts and raw.parts[0] == "outputs" else PROJECT_ROOT / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve prediction path: {path_str}")


def _load_best_trial_paths(config: GroupConfig, selection_metric: str) -> tuple[pd.DataFrame, Path, Path]:
    if config.summary_path is None:
        if config.prediction_path is None or config.metadata_path is None:
            raise ValueError(f"Missing prediction metadata configuration for group {config.group_key}.")
        return pd.DataFrame(), config.prediction_path, config.metadata_path

    summary = pd.read_csv(config.summary_path).sort_values([selection_metric, "trial"], ascending=[True, True])
    best = summary.iloc[0]
    prediction_path = _resolve_prediction_path(str(best["prediction_path"]))
    metadata_name = prediction_path.name.replace("_validation_predictions.parquet", "_validation_metadata.parquet")
    metadata_name = metadata_name.replace("_predictions.parquet", "_metadata.parquet")
    metadata_path = prediction_path.with_name(metadata_name)
    return summary, prediction_path, metadata_path


def _load_panel(cluster_ids: tuple[int, ...]) -> pd.DataFrame:
    train_df = pd.read_parquet(TRAIN_PATH).sort_index()
    train_df.index = pd.to_datetime(train_df.index)
    cluster_table = pd.read_parquet(PROJECT_ROOT / "data" / "extended-clustering-high-cov" / "clusters_3models.parquet")
    selected_meters = cluster_table[cluster_table["cluster_kmeans"].isin(cluster_ids)].index.tolist()
    return train_df.loc[:, train_df.columns.intersection(selected_meters)].copy()


def _prepare_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["forecast_start"] = pd.to_datetime(out["forecast_start"])
    out["phase"] = _assign_period(out["timestamp"])
    out["horizon_step"] = ((out["timestamp"] - out["forecast_start"]) / pd.Timedelta(hours=1)).astype(int) + 1
    out["user_id"] = out["meter_id"].astype(str)
    out["cluster_label"] = out["cluster_id"].map(_cluster_label)
    return out.sort_values(["cluster_id", "user_id", "timestamp"]).reset_index(drop=True)


def _build_baseline_lookup(panel: pd.DataFrame, cluster_ids: tuple[int, ...]) -> pd.DataFrame:
    lag24 = panel.shift(24)
    lookup = (
        lag24.stack()
        .rename("baseline_prediction")
        .reset_index()
        .rename(columns={"level_0": "timestamp", "level_1": "meter_id"})
    )
    cluster_table = pd.read_parquet(PROJECT_ROOT / "data" / "extended-clustering-high-cov" / "clusters_3models.parquet")
    cluster_table = cluster_table[cluster_table["cluster_kmeans"].isin(cluster_ids)][["cluster_kmeans"]].rename(
        columns={"cluster_kmeans": "cluster_id"}
    )
    lookup = lookup.merge(cluster_table, left_on="meter_id", right_index=True, how="left")
    return lookup


def _load_ets_baseline_predictions(cluster_ids: tuple[int, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for cluster_id in cluster_ids:
        candidates = [
            PROJECT_ROOT / "deepar" / "output" / f"ets_direct_cluster_{cluster_id}" / f"ets_cluster_{cluster_id}_validation_predictions.parquet",
            PROJECT_ROOT / "deepar" / "output" / f"ets_cluster_{cluster_id}" / f"ets_cluster_{cluster_id}_validation_predictions.parquet",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(f"Missing ETS baseline predictions for cluster {cluster_id}: {candidates[0]}")
        frame = pd.read_parquet(path).copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frames.append(
            frame[["timestamp", "meter_id", "cluster_id", "y_pred"]].rename(columns={"y_pred": "baseline_prediction"})
        )
    return pd.concat(frames, ignore_index=True)


def build_metrics_overall(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster_id, group in predictions.groupby("cluster_id"):
        rows.append(
            {
                "cluster_id": cluster_id,
                "MAPE_0_100": _mape_0_100(group["y_true"], group["y_pred"]),
                "epsilon_MAPE": float(group["epsilon_ape"].mean()),
                "WMAPE_0_100": _wmape_0_100(group["y_true"], group["y_pred"]),
                "n_obs": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def build_metrics_by_period(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (cluster_id, period), group in predictions.groupby(["cluster_id", "phase"], observed=False):
        rows.append(
            {
                "cluster_id": cluster_id,
                "period": period,
                "MAPE_0_100": _mape_0_100(group["y_true"], group["y_pred"]),
                "WMAPE_0_100": _wmape_0_100(group["y_true"], group["y_pred"]),
                "n_obs": int(len(group)),
            }
        )
    out = pd.DataFrame(rows)
    out["period"] = pd.Categorical(out["period"], categories=PERIOD_ORDER, ordered=True)
    return out.sort_values(["cluster_id", "period"]).reset_index(drop=True)


def build_user_period_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (cluster_id, user_id, period), group in predictions.groupby(["cluster_id", "user_id", "phase"], observed=False):
        rows.append(
            {
                "cluster_id": cluster_id,
                "user_id": user_id,
                "period": period,
                "MAPE_0_100": _mape_0_100(group["y_true"], group["y_pred"]),
                "WMAPE_0_100": _wmape_0_100(group["y_true"], group["y_pred"]),
                "n_obs": int(len(group)),
            }
        )
    out = pd.DataFrame(rows)
    out["period"] = pd.Categorical(out["period"], categories=PERIOD_ORDER, ordered=True)
    return out.sort_values(["cluster_id", "period", "user_id"]).reset_index(drop=True)


def build_cluster_hourly_comparison(predictions: pd.DataFrame, baseline_lookup: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(
        baseline_lookup,
        on=["timestamp", "meter_id", "cluster_id"],
        how="left",
    )
    missing = int(merged["baseline_prediction"].isna().sum())
    if missing:
        raise ValueError(f"Baseline predictions missing for {missing} prediction rows after merge.")
    out = (
        merged.groupby(["cluster_id", "cluster_label", "timestamp", "phase"], as_index=False, observed=False)
        .agg(
            actual=("y_true", "sum"),
            prediction=("y_pred", "sum"),
            baseline_prediction=("baseline_prediction", "sum"),
        )
        .sort_values(["cluster_id", "timestamp"])
    )
    out["date"] = out["timestamp"].dt.normalize()
    out["baseline_residual"] = out["actual"] - out["baseline_prediction"]
    out["model_residual"] = out["actual"] - out["prediction"]
    out["baseline_ape_0_100"] = np.clip(
        np.abs(out["actual"] - out["baseline_prediction"]) / np.maximum(np.abs(out["actual"]), 1.0) * 100.0,
        0.0,
        100.0,
    )
    out["model_ape_0_100"] = np.clip(
        np.abs(out["actual"] - out["prediction"]) / np.maximum(np.abs(out["actual"]), 1.0) * 100.0,
        0.0,
        100.0,
    )
    return out


def build_cluster_daily_aggregate(predictions: pd.DataFrame) -> pd.DataFrame:
    out = (
        predictions.assign(date=predictions["timestamp"].dt.normalize())
        .groupby(["cluster_id", "date"], as_index=False)
        .agg(actual=("y_true", "sum"), prediction=("y_pred", "sum"))
        .sort_values(["cluster_id", "date"])
    )
    out["actual_roll7"] = out.groupby("cluster_id")["actual"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    out["prediction_roll7"] = out.groupby("cluster_id")["prediction"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    return out


def build_horizon_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (cluster_id, horizon_step), group in predictions.groupby(["cluster_id", "horizon_step"]):
        rows.append(
            {
                "cluster_id": cluster_id,
                "horizon_step": horizon_step,
                "horizon_day": int((int(horizon_step) - 1) // 24 + 1),
                "MAPE_0_100": _mape_0_100(group["y_true"], group["y_pred"]),
                "WMAPE_0_100": _wmape_0_100(group["y_true"], group["y_pred"]),
                "n_obs": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["cluster_id", "horizon_step"]).reset_index(drop=True)


def build_baseline_metrics_overall(cluster_hourly_comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster_id, group in cluster_hourly_comparison.groupby("cluster_id"):
        baseline_epsilon = (
            np.abs(group["actual"] - group["baseline_prediction"]) / (np.abs(group["actual"]) + 1.0) * 100.0
        )
        rows.append(
            {
                "cluster_id": cluster_id,
                "MAPE_0_100": _mape_0_100(group["actual"], group["baseline_prediction"]),
                "epsilon_MAPE": float(baseline_epsilon.mean()),
                "WMAPE_0_100": _wmape_0_100(group["actual"], group["baseline_prediction"]),
                "n_obs": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def build_baseline_metrics_by_period(cluster_hourly_comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (cluster_id, period), group in cluster_hourly_comparison.groupby(["cluster_id", "phase"], observed=False):
        rows.append(
            {
                "cluster_id": cluster_id,
                "period": period,
                "MAPE_0_100": _mape_0_100(group["actual"], group["baseline_prediction"]),
                "WMAPE_0_100": _wmape_0_100(group["actual"], group["baseline_prediction"]),
                "n_obs": int(len(group)),
            }
        )
    out = pd.DataFrame(rows)
    out["period"] = pd.Categorical(out["period"], categories=PERIOD_ORDER, ordered=True)
    return out.sort_values(["cluster_id", "period"]).reset_index(drop=True)


def build_evaluation_summary_table(
    model_metrics_overall: pd.DataFrame,
    model_metrics_by_period: pd.DataFrame,
    baseline_metrics_overall: pd.DataFrame,
    baseline_metrics_by_period: pd.DataFrame,
) -> pd.DataFrame:
    model_overall = model_metrics_overall.rename(
        columns={"epsilon_MAPE": "overall_epsilon_mape", "WMAPE_0_100": "overall_wmape"}
    )
    baseline_overall = baseline_metrics_overall.rename(
        columns={"epsilon_MAPE": "overall_epsilon_mape", "WMAPE_0_100": "overall_wmape"}
    )

    model_period = (
        model_metrics_by_period.pivot(index="cluster_id", columns="period", values="MAPE_0_100")
        .rename(columns={period: f"{period}_mape" for period in PERIOD_ORDER})
        .reset_index()
    )
    baseline_period = (
        baseline_metrics_by_period.pivot(index="cluster_id", columns="period", values="MAPE_0_100")
        .rename(columns={period: f"{period}_mape" for period in PERIOD_ORDER})
        .reset_index()
    )

    model_table = model_overall.merge(model_period, on="cluster_id", how="left").assign(model="DeepAR")
    baseline_table = baseline_overall.merge(baseline_period, on="cluster_id", how="left").assign(model=ETS_BASELINE_LABEL)
    summary = pd.concat([baseline_table, model_table], ignore_index=True)

    ordered_cols = ["cluster_id", "model", "overall_epsilon_mape", "overall_wmape", "P1_mape", "P2_mape", "P3_mape"]
    rename_map = {
        "cluster_id": "Cluster",
        "model": "Model",
        "overall_epsilon_mape": "Overall epsilon-MAPE",
        "overall_wmape": "Overall WMAPE",
        "P1_mape": "P1 MAPE",
        "P2_mape": "P2 MAPE",
        "P3_mape": "P3 MAPE",
    }
    summary = summary[ordered_cols].rename(columns=rename_map).sort_values(["Cluster", "Model"]).reset_index(drop=True)
    return summary


def select_random_user(predictions: pd.DataFrame, random_seed: int = 42) -> dict[str, Any]:
    user_map = predictions[["user_id", "cluster_id"]].drop_duplicates().sort_values(["cluster_id", "user_id"])
    sampled = user_map.sample(n=1, random_state=random_seed).iloc[0]
    return {"user_id": str(sampled["user_id"]), "cluster_id": int(sampled["cluster_id"])}


def render_evaluation_summary_table(summary_table: pd.DataFrame, out_path: Path, title: str, show: bool = True) -> None:
    _set_style()
    display_df = summary_table.copy()
    metric_cols = [col for col in display_df.columns if col not in {"Group", "Cluster", "Model"}]
    for col in metric_cols:
        display_df[col] = display_df[col].map(lambda v: f"{float(v):.2f}%")

    fig_height = 0.65 + 0.42 * (len(display_df) + 1)
    fig_width = 12.8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    if "Group" in display_df.columns:
        col_widths = [0.09, 0.08, 0.13, 0.17, 0.15, 0.10, 0.10, 0.10]
    else:
        col_widths = [0.09, 0.13, 0.18, 0.16, 0.11, 0.11, 0.11]
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        cell.visible_edges = "horizontal"
        cell.set_linewidth(0.8 if row == 0 else 0.5)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f2f2")
        elif display_df.iloc[row - 1]["Model"] == "DeepAR":
            cell.set_facecolor("#f7fbff")
        else:
            cell.set_facecolor("white")

    ax.set_title(title, fontsize=12, pad=16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_overall_metrics(metrics_overall: pd.DataFrame, out_dir: Path, title_prefix: str, cluster_ids: list[int], show: bool) -> None:
    _set_style()
    plot_df = metrics_overall.copy()
    plot_df["cluster_label"] = plot_df["cluster_id"].map(_cluster_label)
    palette = _cluster_palette(cluster_ids)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.barplot(data=plot_df, x="cluster_label", y="MAPE_0_100", hue="cluster_label", palette=palette, legend=False, ax=axes[0])
    axes[0].set_title(f"{title_prefix}: Overall Clipped MAPE by Cluster")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("MAPE_0_100 (%)")

    sns.barplot(data=plot_df, x="cluster_label", y="WMAPE_0_100", hue="cluster_label", palette=palette, legend=False, ax=axes[1])
    axes[1].set_title(f"{title_prefix}: Overall WMAPE by Cluster")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("WMAPE_0_100 (%)")
    fig.tight_layout()
    _finish_fig(fig, "01_overall_metrics", out_dir, show=show)


def plot_period_metrics(metrics_by_period: pd.DataFrame, out_dir: Path, title_prefix: str, cluster_ids: list[int], show: bool) -> None:
    _set_style()
    plot_df = metrics_by_period.copy()
    plot_df["cluster_label"] = plot_df["cluster_id"].map(_cluster_label)
    palette = _cluster_palette(cluster_ids)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    sns.barplot(data=plot_df, x="period", y="MAPE_0_100", hue="cluster_label", palette=palette, order=PERIOD_ORDER, ax=axes[0])
    axes[0].set_title(f"{title_prefix}: MAPE_0_100 by Period")
    axes[0].set_xlabel("Period")
    axes[0].set_ylabel("MAPE_0_100 (%)")
    axes[0].legend(title="Cluster")

    sns.barplot(data=plot_df, x="period", y="WMAPE_0_100", hue="cluster_label", palette=palette, order=PERIOD_ORDER, ax=axes[1])
    axes[1].set_title(f"{title_prefix}: WMAPE by Period")
    axes[1].set_xlabel("Period")
    axes[1].set_ylabel("WMAPE_0_100 (%)")
    axes[1].legend(title="Cluster")
    fig.tight_layout()
    _finish_fig(fig, "02_period_metrics", out_dir, show=show)


def plot_cluster_daily_aggregate(cluster_daily_aggregate: pd.DataFrame, out_dir: Path, cluster_ids: list[int], show: bool) -> None:
    _set_style()
    palette = _cluster_palette(cluster_ids)
    fig, axes = plt.subplots(len(cluster_ids), 1, figsize=(14, 4 * len(cluster_ids)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, cluster_id in zip(axes, cluster_ids):
        sub = cluster_daily_aggregate[cluster_daily_aggregate["cluster_id"] == cluster_id].copy()
        color = palette[_cluster_label(cluster_id)]
        ax.plot(sub["date"], sub["actual"], color=color, alpha=0.25, linewidth=1.2, label="Actual daily")
        ax.plot(sub["date"], sub["prediction"], color=MODEL_COLOR, alpha=0.25, linewidth=1.2, label="Predicted daily")
        ax.plot(sub["date"], sub["actual_roll7"], color=color, linewidth=2.2, label="Actual 7d mean")
        ax.plot(sub["date"], sub["prediction_roll7"], color=MODEL_COLOR, linewidth=2.2, label="Predicted 7d mean")
        ax.set_title(f"{_cluster_label(cluster_id)} Daily Aggregate: Actual vs DeepAR Prediction")
        ax.set_ylabel("Daily total load")
        ax.legend(ncol=2, fontsize=8)

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    _finish_fig(fig, "03_cluster_daily_aggregate", out_dir, show=show)


def plot_user_period_distribution(user_period_metrics: pd.DataFrame, out_dir: Path, cluster_ids: list[int], title_prefix: str, show: bool) -> pd.DataFrame:
    _set_style()
    plot_df = user_period_metrics.copy()
    plot_df["cluster_label"] = plot_df["cluster_id"].map(_cluster_label)
    palette = _cluster_palette(cluster_ids)
    summary = (
        plot_df.groupby(["cluster_id", "period"], as_index=False, observed=False)
        .agg(
            median_MAPE_0_100=("MAPE_0_100", "median"),
            mean_MAPE_0_100=("MAPE_0_100", "mean"),
            median_WMAPE_0_100=("WMAPE_0_100", "median"),
            n_users=("user_id", "nunique"),
        )
        .sort_values(["cluster_id", "period"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.boxplot(data=plot_df, x="period", y="MAPE_0_100", hue="cluster_label", order=PERIOD_ORDER, palette=palette, ax=axes[0])
    axes[0].set_title(f"{title_prefix}: User-level MAPE_0_100 Distribution")
    axes[0].set_xlabel("Period")
    axes[0].set_ylabel("MAPE_0_100 (%)")
    axes[0].legend(title="Cluster")

    sns.boxplot(data=plot_df, x="period", y="WMAPE_0_100", hue="cluster_label", order=PERIOD_ORDER, palette=palette, ax=axes[1])
    axes[1].set_title(f"{title_prefix}: User-level WMAPE Distribution")
    axes[1].set_xlabel("Period")
    axes[1].set_ylabel("WMAPE_0_100 (%)")
    axes[1].legend(title="Cluster")
    fig.tight_layout()
    _finish_fig(fig, "04_user_period_distribution", out_dir, show=show)
    return summary


def plot_horizon_error_profile(horizon_metrics: pd.DataFrame, out_dir: Path, cluster_ids: list[int], title_prefix: str, show: bool) -> None:
    _set_style()
    plot_df = horizon_metrics.copy()
    plot_df["cluster_label"] = plot_df["cluster_id"].map(_cluster_label)
    palette = _cluster_palette(cluster_ids)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    sns.lineplot(data=plot_df, x="horizon_step", y="MAPE_0_100", hue="cluster_label", palette=palette, ax=axes[0])
    axes[0].set_title(f"{title_prefix}: Clipped MAPE by Forecast Horizon")
    axes[0].set_xlabel("Horizon step (hour)")
    axes[0].set_ylabel("MAPE_0_100 (%)")
    axes[0].legend(title="Cluster")

    sns.lineplot(data=plot_df, x="horizon_step", y="WMAPE_0_100", hue="cluster_label", palette=palette, ax=axes[1])
    axes[1].set_title(f"{title_prefix}: WMAPE by Forecast Horizon")
    axes[1].set_xlabel("Horizon step (hour)")
    axes[1].set_ylabel("WMAPE_0_100 (%)")
    axes[1].legend(title="Cluster")
    fig.tight_layout()
    _finish_fig(fig, "05_horizon_error_profile", out_dir, show=show)


def plot_cluster_scatter_comparison(cluster_hourly_comparison: pd.DataFrame, cluster_id: int, out_dir: Path, show: bool) -> None:
    _set_style()
    sub = cluster_hourly_comparison[cluster_hourly_comparison["cluster_id"] == cluster_id].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharex=True, sharey=True)
    max_value = float(max(sub["actual"].max(), sub["baseline_prediction"].max(), sub["prediction"].max()) * 1.02)
    plot_specs = [("baseline_prediction", "ETS Baseline", BASELINE_COLOR), ("prediction", "DeepAR Forecast", MODEL_COLOR)]

    for ax, (col, label, color) in zip(axes, plot_specs):
        ax.scatter(sub["actual"], sub[col], alpha=0.45, s=16, color=color)
        ax.plot([0, max_value], [0, max_value], linestyle="--", color="red", linewidth=1.5)
        ax.set_title(f"{_cluster_label(cluster_id)}: Actual vs Predicted for {label}")
        ax.set_xlabel("Actual load")
        ax.set_ylabel("Predicted load")
        ax.ticklabel_format(style="plain", axis="both", useOffset=False)

    fig.tight_layout()
    _finish_fig(fig, f"{cluster_id:02d}_cluster_{cluster_id}_scatter_comparison", out_dir, show=show)


def plot_cluster_residual_trend(cluster_hourly_comparison: pd.DataFrame, cluster_id: int, out_dir: Path, show: bool) -> None:
    _set_style()
    sub = cluster_hourly_comparison[cluster_hourly_comparison["cluster_id"] == cluster_id].copy()
    sub["baseline_residual_roll"] = sub["baseline_residual"].rolling(24, min_periods=24).mean()
    sub["model_residual_roll"] = sub["model_residual"].rolling(24, min_periods=24).mean()

    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.85)
    ax.plot(sub["timestamp"], sub["baseline_residual"], color="#9ecae1", alpha=0.55, linewidth=0.9, label="Baseline residual")
    ax.plot(sub["timestamp"], sub["model_residual"], color="#a1d99b", alpha=0.55, linewidth=0.9, label="Model residual")
    ax.plot(sub["timestamp"], sub["baseline_residual_roll"], color=BASELINE_COLOR, linewidth=2.0, label="Baseline residual 24h mean")
    ax.plot(sub["timestamp"], sub["model_residual_roll"], color=MODEL_COLOR, linewidth=2.0, label="Model residual 24h mean")
    ax.set_title(f"{_cluster_label(cluster_id)}: Residual Trend (Actual Minus Forecast)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Residual")
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    _finish_fig(fig, f"{cluster_id:02d}_cluster_{cluster_id}_residual_trend", out_dir, show=show)


def plot_cluster_actual_vs_forecast_with_intervals(cluster_hourly_comparison: pd.DataFrame, cluster_id: int, out_dir: Path, show: bool) -> None:
    _set_style()
    sub = cluster_hourly_comparison[cluster_hourly_comparison["cluster_id"] == cluster_id].copy()
    z95 = 1.96
    baseline_sigma = sub["baseline_residual"].expanding(min_periods=24).std().shift(1).fillna(float(sub["baseline_residual"].std(ddof=0)))
    model_sigma = sub["model_residual"].expanding(min_periods=24).std().shift(1).fillna(float(sub["model_residual"].std(ddof=0)))

    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.plot(sub["timestamp"], sub["actual"], color="#4c72b0", linewidth=1.6, label="Actual load")
    ax.plot(sub["timestamp"], sub["baseline_prediction"], color=BASELINE_COLOR, linewidth=1.4, label="ETS Baseline")
    ax.fill_between(sub["timestamp"], np.clip(sub["baseline_prediction"] - z95 * baseline_sigma, 0.0, None), sub["baseline_prediction"] + z95 * baseline_sigma, color=BASELINE_COLOR, alpha=0.18)
    ax.plot(sub["timestamp"], sub["prediction"], color=MODEL_COLOR, linewidth=1.4, label="DeepAR Forecast")
    ax.fill_between(sub["timestamp"], np.clip(sub["prediction"] - z95 * model_sigma, 0.0, None), sub["prediction"] + z95 * model_sigma, color=MODEL_COLOR, alpha=0.18)
    ax.set_title(f"{_cluster_label(cluster_id)}: Actual vs Forecast with Residual-Based 95% Intervals")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Hourly electricity consumption")
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    ax.legend()
    fig.tight_layout()
    _finish_fig(fig, f"{cluster_id:02d}_cluster_{cluster_id}_actual_vs_forecast_intervals", out_dir, show=show)


def plot_cluster_rolling_mape(cluster_hourly_comparison: pd.DataFrame, cluster_id: int, out_dir: Path, show: bool) -> None:
    _set_style()
    sub = cluster_hourly_comparison[cluster_hourly_comparison["cluster_id"] == cluster_id].copy()
    sub["baseline_rolling_mape"] = sub["baseline_ape_0_100"].rolling(24, min_periods=24).mean()
    sub["model_rolling_mape"] = sub["model_ape_0_100"].rolling(24, min_periods=24).mean()

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(sub["timestamp"], sub["baseline_rolling_mape"], color=BASELINE_COLOR, linewidth=1.8, label="ETS Baseline")
    ax.plot(sub["timestamp"], sub["model_rolling_mape"], color=MODEL_COLOR, linewidth=1.8, label="DeepAR Forecast")
    ax.set_title(f"{_cluster_label(cluster_id)}: Rolling 24-Hour MAPE")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Rolling MAPE (%)")
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    ax.legend(title="Forecast")
    fig.tight_layout()
    _finish_fig(fig, f"{cluster_id:02d}_cluster_{cluster_id}_rolling_24h_mape", out_dir, show=show)


def plot_cluster_error_distribution_by_period(cluster_hourly_comparison: pd.DataFrame, cluster_id: int, out_dir: Path, show: bool) -> pd.DataFrame:
    _set_style()
    sub = cluster_hourly_comparison[cluster_hourly_comparison["cluster_id"] == cluster_id].copy()
    plot_df = pd.concat(
        [
            sub[["phase", "baseline_ape_0_100"]].rename(columns={"baseline_ape_0_100": "ape_0_100"}).assign(forecast="ETS Baseline"),
            sub[["phase", "model_ape_0_100"]].rename(columns={"model_ape_0_100": "ape_0_100"}).assign(forecast="DeepAR Forecast"),
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(12, 4.8))
    sns.boxplot(data=plot_df, x="phase", y="ape_0_100", hue="forecast", order=PERIOD_ORDER, palette=[BASELINE_COLOR, MODEL_COLOR], ax=ax)
    ax.set_title(f"{_cluster_label(cluster_id)}: Forecast Error Distribution by Period")
    ax.set_xlabel("Period")
    ax.set_ylabel("APE clipped to 0-100 (%)")
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    ax.legend(title="Forecast")
    fig.tight_layout()
    _finish_fig(fig, f"{cluster_id:02d}_cluster_{cluster_id}_error_distribution_by_period", out_dir, show=show)

    summary = (
        plot_df.groupby(["forecast", "phase"], as_index=False, observed=False)
        .agg(
            mean_ape_0_100=("ape_0_100", "mean"),
            median_ape_0_100=("ape_0_100", "median"),
            p90_ape_0_100=("ape_0_100", lambda s: float(np.quantile(s, 0.90))),
            n_obs=("ape_0_100", "size"),
        )
        .sort_values(["phase", "forecast"])
    )
    summary["cluster_id"] = cluster_id
    return summary


def plot_random_user_history_and_forecast(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    future_predictions: pd.DataFrame,
    user_id: str,
    cluster_id: int,
    out_dir: Path,
    history_days: int,
    show: bool,
) -> None:
    _set_style()
    meter_series = panel[user_id].dropna().rename("target_load").reset_index().rename(columns={"index": "timestamp"})
    eval_start = predictions["timestamp"].min()
    history = meter_series[meter_series["timestamp"] < eval_start].copy()
    history_end = history["timestamp"].max()
    history_start = history_end - pd.Timedelta(days=history_days)
    history_window = history[history["timestamp"] >= history_start].copy()

    test_sub = predictions[(predictions["user_id"] == user_id) & (predictions["cluster_id"] == cluster_id)].sort_values("timestamp").copy()
    future_sub = future_predictions[(future_predictions["meter_id"] == user_id) & (future_predictions["cluster_id"] == cluster_id)].sort_values("timestamp").copy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    axes[0].plot(history_window["timestamp"], history_window["target_load"], color="#7f7f7f", linewidth=1.2, label="History")
    axes[0].plot(test_sub["timestamp"], test_sub["y_true"], color="#4c72b0", linewidth=1.8, label=f"{predictions['phase'].iloc[0]} actual")
    axes[0].plot(test_sub["timestamp"], test_sub["y_pred"], color=MODEL_COLOR, linewidth=1.8, label="DeepAR prediction")
    axes[0].set_title(f"{_cluster_label(cluster_id)} Random User {user_id}: History and Evaluation Forecast")
    axes[0].set_xlabel("Timestamp")
    axes[0].set_ylabel("Load")
    axes[0].legend()

    tail = test_sub.tail(24 * 7).copy()
    axes[1].plot(tail["timestamp"], tail["y_true"], color="#4c72b0", linewidth=1.8, label="Eval actual (last 7d)")
    axes[1].plot(tail["timestamp"], tail["y_pred"], color=MODEL_COLOR, linewidth=1.8, label="Eval prediction (last 7d)")
    axes[1].plot(future_sub["timestamp"], future_sub["y_pred"], color=FUTURE_COLOR, linewidth=2.0, label="Future 3-month prediction")
    axes[1].set_title(f"{_cluster_label(cluster_id)} Random User {user_id}: Future Forecast Continuation")
    axes[1].set_xlabel("Timestamp")
    axes[1].set_ylabel("Load")
    axes[1].legend()
    fig.tight_layout()
    _finish_fig(fig, "06_random_user_history_and_future", out_dir, show=show)


def run_group_analysis(
    group_key: str,
    image_root: str | Path = PROJECT_ROOT / "images" / "deepar",
    selection_metric: str = "overall_wmape",
    random_seed: int = 42,
    history_days: int = 60,
    show: bool = True,
) -> dict[str, Any]:
    config = GROUP_CONFIGS[group_key]
    summary, prediction_path, metadata_path = _load_best_trial_paths(config, selection_metric=selection_metric)
    predictions = _prepare_predictions(pd.read_parquet(prediction_path))
    metadata = pd.read_parquet(metadata_path)
    future_predictions = pd.read_parquet(config.future_path)
    future_predictions["timestamp"] = pd.to_datetime(future_predictions["timestamp"])

    panel = _load_panel(config.cluster_ids)
    baseline_lookup = _load_ets_baseline_predictions(config.cluster_ids)
    metrics_overall = build_metrics_overall(predictions)
    metrics_by_period = build_metrics_by_period(predictions)
    user_period_metrics = build_user_period_metrics(predictions)
    cluster_hourly_comparison = build_cluster_hourly_comparison(predictions, baseline_lookup)
    baseline_metrics_overall = build_baseline_metrics_overall(cluster_hourly_comparison=cluster_hourly_comparison)
    baseline_metrics_by_period = build_baseline_metrics_by_period(cluster_hourly_comparison=cluster_hourly_comparison)
    cluster_daily_aggregate = build_cluster_daily_aggregate(predictions)
    horizon_metrics = build_horizon_metrics(predictions)
    selected_user = select_random_user(predictions, random_seed=random_seed)

    out_dir = Path(image_root) / group_key
    cluster_ids = list(config.cluster_ids)
    title_prefix = f"DeepAR {group_key}"
    evaluation_summary_table = build_evaluation_summary_table(
        model_metrics_overall=metrics_overall,
        model_metrics_by_period=metrics_by_period,
        baseline_metrics_overall=baseline_metrics_overall,
        baseline_metrics_by_period=baseline_metrics_by_period,
    )

    plot_overall_metrics(metrics_overall, out_dir, title_prefix, cluster_ids, show)
    plot_period_metrics(metrics_by_period, out_dir, title_prefix, cluster_ids, show)
    plot_cluster_daily_aggregate(cluster_daily_aggregate, out_dir, cluster_ids, show)
    user_period_summary = plot_user_period_distribution(user_period_metrics, out_dir, cluster_ids, title_prefix, show)
    plot_horizon_error_profile(horizon_metrics, out_dir, cluster_ids, title_prefix, show)

    cluster_period_error_summaries = []
    for cluster_id in cluster_ids:
        plot_cluster_scatter_comparison(cluster_hourly_comparison, cluster_id, out_dir, show)
        plot_cluster_residual_trend(cluster_hourly_comparison, cluster_id, out_dir, show)
        plot_cluster_actual_vs_forecast_with_intervals(cluster_hourly_comparison, cluster_id, out_dir, show)
        plot_cluster_rolling_mape(cluster_hourly_comparison, cluster_id, out_dir, show)
        cluster_period_error_summaries.append(plot_cluster_error_distribution_by_period(cluster_hourly_comparison, cluster_id, out_dir, show))

    plot_random_user_history_and_forecast(
        panel=panel,
        predictions=predictions,
        future_predictions=future_predictions,
        user_id=selected_user["user_id"],
        cluster_id=selected_user["cluster_id"],
        out_dir=out_dir,
        history_days=history_days,
        show=show,
    )
    render_evaluation_summary_table(
        summary_table=evaluation_summary_table,
        out_path=out_dir / "00_evaluation_summary_table.png",
        title=f"DeepAR evaluation summary against the ETS baseline ({group_key})",
        show=show,
    )

    return {
        "group_key": group_key,
        "eval_label": config.eval_label,
        "selection_metric": selection_metric,
        "selected_prediction_path": prediction_path,
        "trial_summary": summary,
        "metadata": metadata,
        "predictions": predictions,
        "future_predictions": future_predictions,
        "metrics_overall": metrics_overall,
        "metrics_by_period": metrics_by_period,
        "baseline_metrics_overall": baseline_metrics_overall,
        "baseline_metrics_by_period": baseline_metrics_by_period,
        "evaluation_summary_table": evaluation_summary_table,
        "user_period_metrics": user_period_metrics,
        "user_period_summary": user_period_summary,
        "daily_cluster_aggregate": cluster_daily_aggregate,
        "cluster_hourly_comparison": cluster_hourly_comparison,
        "cluster_period_error_summary": pd.concat(cluster_period_error_summaries, ignore_index=True),
        "horizon_metrics": horizon_metrics,
        "selected_user": selected_user,
        "image_dir": out_dir,
    }


def run_full_analysis(
    image_root: str | Path = PROJECT_ROOT / "images" / "deepar",
    selection_metric: str = "overall_wmape",
    random_seed: int = 42,
    history_days: int = 60,
    show: bool = True,
) -> dict[str, dict[str, Any]]:
    bundle = {
        group_key: run_group_analysis(
            group_key=group_key,
            image_root=image_root,
            selection_metric=selection_metric,
            random_seed=random_seed,
            history_days=history_days,
            show=show,
        )
        for group_key in GROUP_CONFIGS
    }
    combined_summary_table = build_combined_evaluation_summary_table(bundle)
    render_evaluation_summary_table(
        summary_table=combined_summary_table,
        out_path=Path(image_root) / "deepar_evaluation_summary_table.png",
        title="DeepAR evaluation summary against the ETS baseline",
        show=show,
    )
    return bundle


def build_summary_tables(bundle: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows = []
    period_rows = []
    for group_key, group_bundle in bundle.items():
        overall = group_bundle["metrics_overall"].copy()
        overall["group_key"] = group_key
        overall["eval_label"] = group_bundle["eval_label"]
        model_rows.append(overall)

        period = group_bundle["metrics_by_period"].copy()
        period["group_key"] = group_key
        period["eval_label"] = group_bundle["eval_label"]
        period_rows.append(period)

    return (
        pd.concat(model_rows, ignore_index=True).sort_values(["group_key", "cluster_id"]).reset_index(drop=True),
        pd.concat(period_rows, ignore_index=True).sort_values(["group_key", "cluster_id", "period"]).reset_index(drop=True),
    )


def build_combined_evaluation_summary_table(bundle: dict[str, dict[str, Any]]) -> pd.DataFrame:
    frames = []
    for group_key, group_bundle in bundle.items():
        frame = group_bundle["evaluation_summary_table"].copy()
        frame["Group"] = group_key
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    return combined[["Group", "Cluster", "Model", "Overall epsilon-MAPE", "Overall WMAPE", "P1 MAPE", "P2 MAPE", "P3 MAPE"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze DeepAR forecasting outputs and save figures.")
    parser.add_argument("--group", default="all", choices=["all", *GROUP_CONFIGS.keys()], help="Model group to analyze.")
    parser.add_argument("--image-root", default=str(PROJECT_ROOT / "images" / "deepar"), help="Directory for saved images.")
    parser.add_argument("--selection-metric", default="overall_wmape", help="Metric used to select the best random-search trial.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for sampled user visualization.")
    parser.add_argument("--history-days", type=int, default=60, help="History window for the sampled user plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.group == "all":
        run_full_analysis(
            image_root=args.image_root,
            selection_metric=args.selection_metric,
            random_seed=args.random_seed,
            history_days=args.history_days,
            show=False,
        )
    else:
        run_group_analysis(
            group_key=args.group,
            image_root=args.image_root,
            selection_metric=args.selection_metric,
            random_seed=args.random_seed,
            history_days=args.history_days,
            show=False,
        )


if __name__ == "__main__":
    main()
