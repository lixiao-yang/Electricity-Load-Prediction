from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PERIOD_ORDER = ["P1", "P2", "P3"]
CLUSTER_ORDER = [10, 12]
CLUSTER_LABELS = {10: "Cluster 10", 12: "Cluster 12"}
CLUSTER_COLORS = {10: "#1f77b4", 12: "#ff7f0e"}
CLUSTER_PALETTE = {CLUSTER_LABELS[10]: CLUSTER_COLORS[10], CLUSTER_LABELS[12]: CLUSTER_COLORS[12]}
PREDICTION_COLOR = "#2ca02c"
FUTURE_COLOR = "#d62728"


@dataclass
class AnalysisArtifacts:
    test_predictions_raw: pd.DataFrame
    future_predictions: pd.DataFrame
    metrics_by_period: pd.DataFrame
    metrics_overall: pd.DataFrame
    user_period_metrics: pd.DataFrame
    cluster_panels: dict[int, pd.DataFrame]


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.titleweight"] = "bold"


def _save_fig(fig: plt.Figure, file_stem: str, out_dir: Path, save_svg: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{file_stem}.png", dpi=180, bbox_inches="tight")
    if save_svg:
        fig.savefig(out_dir / f"{file_stem}.svg", format="svg", bbox_inches="tight")


def _finish_fig(
    fig: plt.Figure,
    file_stem: str,
    out_dir: Path,
    save_svg: bool = False,
    show: bool = True,
) -> None:
    _save_fig(fig, file_stem=file_stem, out_dir=out_dir, save_svg=save_svg)
    if show:
        plt.show()
    plt.close(fig)


def _mape_0_100(y_true: pd.Series, y_pred: pd.Series, eps: float = 1.0) -> float:
    ape_pct = np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps) * 100.0
    return float(np.mean(np.clip(ape_pct, 0.0, 100.0)))


def _wmape_0_100(y_true: pd.Series, y_pred: pd.Series, eps: float = 1.0) -> float:
    return float(100.0 * np.sum(np.abs(y_true - y_pred)) / max(float(np.sum(np.abs(y_true))), eps))


def _format_period(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "period" in out.columns:
        out["period"] = pd.Categorical(out["period"], categories=PERIOD_ORDER, ordered=True)
    return out


def _with_cluster_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cluster_label"] = out["cluster_id"].map(CLUSTER_LABELS).fillna(out["cluster_id"].astype(str))
    return out


def load_analysis_artifacts(artifacts_dir: str | Path = "tft/artifacts") -> AnalysisArtifacts:
    base_dir = Path(artifacts_dir)
    final_dir = base_dir / "final"
    eval_dir = base_dir / "eval"
    data_dir = base_dir / "data"

    test_predictions_raw = pd.read_parquet(final_dir / "user_level_test_predictions.parquet")
    future_predictions = pd.read_parquet(final_dir / "user_level_future_predictions_14d.parquet")
    metrics_by_period = pd.read_parquet(final_dir / "multiphase_metrics_by_period.parquet")
    metrics_overall = pd.read_parquet(final_dir / "multiphase_metrics_overall.parquet")

    user_period_frames = sorted(eval_dir.glob("*/*_test_metrics_by_user_period.parquet"))
    if not user_period_frames:
        raise FileNotFoundError(f"No per-user period metrics found under {eval_dir}")
    user_period_metrics = pd.concat((pd.read_parquet(path) for path in user_period_frames), ignore_index=True)

    for frame in [
        test_predictions_raw,
        future_predictions,
        metrics_by_period,
        metrics_overall,
        user_period_metrics,
    ]:
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        if "forecast_origin" in frame.columns:
            frame["forecast_origin"] = pd.to_datetime(frame["forecast_origin"])

    cluster_panels = {
        10: pd.read_parquet(data_dir / "cluster_10_panel.parquet"),
        12: pd.read_parquet(data_dir / "cluster_12_panel.parquet"),
    }
    for panel in cluster_panels.values():
        panel["timestamp"] = pd.to_datetime(panel["timestamp"])

    return AnalysisArtifacts(
        test_predictions_raw=test_predictions_raw,
        future_predictions=future_predictions,
        metrics_by_period=_format_period(metrics_by_period),
        metrics_overall=metrics_overall.copy(),
        user_period_metrics=_format_period(user_period_metrics),
        cluster_panels=cluster_panels,
    )


def aggregate_test_predictions(test_predictions_raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model_name", "cluster_id", "user_id", "timestamp", "split", "phase", "horizon_step"]
    agg = (
        test_predictions_raw.groupby(group_cols, as_index=False)
        .agg(
            actual=("actual", "first"),
            prediction=("prediction", "median"),
            n_forecasts=("prediction", "size"),
        )
        .sort_values(["cluster_id", "user_id", "timestamp", "horizon_step"])
    )
    agg["date"] = agg["timestamp"].dt.normalize()
    return agg


def aggregate_user_timestamp_predictions(test_predictions_raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model_name", "cluster_id", "user_id", "timestamp", "split", "phase"]
    agg = (
        test_predictions_raw.groupby(group_cols, as_index=False)
        .agg(
            actual=("actual", "first"),
            prediction=("prediction", "median"),
            n_forecasts=("prediction", "size"),
        )
        .sort_values(["cluster_id", "user_id", "timestamp"])
    )
    agg["date"] = agg["timestamp"].dt.normalize()
    return agg


def add_seasonal_naive_24h_baseline(
    test_user_timestamp_predictions: pd.DataFrame,
    cluster_panels: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    baseline_source = pd.concat(
        [
            panel[["timestamp", "user_id", "cluster_id", "lag_24"]].copy()
            for panel in cluster_panels.values()
        ],
        ignore_index=True,
    )
    out = test_user_timestamp_predictions.merge(
        baseline_source,
        on=["timestamp", "user_id", "cluster_id"],
        how="left",
    )
    out = out.rename(columns={"lag_24": "baseline_prediction"})
    return out


def build_cluster_daily_aggregate(test_user_timestamp_predictions: pd.DataFrame) -> pd.DataFrame:
    out = (
        test_user_timestamp_predictions.groupby(["cluster_id", "date"], as_index=False)
        .agg(
            actual=("actual", "sum"),
            prediction=("prediction", "sum"),
        )
        .sort_values(["cluster_id", "date"])
    )
    out["actual_roll7"] = out.groupby("cluster_id")["actual"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    out["prediction_roll7"] = out.groupby("cluster_id")["prediction"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    return out


def build_cluster_hourly_comparison(test_user_timestamp_predictions: pd.DataFrame) -> pd.DataFrame:
    out = (
        test_user_timestamp_predictions.groupby(["cluster_id", "timestamp", "phase"], as_index=False)
        .agg(
            actual=("actual", "sum"),
            baseline_prediction=("baseline_prediction", "sum"),
            prediction=("prediction", "sum"),
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


def build_horizon_metrics(test_predictions_raw: pd.DataFrame, eps: float = 1.0) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (cluster_id, horizon_step), group in test_predictions_raw.groupby(["cluster_id", "horizon_step"]):
        rows.append(
            {
                "cluster_id": cluster_id,
                "horizon_step": horizon_step,
                "horizon_day": int((int(horizon_step) - 1) // 24 + 1),
                "MAPE_0_100": _mape_0_100(group["actual"], group["prediction"], eps=eps),
                "WMAPE_0_100": _wmape_0_100(group["actual"], group["prediction"], eps=eps),
                "n_obs": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["cluster_id", "horizon_step"]).reset_index(drop=True)


def select_random_user(
    test_user_timestamp_predictions: pd.DataFrame,
    user_id: str | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    user_map = (
        test_user_timestamp_predictions[["user_id", "cluster_id"]]
        .drop_duplicates()
        .sort_values(["cluster_id", "user_id"])
        .reset_index(drop=True)
    )
    if user_map.empty:
        raise ValueError("No users found in test predictions.")
    if user_id is None:
        sampled = user_map.sample(n=1, random_state=random_seed).iloc[0]
    else:
        matched = user_map[user_map["user_id"] == user_id]
        if matched.empty:
            raise ValueError(f"user_id not found in test predictions: {user_id}")
        sampled = matched.iloc[0]
    return {"user_id": str(sampled["user_id"]), "cluster_id": int(sampled["cluster_id"])}


def _cluster_plot_label(cluster_id: int) -> str:
    return CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")


def plot_cluster_scatter_comparison(
    cluster_hourly_comparison: pd.DataFrame,
    cluster_id: int,
    out_dir: Path,
    save_svg: bool = False,
    show: bool = True,
) -> None:
    _set_style()
    sub = cluster_hourly_comparison[cluster_hourly_comparison["cluster_id"] == cluster_id].copy()
    cluster_label = _cluster_plot_label(cluster_id)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharex=True, sharey=True)
    plot_specs = [
        ("baseline_prediction", "Seasonal Naive 24-Hour Baseline", "#1f77b4"),
        ("prediction", "TFT Forecast", "#1f77b4"),
    ]
    max_value = float(
        max(
            sub["actual"].max(),
            sub["baseline_prediction"].max(),
            sub["prediction"].max(),
        )
        * 1.02
    )

    for ax, (col, title_suffix, color) in zip(axes, plot_specs):
        ax.scatter(sub["actual"], sub[col], alpha=0.45, s=16, color=color)
        ax.plot([0, max_value], [0, max_value], linestyle="--", color="red", linewidth=1.5)
        ax.set_title(f"{cluster_label}: Actual vs Predicted for {title_suffix}")
        ax.set_xlabel("Actual load")
        ax.set_ylabel("Predicted load")
        ax.ticklabel_format(style="plain", axis="both", useOffset=False)

    fig.tight_layout()
    _finish_fig(
        fig,
        f"{cluster_id:02d}_cluster_{cluster_id}_scatter_comparison",
        out_dir=out_dir,
        save_svg=save_svg,
        show=show,
    )


def plot_cluster_residual_trend(
    cluster_hourly_comparison: pd.DataFrame,
    cluster_id: int,
    out_dir: Path,
    rolling_hours: int = 24,
    save_svg: bool = False,
    show: bool = True,
) -> None:
    _set_style()
    sub = cluster_hourly_comparison[cluster_hourly_comparison["cluster_id"] == cluster_id].copy()
    cluster_label = _cluster_plot_label(cluster_id)
    sub["baseline_residual_roll"] = sub["baseline_residual"].rolling(rolling_hours, min_periods=rolling_hours).mean()
    sub["model_residual_roll"] = sub["model_residual"].rolling(rolling_hours, min_periods=rolling_hours).mean()

    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.85)
    ax.plot(
        sub["timestamp"],
        sub["baseline_residual"],
        color="#ffbb78",
        alpha=0.55,
        linewidth=0.9,
        label="Baseline residual",
    )
    ax.plot(
        sub["timestamp"],
        sub["model_residual"],
        color="#98df8a",
        alpha=0.55,
        linewidth=0.9,
        label="Model residual",
    )
    ax.plot(
        sub["timestamp"],
        sub["baseline_residual_roll"],
        color="#ff7f0e",
        linewidth=2.0,
        label=f"Baseline residual {rolling_hours}h mean",
    )
    ax.plot(
        sub["timestamp"],
        sub["model_residual_roll"],
        color="#2ca02c",
        linewidth=2.0,
        label=f"Model residual {rolling_hours}h mean",
    )
    ax.set_title(f"{cluster_label}: Residual Trend (Actual Minus Forecast)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Residual")
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    _finish_fig(
        fig,
        f"{cluster_id:02d}_cluster_{cluster_id}_residual_trend",
        out_dir=out_dir,
        save_svg=save_svg,
        show=show,
    )


def plot_cluster_actual_vs_forecast_with_intervals(
    cluster_hourly_comparison: pd.DataFrame,
    cluster_id: int,
    out_dir: Path,
    save_svg: bool = False,
    show: bool = True,
) -> None:
    _set_style()
    sub = cluster_hourly_comparison[cluster_hourly_comparison["cluster_id"] == cluster_id].copy()
    cluster_label = _cluster_plot_label(cluster_id)
    z95 = 1.96

    baseline_sigma = sub["baseline_residual"].expanding(min_periods=24).std().shift(1)
    model_sigma = sub["model_residual"].expanding(min_periods=24).std().shift(1)
    baseline_sigma = baseline_sigma.fillna(float(sub["baseline_residual"].std(ddof=0)))
    model_sigma = model_sigma.fillna(float(sub["model_residual"].std(ddof=0)))

    baseline_low = np.clip(sub["baseline_prediction"] - z95 * baseline_sigma, 0.0, None)
    baseline_up = sub["baseline_prediction"] + z95 * baseline_sigma
    model_low = np.clip(sub["prediction"] - z95 * model_sigma, 0.0, None)
    model_up = sub["prediction"] + z95 * model_sigma

    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.plot(sub["timestamp"], sub["actual"], color="#1f77b4", linewidth=1.6, label="Actual Electricity Load")
    ax.plot(
        sub["timestamp"],
        sub["baseline_prediction"],
        color="#ff7f0e",
        linewidth=1.4,
        label="Seasonal Naive 24-Hour Baseline",
    )
    ax.fill_between(sub["timestamp"], baseline_low, baseline_up, color="#ffbb78", alpha=0.22)
    ax.plot(sub["timestamp"], sub["prediction"], color="#2ca02c", linewidth=1.4, label="TFT Forecast")
    ax.fill_between(sub["timestamp"], model_low, model_up, color="#98df8a", alpha=0.22)
    ax.set_title(f"{cluster_label}: Actual vs Forecast with Residual-Based 95% Intervals")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Hourly electricity consumption")
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    ax.legend()
    fig.tight_layout()
    _finish_fig(
        fig,
        f"{cluster_id:02d}_cluster_{cluster_id}_actual_vs_forecast_intervals",
        out_dir=out_dir,
        save_svg=save_svg,
        show=show,
    )


def plot_cluster_rolling_mape(
    cluster_hourly_comparison: pd.DataFrame,
    cluster_id: int,
    out_dir: Path,
    rolling_hours: int = 24,
    save_svg: bool = False,
    show: bool = True,
) -> None:
    _set_style()
    sub = cluster_hourly_comparison[cluster_hourly_comparison["cluster_id"] == cluster_id].copy()
    cluster_label = _cluster_plot_label(cluster_id)
    sub["baseline_rolling_mape"] = sub["baseline_ape_0_100"].rolling(rolling_hours, min_periods=rolling_hours).mean()
    sub["model_rolling_mape"] = sub["model_ape_0_100"].rolling(rolling_hours, min_periods=rolling_hours).mean()

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(
        sub["timestamp"],
        sub["baseline_rolling_mape"],
        color="#4c72b0",
        linewidth=1.8,
        label="Seasonal Naive 24-Hour Baseline",
    )
    ax.plot(
        sub["timestamp"],
        sub["model_rolling_mape"],
        color="#dd8452",
        linewidth=1.8,
        label="TFT Forecast",
    )
    ax.set_title(f"{cluster_label}: Rolling {rolling_hours}-Hour MAPE")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Rolling MAPE (%)")
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    ax.legend(title="Forecast")
    fig.tight_layout()
    _finish_fig(
        fig,
        f"{cluster_id:02d}_cluster_{cluster_id}_rolling_{rolling_hours}h_mape",
        out_dir=out_dir,
        save_svg=save_svg,
        show=show,
    )


def plot_cluster_error_distribution_by_period(
    cluster_hourly_comparison: pd.DataFrame,
    cluster_id: int,
    out_dir: Path,
    save_svg: bool = False,
    show: bool = True,
) -> pd.DataFrame:
    _set_style()
    sub = cluster_hourly_comparison[cluster_hourly_comparison["cluster_id"] == cluster_id].copy()
    cluster_label = _cluster_plot_label(cluster_id)
    plot_df = pd.concat(
        [
            sub[["phase", "baseline_ape_0_100"]].rename(columns={"baseline_ape_0_100": "ape_0_100"}).assign(
                forecast="Seasonal Naive 24-Hour Baseline"
            ),
            sub[["phase", "model_ape_0_100"]].rename(columns={"model_ape_0_100": "ape_0_100"}).assign(
                forecast="TFT Forecast"
            ),
        ],
        ignore_index=True,
    )
    plot_df["phase"] = pd.Categorical(plot_df["phase"], categories=PERIOD_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    sns.boxplot(
        data=plot_df,
        x="phase",
        y="ape_0_100",
        hue="forecast",
        order=PERIOD_ORDER,
        palette=["#4c72b0", "#dd8452"],
        ax=ax,
    )
    ax.set_title(f"{cluster_label}: Forecast Error Distribution by Test Period")
    ax.set_xlabel("Test period")
    ax.set_ylabel("APE component clipped to 0-100 (%)")
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    ax.legend(title="Forecast")
    fig.tight_layout()
    _finish_fig(
        fig,
        f"{cluster_id:02d}_cluster_{cluster_id}_error_distribution_by_period",
        out_dir=out_dir,
        save_svg=save_svg,
        show=show,
    )

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


def plot_overall_metrics(
    metrics_overall: pd.DataFrame,
    out_dir: Path,
    save_svg: bool = False,
    show: bool = True,
) -> None:
    _set_style()
    plot_df = _with_cluster_label(metrics_overall).sort_values("cluster_id")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.barplot(
        data=plot_df,
        x="cluster_label",
        y="MAPE_0_100",
        hue="cluster_label",
        palette=CLUSTER_PALETTE,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Overall Clipped MAPE by Cluster")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("MAPE_0_100 (%)")

    sns.barplot(
        data=plot_df,
        x="cluster_label",
        y="WMAPE_0_100",
        hue="cluster_label",
        palette=CLUSTER_PALETTE,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Overall WMAPE by Cluster")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("WMAPE_0_100 (%)")

    fig.tight_layout()
    _finish_fig(fig, "01_overall_metrics", out_dir=out_dir, save_svg=save_svg, show=show)


def plot_period_metrics(
    metrics_by_period: pd.DataFrame,
    out_dir: Path,
    save_svg: bool = False,
    show: bool = True,
) -> None:
    _set_style()
    plot_df = _with_cluster_label(metrics_by_period).sort_values(["period", "cluster_id"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    sns.barplot(
        data=plot_df,
        x="period",
        y="MAPE_0_100",
        hue="cluster_label",
        palette=CLUSTER_PALETTE,
        order=PERIOD_ORDER,
        ax=axes[0],
    )
    axes[0].set_title("MAPE_0_100 by Test Period")
    axes[0].set_xlabel("Period")
    axes[0].set_ylabel("MAPE_0_100 (%)")
    axes[0].legend(title="Cluster")

    sns.barplot(
        data=plot_df,
        x="period",
        y="WMAPE_0_100",
        hue="cluster_label",
        palette=CLUSTER_PALETTE,
        order=PERIOD_ORDER,
        ax=axes[1],
    )
    axes[1].set_title("WMAPE_0_100 by Test Period")
    axes[1].set_xlabel("Period")
    axes[1].set_ylabel("WMAPE_0_100 (%)")
    axes[1].legend(title="Cluster")

    fig.tight_layout()
    _finish_fig(fig, "02_period_metrics", out_dir=out_dir, save_svg=save_svg, show=show)


def plot_cluster_daily_aggregate(
    cluster_daily_aggregate: pd.DataFrame,
    out_dir: Path,
    save_svg: bool = False,
    show: bool = True,
) -> None:
    _set_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, cluster_id in zip(axes, CLUSTER_ORDER):
        sub = cluster_daily_aggregate[cluster_daily_aggregate["cluster_id"] == cluster_id].copy()
        ax.plot(sub["date"], sub["actual"], color=CLUSTER_COLORS[cluster_id], alpha=0.25, linewidth=1.2, label="Actual daily")
        ax.plot(
            sub["date"],
            sub["prediction"],
            color=PREDICTION_COLOR,
            alpha=0.25,
            linewidth=1.2,
            label="Predicted daily",
        )
        ax.plot(
            sub["date"],
            sub["actual_roll7"],
            color=CLUSTER_COLORS[cluster_id],
            linewidth=2.2,
            label="Actual 7d mean",
        )
        ax.plot(
            sub["date"],
            sub["prediction_roll7"],
            color=PREDICTION_COLOR,
            linewidth=2.2,
            label="Predicted 7d mean",
        )
        ax.set_title(f"{CLUSTER_LABELS[cluster_id]} Daily Aggregate: Actual vs Prediction")
        ax.set_ylabel("Daily total load")
        ax.legend(ncol=2, fontsize=8)

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    _finish_fig(fig, "03_cluster_daily_aggregate", out_dir=out_dir, save_svg=save_svg, show=show)


def plot_user_period_distribution(
    user_period_metrics: pd.DataFrame,
    out_dir: Path,
    save_svg: bool = False,
    show: bool = True,
) -> pd.DataFrame:
    _set_style()
    plot_df = _with_cluster_label(user_period_metrics).sort_values(["cluster_id", "period", "user_id"])
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
    sns.boxplot(
        data=plot_df,
        x="period",
        y="MAPE_0_100",
        hue="cluster_label",
        order=PERIOD_ORDER,
        palette=CLUSTER_PALETTE,
        ax=axes[0],
    )
    axes[0].set_title("User-level MAPE_0_100 Distribution")
    axes[0].set_xlabel("Period")
    axes[0].set_ylabel("MAPE_0_100 (%)")
    axes[0].legend(title="Cluster")

    sns.boxplot(
        data=plot_df,
        x="period",
        y="WMAPE_0_100",
        hue="cluster_label",
        order=PERIOD_ORDER,
        palette=CLUSTER_PALETTE,
        ax=axes[1],
    )
    axes[1].set_title("User-level WMAPE_0_100 Distribution")
    axes[1].set_xlabel("Period")
    axes[1].set_ylabel("WMAPE_0_100 (%)")
    axes[1].legend(title="Cluster")

    fig.tight_layout()
    _finish_fig(fig, "04_user_period_distribution", out_dir=out_dir, save_svg=save_svg, show=show)
    return summary


def plot_horizon_error_profile(
    horizon_metrics: pd.DataFrame,
    out_dir: Path,
    save_svg: bool = False,
    show: bool = True,
) -> None:
    _set_style()
    plot_df = _with_cluster_label(horizon_metrics)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)

    sns.lineplot(
        data=plot_df,
        x="horizon_step",
        y="MAPE_0_100",
        hue="cluster_label",
        palette=CLUSTER_PALETTE,
        ax=axes[0],
    )
    axes[0].set_title("Clipped MAPE by Forecast Horizon")
    axes[0].set_xlabel("Horizon step (hour)")
    axes[0].set_ylabel("MAPE_0_100 (%)")
    axes[0].legend(title="Cluster")

    sns.lineplot(
        data=plot_df,
        x="horizon_step",
        y="WMAPE_0_100",
        hue="cluster_label",
        palette=CLUSTER_PALETTE,
        ax=axes[1],
    )
    axes[1].set_title("WMAPE by Forecast Horizon")
    axes[1].set_xlabel("Horizon step (hour)")
    axes[1].set_ylabel("WMAPE_0_100 (%)")
    axes[1].legend(title="Cluster")

    fig.tight_layout()
    _finish_fig(fig, "05_horizon_error_profile", out_dir=out_dir, save_svg=save_svg, show=show)


def plot_random_user_history_and_forecast(
    test_user_timestamp_predictions: pd.DataFrame,
    future_predictions: pd.DataFrame,
    cluster_panels: dict[int, pd.DataFrame],
    user_id: str,
    cluster_id: int,
    out_dir: Path,
    history_days: int = 60,
    save_svg: bool = False,
    show: bool = True,
) -> None:
    _set_style()

    panel = cluster_panels[cluster_id].copy()
    panel = panel[panel["user_id"] == user_id].sort_values("timestamp")
    history = panel[panel["split"] != "test"].copy()
    if history.empty:
        raise ValueError(f"No history found for user_id={user_id} in cluster {cluster_id}.")

    history_end = history["timestamp"].max()
    history_start = history_end - pd.Timedelta(days=history_days)
    history_window = history[history["timestamp"] >= history_start].copy()

    test_sub = (
        test_user_timestamp_predictions[
            (test_user_timestamp_predictions["user_id"] == user_id)
            & (test_user_timestamp_predictions["cluster_id"] == cluster_id)
        ]
        .sort_values("timestamp")
        .copy()
    )
    future_sub = (
        future_predictions[
            (future_predictions["user_id"] == user_id)
            & (future_predictions["cluster_id"] == cluster_id)
        ]
        .sort_values("timestamp")
        .copy()
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    axes[0].plot(
        history_window["timestamp"],
        history_window["target_load"],
        color="#7f7f7f",
        linewidth=1.2,
        label="History",
    )
    axes[0].plot(
        test_sub["timestamp"],
        test_sub["actual"],
        color=CLUSTER_COLORS[cluster_id],
        linewidth=1.8,
        label="Test actual",
    )
    axes[0].plot(
        test_sub["timestamp"],
        test_sub["prediction"],
        color=PREDICTION_COLOR,
        linewidth=1.8,
        label="Test prediction",
    )
    axes[0].set_title(f"{CLUSTER_LABELS[cluster_id]} Random User {user_id}: History and Test Forecast")
    axes[0].set_xlabel("Timestamp")
    axes[0].set_ylabel("Load")
    axes[0].legend()

    if not future_sub.empty:
        test_tail = test_sub.tail(24 * 7).copy()
        axes[1].plot(
            test_tail["timestamp"],
            test_tail["actual"],
            color=CLUSTER_COLORS[cluster_id],
            linewidth=1.8,
            label="Test actual (last 7d)",
        )
        axes[1].plot(
            test_tail["timestamp"],
            test_tail["prediction"],
            color=PREDICTION_COLOR,
            linewidth=1.8,
            label="Test prediction (last 7d)",
        )
        axes[1].plot(
            future_sub["timestamp"],
            future_sub["y_pred_p50"],
            color=FUTURE_COLOR,
            linewidth=2.0,
            label="Future p50",
        )
        axes[1].fill_between(
            future_sub["timestamp"],
            future_sub["y_pred_p10"],
            future_sub["y_pred_p90"],
            color=FUTURE_COLOR,
            alpha=0.18,
            label="Future p10-p90",
        )
        axes[1].set_title(f"{CLUSTER_LABELS[cluster_id]} Random User {user_id}: Future 14-day Quantile Forecast")
        axes[1].set_xlabel("Timestamp")
        axes[1].set_ylabel("Load")
        axes[1].legend()

    fig.tight_layout()
    _finish_fig(fig, "06_random_user_history_and_future", out_dir=out_dir, save_svg=save_svg, show=show)


def run_full_analysis(
    artifacts_dir: str | Path = "tft/artifacts",
    image_dir: str | Path = "images/tft",
    user_id: str | None = None,
    random_seed: int = 42,
    history_days: int = 60,
    save_svg: bool = False,
    show: bool = True,
) -> dict[str, Any]:
    artifacts = load_analysis_artifacts(artifacts_dir=artifacts_dir)
    out_dir = Path(image_dir)

    test_user_timestamp_predictions = aggregate_user_timestamp_predictions(artifacts.test_predictions_raw)
    test_user_timestamp_predictions = add_seasonal_naive_24h_baseline(
        test_user_timestamp_predictions=test_user_timestamp_predictions,
        cluster_panels=artifacts.cluster_panels,
    )
    cluster_hourly_comparison = build_cluster_hourly_comparison(test_user_timestamp_predictions)
    cluster_daily_aggregate = build_cluster_daily_aggregate(test_user_timestamp_predictions)
    horizon_metrics = build_horizon_metrics(artifacts.test_predictions_raw)
    selected_user = select_random_user(
        test_user_timestamp_predictions=test_user_timestamp_predictions,
        user_id=user_id,
        random_seed=random_seed,
    )

    plot_overall_metrics(artifacts.metrics_overall, out_dir=out_dir, save_svg=save_svg, show=show)
    plot_period_metrics(artifacts.metrics_by_period, out_dir=out_dir, save_svg=save_svg, show=show)
    plot_cluster_daily_aggregate(cluster_daily_aggregate, out_dir=out_dir, save_svg=save_svg, show=show)
    user_period_summary = plot_user_period_distribution(
        artifacts.user_period_metrics,
        out_dir=out_dir,
        save_svg=save_svg,
        show=show,
    )
    plot_horizon_error_profile(horizon_metrics, out_dir=out_dir, save_svg=save_svg, show=show)
    cluster_period_error_summaries = []
    for cluster_id in CLUSTER_ORDER:
        plot_cluster_scatter_comparison(
            cluster_hourly_comparison=cluster_hourly_comparison,
            cluster_id=cluster_id,
            out_dir=out_dir,
            save_svg=save_svg,
            show=show,
        )
        plot_cluster_residual_trend(
            cluster_hourly_comparison=cluster_hourly_comparison,
            cluster_id=cluster_id,
            out_dir=out_dir,
            rolling_hours=24,
            save_svg=save_svg,
            show=show,
        )
        plot_cluster_actual_vs_forecast_with_intervals(
            cluster_hourly_comparison=cluster_hourly_comparison,
            cluster_id=cluster_id,
            out_dir=out_dir,
            save_svg=save_svg,
            show=show,
        )
        plot_cluster_rolling_mape(
            cluster_hourly_comparison=cluster_hourly_comparison,
            cluster_id=cluster_id,
            out_dir=out_dir,
            rolling_hours=24,
            save_svg=save_svg,
            show=show,
        )
        cluster_period_error_summaries.append(
            plot_cluster_error_distribution_by_period(
                cluster_hourly_comparison=cluster_hourly_comparison,
                cluster_id=cluster_id,
                out_dir=out_dir,
                save_svg=save_svg,
                show=show,
            )
        )
    plot_random_user_history_and_forecast(
        test_user_timestamp_predictions=test_user_timestamp_predictions,
        future_predictions=artifacts.future_predictions,
        cluster_panels=artifacts.cluster_panels,
        user_id=selected_user["user_id"],
        cluster_id=selected_user["cluster_id"],
        out_dir=out_dir,
        history_days=history_days,
        save_svg=save_svg,
        show=show,
    )

    return {
        "metrics_overall": artifacts.metrics_overall,
        "metrics_by_period": artifacts.metrics_by_period,
        "user_period_metrics": artifacts.user_period_metrics,
        "user_period_summary": user_period_summary,
        "daily_cluster_aggregate": cluster_daily_aggregate,
        "cluster_hourly_comparison": cluster_hourly_comparison,
        "cluster_period_error_summary": pd.concat(cluster_period_error_summaries, ignore_index=True),
        "horizon_metrics": horizon_metrics,
        "test_user_timestamp_predictions": test_user_timestamp_predictions,
        "future_predictions": artifacts.future_predictions,
        "cluster_panels": artifacts.cluster_panels,
        "selected_user": selected_user,
        "image_dir": out_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze TFT forecasting outputs and save figures.")
    parser.add_argument("--artifacts-dir", default="tft/artifacts", help="Base TFT artifacts directory.")
    parser.add_argument("--image-dir", default="images/tft", help="Directory for saved images.")
    parser.add_argument("--user-id", default=None, help="Optional fixed user_id for the detailed user plot.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for user sampling.")
    parser.add_argument("--history-days", type=int, default=60, help="History window for the user plot.")
    parser.add_argument("--save-svg", action="store_true", help="Also save SVG copies of each figure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_full_analysis(
        artifacts_dir=args.artifacts_dir,
        image_dir=args.image_dir,
        user_id=args.user_id,
        random_seed=args.random_seed,
        history_days=args.history_days,
        save_svg=args.save_svg,
        show=False,
    )


if __name__ == "__main__":
    main()
