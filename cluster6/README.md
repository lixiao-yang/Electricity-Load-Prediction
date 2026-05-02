# Cluster 6 Workspace

This directory contains the standalone Cluster 6 forecasting workflow for `MT_362`.

Configs, notebooks, artifacts, plots, and logs live under one workflow directory. Combined comparison notebooks, combined notebook outputs, and the downstream agent app are intentionally not included here.

## What Lives Here

```text
cluster6/
  configs/
  notebooks/
  artifacts/
    data/
    eval/direct_trend/
    plots/direct_trend/
    plots/visualization/
  logs/
```

## Upstream Inputs

The workflow expects the upstream preprocessing outputs from `cluster_analysis.ipynb`:

- `data/train_hourly_preprocessed.parquet`
- `data/test_hourly_preprocessed.parquet`
- `data/extended-clustering-high-cov/clusters_3models.parquet`

## Main Notebooks

- `cluster6/notebooks/cluster6_forecasting.ipynb`
  - model selection and final direct-model test evaluation
- `cluster6/notebooks/cluster6_visualization.ipynb`
  - C1/C3-style diagnostics using the selected direct model output
- `cluster6/notebooks/cluster6_future_14d_forecast.ipynb`
  - 14-day future forecast generation using the selected final model

## Artifact Contract

All tabular outputs are parquet:

- `cluster6/artifacts/eval/direct_trend/model_comparison_validation.parquet`
- `cluster6/artifacts/eval/direct_trend/rolling_validation_details.parquet`
- `cluster6/artifacts/eval/direct_trend/final_test_results.parquet`
- `cluster6/artifacts/eval/direct_trend/final_test_forecast_detail.parquet`
- `cluster6/artifacts/data/c6_prediction.parquet`

Non-tabular outputs:

- `cluster6/artifacts/eval/direct_trend/modeling_summary.md`
- `cluster6/artifacts/eval/direct_trend/eval_manifest.json`
- `cluster6/artifacts/plots/direct_trend/*.png`
- `cluster6/artifacts/plots/visualization/*.svg`

## Metrics

The final test output includes the original plain MAPE plus additional MAPE variants for scale-aware evaluation:

- `test_mape_0_100`
- `test_epsilon_mape_pct`
- `test_wmape_0_100`
- `mape_epsilon`
- `test_n_obs`
- `test_n_positive`

## Commands

Run the full Cluster 6 direct model notebook:

```bash
conda run -n AML jupyter nbconvert --to notebook --execute --inplace cluster6/notebooks/cluster6_forecasting.ipynb --ExecutePreprocessor.timeout=600
```

Run the visualization notebook after the direct model notebook has written its parquet outputs:

```bash
conda run -n AML jupyter nbconvert --to notebook --execute --inplace cluster6/notebooks/cluster6_visualization.ipynb --ExecutePreprocessor.timeout=600
```

Run the future 14-day forecast notebook after the direct model notebook has written its model artifacts:

```bash
conda run -n AML jupyter nbconvert --to notebook --execute --inplace cluster6/notebooks/cluster6_future_14d_forecast.ipynb --ExecutePreprocessor.timeout=600
```
