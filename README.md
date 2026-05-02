# Electricity Load Prediction

This repository contains a two-stage electricity load workflow:

1. clustering and preprocessing in `cluster_analysis.ipynb`
2. downstream forecasting with a Temporal Fusion Transformer (TFT) for `cluster_id in {10, 12}`

The TFT implementation lives under `tft/`, but this root README is the primary project guide for the full pipeline.

## Workflow Summary

The full workflow implements:

- hourly preprocessing and train-only cohort construction
- meter clustering from train-only history
- export of filtered hourly train / test forecasting panels
- shared TFT pretraining on clusters `10 + 12`
- per-cluster TFT finetuning for cluster `10`
- per-cluster TFT finetuning for cluster `12`
- rolling test evaluation with fixed multi-phase monthly metrics `P1/P2/P3`
- recursive future 14-day forecasting with quantile outputs
- final user-level parquet deliverables

The forecasting stage specifically implements:

- shared pretraining on clusters `10 + 12`
- per-cluster finetuning for cluster `10`
- per-cluster finetuning for cluster `12`
- rolling test evaluation with fixed multi-phase monthly metrics `P1/P2/P3`
- recursive future 14-day forecasting with quantile outputs
- final user-level parquet deliverables

The design is leakage-aware:

- time splitting happens before feature construction
- lag / rolling features are computed after `shift(1)`
- encoders / normalizers are fit on train only
- validation is used for early stopping and scheduler decisions
- test is reserved for final evaluation
- future forecasting never uses future true targets

## Project Layout

```text
cluster_analysis.ipynb
README.md
tft/
  configs/
  src/
    data/
    model/
    eval/
    infer/
    postprocess/
    agent_bridge/
  artifacts/
  sanity_check.ipynb
```

## Clustering and Preprocessing Pipeline

The upstream clustering workflow is implemented in:

- `cluster_analysis.ipynb`

This notebook is not a lightweight helper. It defines the data window, active-meter cohort, preprocessing policy, clustering features, model selection process, and the cluster labels later consumed by the TFT pipeline.

### Clustering notebook goals

The notebook does the following:

- convert the original panel to parquet-backed storage if needed
- aggregate raw 15-minute load to hourly sums
- derive a rolling point-in-time split from the latest available timestamp
- reserve the final `3` natural months as test
- reserve the immediately preceding `2` years as train for clustering and later forecasting
- identify eligible meters using train-only history
- remove meters that are too short, too sparse, inactive at the cohort anchor, or variance-free
- engineer train-only clustering features
- standardize clustering inputs
- compare clustering model families and select `K`
- fit clustering models and export labels and cluster summaries

### Split logic used by the notebook

The implemented split is derived dynamically from the latest timestamp in the hourly panel:

- test window: final `3` natural calendar months
- train window: previous `2` years ending one hour before the test start

For the current processed panel, the notebook reports:

- train hourly rows: `17,520` from `2012-10-01 00:00:00` to `2014-09-30 23:00:00`
- test hourly rows: `2,208` from `2014-10-01 00:00:00` to `2014-12-31 23:00:00`

### Meter eligibility rules

The clustering notebook keeps a meter only if all of the following implemented code rules are satisfied:

- it is active on the cohort reference date `TRAIN_START`
- `coverage_ratio >= 0.50`
- `positive_rate_in_active_window >= 0.50`
- `std_load_in_active_window > 0`

Additional preprocessing details:

- active windows are detected using `load > LOAD_EPSILON`
- only the train window is used for eligibility and feature engineering
- internal missing values are handled with capped causal forward fill only
- `ffill(limit=24)` is allowed
- `bfill` is intentionally not used

The notebook also writes descriptive reliability bands such as `exclude`, `low`, `medium`, and `high`, but those are diagnostics rather than direct cluster-assignment rules.

### Clustering features

The notebook engineers train-only meter features from the cleaned hourly series, including:

- level and scale features
- volatility and stability features
- load-shape features
- calendar-normalized profiles such as:
  - month
  - day of week
  - hour of day

These features are standardized before model selection and fitting.

### Model selection and fitted cluster labels

The notebook compares clustering choices using:

- KMeans inertia
- KMeans silhouette score
- Gaussian mixture BIC
- seed-based stability diagnostics using ARI

The notebook reports:

- recommended K for KMeans by best silhouette: `13`
- recommended K for GMM by lowest BIC

It then fits:

- `cluster_kmeans`
- `cluster_gmm`
- `cluster_hdbscan`

The downstream TFT workflow currently uses the KMeans labels, specifically clusters `10` and `12`.

### Main notebook outputs

The clustering notebook produces both root-level forecasting inputs and extended clustering artifacts.

Root-level outputs used directly by TFT:

- `data/train_hourly_preprocessed.parquet`
- `data/test_hourly_preprocessed.parquet`

Extended clustering outputs under `data/extended-clustering-high-cov/` include at least:

- `meter_activity_summary.parquet`
- `clustering_history_preprocessed.parquet`
- `train_hourly_preprocessed.parquet`
- `test_hourly_preprocessed.parquet`
- `clusters_3models.parquet`
- `feature_table_extended.parquet`
- cluster aggregate / mean time-series parquet files
- cluster profile and diagnostic exports used for analysis

### Relationship to TFT

The handoff from clustering to TFT is:

1. `cluster_analysis.ipynb` defines the train / test boundary and the eligible meter cohort.
2. It exports the filtered hourly train / test panels.
3. It exports `clusters_3models.parquet`.
4. The TFT pipeline reads those artifacts and keeps only meters assigned to KMeans clusters `10` and `12`.

## Forecasting Input Handoff

The TFT pipeline does not start from raw meter tables. It starts from the clustering outputs above.

Required upstream files:

- `data/train_hourly_preprocessed.parquet`
- `data/test_hourly_preprocessed.parquet`
- `data/extended-clustering-high-cov/clusters_3models.parquet`

The split assumption for the TFT stage is:

- train window used for TFT development: last `18` months before validation
- validation: `3` months
- test: last `3` natural months
- future inference horizon: `14` days

If the upstream test parquet is not exactly three natural months, downstream evaluation is designed to fail fast instead of silently producing misaligned period metrics.

## TFT Pipeline Overview

### 1. Build TFT dataset panels

Script:

- `tft/src/data/build_tft_dataset.py`

Purpose:

- read the notebook-produced wide hourly train / test tables
- keep only cluster `10` and cluster `12`
- reshape to long format
- build leakage-safe known-future and observed feature tables
- export shared and per-cluster panel data
- write split and leakage audit files

Main outputs:

- `tft/artifacts/data/features_train.parquet`
- `tft/artifacts/data/features_val.parquet`
- `tft/artifacts/data/features_test.parquet`
- `tft/artifacts/data/features_future_known.parquet`
- `tft/artifacts/data/shared_panel.parquet`
- `tft/artifacts/data/cluster_10_panel.parquet`
- `tft/artifacts/data/cluster_12_panel.parquet`
- `tft/artifacts/data/split_boundaries.json`
- `tft/artifacts/data/leakage_audit.json`

### 2. Build PyTorch Forecasting dataset metadata

Script:

- `tft/src/data/build_tft_timeseries_dataset.py`

Purpose:

- validate split boundaries and calendar windows
- create continuous `time_idx`
- fit dataset encoders / normalizers on train only
- save reusable dataset metadata and parameters

Main outputs:

- `tft/artifacts/data/tft_dataset_meta.json`
- `tft/artifacts/data/tft_dataset_parameters.pt`
- `tft/configs/dataloader.yaml`

### 3. Train shared pretrain model

Script:

- `tft/src/model/train_shared_pretrain.py`

Purpose:

- train a single shared TFT on the combined cluster `10 + 12` panel

Main outputs:

- `tft/artifacts/models/tft_shared_pretrain/checkpoints/best.ckpt`
- `tft/artifacts/models/tft_shared_pretrain/dataset_params.pt`
- `tft/artifacts/models/tft_shared_pretrain/resolved_config.yaml`
- `tft/artifacts/models/tft_shared_pretrain/train_manifest.json`

### 4. Finetune per cluster

Script:

- `tft/src/model/finetune_cluster.py`

Purpose:

- load the shared checkpoint
- finetune one model for cluster `10`
- finetune one model for cluster `12`

Main outputs:

- `tft/artifacts/models/tft_c10_ft/checkpoints/best.ckpt`
- `tft/artifacts/models/tft_c12_ft/checkpoints/best.ckpt`
- corresponding finetune manifests and resolved configs

### 5. Evaluate on rolling test windows

Script:

- `tft/src/eval/evaluate_tft.py`

Purpose:

- run rolling test forecasts using only information available before each forecast origin
- map the three test months to fixed periods:
  - first month -> `P1`
  - second month -> `P2`
  - third month -> `P3`
- compute overall and multi-phase metrics

Metrics:

- `MAPE_0_100`
- `EPSILON_MAPE_PCT`
- `WMAPE_0_100`
- `n_obs`
- `n_positive`

Main outputs per cluster:

- `..._test_predictions.parquet`
- `..._test_metrics_overall.parquet`
- `..._test_metrics_by_period.parquet`
- `..._test_metrics_by_user_period.parquet`

### 6. Infer future 14-day forecasts

Script:

- `tft/src/infer/predict_future.py`

Purpose:

- recursively forecast the next 14 days
- emit quantile predictions only

Quantile outputs:

- `y_pred_p10`
- `y_pred_p50`
- `y_pred_p90`

Main outputs per cluster:

- `..._future_14d_predictions.parquet`

### 7. Build final user-level deliverables

Script:

- `tft/src/postprocess/build_final_user_parquets.py`

Purpose:

- merge cluster `10` and `12` evaluation / inference outputs
- produce final user-level parquet files for downstream analysis

Final outputs:

- `tft/artifacts/final/user_level_test_predictions.parquet`
- `tft/artifacts/final/user_level_future_predictions_14d.parquet`
- `tft/artifacts/final/user_level_predictions_all.parquet`
- `tft/artifacts/final/multiphase_metrics_by_period.parquet`
- `tft/artifacts/final/multiphase_metrics_overall.parquet`

## Input / Output Contract

### Upstream inputs

- `data/train_hourly_preprocessed.parquet`
- `data/test_hourly_preprocessed.parquet`
- `data/extended-clustering-high-cov/clusters_3models.parquet`

### Final outputs

- cluster-specific checkpoints under `tft/artifacts/models/`
- rolling test predictions and metrics under `tft/artifacts/eval/`
- future quantile forecasts under `tft/artifacts/infer/`
- final merged user-level parquets under `tft/artifacts/final/`

## Model Inputs

The current TFT model intentionally excludes target-derived decoder covariates from the model input to avoid leakage during multi-step prediction.

Used model inputs:

- group id: `user_id`
- target: `target_load`
- static categoricals:
  - `user_id`
  - `cluster_id_cat`
- known future categoricals:
  - `hour_cat`
  - `day_of_week_cat`
  - `month_cat`
  - `is_weekend_cat`
  - `holiday_flag_cat`
- unknown reals:
  - `target_load`

The parquet features still contain lag / rolling columns for auditing and optional experimentation, but the active production configuration sets:

- `data.use_target_derived_covariates: false`

## Training Schedule

The current training code uses:

- linear warmup for the first few hundred optimization steps
- validation-driven `ReduceLROnPlateau` after warmup

This is implemented in `tft/src/model/tft_utils.py` through a thin TFT subclass so the rest of the pipeline and CLI entry points remain unchanged.

Current learning-rate policy:

- shared pretrain:
  - `learning_rate: 0.0006`
  - `warmup_steps: 400`
  - `warmup_init_factor: 0.2`
- cluster 10 finetune:
  - `learning_rate: 0.00025`
  - `warmup_steps: 100`
  - `warmup_init_factor: 0.2`
- cluster 12 finetune:
  - `learning_rate: 0.0003`
  - `warmup_steps: 100`
  - `warmup_init_factor: 0.2`

## Current Formal Configuration

The default formal run is defined mainly in:

- `tft/configs/data.yaml`
- `tft/configs/tft_shared_pretrain.yaml`
- `tft/configs/tft_c10_finetune.yaml`
- `tft/configs/tft_c12_finetune.yaml`
- `tft/configs/eval.yaml`
- `tft/configs/infer.yaml`

Current key model settings:

- `min_encoder_length: 168`
- `max_encoder_length: 672`
- `max_prediction_length: 336`
- `hidden_size: 48`
- `attention_head_size: 4`
- `hidden_continuous_size: 24`
- `dropout: 0.15`
- `precision: 16-mixed`
- `accelerator: gpu`
- `devices: 1`
- `batch_size: 40` for shared pretrain
- `batch_size: 56` for finetune

## Full Training / Evaluation Commands

Run the full formal pipeline in this order:

```bash
python -m tft.src.data.build_tft_dataset --config tft/configs/data.yaml
python -m tft.src.data.build_tft_timeseries_dataset --config tft/configs/data.yaml
python -m tft.src.model.train_shared_pretrain --config tft/configs/tft_shared_pretrain.yaml
python -m tft.src.model.finetune_cluster --config tft/configs/tft_c10_finetune.yaml
python -m tft.src.model.finetune_cluster --config tft/configs/tft_c12_finetune.yaml
python -m tft.src.eval.evaluate_tft --config tft/configs/data.yaml --panel-path tft/artifacts/data/cluster_10_panel.parquet --dataset-params-path tft/artifacts/models/tft_shared_pretrain/dataset_params.pt --checkpoint-path tft/artifacts/models/tft_c10_ft/checkpoints/best.ckpt --cluster-id 10 --model-name tft_c10_ft --output-dir tft/artifacts/eval/tft_c10_ft
python -m tft.src.eval.evaluate_tft --config tft/configs/data.yaml --panel-path tft/artifacts/data/cluster_12_panel.parquet --dataset-params-path tft/artifacts/models/tft_shared_pretrain/dataset_params.pt --checkpoint-path tft/artifacts/models/tft_c12_ft/checkpoints/best.ckpt --cluster-id 12 --model-name tft_c12_ft --output-dir tft/artifacts/eval/tft_c12_ft
python -m tft.src.infer.predict_future --config tft/configs/data.yaml --panel-path tft/artifacts/data/cluster_10_panel.parquet --dataset-params-path tft/artifacts/models/tft_shared_pretrain/dataset_params.pt --checkpoint-path tft/artifacts/models/tft_c10_ft/checkpoints/best.ckpt --cluster-id 10 --model-name tft_c10_ft --output-dir tft/artifacts/infer/tft_c10_ft
python -m tft.src.infer.predict_future --config tft/configs/data.yaml --panel-path tft/artifacts/data/cluster_12_panel.parquet --dataset-params-path tft/artifacts/models/tft_shared_pretrain/dataset_params.pt --checkpoint-path tft/artifacts/models/tft_c12_ft/checkpoints/best.ckpt --cluster-id 12 --model-name tft_c12_ft --output-dir tft/artifacts/infer/tft_c12_ft
python -m tft.src.postprocess.build_final_user_parquets --eval-config tft/configs/eval.yaml --infer-config tft/configs/infer.yaml --output-dir tft/artifacts/final
```

## Quick Smoke Test

Quick configs exist under:

- `tft/configs/data_quick.yaml`
- `tft/configs/tft_shared_pretrain_quick.yaml`
- `tft/configs/tft_c10_finetune_quick.yaml`
- `tft/configs/tft_c12_finetune_quick.yaml`
- `tft/configs/eval_quick.yaml`
- `tft/configs/infer_quick.yaml`

A minimal quick chain is:

```bash
python -m py_compile tft\src\model\tft_utils.py tft\src\model\train_shared_pretrain.py tft\src\model\finetune_cluster.py tft\src\eval\evaluate_tft.py tft\src\infer\predict_future.py
python -m tft.src.model.train_shared_pretrain --config tft/configs/tft_shared_pretrain_quick.yaml
python -m tft.src.model.finetune_cluster --config tft/configs/tft_c10_finetune_quick.yaml
python -m tft.src.eval.evaluate_tft --config tft/configs/data_quick.yaml --panel-path tft/artifacts_quick/data/cluster_10_panel.parquet --dataset-params-path tft/artifacts_quick/models/tft_shared_pretrain/dataset_params.pt --checkpoint-path tft/artifacts_quick/models/tft_c10_ft/checkpoints/best.ckpt --cluster-id 10 --model-name tft_c10_ft_quick --output-dir tft/artifacts_quick/eval/tft_c10_ft
```

## Files and Their Roles

### Main notebooks

- `cluster_analysis.ipynb`
  - upstream clustering and hourly parquet generation
- `tft/sanity_check.ipynb`
  - post-run sanity checks on formal outputs and a random-user train / actual / prediction plot

### Configuration

- `tft/configs/data.yaml`
  - shared data paths and default formal model / trainer settings
- `tft/configs/tft_shared_pretrain.yaml`
  - shared pretrain run entry
- `tft/configs/tft_c10_finetune.yaml`
  - cluster 10 finetune entry
- `tft/configs/tft_c12_finetune.yaml`
  - cluster 12 finetune entry
- `tft/configs/eval.yaml`
  - cluster-specific evaluation paths
- `tft/configs/infer.yaml`
  - cluster-specific future inference paths

### Source modules

- `tft/src/data/build_tft_dataset.py`
  - feature tables, panels, split boundaries, leakage audit
- `tft/src/data/build_tft_timeseries_dataset.py`
  - dataset metadata and train-fitted dataset parameters
- `tft/src/model/tft_utils.py`
  - dataset helpers, TFT subclass, warmup + plateau schedule, trainer utilities
- `tft/src/model/train_shared_pretrain.py`
  - shared training entry
- `tft/src/model/finetune_cluster.py`
  - per-cluster finetuning entry
- `tft/src/eval/evaluate_tft.py`
  - rolling test evaluation and multi-phase metrics
- `tft/src/infer/predict_future.py`
  - recursive 14-day quantile inference
- `tft/src/postprocess/build_final_user_parquets.py`
  - final merged deliverables
- `tft/src/agent_bridge/export_agent_bundle.py`
  - optional bundle builder for downstream agent workflows

## Recommended Output Retention

Keep:

- `tft/artifacts/models/`
- `tft/artifacts/eval/`
- `tft/artifacts/infer/`
- `tft/artifacts/final/`

Optional / disposable:

- `lightning_logs/`
- ad hoc terminal log files

The training code now disables the default Lightning logger, so future runs should not keep expanding `lightning_logs/` unless another logger is explicitly enabled.

## Notes

- The formal pipeline targets local single-GPU training.
- The current configuration is tuned for an RTX 5070-class local setup and prioritizes stability over absolute maximum throughput.
- `triton` is not required for correctness; related warnings can be ignored.
- `persistent_workers=True` is intentionally not enabled because the local Windows environment was observed to be unstable with it.
