# TFT Workspace

This directory contains the second-stage forecasting workflow used by the project.

The first stage is clustering and preprocessing in:

- `cluster_analysis.ipynb`

Only after that notebook writes the filtered hourly forecasting inputs does the TFT workflow in `tft/` begin.

The primary documentation now lives in the root [README.md](/d:/MSDS/2026spring/Forecasting/Project2/Electricity-Load-Prediction/README.md). Use that file for:

- end-to-end pipeline flow
- input / output contracts
- formal training parameters
- full command sequence
- file responsibilities

## What Lives Here

```text
tft/
  configs/
  src/
  artifacts/
  sanity_check.ipynb
```

## Main Subdirectories

- `configs/`
  - formal and quick YAML configs
- `src/data/`
  - dataset and TimeSeriesDataSet builders
- `src/model/`
  - shared pretrain, finetune, trainer, warmup + plateau schedule
- `src/eval/`
  - rolling test evaluation and P1/P2/P3 metrics
- `src/infer/`
  - recursive future 14-day quantile inference
- `src/postprocess/`
  - final user-level parquet builder
- `src/agent_bridge/`
  - optional bundle export for downstream agent workflows
- `artifacts/`
  - formal outputs

## Main Entry Points

- `tft/src/data/build_tft_dataset.py`
- `tft/src/data/build_tft_timeseries_dataset.py`
- `tft/src/model/train_shared_pretrain.py`
- `tft/src/model/finetune_cluster.py`
- `tft/src/eval/evaluate_tft.py`
- `tft/src/infer/predict_future.py`
- `tft/src/postprocess/build_final_user_parquets.py`

## Upstream Clustering Prerequisite

The TFT workflow starts after the upstream clustering notebook finishes:

- `cluster_analysis.ipynb`

That notebook is responsible for:

- aggregating the raw 15-minute panel to hourly load
- defining the rolling split with the final `3` natural months as test
- keeping only eligible active meters using train-only history
- fitting clustering models and exporting meter-level cluster labels
- writing the hourly train / test parquet files consumed by TFT

The minimum required upstream files are:

- `data/train_hourly_preprocessed.parquet`
- `data/test_hourly_preprocessed.parquet`
- `data/extended-clustering-high-cov/clusters_3models.parquet`

The downstream TFT pipeline then filters the KMeans labels to clusters `10` and `12`.

## Formal Commands

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

## Quick Smoke Commands

```bash
python -m py_compile tft\src\model\tft_utils.py tft\src\model\train_shared_pretrain.py tft\src\model\finetune_cluster.py tft\src\eval\evaluate_tft.py tft\src\infer\predict_future.py
python -m tft.src.model.train_shared_pretrain --config tft/configs/tft_shared_pretrain_quick.yaml
python -m tft.src.model.finetune_cluster --config tft/configs/tft_c10_finetune_quick.yaml
python -m tft.src.eval.evaluate_tft --config tft/configs/data_quick.yaml --panel-path tft/artifacts_quick/data/cluster_10_panel.parquet --dataset-params-path tft/artifacts_quick/models/tft_shared_pretrain/dataset_params.pt --checkpoint-path tft/artifacts_quick/models/tft_c10_ft/checkpoints/best.ckpt --cluster-id 10 --model-name tft_c10_ft_quick --output-dir tft/artifacts_quick/eval/tft_c10_ft
```

## Notebook Check

After a formal run finishes, use:

- `tft/sanity_check.ipynb`

It checks formal artifacts and plots one random user's train history, test actuals, and aggregated test predictions.
