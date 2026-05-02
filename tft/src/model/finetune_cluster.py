from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tft.src.model.tft_utils import (
    build_dataset_from_parameters,
    build_trainer,
    load_dataset_parameters,
    load_tft_checkpoint,
    observed_only,
    split_time_index_bounds,
)
from tft.src.pipeline_utils import ensure_dir, load_yaml_config, set_seed, write_json, write_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune one TFT model per cluster from shared pretrain checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to per-cluster finetune YAML config.")
    parser.add_argument("--panel-path", type=str, default=None, help="Override cluster panel parquet path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override finetune output directory.")
    parser.add_argument("--pretrained-checkpoint-path", type=str, default=None, help="Override shared checkpoint path.")
    parser.add_argument("--shared-dataset-params-path", type=str, default=None, help="Override shared dataset params path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    set_seed(int(config["project"]["seed"]))

    panel_path = Path(args.panel_path or config["run"]["panel_path"])
    output_dir = ensure_dir(args.output_dir or config["run"]["output_dir"])
    pretrained_checkpoint_path = Path(args.pretrained_checkpoint_path or config["run"]["pretrained_checkpoint_path"])
    shared_dataset_params_path = Path(args.shared_dataset_params_path or config["run"]["shared_dataset_params_path"])

    panel_df = pd.read_parquet(panel_path)
    observed_df = observed_only(panel_df)
    split_bounds = split_time_index_bounds(panel_df)
    shared_dataset_parameters = load_dataset_parameters(shared_dataset_params_path)

    train_source = observed_df.loc[observed_df["time_idx"] <= split_bounds["train_end_idx"]].copy()
    val_source = observed_df.loc[observed_df["time_idx"] <= split_bounds["val_end_idx"]].copy()

    train_dataset = build_dataset_from_parameters(shared_dataset_parameters, train_source, predict=False, stop_randomization=False)
    val_dataset = build_dataset_from_parameters(
        shared_dataset_parameters,
        val_source,
        predict=False,
        stop_randomization=True,
        min_prediction_idx=split_bounds["val_start_idx"],
    )

    batch_size = int(config["model"]["batch_size"])
    num_workers = int(config["model"]["num_workers"])
    train_loader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

    model = load_tft_checkpoint(pretrained_checkpoint_path, config=config)

    trainer, checkpoint_callback = build_trainer(config, output_dir)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    write_yaml_config(config, output_dir / "resolved_config.yaml")
    write_json(
        {
            "run_name": config["run"]["name"],
            "cluster_id": int(config["run"]["cluster_id"]),
            "panel_path": str(panel_path),
            "shared_dataset_params_path": str(shared_dataset_params_path),
            "pretrained_checkpoint_path": str(pretrained_checkpoint_path),
            "best_checkpoint_path": checkpoint_callback.best_model_path,
            "train_end_idx": split_bounds["train_end_idx"],
            "val_start_idx": split_bounds["val_start_idx"],
            "val_end_idx": split_bounds["val_end_idx"],
        },
        output_dir / "finetune_manifest.json",
    )


if __name__ == "__main__":
    main()
