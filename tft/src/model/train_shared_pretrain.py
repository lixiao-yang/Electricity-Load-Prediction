from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tft.src.model.tft_utils import (
    build_eval_dataset_from_training,
    build_tft_model,
    build_trainer,
    build_training_dataset,
    observed_only,
    save_dataset_parameters,
    split_time_index_bounds,
)
from tft.src.pipeline_utils import ensure_dir, load_yaml_config, set_seed, write_json, write_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train shared TFT pretraining model on clusters 10+12.")
    parser.add_argument("--config", type=str, required=True, help="Path to shared pretrain YAML config.")
    parser.add_argument("--panel-path", type=str, default=None, help="Override shared panel parquet path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override model output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    set_seed(int(config["project"]["seed"]))

    panel_path = Path(args.panel_path or config["run"]["panel_path"])
    output_dir = ensure_dir(args.output_dir or config["run"]["output_dir"])
    panel_df = pd.read_parquet(panel_path)
    observed_df = observed_only(panel_df)
    split_bounds = split_time_index_bounds(panel_df)

    training_dataset = build_training_dataset(observed_df, config, split_bounds["train_end_idx"])
    validation_source = observed_df.loc[observed_df["time_idx"] <= split_bounds["val_end_idx"]].copy()
    validation_dataset = build_eval_dataset_from_training(training_dataset, validation_source, split_bounds["val_start_idx"])

    batch_size = int(config["model"]["batch_size"])
    num_workers = int(config["model"]["num_workers"])
    train_loader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = validation_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

    model = build_tft_model(training_dataset, config)
    trainer, checkpoint_callback = build_trainer(config, output_dir)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    save_dataset_parameters(training_dataset, output_dir / "dataset_params.pt")
    write_yaml_config(config, output_dir / "resolved_config.yaml")
    write_json(
        {
            "run_name": config["run"]["name"],
            "panel_path": str(panel_path),
            "dataset_params_path": str(output_dir / "dataset_params.pt"),
            "best_checkpoint_path": checkpoint_callback.best_model_path,
            "train_end_idx": split_bounds["train_end_idx"],
            "val_start_idx": split_bounds["val_start_idx"],
            "val_end_idx": split_bounds["val_end_idx"],
        },
        output_dir / "train_manifest.json",
    )


if __name__ == "__main__":
    main()

