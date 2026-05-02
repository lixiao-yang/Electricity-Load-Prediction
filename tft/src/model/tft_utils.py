from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Missing torch. Install PyTorch before running the TFT pipeline.") from exc

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
    from pytorch_forecasting.metrics import QuantileLoss
    try:
        from pytorch_forecasting.models.base._base_model import PredictCallback
    except ImportError:  # pragma: no cover
        from pytorch_forecasting.callbacks.predict import PredictCallback
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing pytorch-forecasting. Install pytorch-forecasting and lightning before running the TFT pipeline."
    ) from exc


class WarmupReduceOnPlateauTFT(TemporalFusionTransformer):
    def __init__(
        self,
        *args,
        warmup_steps: int = 0,
        warmup_init_factor: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.warmup_steps = int(warmup_steps)
        self.warmup_init_factor = float(warmup_init_factor)
        self.save_hyperparameters(
            {
                "warmup_steps": self.warmup_steps,
                "warmup_init_factor": self.warmup_init_factor,
            }
        )

    def _target_learning_rate(self) -> float:
        learning_rate = self.hparams.learning_rate
        if isinstance(learning_rate, (list, tuple)):
            learning_rate = learning_rate[0]
        return float(learning_rate)

    def _apply_warmup(self, optimizer) -> None:
        if optimizer is None or self.warmup_steps <= 0 or self.global_step >= self.warmup_steps:
            return
        progress = min(float(self.global_step + 1) / float(self.warmup_steps), 1.0)
        scale = self.warmup_init_factor + (1.0 - self.warmup_init_factor) * progress
        warmed_lr = self._target_learning_rate() * scale
        for param_group in optimizer.param_groups:
            param_group["lr"] = warmed_lr

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs.get("optimizer")
        if optimizer is None and len(args) >= 3:
            optimizer = args[2]
        self._apply_warmup(optimizer)
        return super().optimizer_step(*args, **kwargs)


def observed_only(panel_df: pd.DataFrame) -> pd.DataFrame:
    return panel_df.loc[panel_df["split"] != "future"].copy()


def split_time_index_bounds(panel_df: pd.DataFrame) -> Dict[str, int]:
    observed_df = observed_only(panel_df)
    train_end_idx = int(observed_df.loc[observed_df["split"] == "train", "time_idx"].max())
    val_start_idx = int(observed_df.loc[observed_df["split"] == "val", "time_idx"].min())
    val_end_idx = int(observed_df.loc[observed_df["split"] == "val", "time_idx"].max())
    test_start_idx = int(observed_df.loc[observed_df["split"] == "test", "time_idx"].min())
    test_end_idx = int(observed_df.loc[observed_df["split"] == "test", "time_idx"].max())
    return {
        "train_end_idx": train_end_idx,
        "val_start_idx": val_start_idx,
        "val_end_idx": val_end_idx,
        "test_start_idx": test_start_idx,
        "test_end_idx": test_end_idx,
    }


def feature_spec(config: Dict[str, Any], panel_df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
    static_categoricals = ["user_id", "cluster_id_cat"]
    time_varying_known_categoricals = ["hour_cat", "day_of_week_cat", "month_cat", "is_weekend_cat", "holiday_flag_cat"]
    time_varying_known_reals: List[str] = []
    time_varying_unknown_reals = [config["data"]["target_col"]]
    if bool(config["data"].get("use_target_derived_covariates", False)):
        observed_feature_candidates = [
            "lag_24",
            "lag_48",
            "lag_168",
            "roll_mean_24",
            "roll_std_24",
            "roll_mean_168",
        ]
        available = [column for column in observed_feature_candidates if column in panel_df.columns]
        time_varying_unknown_reals.extend(available)
    return (
        static_categoricals,
        time_varying_known_categoricals,
        time_varying_known_reals,
        time_varying_unknown_reals,
    )


def build_training_dataset(panel_df: pd.DataFrame, config: Dict[str, Any], train_cutoff_time_idx: int) -> TimeSeriesDataSet:
    target_col = config["data"]["target_col"]
    min_encoder_length = int(config["model"]["min_encoder_length"])
    max_encoder_length = int(config["model"]["max_encoder_length"])
    max_prediction_length = int(config["model"]["max_prediction_length"])
    (
        static_categoricals,
        time_varying_known_categoricals,
        time_varying_known_reals,
        time_varying_unknown_reals,
    ) = feature_spec(config, panel_df)

    train_df = panel_df.loc[panel_df["time_idx"] <= train_cutoff_time_idx].copy()
    train_df = train_df.sort_values(["user_id", "time_idx"]).reset_index(drop=True)

    return TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=target_col,
        group_ids=["user_id"],
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=["user_id"]),
        categorical_encoders={
            "user_id": NaNLabelEncoder(add_nan=False),
            "cluster_id_cat": NaNLabelEncoder(add_nan=False),
            "hour_cat": NaNLabelEncoder(add_nan=False),
            "day_of_week_cat": NaNLabelEncoder(add_nan=False),
            "month_cat": NaNLabelEncoder(add_nan=False),
            "is_weekend_cat": NaNLabelEncoder(add_nan=False),
            "holiday_flag_cat": NaNLabelEncoder(add_nan=False),
        },
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )


def build_eval_dataset_from_training(
    training_dataset: TimeSeriesDataSet,
    panel_df: pd.DataFrame,
    prediction_start_idx: int,
) -> TimeSeriesDataSet:
    return TimeSeriesDataSet.from_dataset(
        training_dataset,
        panel_df.copy(),
        min_prediction_idx=prediction_start_idx,
        predict=False,
        stop_randomization=True,
    )


def build_dataset_from_parameters(
    dataset_parameters: Dict[str, Any],
    panel_df: pd.DataFrame,
    *,
    predict: bool,
    stop_randomization: bool,
    min_prediction_idx: int | None = None,
) -> TimeSeriesDataSet:
    kwargs: Dict[str, Any] = {
        "predict": predict,
        "stop_randomization": stop_randomization,
    }
    if min_prediction_idx is not None:
        kwargs["min_prediction_idx"] = min_prediction_idx
    return TimeSeriesDataSet.from_parameters(dataset_parameters, panel_df.copy(), **kwargs)


def build_tft_model(training_dataset: TimeSeriesDataSet, config: Dict[str, Any]) -> WarmupReduceOnPlateauTFT:
    return WarmupReduceOnPlateauTFT.from_dataset(
        training_dataset,
        learning_rate=float(config["model"]["learning_rate"]),
        hidden_size=int(config["model"]["hidden_size"]),
        attention_head_size=int(config["model"]["attention_head_size"]),
        dropout=float(config["model"]["dropout"]),
        hidden_continuous_size=int(config["model"]["hidden_continuous_size"]),
        loss=QuantileLoss(config["model"]["quantiles"]),
        reduce_on_plateau_patience=int(config["model"]["reduce_on_plateau_patience"]),
        warmup_steps=int(config["model"].get("warmup_steps", 0)),
        warmup_init_factor=float(config["model"].get("warmup_init_factor", 0.2)),
        # Half precision can overflow on the default large negative mask bias.
        mask_bias=-float("inf"),
    )


def apply_training_schedule_overrides(model: WarmupReduceOnPlateauTFT, config: Dict[str, Any]) -> WarmupReduceOnPlateauTFT:
    learning_rate = float(config["model"]["learning_rate"])
    warmup_steps = int(config["model"].get("warmup_steps", getattr(model, "warmup_steps", 0)))
    warmup_init_factor = float(
        config["model"].get("warmup_init_factor", getattr(model, "warmup_init_factor", 0.2))
    )
    model.hparams.learning_rate = learning_rate
    model.hparams["warmup_steps"] = warmup_steps
    model.hparams["warmup_init_factor"] = warmup_init_factor
    model.warmup_steps = warmup_steps
    model.warmup_init_factor = warmup_init_factor
    return model


def load_tft_checkpoint(
    checkpoint_path: str | Path,
    config: Dict[str, Any] | None = None,
) -> WarmupReduceOnPlateauTFT:
    model = WarmupReduceOnPlateauTFT.load_from_checkpoint(str(checkpoint_path))
    if config is not None:
        model = apply_training_schedule_overrides(model, config)
    return model


def build_trainer(config: Dict[str, Any], output_dir: str | Path) -> Tuple[pl.Trainer, ModelCheckpoint]:
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=int(config["model"]["early_stop_patience"]),
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    precision = config["model"]["precision"]
    if precision == 16:
        precision = "16-mixed"

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    trainer = pl.Trainer(
        max_epochs=int(config["model"]["max_epochs"]),
        accelerator=config["model"]["accelerator"],
        devices=config["model"]["devices"],
        precision=precision,
        logger=False,
        gradient_clip_val=float(config["model"]["gradient_clip_val"]),
        limit_train_batches=float(config["model"]["limit_train_batches"]),
        limit_val_batches=float(config["model"]["limit_val_batches"]),
        num_sanity_val_steps=int(config["model"].get("num_sanity_val_steps", 2)),
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        default_root_dir=str(output_dir),
        log_every_n_steps=int(config["model"].get("log_every_n_steps", 10)),
    )
    return trainer, checkpoint_callback


def build_predict_trainer(config: Dict[str, Any]) -> pl.Trainer:
    precision = config["model"]["precision"]
    if precision == 16:
        precision = "16-mixed"

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    return pl.Trainer(
        accelerator=config["model"]["accelerator"],
        devices=config["model"]["devices"],
        precision=precision,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def predict_with_trainer(
    trainer: pl.Trainer,
    model: TemporalFusionTransformer,
    dataloader,
    *,
    mode: str,
    mode_kwargs: Dict[str, Any] | None = None,
):
    callback_signature = inspect.signature(PredictCallback.__init__)
    callback_parameters = callback_signature.parameters
    if "return_info" in callback_parameters:
        predict_callback = PredictCallback(
            mode=mode,
            return_info=[],
            mode_kwargs=mode_kwargs or {},
        )
    else:
        predict_callback = PredictCallback(
            mode=mode,
            return_index=False,
            return_decoder_lengths=False,
            write_interval="batch",
            return_x=False,
            mode_kwargs=mode_kwargs or {},
            output_dir=None,
            predict_kwargs={},
            return_y=False,
        )
    trainer.callbacks.append(predict_callback)
    try:
        trainer.predict(model, dataloaders=dataloader)
        return predict_callback.result
    finally:
        trainer.callbacks = [callback for callback in trainer.callbacks if callback is not predict_callback]


def save_dataset_parameters(dataset: TimeSeriesDataSet, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset.get_parameters(), output_path)


def load_dataset_parameters(input_path: str | Path) -> Dict[str, Any]:
    # TimeSeriesDataSet parameters are a trusted local Python object, not a pure tensor state_dict.
    return torch.load(Path(input_path), map_location="cpu", weights_only=False)
