from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from electricity_agent import AGENT_ROOT, PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

ARTIFACTS_DIR = AGENT_ROOT / "artifacts"
REGISTRY_PATH = ARTIFACTS_DIR / "meter_registry.csv"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
CHAT_HISTORY_PATH = ARTIFACTS_DIR / "chat_history.json"
FORECAST_BUNDLE_PATH = ARTIFACTS_DIR / "forecast_bundle.csv"
METRICS_BUNDLE_PATH = ARTIFACTS_DIR / "metrics_bundle.csv"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_SUMMARY_MODEL = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
OPENAI_INTENT_MODEL = os.getenv("OPENAI_INTENT_MODEL", OPENAI_SUMMARY_MODEL).strip() or OPENAI_SUMMARY_MODEL

DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "train_hourly_preprocessed.parquet"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "test_hourly_preprocessed.parquet"

TFT_FUTURE_PREDICTIONS_PATH = PROJECT_ROOT / "tft" / "artifacts" / "final" / "user_level_future_predictions_14d.parquet"
TFT_TEST_PREDICTIONS_PATH = PROJECT_ROOT / "tft" / "artifacts" / "final" / "user_level_test_predictions.parquet"
TFT_METRICS_PATH = PROJECT_ROOT / "tft" / "artifacts" / "final" / "multiphase_metrics_overall.parquet"

CLUSTER6_FUTURE_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "cluster6"
    / "artifacts"
    / "infer"
    / "direct_trend"
    / "cluster6_final_model_future_14d_predictions.parquet"
)
CLUSTER6_TEST_PREDICTIONS_PATH = (
    PROJECT_ROOT / "cluster6" / "artifacts" / "eval" / "direct_trend" / "final_test_forecast_detail.parquet"
)
CLUSTER6_METRICS_PATH = (
    PROJECT_ROOT / "cluster6" / "artifacts" / "eval" / "direct_trend" / "final_test_results.parquet"
)

DEEPAR_FUTURE_PREDICTIONS_PATHS = (
    PROJECT_ROOT / "deepar" / "output" / "future_3months_predictions_1_11.parquet",
    PROJECT_ROOT / "deepar" / "output" / "future_3months_predictions_2_3.parquet",
    PROJECT_ROOT / "deepar" / "output" / "future_3months_predictions_7.parquet",
)
DEEPAR_EVALUATION_PREDICTIONS_PATHS = (
    PROJECT_ROOT
    / "deepar"
    / "output"
    / "deepar_random_search_1_11"
    / "trial_07"
    / "deepar_clusters_1_11_predictions.parquet",
    PROJECT_ROOT / "deepar" / "output" / "deepar_clusters_2_3" / "deepar_clusters_2_3_predictions.parquet",
    PROJECT_ROOT
    / "deepar"
    / "output"
    / "deepar_random_search_7"
    / "trial_01"
    / "deepar_clusters_7_validation_predictions.parquet",
)
DEEPAR_MODEL_NAME = "deepar"

DEFAULT_HORIZON_HOURS = 24 * 14
MAX_HORIZON_HOURS = 24 * 90


def ensure_agent_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
