# Electricity Load Forecast Query Agent

This is a project-specific Streamlit agent for the electricity load forecasting project.

The UI follows the previous retail forecasting agent structure:

- sidebar mode toggle
- artifact status panel
- cluster guide
- current chat controls
- conversation JSON download
- recent queries
- query examples
- chat-based forecast lookup

Intentional differences:

- Rebuild buttons and asset build flows are removed.
- Forecast rows are connected through standardized CSV bundles in `electricity_agent/artifacts/`.
- `electricity_agent.tools.get_meter_forecast()` routes requests to cluster 6, TFT cluster 10, or TFT cluster 12 outputs.
- DeepAR outputs are read dynamically from `deepar/output/*.parquet` and merged with the existing agent registry, forecast rows, and metrics at runtime. They are not written back into the large CSV bundle.
- When `OPENAI_API_KEY` is configured, the LLM parses user intent before artifact lookup. The deterministic registry and forecast bundle still validate meter IDs, modes, horizons, and returned forecast rows.
- If the LLM is unavailable, the agent falls back to the rule-based parser and reports `Intent parser: rules`.

Connected forecast artifacts:

- DeepAR future: `deepar/output/future_3months_predictions_1_11.parquet`
- DeepAR future: `deepar/output/future_3months_predictions_2_3.parquet`
- DeepAR future: `deepar/output/future_3months_predictions_7.parquet`
- DeepAR evaluation/test: selected DeepAR prediction parquet files under `deepar/output/`
- Cluster 6: `cluster6/artifacts/infer/direct_trend/cluster6_final_model_future_14d_predictions.parquet`
- Cluster 6 evaluation: `cluster6/artifacts/eval/direct_trend/final_test_forecast_detail.parquet`
- TFT future: `tft/artifacts/final/user_level_future_predictions_14d.parquet`
- TFT evaluation: `tft/artifacts/final/user_level_test_predictions.parquet`

Agent runtime artifacts:

- `electricity_agent/artifacts/meter_registry.csv`
- `electricity_agent/artifacts/forecast_bundle.csv`
- `electricity_agent/artifacts/metrics_bundle.csv`
- `electricity_agent/artifacts/manifest.json`

Run locally:

```bash
streamlit run electricity_agent/app.py
```

Optional environment variables:

- `OPENAI_API_KEY`: enables LLM intent parsing and artifact-grounded summaries.
- `OPENAI_INTENT_MODEL`: model used for structured intent parsing.
- `OPENAI_SUMMARY_MODEL`: model used for final forecast explanation.

If the UI reports `LLM disabled: openai_import_failed:ModuleNotFoundError`, install the OpenAI Python package in the same environment used to launch Streamlit:

```bash
python -m pip install openai
```
