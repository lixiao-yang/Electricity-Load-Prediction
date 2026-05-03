from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from electricity_agent.config import CHAT_HISTORY_PATH, OPENAI_API_KEY, OPENAI_SUMMARY_MODEL
from electricity_agent.prompts import FORECAST_AGENT_SYSTEM_PROMPT, build_summary_payload
from electricity_agent.query_parser import parse_request_options_with_llm
from electricity_agent.tools import (
    get_meter_forecast,
    get_meter_history,
    load_manifest,
    resolve_meter_strict,
)


st.set_page_config(page_title="Electricity Load Forecast Query Agent", layout="wide")

MAX_SAVED_QUERIES = 5

CLUSTER_GUIDE: List[Dict[str, str]] = [
    {
        "cluster": "C1 / C11",
        "title": "DeepAR Customer Groups",
        "model": "deepar",
        "description": "Clusters 1 and 11 use DeepAR user-level evaluation rows and 3-month future prediction artifacts.",
    },
    {
        "cluster": "C2 / C3",
        "title": "DeepAR Customer Groups",
        "model": "deepar",
        "description": "Clusters 2 and 3 use DeepAR user-level test rows and 3-month future prediction artifacts.",
    },
    {
        "cluster": "C6",
        "title": "Single High-Load Customer",
        "model": "direct_xgboost",
        "description": "Cluster 6 contains MT_362, a high-load customer served by the direct trend model.",
    },
    {
        "cluster": "C7",
        "title": "DeepAR Customer Group",
        "model": "deepar",
        "description": "Cluster 7 uses DeepAR validation rows and 3-month future prediction artifacts.",
    },
    {
        "cluster": "C10 / C12",
        "title": "TFT Customer Groups",
        "model": "tft_c10_ft / tft_c12_ft",
        "description": "Clusters 10 and 12 use the finalized TFT user-level test and 14-day future prediction artifacts.",
    },
]


def main() -> None:
    if "forecast_mode" not in st.session_state:
        st.session_state["forecast_mode"] = "evaluation"

    st.title("Electricity Load Forecast Query Agent")
    st.caption("Ask for an electricity load forecast by meter ID. Connected artifacts include DeepAR, cluster 6, and TFT outputs.")

    with st.sidebar:
        st.markdown(
            """
            <div style="padding:0.9rem 1rem;border:2px solid #facc15;border-radius:0.9rem;background:#fff9db;margin-bottom:0.75rem;">
                <div style="font-size:0.82rem;font-weight:700;color:#92400e;letter-spacing:0.03em;text-transform:uppercase;">
                    Forecast Mode
                </div>
                <div style="font-size:0.95rem;color:#5b3b00;margin-top:0.2rem;">
                    Future mode uses available future forecast artifacts. Evaluation mode uses held-out test or validation predictions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        future_mode_enabled = st.toggle(
            "Future mode",
            value=st.session_state.get("forecast_mode", "future") == "future",
            help="Off: evaluation rows from test or validation predictions. On: future forecast rows.",
        )
        st.session_state["forecast_mode"] = "future" if future_mode_enabled else "evaluation"
        st.caption("Mode: Future forecast" if future_mode_enabled else "Mode: Evaluation forecast")

        st.subheader("Artifacts")
        manifest = load_manifest()
        if manifest is None:
            st.warning("No connected forecast artifacts yet.")
        else:
            st.success(manifest.get("status", "Agent registry ready"))
            st.caption(f"Meters registered: {manifest.get('meters', 0)}")
            st.caption(f"Forecast interface: {manifest.get('forecast_interface', 'connected')}")

        llm_status = _get_llm_runtime_status()
        if llm_status["available"]:
            st.success(f"LLM enabled: {llm_status['summary_model']}")
        else:
            st.warning(f"LLM disabled: {llm_status['reason']}")

        st.info("Rebuild controls are disabled. The agent reads standardized forecast bundles in `electricity_agent/artifacts/`.")

        st.divider()
        st.subheader("Cluster Guide")
        for item in CLUSTER_GUIDE:
            st.markdown(
                (
                    f"**{item['cluster']} · {item['title']}**  \n"
                    f"Model: `{item['model']}`  \n"
                    f"{item['description']}"
                )
            )

        if st.button("Clear current chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        transcript_bytes = _conversation_to_json_bytes(st.session_state.get("messages", []))
        st.download_button(
            "Download conversation JSON",
            data=transcript_bytes,
            file_name="electricity_forecast_agent_conversation.json",
            mime="application/json",
            use_container_width=True,
        )

        st.divider()
        st.subheader("Recent queries")
        history_items = _load_query_history()
        if not history_items:
            st.caption("No saved query history yet.")
        else:
            if st.button("Clear recent queries", key="clear_recent_queries", use_container_width=True):
                _clear_query_history()
                st.rerun()
        for idx, item in enumerate(reversed(history_items)):
            request_options = item.get("request_options") or {}
            mode = str(request_options.get("mode", "future"))
            mode_prefix = "[Future]" if mode == "future" else "[Eval]"
            label = f"{mode_prefix} {item.get('user_query', 'query')} | {item.get('meter_id', '')}"
            if st.button(label, key=f"history_query_{idx}", use_container_width=True):
                _replay_history_item(item)
                st.rerun()

        st.divider()
        st.markdown("Query examples")
        st.code("Evaluate actual vs predicted for MT_098 over 48 hours")
        st.code("Show me a daily future forecast for MT_003 for 14 days")
        st.code("Why does the MT_002 DeepAR forecast change at the end of evaluation?")
        st.code("Evaluate actual vs predicted for MT_090 over 48 hours")
        st.code("Compare the evaluation forecast for MT_013 over 7 days")
        st.code("Show me a daily forecast for MT_362 for 14 days")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for idx, message in enumerate(st.session_state.messages):
        _render_chat_message(message, idx)

    user_query = st.chat_input("Enter a meter ID, for example: hourly forecast for MT_362 for 48 hours")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        _handle_query(user_query)
        st.rerun()


def _handle_query(user_query: str) -> None:
    selected_mode = st.session_state.get("forecast_mode", "future")
    request_options = parse_request_options_with_llm(
        user_query,
        default_mode=selected_mode,
    )
    request_options["mode"] = selected_mode
    request_options["mode_source"] = "ui"
    lookup_query = str(request_options.get("meter_query") or user_query).strip()
    context = _get_last_assistant_context()

    resolution = resolve_meter_strict(lookup_query)
    if resolution["status"] != "resolved":
        context_meter_id = str(context.get("meter_id") or "")
        if context_meter_id:
            resolution = resolve_meter_strict(context_meter_id)
            request_options["meter_query"] = context_meter_id
            request_options["meter_source"] = "conversation_context"

    if resolution["status"] != "resolved":
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "I could not determine which meter this question refers to. "
                    "Please include a registered meter ID, for example `MT_362`."
                ),
            }
        )
        return

    if request_options.get("analysis_type") in {"diagnostics", "comparison", "evaluation"}:
        context_options = context.get("request_options") or {}
        if request_options.get("horizon_source") == "default" and context_options.get("horizon_hours"):
            request_options["horizon_hours"] = context_options["horizon_hours"]
            request_options["horizon_source"] = "conversation_context"
        if request_options.get("granularity_source") == "default" and context_options.get("granularity"):
            request_options["granularity"] = context_options["granularity"]
            request_options["granularity_source"] = "conversation_context"

    response = _build_forecast_response(
        meter_id=resolution["match"]["meter_id"],
        user_query=user_query,
        request_options=request_options,
    )
    st.session_state.messages.append(response)
    _append_query_history(response)


def _build_forecast_response(meter_id: str, user_query: str, request_options: Dict[str, object]) -> Dict[str, Any]:
    horizon_hours = int(request_options["horizon_hours"])
    granularity = str(request_options["granularity"])
    mode = str(request_options.get("mode", "future"))
    result = get_meter_forecast(meter_id=meter_id, horizon_hours=horizon_hours, mode=mode)
    metadata = result["metadata"]
    forecast_df = result["forecast"].copy()
    display_df = _prepare_display_frame(forecast_df=forecast_df, granularity=granularity)
    history_df = get_meter_history(meter_id=meter_id)
    chart_df = _prepare_chart_frame(forecast_df=forecast_df, granularity=granularity, history_df=history_df)
    computed_insights = _compute_forecast_insights(
        forecast_df=display_df,
        history_df=history_df,
        granularity=granularity,
        mode=mode,
    )
    summary = _generate_summary(
        user_query=user_query,
        metadata=metadata,
        forecast_df=display_df,
        granularity=granularity,
        placeholder_message=result.get("message", ""),
        request_options=request_options,
        computed_insights=computed_insights,
    )

    info_lines = [
        f"**Meter**: {_display_value(metadata.get('meter_id'))}",
        f"**Cluster**: {_display_value(metadata.get('cluster'))}",
        f"**Model**: {_display_value(metadata.get('model_name'))}",
        f"**Mode**: {mode}",
        f"**Mode source**: {_display_value(request_options.get('mode_source'))}",
        f"**Meter source**: {_display_value(request_options.get('meter_source', 'query'))}",
        f"**Requested horizon**: {horizon_hours} hours",
        f"**Horizon source**: {_display_value(request_options.get('horizon_source', 'query'))}",
        f"**Returned periods**: {len(display_df)}",
        f"**Display granularity**: {granularity}",
        f"**Granularity source**: {_display_value(request_options.get('granularity_source', 'query'))}",
        f"**Analysis type**: {_display_value(request_options.get('analysis_type'))}",
        f"**Intent parser**: {_display_value(request_options.get('intent_source'))}",
        f"**LLM runtime**: {_display_value(_get_llm_runtime_status()['reason'])}",
        f"**Forecast interface**: {_display_value(metadata.get('forecast_interface_status'))}",
    ]
    if request_options.get("intent_fallback_reason"):
        info_lines.append(f"**Intent fallback reason**: {_display_value(request_options.get('intent_fallback_reason'))}")
    if request_options.get("intent_confidence") is not None:
        info_lines.append(f"**Intent confidence**: {float(request_options['intent_confidence']):.2f}")
    metric_mape = metadata.get("metric_test_mape_0_100")
    metric_wmape = metadata.get("metric_test_wmape_0_100")
    if metric_mape is not None and not pd.isna(metric_mape):
        info_lines.append(f"**Test MAPE (0-100 capped)**: {float(metric_mape):.2f}%")
    if metric_wmape is not None and not pd.isna(metric_wmape):
        info_lines.append(f"**Test WMAPE (0-100 capped)**: {float(metric_wmape):.2f}%")

    return {
        "role": "assistant",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_query": user_query,
        "meter_id": metadata.get("meter_id"),
        "request_options": {
            "horizon_hours": horizon_hours,
            "granularity": granularity,
            "mode": mode,
            "mode_source": request_options.get("mode_source"),
            "meter_source": request_options.get("meter_source", "query"),
            "horizon_source": request_options.get("horizon_source"),
            "granularity_source": request_options.get("granularity_source"),
            "analysis_type": request_options.get("analysis_type"),
            "intent_source": request_options.get("intent_source"),
        },
        "content": "  \n".join(info_lines) + "\n\n" + summary,
        "table": _serialize_records(display_df.to_dict(orient="records")),
        "chart": _serialize_records(chart_df.to_dict(orient="records")),
        "chart_granularity": granularity,
    }


def _generate_summary(
    user_query: str,
    metadata: Dict[str, Any],
    forecast_df: pd.DataFrame,
    granularity: str,
    placeholder_message: str,
    request_options: Dict[str, Any],
    computed_insights: Dict[str, Any],
) -> str:
    if forecast_df.empty:
        return placeholder_message or "Forecast data interface is not connected yet."

    try:
        from openai import OpenAI
    except Exception:
        return _fallback_summary(
            forecast_df=forecast_df,
            granularity=granularity,
            request_options=request_options,
            computed_insights=computed_insights,
        )

    if not OPENAI_API_KEY:
        return _fallback_summary(
            forecast_df=forecast_df,
            granularity=granularity,
            request_options=request_options,
            computed_insights=computed_insights,
        )

    client = OpenAI(api_key=OPENAI_API_KEY)
    payload = build_summary_payload(
        user_query=user_query,
        metadata=metadata,
        forecast_rows=forecast_df.to_dict(orient="records"),
        request_options=request_options,
        computed_insights=computed_insights,
    )
    try:
        response = client.responses.create(
            model=OPENAI_SUMMARY_MODEL,
            input=[
                {"role": "system", "content": FORECAST_AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
        )
        text = getattr(response, "output_text", "") or ""
        if text.strip():
            return text.strip()
    except Exception:
        pass
    return _fallback_summary(
        forecast_df=forecast_df,
        granularity=granularity,
        request_options=request_options,
        computed_insights=computed_insights,
    )


def _prepare_display_frame(forecast_df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if forecast_df.empty:
        return pd.DataFrame(columns=["forecast_timestamp", "forecast_value", "model_name"])

    rows = forecast_df.copy()
    rows["forecast_timestamp"] = pd.to_datetime(rows["forecast_timestamp"], errors="coerce")
    rows["forecast_value"] = pd.to_numeric(rows["forecast_value"], errors="coerce")
    if "actual_value" in rows.columns:
        rows["actual_value"] = pd.to_numeric(rows["actual_value"], errors="coerce")
    if "prediction_lower" in rows.columns:
        rows["prediction_lower"] = pd.to_numeric(rows["prediction_lower"], errors="coerce")
    if "prediction_upper" in rows.columns:
        rows["prediction_upper"] = pd.to_numeric(rows["prediction_upper"], errors="coerce")

    if granularity == "daily":
        return (
            rows.set_index("forecast_timestamp")
            .resample("D")
            .agg(
                {
                    "forecast_value": _sum_preserve_missing,
                    "actual_value": _sum_preserve_missing,
                    "prediction_lower": _sum_preserve_missing,
                    "prediction_upper": _sum_preserve_missing,
                    "model_name": "first",
                }
            )
            .reset_index()
        )
    if granularity == "weekly":
        return (
            rows.set_index("forecast_timestamp")
            .resample("W-MON", label="left", closed="left")
            .agg(
                {
                    "forecast_value": _sum_preserve_missing,
                    "actual_value": _sum_preserve_missing,
                    "prediction_lower": _sum_preserve_missing,
                    "prediction_upper": _sum_preserve_missing,
                    "model_name": "first",
                }
            )
            .reset_index()
        )
    display_cols = [
        "forecast_timestamp",
        "forecast_value",
        "actual_value",
        "prediction_lower",
        "prediction_upper",
        "model_name",
    ]
    return rows[[col for col in display_cols if col in rows.columns]]


def _sum_preserve_missing(values: pd.Series) -> float:
    return values.sum(min_count=1)


def _prepare_chart_frame(
    forecast_df: pd.DataFrame,
    granularity: str,
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    history = history_df.copy()
    if not history.empty:
        history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")
        history["actual_value"] = pd.to_numeric(history["actual_value"], errors="coerce")
        history["forecast_value"] = float("nan")
        history["segment"] = "history"
        history = history.rename(columns={"timestamp": "forecast_timestamp"})
    else:
        history = pd.DataFrame(columns=["forecast_timestamp", "actual_value", "forecast_value", "segment"])

    forecast_display = _prepare_display_frame(forecast_df=forecast_df, granularity=granularity).copy()
    if not forecast_display.empty:
        if "actual_value" not in forecast_display.columns:
            forecast_display["actual_value"] = float("nan")
        forecast_display["segment"] = "forecast"
    else:
        forecast_display = pd.DataFrame(columns=["forecast_timestamp", "forecast_value", "model_name", "actual_value", "segment"])

    combined = pd.concat(
        [
            history[["forecast_timestamp", "actual_value", "forecast_value", "segment"]],
            forecast_display[["forecast_timestamp", "actual_value", "forecast_value", "segment"]],
        ],
        ignore_index=True,
    )
    return combined.sort_values(["forecast_timestamp", "segment"]).reset_index(drop=True)


def _compute_forecast_insights(
    forecast_df: pd.DataFrame,
    history_df: pd.DataFrame,
    granularity: str,
    mode: str,
) -> Dict[str, Any]:
    if forecast_df.empty:
        return {"row_count": 0, "granularity": granularity, "mode": mode}

    rows = forecast_df.copy()
    rows["forecast_timestamp"] = pd.to_datetime(rows["forecast_timestamp"], errors="coerce")
    values = pd.to_numeric(rows["forecast_value"], errors="coerce")
    insights: Dict[str, Any] = {
        "row_count": int(len(rows)),
        "granularity": granularity,
        "mode": mode,
        "start": rows["forecast_timestamp"].min(),
        "end": rows["forecast_timestamp"].max(),
        "forecast_total": float(values.sum(skipna=True)),
        "forecast_average": float(values.mean(skipna=True)),
        "forecast_min": float(values.min(skipna=True)),
        "forecast_max": float(values.max(skipna=True)),
    }
    if len(values.dropna()) >= 2:
        insights["forecast_first_value"] = float(values.dropna().iloc[0])
        insights["forecast_last_value"] = float(values.dropna().iloc[-1])
        insights["forecast_last_minus_first"] = float(values.dropna().iloc[-1] - values.dropna().iloc[0])
    if len(values.dropna()) >= 24:
        insights["forecast_last_24h_average"] = float(values.dropna().tail(24).mean())
        insights["forecast_previous_24h_average"] = float(values.dropna().iloc[-48:-24].mean()) if len(values.dropna()) >= 48 else None

    if "actual_value" in rows.columns and rows["actual_value"].notna().any():
        actual = pd.to_numeric(rows["actual_value"], errors="coerce")
        absolute_error = (actual - values).abs()
        denom = actual.abs().replace(0, pd.NA)
        ape = (absolute_error / denom * 100.0).dropna()
        insights.update(
            {
                "actual_total": float(actual.sum(skipna=True)),
                "actual_average": float(actual.mean(skipna=True)),
                "mean_absolute_error": float(absolute_error.mean(skipna=True)),
                "mean_absolute_percentage_error": float(ape.mean()) if not ape.empty else None,
            }
        )
        actual_nonmissing = actual.dropna()
        if len(actual_nonmissing) >= 2:
            insights["actual_first_value"] = float(actual_nonmissing.iloc[0])
            insights["actual_last_value"] = float(actual_nonmissing.iloc[-1])
            insights["actual_last_minus_first"] = float(actual_nonmissing.iloc[-1] - actual_nonmissing.iloc[0])
        if len(actual_nonmissing) >= 24:
            insights["actual_last_24h_average"] = float(actual_nonmissing.tail(24).mean())
            insights["actual_previous_24h_average"] = float(actual_nonmissing.iloc[-48:-24].mean()) if len(actual_nonmissing) >= 48 else None

    if not history_df.empty:
        history_values = pd.to_numeric(history_df["actual_value"], errors="coerce").dropna()
        if not history_values.empty:
            insights.update(
                {
                    "history_periods_used_for_chart": int(len(history_values)),
                    "recent_history_average": float(history_values.mean()),
                    "recent_history_last_value": float(history_values.iloc[-1]),
                }
            )

    return insights


def _fallback_summary(
    forecast_df: pd.DataFrame,
    granularity: str,
    request_options: Dict[str, Any] | None = None,
    computed_insights: Dict[str, Any] | None = None,
) -> str:
    if forecast_df.empty:
        return "Forecast rows are not available yet. The meter registry is ready, but the prediction data interface is still a placeholder."

    request_options = request_options or {}
    computed_insights = computed_insights or {}
    analysis_type = str(request_options.get("analysis_type") or "forecast")
    if analysis_type in {"diagnostics", "comparison", "evaluation"}:
        start = _display_value(computed_insights.get("start"))
        end = _display_value(computed_insights.get("end"))
        forecast_avg = computed_insights.get("forecast_average")
        actual_avg = computed_insights.get("actual_average")
        mape_value = computed_insights.get("mean_absolute_percentage_error")
        mae_value = computed_insights.get("mean_absolute_error")
        pieces = [
            f"This is an artifact-grounded diagnostic over {len(forecast_df)} {granularity} periods from {start} to {end}.",
        ]
        if forecast_avg is not None and actual_avg is not None:
            direction = "above" if float(forecast_avg) > float(actual_avg) else "below"
            pieces.append(
                f"The forecast average is {float(forecast_avg):.2f}, which is {direction} the actual average of {float(actual_avg):.2f}."
            )
        if mae_value is not None:
            pieces.append(f"The mean absolute error over this window is {float(mae_value):.2f}.")
        if mape_value is not None:
            pieces.append(f"The window-level MAPE is {float(mape_value):.2f}%.")
        last_24 = computed_insights.get("forecast_last_24h_average")
        prev_24 = computed_insights.get("forecast_previous_24h_average")
        if last_24 is not None and prev_24 is not None:
            direction = "higher" if float(last_24) > float(prev_24) else "lower"
            pieces.append(
                f"In the forecast rows, the last 24-hour average is {float(last_24):.2f}, {direction} than the previous 24-hour average of {float(prev_24):.2f}."
            )
        actual_last_24 = computed_insights.get("actual_last_24h_average")
        actual_prev_24 = computed_insights.get("actual_previous_24h_average")
        if actual_last_24 is not None and actual_prev_24 is not None:
            direction = "higher" if float(actual_last_24) > float(actual_prev_24) else "lower"
            pieces.append(
                f"In the actual rows, the last 24-hour average is {float(actual_last_24):.2f}, {direction} than the previous 24-hour average of {float(actual_prev_24):.2f}."
            )
        pieces.append(
            "The late-window rise should therefore be read as model behavior in the selected evaluation slice, not as a new longer-horizon forecast."
        )
        pieces.append(
            "For a causal explanation, enable the OpenAI API key so the LLM can reason over the supplied rows, metrics, and recent history without inventing values."
        )
        return " ".join(pieces)

    values = pd.to_numeric(forecast_df["forecast_value"], errors="coerce").fillna(0.0)
    return f"Across {len(values)} {granularity} periods, expected load totals {values.sum():.2f} with an average of {values.mean():.2f}."


def _get_llm_runtime_status() -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return {
            "available": False,
            "reason": "missing_openai_api_key",
            "summary_model": OPENAI_SUMMARY_MODEL,
        }
    try:
        import openai  # noqa: F401
    except Exception as exc:
        return {
            "available": False,
            "reason": f"openai_import_failed:{type(exc).__name__}",
            "summary_model": OPENAI_SUMMARY_MODEL,
        }
    return {
        "available": True,
        "reason": "available",
        "summary_model": OPENAI_SUMMARY_MODEL,
    }


def _render_chat_message(message: Dict[str, Any], idx: int) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message.get("content", ""))
        table = message.get("table")
        if table:
            st.dataframe(pd.DataFrame(table), use_container_width=True)
        chart = message.get("chart")
        if chart:
            chart_df = pd.DataFrame(chart)
            if not chart_df.empty and "forecast_timestamp" in chart_df.columns:
                chart_df["forecast_timestamp"] = pd.to_datetime(chart_df["forecast_timestamp"], errors="coerce")
                plot_df = chart_df.set_index("forecast_timestamp")
                columns = [col for col in ["actual_value", "forecast_value"] if col in plot_df.columns]
                if columns:
                    st.line_chart(plot_df[columns])


def _get_last_assistant_context() -> Dict[str, Any]:
    for message in reversed(st.session_state.get("messages", [])):
        meter_id = str(message.get("meter_id") or "").strip()
        if meter_id:
            return {
                "meter_id": meter_id,
                "request_options": message.get("request_options") or {},
            }
    return {}


def _conversation_to_json_bytes(messages: List[Dict[str, Any]]) -> bytes:
    return json.dumps(messages, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def _load_query_history() -> List[Dict[str, Any]]:
    if not CHAT_HISTORY_PATH.exists():
        return []
    try:
        with CHAT_HISTORY_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _save_query_history(items: List[Dict[str, Any]]) -> None:
    CHAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHAT_HISTORY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(items[-MAX_SAVED_QUERIES:], handle, ensure_ascii=False, indent=2, default=str)


def _append_query_history(response: Dict[str, Any]) -> None:
    items = _load_query_history()
    items.append(
        {
            "timestamp": response.get("timestamp"),
            "user_query": response.get("user_query"),
            "meter_id": response.get("meter_id"),
            "request_options": response.get("request_options"),
        }
    )
    _save_query_history(items)


def _clear_query_history() -> None:
    _save_query_history([])


def _replay_history_item(item: Dict[str, Any]) -> None:
    query = str(item.get("user_query") or "")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        _handle_query(query)


def _display_value(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass
    return str(value)


def _serialize_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return json.loads(json.dumps(records, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
