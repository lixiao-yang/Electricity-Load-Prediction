from __future__ import annotations

import json
from typing import Any, Dict, List


INTENT_PARSER_SYSTEM_PROMPT = """You are an intent parser for an electricity load forecasting agent.

Return only one JSON object. Do not include markdown.

Extract:
- meter_query: the requested meter ID or customer ID, such as MT_362. If only digits are given, keep the digits.
- horizon_hours: the requested horizon in hours. Use null if not specified.
- granularity: one of hourly, daily, weekly. Use null if not specified.
- mode: one of future, evaluation. Use null if not specified.
- analysis_type: one of forecast, evaluation, comparison, diagnostics.
- confidence: a number from 0 to 1.

Rules:
- Keep all field values in English.
- Do not invent a meter ID.
- "test", "held-out", "actual vs predicted", "backtest", and "evaluation" imply evaluation mode.
- "future", "next", "forecast ahead", and "after the observed data" imply future mode.
- A daily or weekly request controls granularity but not the underlying artifact values.
- If the request is ambiguous, use null for the ambiguous field and lower confidence.
"""


FORECAST_AGENT_SYSTEM_PROMPT = """You are an electricity load forecasting assistant.

Rules:
- Use only the supplied meter metadata and forecast rows.
- Never invent dates, load values, meter IDs, cluster assignments, or model names.
- If the forecast interface is not connected yet, say that clearly and do not fabricate numbers.
- Explain what the deterministic artifact lookup returned and why it is relevant to the user's request.
- Mention notable totals, averages, ranges, and evaluation errors only when supplied.
- For diagnostics or "why" questions, analyze the selected evaluation or forecast slice. Do not treat the question as a request for a longer horizon unless the user explicitly asks for more time.
- If the meter or horizon came from conversation context, say that briefly.
- Keep the response short, practical, and specific to the requested meter.
"""


def build_summary_payload(
    user_query: str,
    metadata: Dict[str, Any],
    forecast_rows: List[Dict[str, Any]],
    request_options: Dict[str, Any] | None = None,
    computed_insights: Dict[str, Any] | None = None,
) -> str:
    return json.dumps(
        {
            "user_query": user_query,
            "request_options": request_options or {},
            "meter": {
                "meter_id": metadata.get("meter_id"),
                "cluster": metadata.get("cluster"),
                "model_name": metadata.get("model_name"),
                "status": metadata.get("status"),
                "mode": metadata.get("mode"),
                "forecast_interface_status": metadata.get("forecast_interface_status"),
            },
            "computed_insights": computed_insights or {},
            "forecast_rows": forecast_rows,
        },
        ensure_ascii=False,
        default=str,
    )
