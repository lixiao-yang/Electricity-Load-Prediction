from __future__ import annotations

import json
import re
from typing import Any, Dict

from electricity_agent.config import DEFAULT_HORIZON_HOURS, MAX_HORIZON_HOURS, OPENAI_API_KEY, OPENAI_INTENT_MODEL
from electricity_agent.prompts import INTENT_PARSER_SYSTEM_PROMPT

_LAST_INTENT_FALLBACK_REASON = ""


def parse_request_options(query: str) -> Dict[str, object]:
    text = (query or "").strip()
    text_lower = text.lower()
    horizon_hours = DEFAULT_HORIZON_HOURS
    granularity = "hourly"
    horizon_source = "default"
    granularity_source = "default"

    hour_match = re.search(r"(\d+)\s*(?:hours?|hour|h)\b", text_lower)
    day_match = re.search(r"(\d+)\s*(?:days?|day|d)\b", text_lower)
    week_match = re.search(r"(\d+)\s*(?:weeks?|week|w)\b", text_lower)
    month_match = re.search(r"(\d+)\s*(?:months?|month|m)\b", text_lower)

    if hour_match:
        horizon_hours = int(hour_match.group(1))
        horizon_source = "query"
    elif day_match:
        horizon_hours = int(day_match.group(1)) * 24
        horizon_source = "query"
    elif week_match:
        horizon_hours = int(week_match.group(1)) * 24 * 7
        granularity = "daily"
        horizon_source = "query"
        granularity_source = "query"
    elif month_match:
        horizon_hours = int(month_match.group(1)) * 24 * 30
        granularity = "daily"
        horizon_source = "query"
        granularity_source = "query"

    if any(token in text_lower for token in ["daily", "per day", "day-level"]):
        granularity = "daily"
        granularity_source = "query"
    elif any(token in text_lower for token in ["weekly", "per week"]):
        granularity = "weekly"
        granularity_source = "query"
    elif any(token in text_lower for token in ["hourly", "per hour"]):
        granularity = "hourly"
        granularity_source = "query"

    horizon_hours = max(1, min(int(horizon_hours), MAX_HORIZON_HOURS))
    return {
        "horizon_hours": horizon_hours,
        "horizon_source": horizon_source,
        "granularity": granularity,
        "granularity_source": granularity_source,
        "meter_query": _extract_meter_query(text),
        "mode": _extract_mode(text),
        "analysis_type": _extract_analysis_type(text),
        "intent_source": "rules",
    }


def parse_request_options_with_llm(query: str, default_mode: str = "future") -> Dict[str, object]:
    global _LAST_INTENT_FALLBACK_REASON
    _LAST_INTENT_FALLBACK_REASON = ""
    fallback = parse_request_options(query)
    fallback["mode"] = fallback.get("mode") or default_mode

    llm_intent = _parse_request_options_llm(query)
    if not llm_intent:
        fallback["intent_fallback_reason"] = _LAST_INTENT_FALLBACK_REASON or "llm_intent_unavailable"
        return fallback

    merged = {
        **fallback,
        "intent_source": "llm",
        "intent_confidence": llm_intent.get("confidence"),
    }
    meter_query = _clean_meter_query(llm_intent.get("meter_query"))
    if meter_query:
        merged["meter_query"] = meter_query

    horizon_hours = _coerce_horizon(llm_intent.get("horizon_hours"))
    if horizon_hours is not None:
        merged["horizon_hours"] = horizon_hours
        merged["horizon_source"] = "llm"

    granularity = str(llm_intent.get("granularity") or "").strip().lower()
    if granularity in {"hourly", "daily", "weekly"}:
        merged["granularity"] = granularity
        merged["granularity_source"] = "llm"

    mode = str(llm_intent.get("mode") or "").strip().lower()
    if mode in {"future", "evaluation"}:
        merged["mode"] = mode

    analysis_type = str(llm_intent.get("analysis_type") or "").strip().lower()
    if analysis_type in {"forecast", "evaluation", "comparison", "diagnostics"}:
        merged["analysis_type"] = analysis_type

    return merged


def _extract_meter_query(query: str) -> str:
    text = (query or "").strip()
    explicit = re.search(r"\b(MT[_\-\s]?\d{1,4})\b", text, flags=re.IGNORECASE)
    if explicit:
        return explicit.group(1).upper().replace("-", "_").replace(" ", "_")

    explicit_meter = re.search(
        r"\b(?:meter|client|customer|user)\s*(?:id)?\s*[:#]?\s*([A-Z]{0,3}[_\-\s]?\d{1,4})\b",
        text,
        flags=re.IGNORECASE,
    )
    if explicit_meter:
        value = explicit_meter.group(1).upper().replace("-", "_").replace(" ", "_")
        if value.startswith("MT_"):
            return value
        digits = re.sub(r"\D", "", value)
        if digits:
            return f"MT_{int(digits):03d}"

    digits_only = re.fullmatch(r"\s*(\d{1,4})\s*", text)
    if digits_only:
        return f"MT_{int(digits_only.group(1)):03d}"

    cleaned = re.sub(
        r"\b(?:forecast|load|electricity|consumption|for|next|hours?|days?|weeks?|months?|daily|weekly|hourly|show|give|me|please)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[^A-Za-z0-9_]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_mode(query: str) -> str | None:
    text_lower = (query or "").lower()
    if any(token in text_lower for token in ["evaluation", "evaluate", "test", "held-out", "backtest", "actual vs predicted"]):
        return "evaluation"
    if any(token in text_lower for token in ["future", "next", "forecast ahead"]):
        return "future"
    return None


def _extract_analysis_type(query: str) -> str:
    text_lower = (query or "").lower()
    if any(token in text_lower for token in ["compare", "comparison", "versus", "vs"]):
        return "comparison"
    if any(token in text_lower for token in ["diagnostic", "diagnostics", "error", "residual", "why"]):
        return "diagnostics"
    if any(token in text_lower for token in ["evaluation", "evaluate", "test", "actual vs predicted", "backtest"]):
        return "evaluation"
    return "forecast"


def _parse_request_options_llm(query: str) -> Dict[str, Any] | None:
    global _LAST_INTENT_FALLBACK_REASON
    if not OPENAI_API_KEY:
        _LAST_INTENT_FALLBACK_REASON = "missing_openai_api_key"
        return None
    try:
        from openai import OpenAI
    except Exception as exc:
        _LAST_INTENT_FALLBACK_REASON = f"openai_import_failed:{type(exc).__name__}"
        return None

    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.responses.create(
            model=OPENAI_INTENT_MODEL,
            input=[
                {"role": "system", "content": INTENT_PARSER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
    except Exception as exc:
        _LAST_INTENT_FALLBACK_REASON = f"openai_request_failed:{type(exc).__name__}"
        return None

    text = (getattr(response, "output_text", "") or "").strip()
    if not text:
        _LAST_INTENT_FALLBACK_REASON = "openai_empty_response"
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            _LAST_INTENT_FALLBACK_REASON = "openai_invalid_json"
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            _LAST_INTENT_FALLBACK_REASON = "openai_invalid_json"
            return None
    if not isinstance(parsed, dict):
        _LAST_INTENT_FALLBACK_REASON = "openai_json_not_object"
        return None
    return parsed


def _clean_meter_query(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.isdigit():
        return f"MT_{int(text):03d}"
    return text.upper().replace("-", "_").replace(" ", "_")


def _coerce_horizon(value: Any) -> int | None:
    if value is None:
        return None
    try:
        horizon = int(float(value))
    except (TypeError, ValueError):
        return None
    return max(1, min(horizon, MAX_HORIZON_HOURS))
