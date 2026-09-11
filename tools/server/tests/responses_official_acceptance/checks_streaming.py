"""Official streaming event checks."""

from __future__ import annotations

from typing import Any

from .catalog import CONDITIONAL_STREAM_EVENTS, CORE_STREAM_EVENTS
from .http_client import ResponsesHttpClient, parse_sse
from .report import Report
from .validators import validate_event, validate_response


def _types(events: list[tuple[str | None, dict[str, Any]]]) -> list[str]:
    return [t for t, _ in events if t]


def _mark_conditional(
    report: Report,
    observed: set[str],
    forced: dict[str, str],
) -> None:
    """forced: event_type -> detail for events proven by dedicated probes."""
    for ev in CONDITIONAL_STREAM_EVENTS:
        if ev in forced:
            report.add("stream_event", ev, "PASS", forced[ev])
        elif ev in observed:
            report.add("stream_event", ev, "PASS", "observed on happy-path")
        else:
            report.add(
                "stream_event",
                ev,
                "SKIP",
                "not applicable / not forceable in this suite",
            )


def run_streaming_checks(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    code, headers, raw = client.request(
        "POST",
        "/v1/responses",
        {
            "model": model,
            "input": "Reply with exactly: STREAM_OFFICIAL",
            "max_output_tokens": 64,
            "stream": True,
            **extra,
        },
        stream=True,
    )
    ct = headers.get("Content-Type", "")
    report.add(
        "stream_event",
        "content_type_text_event_stream",
        "PASS" if code == 200 and "text/event-stream" in ct else "FAIL",
        f"HTTP {code} ct={ct}",
    )
    events = parse_sse(raw) if code == 200 else []
    types = _types(events)
    seqs = [obj.get("sequence_number") for _, obj in events]

    report.add(
        "create_param",
        "stream",
        "PASS" if code == 200 and "response.completed" in types else "FAIL",
        f"HTTP {code} n={len(events)}",
    )

    for ev in CORE_STREAM_EVENTS:
        if ev not in types:
            report.add("stream_event", ev, "FAIL", "missing from happy-path text stream")
            continue
        errs = []
        for t, obj in events:
            if t != ev:
                continue
            ok, detail = validate_event(ev, obj)
            if not ok:
                errs.append(detail)
        report.add(
            "stream_event",
            ev,
            "PASS" if not errs else "FAIL",
            "observed+sdk_ok" if not errs else f"sdk_err={errs[:1]}",
        )

    forced: dict[str, str] = {}

    # --- force function_call argument stream events ---
    fcode, _, fraw = client.request(
        "POST",
        "/v1/responses",
        {
            "model": model,
            "input": "call get_weather for Paris",
            "max_output_tokens": 64,
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            "tool_choice": {"type": "function", "name": "get_weather"},
            **extra,
        },
        stream=True,
    )
    fevents = parse_sse(fraw) if fcode == 200 else []
    ftypes = set(_types(fevents))
    for ev in (
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
    ):
        if ev not in ftypes:
            report.add("stream_event", f"{ev}.forced", "FAIL", f"missing; types={sorted(ftypes)}")
            continue
        errs = []
        for t, obj in fevents:
            if t != ev:
                continue
            ok, detail = validate_event(ev, obj)
            if not ok:
                errs.append(detail)
        if errs:
            report.add("stream_event", f"{ev}.forced", "FAIL", f"sdk_err={errs[:1]}")
        else:
            forced[ev] = "forced tool_choice stream+sdk_ok"

    # --- force incomplete (max_output_tokens cap) ---
    icode, _, iraw = client.request(
        "POST",
        "/v1/responses",
        {
            "model": model,
            "input": "Write a long detailed essay about mathematics history.",
            "max_output_tokens": 2,
            "temperature": 0,
            "stream": True,
            **extra,
        },
        stream=True,
    )
    ievents = parse_sse(iraw) if icode == 200 else []
    itypes = _types(ievents)
    incomplete = next((obj for t, obj in ievents if t == "response.incomplete"), None)
    if incomplete and isinstance(incomplete.get("response"), dict):
        resp = incomplete["response"]
        ok_schema, detail = validate_response(resp)
        ok_evt, edt = validate_event("response.incomplete", incomplete)
        ok = (
            resp.get("status") == "incomplete"
            and (resp.get("incomplete_details") or {}).get("reason") == "max_output_tokens"
            and ok_schema
            and ok_evt
        )
        if ok:
            forced["response.incomplete"] = "forced max_output_tokens stream+sdk_ok"
        else:
            report.add(
                "stream_event",
                "response.incomplete.forced",
                "FAIL",
                f"status={resp.get('status')!r} details={resp.get('incomplete_details')!r} "
                f"schema={detail} evt={edt}",
            )
    else:
        report.add(
            "stream_event",
            "response.incomplete.forced",
            "FAIL",
            f"HTTP {icode} types={itypes}",
        )

    # --- force reasoning_text.delta (thinking on for this probe only) ---
    think_extra = dict(extra)
    ctk = dict(think_extra.get("chat_template_kwargs") or {})
    ctk["enable_thinking"] = True
    think_extra["chat_template_kwargs"] = ctk
    rcode, _, rraw = client.request(
        "POST",
        "/v1/responses",
        {
            "model": model,
            "input": "Think one short sentence then reply: HI",
            "max_output_tokens": 64,
            "stream": True,
            **think_extra,
            "reasoning": {"effort": "low"},
        },
        stream=True,
    )
    revents = parse_sse(rraw) if rcode == 200 else []
    if "response.reasoning_text.delta" in _types(revents):
        forced["response.reasoning_text.delta"] = "forced enable_thinking stream"
    else:
        report.add(
            "stream_event",
            "response.reasoning_text.delta.forced",
            "FAIL",
            f"HTTP {rcode} types={_types(revents)}",
        )

    _mark_conditional(report, set(types), forced)

    mono = seqs == list(range(len(seqs))) and len(seqs) > 0
    report.add(
        "stream_event",
        "sequence_number_monotonic_from_zero",
        "PASS" if mono else "FAIL",
        f"len={len(seqs)} first_last={(seqs[0], seqs[-1]) if seqs else None}",
    )

    completed = next((obj for t, obj in events if t == "response.completed"), None)
    if completed and isinstance(completed.get("response"), dict):
        ok, detail = validate_response(completed["response"])
        report.add(
            "stream_event",
            "response.completed.response_schema",
            "PASS" if ok else "FAIL",
            detail,
        )
    else:
        report.add(
            "stream_event",
            "response.completed.response_schema",
            "FAIL",
            "missing completed.response",
        )


def run_response_object_checks(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    from openai.types.responses import Response

    code, data = client.post_json(
        "/v1/responses",
        {
            "model": model,
            "input": "Reply with exactly: OBJ",
            "max_output_tokens": 32,
            **extra,
        },
    )
    if code != 200 or not isinstance(data, dict):
        report.add("response_object", "sdk_Response_validate", "FAIL", f"HTTP {code}")
        return
    ok, detail = validate_response(data)
    report.add(
        "response_object",
        "sdk_Response_validate",
        "PASS" if ok else "FAIL",
        detail,
    )
    for k in [k for k, v in Response.model_fields.items() if v.is_required()]:
        report.add(
            "response_object",
            f"required.{k}",
            "PASS" if k in data else "FAIL",
            repr(data.get(k))[:80],
        )
