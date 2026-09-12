"""Official Chat Completions streaming + completion object checks."""

from __future__ import annotations

from typing import Any

from .catalog import CONDITIONAL_STREAM_CHECKS, CORE_STREAM_CHECKS
from .http_client import ChatHttpClient, parse_sse
from .report import Report
from .validators import validate_chunk, validate_completion


def _chunks(events: list[tuple[str | None, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [obj for t, obj in events if t not in (None, "[DONE]") and obj.get("object") == "chat.completion.chunk"]


def _mark_conditional(report: Report, forced: dict[str, str]) -> None:
    for name in CONDITIONAL_STREAM_CHECKS:
        if name in forced:
            report.add("stream_event", name, "PASS", forced[name])
        else:
            report.add(
                "stream_event",
                name,
                "SKIP",
                "not applicable / not forceable in this suite",
            )


def run_streaming_checks(
    client: ChatHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    code, headers, raw = client.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 64,
            "stream": True,
            "temperature": 0,
            "messages": [{"role": "user", "content": "Reply with exactly: STREAM_OFFICIAL"}],
            **extra,
        },
        stream=True,
    )
    ct = headers.get("Content-Type", "") or headers.get("content-type", "")
    events = parse_sse(raw) if code == 200 else []
    chunks = _chunks(events)
    saw_done = any(t == "[DONE]" for t, _ in events)

    report.add(
        "stream_event",
        "content_type_text_event_stream",
        "PASS" if code == 200 and "text/event-stream" in ct else "FAIL",
        f"HTTP {code} ct={ct}",
    )

    stream_ok = code == 200 and chunks and saw_done
    report.add(
        "create_param",
        "stream",
        "PASS" if stream_ok else "FAIL",
        f"HTTP {code} n_chunks={len(chunks)} done={saw_done}",
    )

    all_objects_ok = all(c.get("object") == "chat.completion.chunk" for c in chunks)
    report.add(
        "stream_event",
        "chunk.object_chat_completion_chunk",
        "PASS" if all_objects_ok and chunks else "FAIL",
        f"n={len(chunks)}",
    )

    saw_role = any(
        (c.get("choices") or [{}])[0].get("delta", {}).get("role") == "assistant"
        for c in chunks
        if c.get("choices")
    )
    report.add(
        "stream_event",
        "delta.role",
        "PASS" if saw_role else "FAIL",
        f"saw_role={saw_role}",
    )

    content_parts: list[str] = []
    for c in chunks:
        choices = c.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
        if isinstance(delta, dict) and delta.get("content"):
            content_parts.append(delta["content"])
    saw_content = bool(content_parts)
    report.add(
        "stream_event",
        "delta.content",
        "PASS" if saw_content else "FAIL",
        f"len={len(''.join(content_parts))}",
    )

    finish = None
    for c in reversed(chunks):
        choices = c.get("choices") or []
        if choices and isinstance(choices[0], dict):
            fr = choices[0].get("finish_reason")
            if fr:
                finish = fr
                break
    report.add(
        "stream_event",
        "finish_reason",
        "PASS" if finish in ("stop", "length", "tool_calls") else "FAIL",
        f"finish={finish!r}",
    )

    report.add(
        "stream_event",
        "stream_done_sentinel",
        "PASS" if saw_done else "FAIL",
        f"saw_done={saw_done}",
    )

    chunk_errs = []
    for c in chunks:
        ok, detail = validate_chunk(c)
        if not ok:
            chunk_errs.append(detail)
    report.add(
        "stream_event",
        "chunk_sdk_validate",
        "PASS" if chunks and not chunk_errs else "FAIL",
        "ok" if not chunk_errs else f"sdk_err={chunk_errs[:1]}",
    )

    forced: dict[str, str] = {}

    # --- force tool_calls delta ---
    fcode, _, fraw = client.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 128,
            "stream": True,
            "temperature": 0,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            **extra,
        },
        stream=True,
    )
    fchunks = _chunks(parse_sse(fraw) if fcode == 200 else [])
    saw_tc = any(
        (c.get("choices") or [{}])[0].get("delta", {}).get("tool_calls")
        for c in fchunks
        if c.get("choices")
    )
    if saw_tc:
        forced["delta.tool_calls"] = "forced tool_choice stream"
    else:
        report.add(
            "stream_event",
            "delta.tool_calls.forced",
            "FAIL",
            f"HTTP {fcode} n_chunks={len(fchunks)}",
        )

    # --- force logprobs ---
    lcode, _, lraw = client.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 32,
            "stream": True,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 2,
            "messages": [{"role": "user", "content": "Reply with exactly: LP_STREAM"}],
            **extra,
        },
        stream=True,
    )
    lchunks = _chunks(parse_sse(lraw) if lcode == 200 else [])
    saw_lp = any(
        isinstance((c.get("choices") or [{}])[0].get("logprobs"), dict)
        for c in lchunks
        if c.get("choices")
    )
    if (saw_lp):
        forced["choice.logprobs"] = "forced logprobs stream"
    else:
        report.add(
            "stream_event",
            "choice.logprobs.forced",
            "FAIL",
            f"HTTP {lcode} n_chunks={len(lchunks)}",
        )

    # --- force reasoning_content deltas (thinking on) ---
    rbody = {
        "model": model,
        "max_tokens": 128,
        "stream": True,
        "temperature": 0,
        "reasoning_effort": "low",
        "messages": [
            {
                "role": "user",
                "content": "Think briefly, then reply with exactly: STREAM_REASON",
            }
        ],
    }
    # Do not inherit enable_thinking=false from suite extra.
    rcode, _, rraw = client.request("POST", "/v1/chat/completions", rbody, stream=True)
    rchunks = _chunks(parse_sse(rraw) if rcode == 200 else [])
    saw_reason = any(
        isinstance((c.get("choices") or [{}])[0].get("delta"), dict)
        and (c.get("choices") or [{}])[0].get("delta", {}).get("reasoning_content")
        for c in rchunks
        if c.get("choices")
    )
    if saw_reason:
        forced["delta.reasoning_content"] = "forced reasoning_effort=low stream"
    else:
        report.add(
            "stream_event",
            "delta.reasoning_content.forced",
            "FAIL",
            f"HTTP {rcode} n_chunks={len(rchunks)}",
        )

    # --- usage final chunk (stream_options.include_usage) ---
    ucode, _, uraw = client.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 16,
            "stream": True,
            "temperature": 0,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "Reply with exactly: USAGE"}],
            **extra,
        },
        stream=True,
    )
    uevents = parse_sse(uraw) if ucode == 200 else []
    usage_chunk = next(
        (obj for t, obj in uevents if isinstance(obj.get("usage"), dict)),
        None,
    )
    if usage_chunk and isinstance(usage_chunk.get("usage"), dict):
        u = usage_chunk["usage"]
        if isinstance(u.get("prompt_tokens"), int) and isinstance(u.get("completion_tokens"), int):
            forced["usage.final_chunk"] = f"prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')}"

    _mark_conditional(report, forced)

    # Ensure catalog rows for core checks
    for name in CORE_STREAM_CHECKS:
        if not any(r.name == name for r in report.rows if r.area == "stream_event"):
            report.add("stream_event", name, "FAIL", "missing from suite execution")


def run_completion_object_checks(
    client: ChatHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    from openai.types.chat import ChatCompletion

    code, data = client.post_json(
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 32,
            "temperature": 0,
            "messages": [{"role": "user", "content": "Reply with exactly: OBJ"}],
            **extra,
        },
    )
    if code != 200 or not isinstance(data, dict):
        report.add("completion_object", "sdk_ChatCompletion_validate", "FAIL", f"HTTP {code}")
        return
    ok, detail = validate_completion(data)
    report.add(
        "completion_object",
        "sdk_ChatCompletion_validate",
        "PASS" if ok else "FAIL",
        detail,
    )
    for k in [k for k, v in ChatCompletion.model_fields.items() if v.is_required()]:
        report.add(
            "completion_object",
            f"required.{k}",
            "PASS" if k in data else "FAIL",
            repr(data.get(k))[:80],
        )
