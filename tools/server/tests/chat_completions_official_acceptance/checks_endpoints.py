"""Official HTTP endpoint checks (Chat Completions core)."""

from __future__ import annotations

import json
from typing import Any

from .http_client import ChatHttpClient, choice_text
from .report import Report
from .validators import validate_completion


def json_preview(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)[:160]
    except Exception:
        return repr(obj)[:160]


def run_endpoint_checks(
    client: ChatHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    body = {
        "model": model,
        "max_tokens": 64,
        "temperature": 0,
        "messages": [{"role": "user", "content": "Reply with exactly: CHAT_CREATE"}],
        **extra,
    }
    code, data = client.post_json("/v1/chat/completions", body)
    if code != 200 or not isinstance(data, dict):
        report.add("endpoint", "POST /v1/chat/completions (create)", "FAIL", f"HTTP {code}")
    else:
        ok, detail = validate_completion(data)
        text = choice_text(data)
        if ok and data.get("object") == "chat.completion" and "CHAT_CREATE" in text:
            report.add(
                "endpoint",
                "POST /v1/chat/completions (create)",
                "PASS",
                f"id={data.get('id')}",
            )
        else:
            report.add(
                "endpoint",
                "POST /v1/chat/completions (create)",
                "FAIL",
                f"sdk={detail} text={text!r}",
            )

    # input_tokens adjunct
    tok_body = {
        "model": model,
        "messages": [{"role": "user", "content": "token probe hello"}],
        **{k: v for k, v in extra.items() if k != "chat_template_kwargs"},
    }
    code, tok = client.post_json("/v1/chat/completions/input_tokens", tok_body)
    if code == 404:
        report.add(
            "endpoint",
            "POST /v1/chat/completions/input_tokens",
            "NOT_IMPLEMENTED",
            "HTTP 404",
        )
    elif code == 200 and isinstance(tok, dict) and isinstance(tok.get("input_tokens"), int):
        report.add(
            "endpoint",
            "POST /v1/chat/completions/input_tokens",
            "PASS",
            json_preview(tok),
        )
    else:
        report.add(
            "endpoint",
            "POST /v1/chat/completions/input_tokens",
            "FAIL" if code >= 400 else "PARTIAL",
            f"HTTP {code} {json_preview(tok)}",
        )

    # stream endpoint (HTTP-level smoke)
    scode, headers, raw = client.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 32,
            "stream": True,
            "temperature": 0,
            "messages": [{"role": "user", "content": "Reply with exactly: CHAT_STREAM"}],
            **extra,
        },
        stream=True,
    )
    ct = headers.get("Content-Type", "") or headers.get("content-type", "")
    stream_ok = scode == 200 and "text/event-stream" in ct and b"[DONE]" in raw
    report.add(
        "endpoint",
        "POST /v1/chat/completions (stream)",
        "PASS" if stream_ok else "FAIL",
        f"HTTP {scode} ct={ct}",
    )
