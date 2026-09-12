"""Official HTTP endpoint checks (Chat Completions core)."""

from __future__ import annotations

import json
import time
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

    # GET /v1/chat/completions — official query contract: `order` enum, default `asc`
    def _store_probe(marker: str):
        return client.post_json(
            "/v1/chat/completions",
            {
                "model": model,
                "store": True,
                "max_tokens": 4,
                "temperature": 0,
                "messages": [{"role": "user", "content": f"Reply with exactly: {marker}"}],
                **extra,
            },
        )

    first_code, first_data = _store_probe("LIST_A")
    time.sleep(1.1)  # `created` is unix seconds; keep the two entries apart
    second_code, second_data = _store_probe("LIST_B")
    first_id = first_data.get("id") if first_code == 200 and isinstance(first_data, dict) else ""
    second_id = second_data.get("id") if second_code == 200 and isinstance(second_data, dict) else ""
    if not (first_id and second_id):
        report.add(
            "endpoint",
            "GET /v1/chat/completions (list)",
            "FAIL",
            f"store setup failed: {first_code}/{second_code}",
        )
        return

    code, page = client.get_json("/v1/chat/completions?limit=100")
    ids = [c.get("id") for c in (page.get("data") or [])] if isinstance(page, dict) else []
    asc_ok = first_id in ids and second_id in ids and ids.index(first_id) < ids.index(second_id)
    report.add(
        "endpoint",
        "GET /v1/chat/completions (order default asc)",
        "PASS" if code == 200 and asc_ok else "FAIL",
        f"HTTP {code} first<second={asc_ok} n={len(ids)}",
    )

    code, page = client.get_json("/v1/chat/completions?order=desc&limit=100")
    ids = [c.get("id") for c in (page.get("data") or [])] if isinstance(page, dict) else []
    desc_ok = first_id in ids and second_id in ids and ids.index(second_id) < ids.index(first_id)
    report.add(
        "endpoint",
        "GET /v1/chat/completions (order=desc)",
        "PASS" if code == 200 and desc_ok else "FAIL",
        f"HTTP {code} second<first={desc_ok}",
    )

    code, data = client.get_json("/v1/chat/completions?order=bogus")
    report.add(
        "endpoint",
        "GET /v1/chat/completions (order=bogus)",
        "PASS" if code == 400 else "FAIL",
        f"HTTP {code} {json_preview(data)}",
    )

    # GET /v1/chat/completions/{id}/messages — official stored-messages contract
    mcode, mdata = client.get_json(f"/v1/chat/completions/{first_id}/messages")
    msgs = mdata.get("data") if isinstance(mdata, dict) else None
    msgs_ok = (
        mcode == 200
        and isinstance(mdata, dict)
        and mdata.get("object") == "list"
        and isinstance(mdata.get("has_more"), bool)
        and isinstance(msgs, list)
        and len(msgs) >= 1
        and mdata.get("first_id") == msgs[0].get("id")
        and mdata.get("last_id") == msgs[-1].get("id")
    )
    msg0_ok = (
        msgs_ok
        and isinstance(msgs[0], dict)
        and isinstance(msgs[0].get("id"), str)
        and msgs[0]["id"].startswith("msg_")
        and msgs[0].get("role") == "assistant"
        and "LIST_A" in str(msgs[0].get("content"))
        and "content_parts" in msgs[0]
    )
    report.add(
        "endpoint",
        "GET /v1/chat/completions/{id}/messages",
        "PASS" if msg0_ok else "FAIL",
        f"HTTP {mcode} n={len(msgs) if isinstance(msgs, list) else None}",
    )

    rcode, rdata = client.get_json(f"/v1/chat/completions/{first_id}/messages?order=desc")
    rmsgs = rdata.get("data") if isinstance(rdata, dict) else []
    asc_ids = [m.get("id") for m in (msgs or [])]
    desc_ids = [m.get("id") for m in (rmsgs or [])]
    report.add(
        "endpoint",
        "GET /v1/chat/completions/{id}/messages (order=desc)",
        "PASS" if rcode == 200 and desc_ids == list(reversed(asc_ids)) else "FAIL",
        f"HTTP {rcode} asc={asc_ids} desc={desc_ids}",
    )

    last_msg_id = asc_ids[-1] if asc_ids else "none"
    acode, adata = client.get_json(
        f"/v1/chat/completions/{first_id}/messages?after={last_msg_id}"
    )
    atail = adata.get("data") if isinstance(adata, dict) else None
    report.add(
        "endpoint",
        "GET /v1/chat/completions/{id}/messages (after)",
        "PASS" if acode == 200 and atail == [] else "FAIL",
        f"HTTP {acode} tail={json_preview(atail)}",
    )

    bcode, bdata = client.get_json(f"/v1/chat/completions/{first_id}/messages?order=bogus")
    report.add(
        "endpoint",
        "GET /v1/chat/completions/{id}/messages (order=bogus)",
        "PASS" if bcode == 400 else "FAIL",
        f"HTTP {bcode} {json_preview(bdata)}",
    )

    ucode, _ = client.get_json("/v1/chat/completions/chatcmpl-not-stored/messages")
    report.add(
        "endpoint",
        "GET /v1/chat/completions/{id}/messages (unknown id)",
        "PASS" if ucode == 404 else "FAIL",
        f"HTTP {ucode}",
    )
