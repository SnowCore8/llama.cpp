"""Official HTTP endpoint checks."""

from __future__ import annotations

import json
from typing import Any

from .http_client import ResponsesHttpClient, output_text
from .report import Report
from .validators import validate_response


def run_endpoint_checks(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> str | None:
    """Returns a live response id for later checks, or None on hard failure."""

    # create
    body = {
        "model": model,
        "input": "Reply with exactly: OFFICIAL_CREATE",
        "max_output_tokens": 64,
        **extra,
    }
    code, data = client.post_json("/v1/responses", body)
    if code != 200 or not isinstance(data, dict):
        report.add("endpoint", "POST /v1/responses (create)", "FAIL", f"HTTP {code}")
        return None
    ok, detail = validate_response(data)
    text = output_text(data)
    if ok and data.get("id", "").startswith("resp_") and "OFFICIAL_CREATE" in text:
        report.add("endpoint", "POST /v1/responses (create)", "PASS", f"id={data['id']}")
    else:
        report.add(
            "endpoint",
            "POST /v1/responses (create)",
            "FAIL",
            f"sdk={detail} text={text!r}",
        )
        if not data.get("id"):
            return None
    rid = data["id"]

    # retrieve
    code, got = client.get_json(f"/v1/responses/{rid}")
    if code == 404:
        report.add("endpoint", "GET /v1/responses/{id} (retrieve)", "NOT_IMPLEMENTED", "HTTP 404")
    elif code == 200 and isinstance(got, dict):
        ok, detail = validate_response(got)
        report.add(
            "endpoint",
            "GET /v1/responses/{id} (retrieve)",
            "PASS" if ok else "FAIL",
            detail,
        )
    else:
        report.add(
            "endpoint",
            "GET /v1/responses/{id} (retrieve)",
            "FAIL",
            f"HTTP {code}",
        )

    # input_tokens
    code, tok = client.post_json(
        "/v1/responses/input_tokens",
        {"model": model, "input": "token probe", **{k: v for k, v in extra.items() if k != "chat_template_kwargs"}},
    )
    # chat_template_kwargs is vendor; omit from official body when possible
    if code == 404:
        report.add("endpoint", "POST /v1/responses/input_tokens", "NOT_IMPLEMENTED", "HTTP 404")
    elif code == 200 and isinstance(tok, dict) and (
        "input_tokens" in tok or "tokens" in tok
    ):
        report.add(
            "endpoint",
            "POST /v1/responses/input_tokens",
            "PASS",
            json_preview(tok),
        )
    else:
        report.add(
            "endpoint",
            "POST /v1/responses/input_tokens",
            "FAIL" if code >= 400 else "PARTIAL",
            f"HTTP {code} {json_preview(tok)}",
        )

    # cancel — OpenAI: only in_progress/queued background responses
    code_bg, bg = client.post_json(
        "/v1/responses",
        {
            "model": model,
            "input": "Write a long poem about the ocean, at least twenty lines.",
            "max_output_tokens": 256,
            "background": True,
            **extra,
        },
    )
    if code_bg == 200 and isinstance(bg, dict) and bg.get("id"):
        bg_id = bg["id"]
        code, cancelled = client.post_json(f"/v1/responses/{bg_id}/cancel", {})
        if code == 404:
            report.add("endpoint", "POST /v1/responses/{id}/cancel", "NOT_IMPLEMENTED", "HTTP 404")
        elif code == 200 and isinstance(cancelled, dict) and cancelled.get("status") == "cancelled":
            report.add(
                "endpoint",
                "POST /v1/responses/{id}/cancel",
                "PASS",
                f"id={cancelled.get('id')} from_status={bg.get('status')}",
            )
        else:
            report.add(
                "endpoint",
                "POST /v1/responses/{id}/cancel",
                "FAIL",
                f"HTTP {code} create_status={bg.get('status')} {json_preview(cancelled)}",
            )
    else:
        report.add(
            "endpoint",
            "POST /v1/responses/{id}/cancel",
            "FAIL",
            f"background create HTTP {code_bg} {json_preview(bg)}",
        )

    # compact — use a fresh previous id (cancel does not remove store entry)
    code, compacted = client.post_json(
        "/v1/responses/compact",
        {"model": model, "input": "compact me", "previous_response_id": rid},
    )
    if code == 404:
        report.add("endpoint", "POST /v1/responses/compact", "NOT_IMPLEMENTED", "HTTP 404")
    elif (
        code == 200
        and isinstance(compacted, dict)
        and compacted.get("object") == "response.compaction"
        and isinstance(compacted.get("output"), list)
    ):
        report.add("endpoint", "POST /v1/responses/compact", "PASS", f"id={compacted.get('id')}")
    else:
        report.add(
            "endpoint",
            "POST /v1/responses/compact",
            "FAIL",
            f"HTTP {code} {json_preview(compacted)}",
        )

    # input_items
    code, items = client.get_json(f"/v1/responses/{rid}/input_items")
    if code == 404:
        report.add("endpoint", "GET /v1/responses/{id}/input_items", "NOT_IMPLEMENTED", "HTTP 404")
    elif (
        code == 200
        and isinstance(items, dict)
        and items.get("object") == "list"
        and isinstance(items.get("data"), list)
    ):
        report.add(
            "endpoint",
            "GET /v1/responses/{id}/input_items",
            "PASS",
            f"n={len(items.get('data') or [])}",
        )
    else:
        report.add(
            "endpoint",
            "GET /v1/responses/{id}/input_items",
            "FAIL",
            f"HTTP {code} {json_preview(items)}",
        )

    # WebSocket connect (openai SDK responses.connect → ws://.../v1/responses)
    from openai.resources.responses.responses import Responses
    from urllib.parse import urlparse

    if not hasattr(Responses, "connect"):
        report.add(
            "endpoint",
            "Responses.connect (WebSocket)",
            "SKIP",
            "not present in installed openai SDK",
        )
    else:
        try:
            from websockets.sync.client import connect as ws_connect
        except Exception as e:
            report.add(
                "endpoint",
                "Responses.connect (WebSocket)",
                "FAIL",
                f"websockets package required for connect probe: {e}",
            )
        else:
            parsed = urlparse(client.base_url)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            scheme = "wss" if parsed.scheme == "https" else "ws"
            ws_url = f"{scheme}://{host}:{port}/v1/responses"
            try:
                with ws_connect(
                    ws_url,
                    additional_headers={"Authorization": f"Bearer {client.api_key}"},
                    open_timeout=30,
                    close_timeout=30,
                ) as ws:
                    create_ev = {
                        "type": "response.create",
                        "model": model,
                        "input": "Reply with exactly: WS_CONNECT_OK",
                        "max_output_tokens": 48,
                        "temperature": 0,
                        "reasoning": {"effort": "none"},
                        **extra,
                    }
                    ws.send(json.dumps(create_ev))
                    saw_completed = False
                    detail = ""
                    for _ in range(200):
                        raw = ws.recv(timeout=30)
                        obj = json.loads(raw)
                        et = obj.get("type")
                        if et == "error":
                            detail = str(obj)[:200]
                            break
                        if et in ("response.completed", "response.incomplete", "response.failed"):
                            if et == "response.completed":
                                saw_completed = True
                            detail = f"id={(obj.get('response') or {}).get('id')} event={et}"
                            break
                    report.add(
                        "endpoint",
                        "Responses.connect (WebSocket)",
                        "PASS" if saw_completed else "FAIL",
                        detail or "no response.completed",
                    )
            except Exception as e:
                report.add(
                    "endpoint",
                    "Responses.connect (WebSocket)",
                    "FAIL",
                    f"{type(e).__name__}: {e}"[:200],
                )

    # SDK helpers — exercise the real openai Python client against this server
    try:
        from openai import OpenAI
        from pydantic import BaseModel

        class _ParseBox(BaseModel):
            ok: bool

        oai = OpenAI(base_url=client.base_url.rstrip("/") + "/v1", api_key=client.api_key)
        parsed = oai.responses.parse(
            model=model,
            input='Output exactly {"ok":true} with no markdown fences or prose.',
            text_format=_ParseBox,
            max_output_tokens=32,
            temperature=0,
            extra_body=extra or None,
        )
        parse_ok = (
            getattr(parsed, "status", None) == "completed"
            and getattr(parsed, "output_parsed", None) is not None
            and getattr(parsed.output_parsed, "ok", None) is True
        )
        report.add(
            "endpoint",
            "SDK responses.parse",
            "PASS" if parse_ok else "FAIL",
            f"status={getattr(parsed, 'status', None)!r} parsed={getattr(parsed, 'output_parsed', None)!r}",
        )
    except Exception as e:
        report.add(
            "endpoint",
            "SDK responses.parse",
            "FAIL",
            f"{type(e).__name__}: {e}"[:240],
        )

    try:
        from openai import OpenAI

        oai = OpenAI(base_url=client.base_url.rstrip("/") + "/v1", api_key=client.api_key)
        saw = []
        with oai.responses.stream(
            model=model,
            input="Reply with exactly: SDK_STREAM_OK",
            max_output_tokens=32,
            temperature=0,
            extra_body=extra or None,
        ) as stream:
            for ev in stream:
                saw.append(getattr(ev, "type", type(ev).__name__))
            final = stream.get_final_response()
        text = getattr(final, "output_text", None) or ""
        stream_ok = "response.completed" in saw and "SDK_STREAM_OK" in text
        report.add(
            "endpoint",
            "SDK responses.stream",
            "PASS" if stream_ok else "FAIL",
            f"events={len(saw)} text={text!r}",
        )
    except Exception as e:
        report.add(
            "endpoint",
            "SDK responses.stream",
            "FAIL",
            f"{type(e).__name__}: {e}"[:240],
        )

    # delete (use a fresh id so rid remains for other suites if needed)
    code, created = client.post_json(
        "/v1/responses",
        {
            "model": model,
            "input": "Reply with exactly: TO_DELETE",
            "max_output_tokens": 32,
            **extra,
        },
    )
    if code == 200 and isinstance(created, dict) and created.get("id"):
        did = created["id"]
        code, deleted = client.delete_json(f"/v1/responses/{did}")
        ok = (
            code == 200
            and isinstance(deleted, dict)
            and deleted.get("deleted") is True
            and deleted.get("object") == "response.deleted"
        )
        if code == 404:
            report.add("endpoint", "DELETE /v1/responses/{id}", "NOT_IMPLEMENTED", "HTTP 404")
        else:
            report.add(
                "endpoint",
                "DELETE /v1/responses/{id}",
                "PASS" if ok else "FAIL",
                f"HTTP {code} {json_preview(deleted)}",
            )
            code2, _ = client.get_json(f"/v1/responses/{did}")
            report.add(
                "endpoint",
                "GET after DELETE is 404",
                "PASS" if code2 == 404 else "FAIL",
                f"HTTP {code2}",
            )
    else:
        report.add("endpoint", "DELETE /v1/responses/{id}", "FAIL", "could not create victim")

    return rid


def json_preview(obj: Any) -> str:
    import json

    try:
        return json.dumps(obj, ensure_ascii=False)[:160]
    except Exception:
        return repr(obj)[:160]
