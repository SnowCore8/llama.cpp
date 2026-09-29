"""Official HTTP endpoint checks."""

from __future__ import annotations

import json
from typing import Any

from .http_client import ResponsesHttpClient, output_text, parse_sse
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

    # input_items pagination: official order defaults to desc; limit 1..100 (default 20)
    code_pg, pg = client.post_json(
        "/v1/responses",
        {
            "model": model,
            "input": [
                {"role": "user", "content": "Reply with exactly: PG1"},
                {"role": "assistant", "content": "PG1"},
                {"role": "user", "content": "Reply with exactly: PG2"},
            ],
            "max_output_tokens": 16,
            **extra,
        },
    )
    if code_pg == 200 and isinstance(pg, dict) and pg.get("id"):
        pid = pg["id"]
        gcode_all, all_items = client.get_json(f"/v1/responses/{pid}/input_items?order=asc&limit=100")
        gcode_def, def_items = client.get_json(f"/v1/responses/{pid}/input_items")
        gcode_l1, l1_items = client.get_json(f"/v1/responses/{pid}/input_items?limit=1")
        asc_ids = [x.get("id") for x in (all_items.get("data") or []) if isinstance(x, dict)]
        def_ids = [x.get("id") for x in (def_items.get("data") or []) if isinstance(x, dict)]
        l1_ids = [x.get("id") for x in (l1_items.get("data") or []) if isinstance(x, dict)]
        gcode_af = 0
        af_ids: list[Any] = []
        if asc_ids:
            gcode_af, af_items = client.get_json(
                f"/v1/responses/{pid}/input_items?order=desc&limit=100&after={asc_ids[-1]}"
            )
            af_ids = [x.get("id") for x in (af_items.get("data") or []) if isinstance(x, dict)]
        code_l0, _ = client.get_json(f"/v1/responses/{pid}/input_items?limit=0")
        code_l101, _ = client.get_json(f"/v1/responses/{pid}/input_items?limit=101")
        ok_pg = (
            gcode_all == 200
            and len(asc_ids) >= 3
            and gcode_def == 200
            and isinstance(def_items, dict)
            and len(def_ids) == len(asc_ids)
            and def_ids == list(reversed(asc_ids))
            and def_items.get("has_more") is False
            and def_items.get("first_id") == asc_ids[-1]
            and def_items.get("last_id") == asc_ids[0]
            and gcode_l1 == 200
            and isinstance(l1_items, dict)
            and l1_ids == [asc_ids[-1]]
            and l1_items.get("has_more") is True
            and gcode_af == 200
            and af_ids == list(reversed(asc_ids[:-1]))
        )
        report.add(
            "endpoint",
            "GET /v1/responses/{id}/input_items?order/limit/after (pagination)",
            "PASS" if ok_pg else "FAIL",
            f"all={gcode_all} n_asc={len(asc_ids)} def={gcode_def} n_def={len(def_ids)} "
            f"has_more={def_items.get('has_more') if isinstance(def_items, dict) else None} "
            f"l1={gcode_l1} l1_ids={l1_ids} af={gcode_af} n_af={len(af_ids)}",
        )
        report.add(
            "endpoint",
            "GET /v1/responses/{id}/input_items (limit bounds)",
            "PASS" if code_l0 == 400 and code_l101 == 400 else "FAIL",
            f"limit0={code_l0} limit101={code_l101}",
        )
    else:
        report.add(
            "endpoint",
            "GET /v1/responses/{id}/input_items?order/limit/after (pagination)",
            "FAIL",
            f"seed create HTTP {code_pg} {json_preview(pg)}",
        )
        report.add(
            "endpoint",
            "GET /v1/responses/{id}/input_items (limit bounds)",
            "FAIL",
            "seed create failed",
        )

    # retrieve include gating: optional fields (reasoning.encrypted_content) follow include
    code_enc, enc_data = client.post_json(
        "/v1/responses",
        {
            "model": model,
            "input": "Think briefly, then reply with exactly: ENC_EP",
            "reasoning": {"effort": "low"},
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": 128,
            "temperature": 0,
            **{k: v for k, v in extra.items() if k != "reasoning"},
        },
    )

    def _enc_present(obj: Any) -> bool:
        return any(
            isinstance(x, dict) and "encrypted_content" in x
            for x in ((obj.get("output") or []) if isinstance(obj, dict) else [])
            if isinstance(x, dict) and x.get("type") == "reasoning"
        )

    if code_enc == 200 and isinstance(enc_data, dict) and enc_data.get("id"):
        rid_enc = enc_data["id"]
        created_enc = _enc_present(enc_data)
        gcode_plain, got_plain = client.get_json(f"/v1/responses/{rid_enc}")
        gcode_inc, got_inc = client.get_json(
            f"/v1/responses/{rid_enc}?include=reasoning.encrypted_content"
        )
        ok_inc = (
            created_enc
            and gcode_plain == 200
            and not _enc_present(got_plain)
            and gcode_inc == 200
            and _enc_present(got_inc)
        )
        report.add(
            "endpoint",
            "GET /v1/responses/{id}?include (gating)",
            "PASS" if ok_inc else "FAIL",
            f"create_enc={created_enc} plain={gcode_plain}/{_enc_present(got_plain)} "
            f"inc={gcode_inc}/{_enc_present(got_inc)}",
        )
    else:
        report.add(
            "endpoint",
            "GET /v1/responses/{id}?include (gating)",
            "FAIL",
            f"create HTTP {code_enc} {json_preview(enc_data)}",
        )

    # retrieve?stream=true replay: include_obfuscation=false drops the obfuscation field
    code_bs, _, raw_bs = client.request(
        "POST",
        "/v1/responses",
        {
            "model": model,
            "input": "Reply with exactly: OBF_STREAM",
            "max_output_tokens": 32,
            "temperature": 0,
            "background": True,
            "stream": True,
            **extra,
        },
        stream=True,
    )
    rid_bs = None
    if code_bs == 200:
        for _t, obj in parse_sse(raw_bs):
            resp_obj = obj.get("response") if isinstance(obj, dict) else None
            if isinstance(resp_obj, dict) and resp_obj.get("id"):
                rid_bs = resp_obj["id"]
                break
    if code_bs == 200 and rid_bs:
        c_keep, _, raw_keep = client.request("GET", f"/v1/responses/{rid_bs}?stream=true", stream=True)
        c_strip, _, raw_strip = client.request(
            "GET", f"/v1/responses/{rid_bs}?stream=true&include_obfuscation=false", stream=True
        )
        keep_events = parse_sse(raw_keep) if c_keep == 200 else []
        strip_events = parse_sse(raw_strip) if c_strip == 200 else []
        keep_has = any(isinstance(obj, dict) and "obfuscation" in obj for _t, obj in keep_events)
        strip_has = any(isinstance(obj, dict) and "obfuscation" in obj for _t, obj in strip_events)
        # stripping must drop only the obfuscation field: same events, same sequence numbers
        keep_seq = [(t, obj.get("sequence_number")) for t, obj in keep_events if isinstance(obj, dict)]
        strip_seq = [(t, obj.get("sequence_number")) for t, obj in strip_events if isinstance(obj, dict)]
        same_events = keep_seq == strip_seq
        ok_obf = (
            c_keep == 200
            and keep_has
            and c_strip == 200
            and len(strip_events) > 0
            and not strip_has
            and same_events
        )
        report.add(
            "endpoint",
            "GET /v1/responses/{id}?stream=true (include_obfuscation)",
            "PASS" if ok_obf else "FAIL",
            f"keep HTTP {c_keep} n={len(keep_events)} has_obf={keep_has}; "
            f"strip HTTP {c_strip} n={len(strip_events)} has_obf={strip_has} same_events={same_events}",
        )
    else:
        report.add(
            "endpoint",
            "GET /v1/responses/{id}?stream=true (include_obfuscation)",
            "FAIL",
            f"background stream create HTTP {code_bs} id={rid_bs!r}",
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
            and deleted.get("object") == "response"
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

    # OpenAI: an unknown model on create is 400 with the official model_not_found body
    nfcode, nfdata = client.post_json(
        "/v1/responses",
        {
            "model": "no-such-model-xyz",
            "input": "hi",
            "max_output_tokens": 4,
        },
    )
    expected_not_found = {
        "code": "model_not_found",
        "message": "The requested model 'no-such-model-xyz' does not exist.",
        "param": "model",
        "type": "invalid_request_error",
    }
    nf_ok = (
        nfcode == 400
        and isinstance(nfdata, dict)
        and nfdata.get("error") == expected_not_found
    )
    report.add(
        "endpoint",
        "POST /v1/responses (unknown model)",
        "PASS" if nf_ok else "FAIL",
        f"HTTP {nfcode} body={json_preview(nfdata)}",
    )

    return rid


def json_preview(obj: Any) -> str:
    import json

    try:
        return json.dumps(obj, ensure_ascii=False)[:160]
    except Exception:
        return repr(obj)[:160]
