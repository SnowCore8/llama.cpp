"""Semantic-level Responses checks (behavior, not mere field echo)."""

from __future__ import annotations

import json
import time
from typing import Any

from .http_client import ResponsesHttpClient, output_text, parse_sse
from .report import Report
from .validators import validate_response


def _create(
    client: ResponsesHttpClient,
    model: str,
    extra: dict[str, Any],
    body: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    payload = {"model": model, "max_output_tokens": 64, **extra, **body}
    if "max_output_tokens" in body:
        payload["max_output_tokens"] = body["max_output_tokens"]
    code, data = client.post_json("/v1/responses", payload)
    return code, data if isinstance(data, dict) else {"_data": data}


def _logprobs_nonempty(data: dict[str, Any]) -> bool:
    for item in data.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            lp = part.get("logprobs")
            if isinstance(lp, list) and len(lp) > 0:
                return True
    return False


def run_semantic_checks(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    # --- include + top_logprobs populate output_text.logprobs ---
    code, data = _create(
        client,
        model,
        extra,
        {
            "input": "Reply with exactly: LOGPROB",
            "temperature": 0,
            "top_logprobs": 2,
            "include": ["message.output_text.logprobs"],
            "max_output_tokens": 32,
        },
    )
    if code != 200:
        report.add("semantic", "include_output_text_logprobs", "FAIL", f"HTTP {code}")
    elif _logprobs_nonempty(data):
        report.add("semantic", "include_output_text_logprobs", "PASS", "logprobs populated")
    else:
        report.add(
            "semantic",
            "include_output_text_logprobs",
            "FAIL",
            "logprobs empty despite include+top_logprobs",
        )

    # --- background returns in_progress then completes via retrieve ---
    code, stub = _create(
        client,
        model,
        extra,
        {
            "background": True,
            "input": "Reply with exactly: BG_DONE",
            "temperature": 0,
            "max_output_tokens": 32,
        },
    )
    if code != 200 or not stub.get("id"):
        report.add("semantic", "background_async_complete", "FAIL", f"HTTP {code}")
    elif stub.get("status") not in ("in_progress", "queued"):
        report.add(
            "semantic",
            "background_async_complete",
            "FAIL",
            f"expected in_progress/queued got status={stub.get('status')!r}",
        )
    else:
        rid = stub["id"]
        final = None
        for _ in range(120):
            time.sleep(0.5)
            gcode, got = client.get_json(f"/v1/responses/{rid}")
            if gcode != 200:
                continue
            st = got.get("status")
            if st in ("completed", "failed", "cancelled", "incomplete"):
                final = got
                break
        if final and final.get("status") == "completed" and "BG_DONE" in output_text(final):
            report.add(
                "semantic",
                "background_async_complete",
                "PASS",
                f"id={rid}",
            )
        elif final and final.get("status") == "completed":
            report.add(
                "semantic",
                "background_async_complete",
                "PARTIAL",
                f"completed text={output_text(final)!r}",
            )
        else:
            report.add(
                "semantic",
                "background_async_complete",
                "FAIL",
                f"final={None if final is None else final.get('status')}",
            )

    # --- cancel rejects completed (non-cancellable) responses ---
    code, done = _create(
        client,
        model,
        extra,
        {"input": "Reply with exactly: NOCANCEL", "temperature": 0},
    )
    if code == 200 and done.get("id"):
        ccode, err = client.post_json(f"/v1/responses/{done['id']}/cancel", {})
        report.add(
            "semantic",
            "cancel_rejects_completed",
            "PASS" if ccode == 400 else "FAIL",
            f"HTTP {ccode} {str(err)[:120]}",
        )
    else:
        report.add("semantic", "cancel_rejects_completed", "FAIL", f"create HTTP {code}")

    # --- cancel succeeds on background in_progress ---
    code, stub = _create(
        client,
        model,
        extra,
        {
            "background": True,
            "input": "Write a very long detailed essay about mathematics history.",
            "max_output_tokens": 512,
        },
    )
    if code == 200 and stub.get("id") and stub.get("status") in ("in_progress", "queued"):
        ccode, cancelled = client.post_json(f"/v1/responses/{stub['id']}/cancel", {})
        ok = ccode == 200 and cancelled.get("status") == "cancelled"
        report.add(
            "semantic",
            "cancel_background_in_progress",
            "PASS" if ok else "FAIL",
            f"HTTP {ccode} status={cancelled.get('status') if isinstance(cancelled, dict) else None}",
        )
    else:
        report.add(
            "semantic",
            "cancel_background_in_progress",
            "FAIL",
            f"HTTP {code} status={stub.get('status') if isinstance(stub, dict) else None}",
        )

    # --- compact: user msgs + compaction item ---
    code, seed = _create(
        client,
        model,
        extra,
        {"input": "Remember secret SEM_COMPACT_9. Reply OK.", "temperature": 0},
    )
    if code == 200 and seed.get("id"):
        ccode, compacted = client.post_json(
            "/v1/responses/compact",
            {
                "model": model,
                "previous_response_id": seed["id"],
                "input": "compact please",
            },
        )
        out = compacted.get("output") if isinstance(compacted, dict) else None
        has_comp = isinstance(out, list) and any(
            isinstance(x, dict) and x.get("type") == "compaction" for x in out
        )
        has_user = isinstance(out, list) and any(
            isinstance(x, dict) and x.get("role") == "user" for x in out
        )
        ok = (
            ccode == 200
            and compacted.get("object") == "response.compaction"
            and has_comp
            and has_user
            and isinstance(compacted.get("usage"), dict)
        )
        report.add(
            "semantic",
            "compact_user_then_compaction_item",
            "PASS" if ok else "FAIL",
            f"HTTP {ccode} has_user={has_user} has_comp={has_comp}",
        )
        # Expand local. via previous_response_id on the compacted response id.
        # Assert via follow input_items (model may spend tokens on reasoning).
        if ok and isinstance(compacted, dict) and compacted.get("id"):
            cid = compacted["id"]
            rcode, got = client.get_json(f"/v1/responses/{cid}")
            stored = rcode == 200 and isinstance(got, dict) and got.get("id") == cid
            fcode, follow = _create(
                client,
                model,
                extra,
                {
                    "previous_response_id": cid,
                    "input": "What secret did I ask you to remember? Reply token only.",
                    "temperature": 0,
                    "max_output_tokens": 32,
                },
            )
            expanded = False
            if fcode == 200 and isinstance(follow, dict) and follow.get("id"):
                _, items = client.get_json(f"/v1/responses/{follow['id']}/input_items")
                try:
                    expanded = "SEM_COMPACT_9" in json.dumps(items, ensure_ascii=False)
                except Exception:
                    expanded = False
            text = output_text(follow) if isinstance(follow, dict) else ""
            expand_ok = stored and expanded
            report.add(
                "semantic",
                "compact_expand_previous_response_id",
                "PASS" if expand_ok else "FAIL",
                f"retrieve={rcode} follow={fcode} expanded={expanded} text={text!r}",
            )
        else:
            report.add(
                "semantic",
                "compact_expand_previous_response_id",
                "FAIL",
                "compact seed missing id",
            )
    else:
        report.add("semantic", "compact_user_then_compaction_item", "FAIL", f"seed HTTP {code}")

    # --- unknown prompt.id → 400 (templates under --openai-files-path/prompts/) ---
    code_miss, _ = _create(
        client,
        model,
        extra,
        {
            "input": "Reply with exactly: PROMPT_BASE",
            "temperature": 0,
            "prompt": {"id": "pmpt_does_not_exist_zzz"},
        },
    )
    report.add(
        "semantic",
        "prompt_id_unknown_rejected",
        "PASS" if code_miss >= 400 else "FAIL",
        f"HTTP {code_miss}",
    )

    # --- stream_options present on streamed completed response ---
    scode, headers, raw = client.request(
        "POST",
        "/v1/responses",
        {
            "model": model,
            "input": "Reply with exactly: SO",
            "max_output_tokens": 32,
            "stream": True,
            "temperature": 0,
            "stream_options": {"include_obfuscation": False},
            **extra,
        },
        stream=True,
    )
    events = parse_sse(raw) if scode == 200 else []
    completed = next((obj for t, obj in events if t == "response.completed"), None)
    resp = (completed or {}).get("response") if isinstance(completed, dict) else None
    so = resp.get("stream_options") if isinstance(resp, dict) else None
    report.add(
        "semantic",
        "stream_options_on_completed",
        "PASS" if scode == 200 and isinstance(so, dict) else "FAIL",
        f"HTTP {scode} stream_options={so!r}",
    )

    # --- input_items lists stored input with ids ---
    code, created = _create(
        client,
        model,
        extra,
        {"input": "Reply with exactly: ITEMS", "temperature": 0},
    )
    if code == 200 and created.get("id"):
        icode, items = client.get_json(f"/v1/responses/{created['id']}/input_items")
        data = items.get("data") if isinstance(items, dict) else None
        ok = (
            icode == 200
            and items.get("object") == "list"
            and isinstance(data, list)
            and len(data) >= 1
            and all(isinstance(x.get("id"), str) and x["id"] for x in data if isinstance(x, dict))
        )
        report.add(
            "semantic",
            "input_items_have_ids",
            "PASS" if ok else "FAIL",
            f"HTTP {icode} n={len(data) if isinstance(data, list) else None}",
        )
    else:
        report.add("semantic", "input_items_have_ids", "FAIL", f"create HTTP {code}")

    # --- max_tool_calls=0 drops the tool-call attempt; no function_call survives ---
    code, data = _create(
        client,
        model,
        extra,
        {
            "max_tool_calls": 0,
            "temperature": 0,
            "tools": [
                {
                    "type": "function",
                    "name": "ping",
                    "description": "ping",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "function", "name": "ping"},
            "input": "call ping",
            "max_output_tokens": 64,
        },
    )
    fcs = [
        x
        for x in (data.get("output") or [])
        if isinstance(x, dict) and x.get("type") == "function_call"
    ]
    details = data.get("incomplete_details") if isinstance(data, dict) else None
    status = data.get("status")
    # Semantics under test: the dropped call leaves no function_call item and the cap is
    # echoed. The model may keep talking past the dropped call up to max_output_tokens,
    # so both "completed" and "incomplete/max_output_tokens" are acceptable terminals.
    terminal_ok = status == "completed" or (
        status == "incomplete"
        and isinstance(details, dict)
        and details.get("reason") == "max_output_tokens"
    )
    ok = (
        code == 200
        and data.get("max_tool_calls") == 0
        and len(fcs) == 0
        and terminal_ok
    )
    report.add(
        "semantic",
        "max_tool_calls_enforced_zero",
        "PASS" if ok else "FAIL",
        f"HTTP {code} status={status!r} incomplete={details!r} n_fc={len(fcs)}",
    )

    # --- max_tool_calls=1 still allows a single forced tool ---
    code, data = _create(
        client,
        model,
        extra,
        {
            "max_tool_calls": 1,
            "temperature": 0,
            "tools": [
                {
                    "type": "function",
                    "name": "ping",
                    "description": "ping",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "function", "name": "ping"},
            "input": "call ping",
            "max_output_tokens": 64,
        },
    )
    fcs = [
        x
        for x in (data.get("output") or [])
        if isinstance(x, dict) and x.get("type") == "function_call"
    ]
    ok = code == 200 and data.get("max_tool_calls") == 1 and len(fcs) == 1
    report.add(
        "semantic",
        "max_tool_calls_with_forced_tool",
        "PASS" if ok else "FAIL",
        f"HTTP {code} max_tool_calls={data.get('max_tool_calls')} n_fc={len(fcs)}",
    )

    # --- max_output_tokens yields incomplete ---
    code, data = _create(
        client,
        model,
        extra,
        {
            "input": "Write a long detailed essay about the history of mathematics world wide.",
            "temperature": 0,
            "max_output_tokens": 3,
        },
    )
    details = data.get("incomplete_details") if isinstance(data, dict) else None
    ok_inc = (
        code == 200
        and data.get("status") == "incomplete"
        and isinstance(details, dict)
        and details.get("reason") == "max_output_tokens"
        and int((data.get("usage") or {}).get("output_tokens") or 0) <= 3
    )
    report.add(
        "semantic",
        "max_output_tokens_incomplete",
        "PASS" if ok_inc else "FAIL",
        f"HTTP {code} status={data.get('status')!r} details={details!r} "
        f"out={(data.get('usage') or {}).get('output_tokens')}",
    )

    # --- input_tokens endpoint matches create usage.input_tokens ---
    # Explicit verbosity pins both probes to the shared explicit hint path: the
    # omitted-default injection applies to the create path only, so an omitted
    # probe would count the hint on one side and not the other.
    probe_input = "token parity probe " + ("alpha " * 30)
    tcode, tok = client.post_json(
        "/v1/responses/input_tokens",
        {"model": model, "input": probe_input, "text": {"verbosity": "medium"}, **extra},
    )
    ccode, created = _create(
        client,
        model,
        extra,
        {"input": probe_input, "temperature": 0, "max_output_tokens": 4, "text": {"verbosity": "medium"}},
    )
    tin = tok.get("input_tokens") if isinstance(tok, dict) else None
    cin = (created.get("usage") or {}).get("input_tokens") if isinstance(created, dict) else None
    report.add(
        "semantic",
        "input_tokens_matches_create_usage",
        "PASS" if tcode == 200 and ccode == 200 and tin == cin and tin is not None else "FAIL",
        f"input_tokens={tin} create_usage={cin}",
    )

    # --- metadata / user retrieve round-trip ---
    code, data = _create(
        client,
        model,
        extra,
        {
            "input": "Reply with exactly: META_OK",
            "temperature": 0,
            "metadata": {"suite": "semantic", "n": "1"},
            "user": "semantic-user",
            "safety_identifier": "semantic-safe",
        },
    )
    if code == 200 and data.get("id"):
        gcode, got = client.get_json(f"/v1/responses/{data['id']}")
        ok_rt = (
            gcode == 200
            and isinstance(got, dict)
            and got.get("metadata") == data.get("metadata")
            and got.get("user") == "semantic-user"
            and got.get("safety_identifier") == "semantic-safe"
            and "META_OK" in output_text(got)
        )
        report.add(
            "semantic",
            "metadata_user_retrieve_roundtrip",
            "PASS" if ok_rt else "FAIL",
            f"get_http={gcode} metadata={got.get('metadata') if isinstance(got, dict) else None!r}",
        )
    else:
        report.add("semantic", "metadata_user_retrieve_roundtrip", "FAIL", f"create HTTP {code}")

    # image input (1x1 PNG)
    png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    code, data = _create(
        client,
        model,
        extra,
        {
            "temperature": 0,
            "max_output_tokens": 32,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{png_b64}",
                        },
                        {"type": "input_text", "text": "Reply with exactly: IMAGE_OK"},
                    ],
                }
            ],
        },
    )
    report.add(
        "semantic",
        "input_image_content",
        "PASS" if code == 200 and "IMAGE_OK" in output_text(data) else "FAIL",
        f"HTTP {code} text={output_text(data)!r}",
    )

    # --- context_management compaction shortens long history ---
    prev = None
    for i in range(3):
        body = {
            "input": f"Turn {i}: remember fact F{i} and reply OK{i}",
            "temperature": 0,
            "max_output_tokens": 24,
        }
        if prev:
            body["previous_response_id"] = prev
        code, data = _create(client, model, extra, body)
        if code != 200 or not data.get("id"):
            report.add(
                "semantic",
                "context_management_compacts_history",
                "FAIL",
                f"seed turn {i} HTTP {code}",
            )
            break
        prev = data["id"]
    else:
        code_a, data_a = _create(
            client,
            model,
            extra,
            {
                "previous_response_id": prev,
                "input": "Reply with exactly: AFTER_COMPACT",
                "temperature": 0,
                "max_output_tokens": 24,
            },
        )
        code_b, data_b = _create(
            client,
            model,
            extra,
            {
                "previous_response_id": prev,
                "input": "Reply with exactly: AFTER_COMPACT",
                "temperature": 0,
                "max_output_tokens": 24,
                "context_management": [{"type": "compaction"}],
            },
        )
        tok_a = ((data_a.get("usage") or {}) if isinstance(data_a, dict) else {}).get(
            "input_tokens"
        )
        tok_b = ((data_b.get("usage") or {}) if isinstance(data_b, dict) else {}).get(
            "input_tokens"
        )
        ok = (
            code_a == 200
            and code_b == 200
            and isinstance(tok_a, int)
            and isinstance(tok_b, int)
            and tok_b < tok_a
            and data_b.get("context_management") == [{"type": "compaction"}]
        )
        report.add(
            "semantic",
            "context_management_compacts_history",
            "PASS" if ok else "FAIL",
            f"http={code_a}/{code_b} tokens={tok_a}/{tok_b}",
        )
        # Expanded compaction must still expose folded facts to the model (not a placeholder).
        code_r, data_r = _create(
            client,
            model,
            extra,
            {
                "previous_response_id": data_b.get("id") if isinstance(data_b, dict) else prev,
                "input": "What was fact F0? Reply with exactly: F0",
                "temperature": 0,
                "max_output_tokens": 32,
                "context_management": [{"type": "compaction"}],
            },
        )
        text_r = output_text(data_r if isinstance(data_r, dict) else {})
        status_r = data_r.get("status") if isinstance(data_r, dict) else None
        if code_r != 200 or not text_r or status_r not in ("completed", "incomplete"):
            verdict = "FAIL"
            detail_r = f"HTTP {code_r} status={status_r!r} text={text_r!r}"
        elif "F0" in text_r:
            verdict = "PASS"
            detail_r = f"HTTP {code_r} text={text_r!r}"
        else:
            # Model capability gap, not an API violation: the turn ran, but the model did
            # not recall the folded fact. 0.8B measured; 9B passes
            # (evidence-9B-resp-289-FAIL0.json). The expand mechanism keeps direct
            # coverage in semantic/compact_expand_previous_response_id.
            verdict = "SKIP"
            detail_r = (
                f"model capability: F0 not recalled from expanded history; 0.8B fails, "
                f"9B passes (evidence-9B-resp-289-FAIL0.json); expand mechanism covered by "
                f"semantic/compact_expand_previous_response_id. HTTP {code_r} text={text_r!r}"
            )
        report.add("semantic", "context_management_expands_for_model", verdict, detail_r)

    # --- stream_options.include_obfuscation toggles SSE obfuscation field ---
    def _stream_obf(include: bool) -> tuple[int, bool]:
        body = {
            "model": model,
            "stream": True,
            "stream_options": {"include_obfuscation": include},
            "input": "Reply with exactly: OBF",
            "temperature": 0,
            "max_output_tokens": 16,
            **extra,
        }
        code, _, raw = client.request("POST", "/v1/responses", body, stream=True)
        from responses_official_acceptance.http_client import parse_sse

        has = False
        for _ev, obj in parse_sse(raw):
            if isinstance(obj, dict) and "obfuscation" in obj:
                has = True
                break
        return code, has

    c_true, has_true = _stream_obf(True)
    c_false, has_false = _stream_obf(False)
    ok = c_true == 200 and c_false == 200 and has_true and not has_false
    report.add(
        "semantic",
        "stream_options_obfuscation_toggle",
        "PASS" if ok else "FAIL",
        f"true={c_true}/{has_true} false={c_false}/{has_false}",
    )

    # --- stream_options omitted: obfuscation is on by default ---
    code_def, _, raw_def = client.request(
        "POST",
        "/v1/responses",
        {
            "model": model,
            "stream": True,
            "input": "Reply with exactly: OBF_DEFAULT",
            "temperature": 0,
            "max_output_tokens": 16,
            **extra,
        },
        stream=True,
    )
    n_def_events = 0
    n_def_obf = 0
    for _ev, obj in parse_sse(raw_def):
        if isinstance(obj, dict):
            n_def_events += 1
            if "obfuscation" in obj:
                n_def_obf += 1
    report.add(
        "semantic",
        "stream_options_obfuscation_default",
        "PASS" if code_def == 200 and n_def_obf > 0 else "FAIL",
        f"HTTP {code_def} n_events={n_def_events} n_with_obfuscation={n_def_obf}",
    )

    # --- truncation: disabled rejects; auto drops oldest (tokenizer vs n_ctx budget) ---
    secret = "SECRET_TRUNC_ZZ9"
    pcode, props = client.get_json("/props")
    n_ctx = 2048
    if pcode == 200 and isinstance(props, dict):
        n_ctx = int(props.get("n_ctx") or (props.get("default_generation_settings") or {}).get("n_ctx") or n_ctx)
        if "default_generation_settings" in props and isinstance(props["default_generation_settings"], dict):
            n_ctx = int(props["default_generation_settings"].get("n_ctx") or n_ctx)
    # Exceed real slot budget: long pads so oldest (with secret) drop under truncation=auto.
    item_words = max(128, n_ctx // 5)
    pad = ("tw " * item_words).strip()
    oversized = []
    for i in range(5):
        oversized.append({"role": "user", "content": f"msg{i} remember {secret} {pad}"})
    for i in range(5):
        oversized.append({"role": "user", "content": f"filler{i} {pad}"})
    oversized.append({"role": "user", "content": "Reply with exactly: TA"})
    code_dis, _ = client.post_json(
        "/v1/responses",
        {
            "model": model,
            "truncation": "disabled",
            "input": oversized,
            "max_output_tokens": 8,
            "temperature": 0,
            **extra,
        },
    )
    code_auto, data_auto = _create(
        client,
        model,
        extra,
        {
            "truncation": "auto",
            "input": oversized,
            "max_output_tokens": 16,
            "temperature": 0,
        },
    )
    text_auto = output_text(data_auto)
    ok = (
        code_dis >= 400
        and code_auto == 200
        and "TA" in text_auto
        and secret not in text_auto
    )
    report.add(
        "semantic",
        "truncation_auto_vs_disabled",
        "PASS" if ok else "FAIL",
        f"disabled={code_dis} auto={code_auto} text={text_auto!r}",
    )
    # Follow-up: ask for the early secret after truncation — should not recall it.
    if code_auto == 200 and isinstance(data_auto, dict) and data_auto.get("id"):
        code_ask, data_ask = _create(
            client,
            model,
            extra,
            {
                "previous_response_id": data_auto["id"],
                "truncation": "auto",
                "input": f"What secret code was in the early messages? If unknown reply exactly: UNKNOWN",
                "max_output_tokens": 24,
                "temperature": 0,
            },
        )
        text_ask = output_text(data_ask)
        report.add(
            "semantic",
            "truncation_auto_drops_early_secret",
            "PASS"
            if code_ask == 200 and secret not in text_ask and "UNKNOWN" in text_ask
            else "FAIL",
            f"HTTP {code_ask} text={text_ask!r}",
        )
    else:
        report.add(
            "semantic",
            "truncation_auto_drops_early_secret",
            "FAIL",
            "auto truncation create failed",
        )

    # --- cloud-shaped fields: invalid enum/shape → 400; valid still accepted ---
    bad_cases = [
        ("prompt_missing_id", {"prompt": {"version": "1"}}, "prompt"),
        ("prompt_cache_retention_invalid", {"prompt_cache_retention": "forever"}, "prompt_cache_retention"),
        ("prompt_cache_options_mode_invalid", {"prompt_cache_options": {"mode": "magic"}}, "prompt_cache_options"),
        ("prompt_cache_key_not_string", {"prompt_cache_key": 123}, "prompt_cache_key"),
    ]
    for name, extra_bad, _ in bad_cases:
        code_bad, _ = client.post_json(
            "/v1/responses",
            {
                "model": model,
                "input": "Reply with exactly: X",
                "max_output_tokens": 8,
                "temperature": 0,
                **extra_bad,
                **extra,
            },
        )
        report.add(
            "semantic",
            name,
            "PASS" if code_bad >= 400 else "FAIL",
            f"HTTP {code_bad}",
        )
    code_ok, data_ok = _create(
        client,
        model,
        extra,
        {
            "prompt": {"id": "pmpt_local_echo"},
            "prompt_cache_key": "pck-valid",
            "prompt_cache_retention": "24h",
            "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            "input": "Reply with exactly: CLOUD_SHAPED_OK",
            "max_output_tokens": 16,
            "temperature": 0,
        },
    )
    ok_valid = code_ok == 200 and "CLOUD_SHAPED_OK" in output_text(data_ok)
    report.add(
        "semantic",
        "cloud_shaped_valid_accepted",
        "PASS" if ok_valid else "FAIL",
        f"HTTP {code_ok} text={output_text(data_ok)!r}",
    )
    code_exp, _ = _create(
        client,
        model,
        extra,
        {
            "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            "input": "x",
            "max_output_tokens": 8,
            "temperature": 0,
        },
    )
    report.add(
        "semantic",
        "prompt_cache_options_explicit_requires_key",
        "PASS" if code_exp >= 400 else "FAIL",
        f"HTTP {code_exp}",
    )

    # --- metadata / user / safety_identifier durable retrieve ---
    code, data = _create(
        client,
        model,
        extra,
        {
            "input": "Reply with exactly: META_DUR",
            "temperature": 0,
            "max_output_tokens": 16,
            "metadata": {"suite": "semantic", "k": "durable"},
            "user": "acceptance-user",
            "safety_identifier": "sid-durable-1",
        },
    )
    if code != 200 or not data.get("id"):
        report.add("semantic", "metadata_user_safety_durable", "FAIL", f"HTTP {code}")
    else:
        gcode, got = client.get_json(f"/v1/responses/{data['id']}")
        ok = (
            gcode == 200
            and got.get("metadata") == {"suite": "semantic", "k": "durable"}
            and got.get("user") == "acceptance-user"
            and got.get("safety_identifier") == "sid-durable-1"
            and "META_DUR" in output_text(got)
        )
        report.add(
            "semantic",
            "metadata_user_safety_durable",
            "PASS" if ok else "FAIL",
            f"get={gcode} metadata={got.get('metadata')!r} user={got.get('user')!r} sid={got.get('safety_identifier')!r}",
        )

    # --- include=reasoning.encrypted_content fills local. blob when reasoning present ---
    code, data = _create(
        client,
        model,
        extra,
        {
            "input": "Think briefly then say HI",
            "reasoning": {"effort": "low"},
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": 128,
        },
    )
    encs = [
        x.get("encrypted_content")
        for x in (data.get("output") or [])
        if isinstance(x, dict) and x.get("type") == "reasoning"
    ]
    ok_enc = (
        code == 200
        and any(isinstance(e, str) and e.startswith("local.") for e in encs)
    )
    report.add(
        "semantic",
        "include_reasoning_encrypted_content",
        "PASS" if ok_enc else "FAIL",
        f"HTTP {code} encs={encs!r}",
    )

    # --- without include=reasoning.encrypted_content the field is not emitted ---
    code_plain, data_plain = _create(
        client,
        model,
        extra,
        {
            "input": "Think briefly then say HI",
            "reasoning": {"effort": "low"},
            "max_output_tokens": 128,
        },
    )
    items_plain = [
        x
        for x in (data_plain.get("output") or [])
        if isinstance(x, dict) and x.get("type") == "reasoning"
    ]
    has_enc_key = any("encrypted_content" in x for x in items_plain)
    report.add(
        "semantic",
        "include_reasoning_encrypted_content_absent_by_default",
        "PASS" if code_plain == 200 and items_plain and not has_enc_key else "FAIL",
        f"HTTP {code_plain} n_reasoning={len(items_plain)} enc_key={has_enc_key}",
    )

    # --- parallel_tool_calls=false: at most one function_call emitted ---
    code, data = _create(
        client,
        model,
        extra,
        {
            "parallel_tool_calls": False,
            "reasoning": {"effort": "none"},
            "tools": [
                {
                    "type": "function",
                    "name": "alpha",
                    "description": "A",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "type": "function",
                    "name": "beta",
                    "description": "B",
                    "parameters": {"type": "object", "properties": {}},
                },
            ],
            # Force a named tool so we observe a call even when parallel is false.
            "tool_choice": {"type": "function", "name": "alpha"},
            "input": "Call alpha now.",
            "temperature": 0,
            "max_output_tokens": 128,
        },
    )
    fcs = [
        x
        for x in (data.get("output") or [])
        if isinstance(x, dict) and x.get("type") == "function_call"
    ]
    names = [x.get("name") for x in fcs]
    ok_par = code == 200 and len(fcs) == 1 and names == ["alpha"]
    report.add(
        "semantic",
        "parallel_tool_calls_false_single",
        "PASS" if ok_par else "FAIL",
        f"HTTP {code} n_fc={len(fcs)} names={names!r} status={data.get('status')!r}",
    )

    # --- reasoning.effort ladder: higher effort -> more reasoning text (budget mapping);
    # --- measure it through the official summary channel, raw text is not exposed ---
    def _reasoning_chars(effort: str) -> tuple[int, int, str]:
        c, d = _create(
            client,
            model,
            extra,
            {
                "input": "Solve carefully: what is 17*19?",
                "reasoning": {"effort": effort, "summary": "detailed"},
                "temperature": 0,
                "max_output_tokens": 512,
            },
        )
        chars = 0
        for o in d.get("output") or []:
            if isinstance(o, dict) and o.get("type") == "reasoning":
                for p in o.get("summary") or []:
                    if isinstance(p, dict):
                        chars += len(p.get("text") or "")
        return c, chars, d.get("status") or ""

    c_min, n_min, st_min = _reasoning_chars("minimal")
    c_low, n_low, st_low = _reasoning_chars("low")
    c_high, n_high, st_high = _reasoning_chars("high")
    ok_eff = (
        c_min == 200
        and c_low == 200
        and c_high == 200
        and n_min > 0
        and n_low >= n_min
        and n_high >= n_low
    )
    report.add(
        "semantic",
        "reasoning_effort_budget_ladder",
        "PASS" if ok_eff else "FAIL",
        f"minimal={n_min}/{st_min} low={n_low}/{st_low} high={n_high}/{st_high}",
    )

    # reasoning.mode=pro elevates low effort toward high budget locally
    def _reasoning_chars_mode(effort: str, mode: str) -> tuple[int, int]:
        c, d = _create(
            client,
            model,
            extra,
            {
                "input": "Solve carefully: what is 23*29?",
                "reasoning": {"effort": effort, "mode": mode, "summary": "detailed"},
                "temperature": 0,
                "max_output_tokens": 512,
            },
        )
        chars = 0
        for o in d.get("output") or []:
            if isinstance(o, dict) and o.get("type") == "reasoning":
                for p in o.get("summary") or []:
                    if isinstance(p, dict):
                        chars += len(p.get("text") or "")
        return c, chars

    c_std, n_std = _reasoning_chars_mode("low", "standard")
    c_pro, n_pro = _reasoning_chars_mode("low", "pro")
    ok_pro = c_std == 200 and c_pro == 200 and n_std > 0 and n_pro >= n_std
    report.add(
        "semantic",
        "reasoning_mode_pro_boosts_low_effort",
        "PASS" if ok_pro else "FAIL",
        f"standard_low={n_std} pro_low={n_pro}",
    )
