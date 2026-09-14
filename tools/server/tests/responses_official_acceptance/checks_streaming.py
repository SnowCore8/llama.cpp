"""Official streaming event checks."""

from __future__ import annotations

import json
from typing import Any, Callable

from .catalog import CONDITIONAL_STREAM_EVENTS, CORE_STREAM_EVENTS
from .http_client import (
    ResponsesHttpClient,
    as_dict as _as_dict,
    as_list as _as_list,
    iter_sse,
    parse_sse,
)
from .report import Report
from .validators import validate_event, validate_response


def _types(events: list[tuple[str | None, dict[str, Any]]]) -> list[str]:
    return [t for t, _ in events if t]


def _lp_entry_ok(e: Any) -> bool:
    """Local logprob entry contract: non-empty token, numeric logprob <= 0."""
    return (
        isinstance(e, dict)
        and isinstance(e.get("token"), str)
        and e["token"] != ""
        and isinstance(e.get("logprob"), (int, float))
        and e["logprob"] <= 0
    )


def _lp_top_ok(e: Any) -> bool:
    """Local contract: every top_logprobs entry has a non-empty token."""
    tops = _as_list(_as_dict(e).get("top_logprobs"))
    return bool(tops) and all(
        isinstance(t, dict) and isinstance(t.get("token"), str) and t["token"] != "" for t in tops
    )


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

    # --- function_call output_item.added carries the required output_index (G10) and the
    #     item id matches the position of that call in the final output array ---
    fc_added = [
        obj
        for t, obj in fevents
        if t == "response.output_item.added" and _as_dict(obj.get("item")).get("type") == "function_call"
    ]
    if not fc_added:
        report.add(
            "stream_event",
            "function_call.output_item.added.output_index",
            "FAIL",
            f"no function_call output_item.added in forced stream; types={sorted(ftypes)}",
        )
    else:
        fin_out = _as_list(
            _as_dict(
                next(
                    (obj.get("response") for t, obj in fevents
                     if t in ("response.completed", "response.incomplete")),
                    None,
                )
            ).get("output")
        )
        fin_pos = {
            _as_dict(it).get("id"): i
            for i, it in enumerate(fin_out)
            if _as_dict(it).get("type") == "function_call"
        }
        added_errs = []
        for obj in fc_added:
            ok, detail = validate_event("response.output_item.added", obj)
            if not ok:
                added_errs.append(f"sdk={detail}")
                continue
            item_id = _as_dict(obj.get("item")).get("id")
            if obj.get("output_index") != fin_pos.get(item_id):
                added_errs.append(
                    f"output_index={obj.get('output_index')!r} final_pos={fin_pos.get(item_id)!r}"
                )
        report.add(
            "stream_event",
            "function_call.output_item.added.output_index",
            "PASS" if not added_errs else "FAIL",
            f"n={len(fc_added)}" + ("" if not added_errs else f" errs={added_errs[:2]}"),
        )

    # --- response.function_call_arguments.done carries exactly the official fields (G12) ---
    done_args = [obj for t, obj in fevents if t == "response.function_call_arguments.done"]
    need_done = {"arguments", "item_id", "output_index", "sequence_number", "type"}
    bad_extra = [obj for obj in done_args if "name" in obj]
    bad_missing = [sorted(need_done - set(obj)) for obj in done_args]
    ok_done = bool(done_args) and not bad_extra and not any(bad_missing)
    report.add(
        "stream_event",
        "function_call_arguments.done.official_fields",
        "PASS" if ok_done else "FAIL",
        f"n={len(done_args)} with_name={len(bad_extra)} missing={bad_missing[:1]}",
    )

    # --- parallel tool calls: unique stable fc_ item ids and per-item output_index (G11) ---
    pcode, _, praw = client.request(
        "POST",
        "/v1/responses",
        {
            "model": model,
            "input": "Call get_weather for Paris and get_time for London. Use both tools in one turn.",
            "max_output_tokens": 128,
            "temperature": 0,
            "stream": True,
            "parallel_tool_calls": True,
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
                },
                {
                    "type": "function",
                    "name": "get_time",
                    "description": "time",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            ],
            "tool_choice": "required",
            **extra,
        },
        stream=True,
    )
    pevents = parse_sse(praw) if pcode == 200 else []
    p_out = _as_list(
        _as_dict(
            next(
                (obj.get("response") for t, obj in pevents
                 if t in ("response.completed", "response.incomplete")),
                None,
            )
        ).get("output")
    )
    p_fcs = [(i, _as_dict(it)) for i, it in enumerate(p_out) if _as_dict(it).get("type") == "function_call"]
    if len(p_fcs) >= 2:
        report.add(
            "stream_event",
            "parallel_tool_calls.forced",
            "PASS",
            f"n_fc={len(p_fcs)} (parallel_tool_calls=true + tool_choice=required)",
        )
    else:
        report.add(
            "stream_event",
            "parallel_tool_calls.forced",
            "FAIL",
            f"HTTP {pcode} n_fc={len(p_fcs)}: expected 2 calls from an explicit two-tool prompt",
        )

    ids_final = [it.get("id") for _, it in p_fcs]
    pos_of_id = {it.get("id"): i for i, it in p_fcs}
    p_errs = []
    if not ids_final:
        p_errs.append("no function_call items")
    if not all(isinstance(iid, str) and iid.startswith("fc_") for iid in ids_final):
        p_errs.append(f"ids not fc_-prefixed: {ids_final[:4]}")
    if len(set(ids_final)) != len(ids_final):
        p_errs.append(f"duplicate ids: {ids_final}")
    added_ids = set()
    for t, obj in pevents:
        it = _as_dict(obj.get("item"))
        if t == "response.output_item.added" and it.get("type") == "function_call":
            added_ids.add(it.get("id"))
            if obj.get("output_index") != pos_of_id.get(it.get("id")):
                p_errs.append(
                    f"added {it.get('id')!r} output_index={obj.get('output_index')!r} "
                    f"want={pos_of_id.get(it.get('id'))!r}"
                )
        elif t == "response.output_item.done" and it.get("type") == "function_call":
            if obj.get("output_index") != pos_of_id.get(it.get("id")):
                p_errs.append(
                    f"done {it.get('id')!r} output_index={obj.get('output_index')!r} "
                    f"want={pos_of_id.get(it.get('id'))!r}"
                )
        elif t in ("response.function_call_arguments.delta", "response.function_call_arguments.done"):
            if pos_of_id.get(obj.get("item_id")) != obj.get("output_index"):
                p_errs.append(
                    f"{t} item_id={obj.get('item_id')!r} output_index={obj.get('output_index')!r}"
                )
    if ids_final and added_ids != set(ids_final):
        p_errs.append(f"added ids {sorted(added_ids)} != final ids {sorted(set(ids_final))}")
    report.add(
        "stream_event",
        "parallel_tool_calls.unique_stable_ids",
        "PASS" if not p_errs else "FAIL",
        f"n_fc={len(p_fcs)} ids={ids_final[:4]} errs={p_errs[:3]}",
    )

    # --- message items: phase is optional in the official schema ---
    # local output has no phase source (the field is input-only), so absence is the expected
    # shape; if a phase ever shows up it must carry one of the two official labels
    msg_items = []
    for t, obj in events:
        if t in ("response.output_item.added", "response.output_item.done"):
            it = _as_dict(obj.get("item"))
            if it.get("type") == "message":
                msg_items.append(it)
    happy_out = _as_list(
        _as_dict(
            next(
                (obj.get("response") for t, obj in events
                 if t in ("response.completed", "response.incomplete")),
                None,
            )
        ).get("output")
    )
    for it in happy_out:
        if _as_dict(it).get("type") == "message":
            msg_items.append(_as_dict(it))
    phases = [it["phase"] for it in msg_items if "phase" in it]
    ok_phase = bool(msg_items) and all(p in ("commentary", "final_answer") for p in phases)
    report.add(
        "stream_event",
        "message_phase.optional",
        "PASS" if ok_phase else "FAIL",
        f"items={len(msg_items)} with_phase={len(phases)}; official field is optional "
        f"(local generation has no phase source)",
    )

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

    # --- max_tool_calls cap: excess calls are ignored, the stream still completes ---
    mcode, _, mraw = client.request(
        "POST",
        "/v1/responses",
        {
            "model": model,
            "input": "call ping",
            "max_output_tokens": 64,
            "temperature": 0,
            "stream": True,
            "max_tool_calls": 0,
            "tools": [
                {
                    "type": "function",
                    "name": "ping",
                    "description": "ping",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "function", "name": "ping"},
            **extra,
        },
        stream=True,
    )
    mevents = parse_sse(mraw) if mcode == 200 else []
    mtypes = _types(mevents)
    mfinal = next(
        (obj for t, obj in mevents if t in ("response.completed", "response.incomplete")),
        None,
    )
    mresp = (mfinal or {}).get("response") if isinstance(mfinal, dict) else None
    m_fcs = [
        x
        for x in ((mresp or {}).get("output") or [])
        if isinstance(x, dict) and x.get("type") == "function_call"
    ]
    ok = (
        mcode == 200
        and "response.completed" in mtypes
        and "response.incomplete" not in mtypes
        and isinstance(mresp, dict)
        and mresp.get("status") == "completed"
        and not mresp.get("incomplete_details")
        and not m_fcs
    )
    report.add(
        "stream_event",
        "max_tool_calls_cap_completes",
        "PASS" if ok else "FAIL",
        f"HTTP {mcode} completed={'response.completed' in mtypes} "
        f"incomplete={'response.incomplete' in mtypes} "
        f"status={(mresp or {}).get('status')!r} n_fc={len(m_fcs)}",
    )

    # --- force the official reasoning summary events ---
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
            "reasoning": {"effort": "low", "summary": "detailed"},
        },
        stream=True,
    )
    revents = parse_sse(rraw) if rcode == 200 else []
    rtypes = _types(revents)

    for ev in (
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_summary_part.done",
    ):
        if ev not in rtypes:
            report.add(
                "stream_event",
                f"{ev}.forced",
                "FAIL",
                f"missing; HTTP {rcode} types={sorted(set(rtypes))}",
            )
            continue
        errs = []
        for t, obj in revents:
            if t != ev:
                continue
            ok, detail = validate_event(ev, obj)
            if not ok:
                errs.append(detail)
        if errs:
            report.add("stream_event", f"{ev}.forced", "FAIL", f"sdk_err={errs[:1]}")
        else:
            forced[ev] = "forced reasoning.summary=detailed stream+sdk_ok"

    summary_joined = "".join(
        obj.get("delta", "") for t, obj in revents if t == "response.reasoning_summary_text.delta"
    )
    summary_done = next(
        (obj.get("text") for t, obj in revents if t == "response.reasoning_summary_text.done"),
        None,
    )
    report.add(
        "stream_event",
        "reasoning_summary_text.delta_join_matches_done",
        "PASS" if summary_done is not None and summary_joined == summary_done else "FAIL",
        f"joined={len(summary_joined)} done={None if summary_done is None else len(summary_done)}",
    )

    # the part is marked incomplete only when the generation was cut, see the SDK field doc
    part_done = next(
        (obj for t, obj in revents if t == "response.reasoning_summary_part.done"), None
    )
    final_resp = next(
        (obj.get("response") for t, obj in revents
         if t in ("response.completed", "response.incomplete")),
        None,
    )
    incomplete_reason = ((final_resp or {}).get("incomplete_details") or {}).get("reason")
    pd_status = (part_done or {}).get("status")
    report.add(
        "stream_event",
        "reasoning_summary_part.done.status_matches_response",
        "PASS" if part_done is not None and (pd_status == "incomplete") == (incomplete_reason == "max_output_tokens")
        else "FAIL",
        f"part.done.status={pd_status!r} incomplete_reason={incomplete_reason!r}",
    )

    # aligned with the official API, which never exposes the raw reasoning text
    raw_events = sorted({t for t in rtypes if t.startswith("response.reasoning_text")})
    reasoning_item = next(
        (it for it in ((final_resp or {}).get("output") or [])
         if isinstance(it, dict) and it.get("type") == "reasoning"),
        None,
    )
    report.add(
        "stream_event",
        "reasoning_raw_text_absent",
        "PASS" if not raw_events and reasoning_item is not None and "content" not in reasoning_item
        else "FAIL",
        f"events={raw_events} content={'content' in (reasoning_item or {})}",
    )

    # --- pre-flight rejection: a prompt that exceeds the context is rejected before
    #     response.created, so the client gets a plain JSON error, not SSE frames ---
    # pad size follows /props.n_ctx, so the row holds on any ctx instead of failing for a config reason
    n_ctx = 32768
    pcode, props = client.get_json("/props")
    if pcode == 200 and isinstance(props, dict):
        dgs = props.get("default_generation_settings") or {}
        if isinstance(dgs, dict) and dgs.get("n_ctx"):
            n_ctx = int(dgs["n_ctx"])
        elif props.get("n_ctx"):
            n_ctx = int(props["n_ctx"])
    huge = "test " * max(60000, int(n_ctx * 1.5) + 512)
    xcode, xheaders, xraw = client.request(
        "POST",
        "/v1/responses",
        {
            "model": model,
            "input": huge,
            "max_output_tokens": 8,
            "stream": True,
            **extra,
        },
        stream=True,
    )
    xct = xheaders.get("Content-Type", "")
    try:
        xjson = json.loads(xraw.decode(errors="replace") or "null")
    except Exception:
        xjson = None
    xbody = _as_dict(xjson)
    xinner = _as_dict(xbody.get("error")) or xbody
    pre_ok = (
        xcode == 400
        and "text/event-stream" not in xct
        and isinstance(xjson, dict)
        and isinstance(xinner.get("message"), str)
        and bool(xinner.get("message"))
    )
    report.add(
        "stream_event",
        "pre_created_error_is_plain_json",
        "PASS" if pre_ok else "FAIL",
        f"HTTP {xcode} ct={xct!r} type={xinner.get('type')!r} msg={str(xinner.get('message'))[:90]!r}",
    )

    # --- custom tool call stream: item added -> input delta -> input done -> item done ---
    ccode, _, craw = client.request(
        "POST",
        "/v1/responses",
        {
            "model": model,
            "input": "Call the dj_play tool now.",
            "max_output_tokens": 128,
            "temperature": 0,
            "stream": True,
            "tools": [{"type": "custom", "name": "dj_play", "format": {"type": "text"}}],
            "tool_choice": {"type": "custom", "name": "dj_play"},
            **extra,
        },
        stream=True,
    )
    cevents = parse_sse(craw) if ccode == 200 else []
    c_added = [
        i
        for i, (t, obj) in enumerate(cevents)
        if t == "response.output_item.added"
        and _as_dict(obj.get("item")).get("type") == "custom_tool_call"
    ]
    c_delta = [
        (i, _as_dict(obj))
        for i, (t, obj) in enumerate(cevents)
        if t == "response.custom_tool_call_input.delta"
    ]
    c_done = [
        (i, _as_dict(obj))
        for i, (t, obj) in enumerate(cevents)
        if t == "response.custom_tool_call_input.done"
    ]
    c_item_done = [
        _as_dict(obj.get("item"))
        for t, obj in cevents
        if t == "response.output_item.done"
        and _as_dict(obj.get("item")).get("type") == "custom_tool_call"
    ]
    c_delta_text = "".join(str(obj.get("delta", "")) for _, obj in c_delta)
    c_done_input = c_done[0][1].get("input") if c_done else None
    c_item = c_item_done[0] if c_item_done else {}
    c_order_ok = bool(c_added and c_delta and c_done) and c_added[0] < c_delta[0][0] < c_done[0][0]
    c_ok = (
        ccode == 200
        and c_order_ok
        and all(isinstance(obj.get("delta"), str) for _, obj in c_delta)
        and isinstance(c_done_input, str)
        and c_item.get("name") == "dj_play"
        and isinstance(c_item.get("call_id"), str)
    )
    report.add(
        "stream_event",
        "custom_tool_call_stream",
        "PASS" if c_ok else "FAIL",
        f"HTTP {ccode} added={len(c_added)} delta={len(c_delta)} done={len(c_done)} "
        f"order_ok={c_order_ok} name={c_item.get('name')!r} id={str(c_item.get('id'))[:20]!r} "
        f"call_id={str(c_item.get('call_id'))[:20]!r} delta={c_delta_text[:44]!r} "
        f"input={str(c_done_input)[:32]!r}",
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


def _first_response_id(events: list[tuple[str | None, dict[str, Any]]]) -> str | None:
    for t, obj in events:
        if t == "response.created" and isinstance(obj.get("response"), dict):
            rid = obj["response"].get("id")
            if isinstance(rid, str) and rid:
                return rid
    return None


def run_background_stream_checks(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    """Official background + stream: immediate SSE, resumable by cursor, cancellable."""
    cat = "background_stream"
    body = {
        "model": model,
        # long generation: the client disconnect below must happen while it is still running
        "input": "Write a long detailed essay about the history of mathematics.",
        "max_output_tokens": 256,
        "temperature": 0,
        "background": True,
        "stream": True,
        **extra,
    }
    try:
        resp = client.open_stream("/v1/responses", body, timeout=60.0)
    except Exception as e:
        for name in (
            "background_stream.immediate",
            "background_stream.resume",
            "background_stream.retrieve_after_disconnect",
            "background_stream.cancel",
            "background_stream.non_background_resume_404",
        ):
            report.add(cat, name, "FAIL", f"open failed: {e!r}")
        return

    # read the first events only, then drop the connection: the generation must survive
    first: list[tuple[str | None, dict[str, Any]]] = []
    try:
        for ev in iter_sse(resp, max_events=4):
            first.append(ev)
    finally:
        resp.close()
    types = _types(first)
    created = next((obj for t, obj in first if t == "response.created"), None)
    created_resp = _as_dict(_as_dict(created).get("response"))
    immediate = (
        "response.created" in types
        and isinstance(created, dict)
        and created_resp.get("status") in ("in_progress", "queued")
        and created_resp.get("background") is True
    )
    report.add(
        cat,
        "background_stream.immediate",
        "PASS" if immediate else "FAIL",
        f"types={types[:4]} status={created_resp.get('status')!r}",
    )
    rid = _first_response_id(first)
    cursor = max((s for s in (obj.get("sequence_number") for _, obj in first) if isinstance(s, int)), default=-1)
    if not rid or cursor < 0:
        report.add(cat, "background_stream.resume", "FAIL", f"no id/cursor in first events (rid={rid!r}, cursor={cursor})")
        report.add(cat, "background_stream.retrieve_after_disconnect", "FAIL", "no id")
    else:
        # GET /v1/responses/{id}?stream=true&starting_after=N
        # must replay only events after the cursor, gap-free, through a terminal event
        try:
            resume = client.open_stream(
                f"/v1/responses/{rid}?stream=true&starting_after={cursor}",
                timeout=90.0,
            )
        except Exception as e:
            report.add(cat, "background_stream.resume", "FAIL", f"resume open failed: {e!r}")
        else:
            try:
                rest = list(iter_sse(resume))
            finally:
                resume.close()
            rest_seqs = [obj.get("sequence_number") for _, obj in rest]
            rest_types = _types(rest)
            expected_seqs = list(range(cursor + 1, cursor + 1 + len(rest_seqs)))
            terminal = [t for t in rest_types if t in ("response.completed", "response.incomplete")]
            ok_resume = (
                len(rest) > 0
                and all(isinstance(s, int) for s in rest_seqs)
                and rest_seqs == expected_seqs
                and bool(terminal)
            )
            report.add(
                cat,
                "background_stream.resume",
                "PASS" if ok_resume else "FAIL",
                f"cursor={cursor} n={len(rest)} seq head={rest_seqs[:3]} "
                f"gap_free={rest_seqs == expected_seqs} terminal={terminal[:1]}",
            )
        code, data = client.get_json(f"/v1/responses/{rid}")
        data_d = _as_dict(data)
        ok_get = code == 200 and data_d.get("status") in ("completed", "incomplete")
        report.add(
            cat,
            "background_stream.retrieve_after_disconnect",
            "PASS" if ok_get else "FAIL",
            f"HTTP {code} status={data_d.get('status')!r}",
        )

    # cancel: a live background stream must stop promptly and flip the stored status
    cancel_body = {
        "model": model,
        "input": "Write a long detailed essay about mathematics history.",
        "max_output_tokens": 512,
        "temperature": 0,
        "background": True,
        "stream": True,
        **extra,
    }
    resp2 = None
    try:
        resp2 = client.open_stream("/v1/responses", cancel_body, timeout=60.0)
        state: dict[str, Any] = {}
        head2: list[tuple[str | None, dict[str, Any]]] = []
        for ev in iter_sse(resp2, max_events=2, state=state):
            head2.append(ev)
        rid2 = _first_response_id(head2)
        if not rid2:
            report.add(cat, "background_stream.cancel", "FAIL", "no response id in first events")
        else:
            code_c, _ = client.post_json(f"/v1/responses/{rid2}/cancel", {})
            # the cancelled stream must end: a terminal event or EOF, not a socket timeout
            ended = False
            try:
                for t, _obj in iter_sse(resp2, state=state):
                    if t in ("response.completed", "response.incomplete", "response.failed"):
                        ended = True
                        break
                if not ended and state.get("_eof"):
                    ended = True
            except Exception:
                ended = False
            code_g, data_g = client.get_json(f"/v1/responses/{rid2}")
            status = _as_dict(data_g).get("status")
            ok_cancel = code_c == 200 and ended and status == "cancelled"
            report.add(
                cat,
                "background_stream.cancel",
                "PASS" if ok_cancel else "FAIL",
                f"cancel_http={code_c} stream_ended={ended} status={status!r}",
            )
    except Exception as e:
        report.add(cat, "background_stream.cancel", "FAIL", f"{e!r}")
    finally:
        if resp2 is not None:
            resp2.close()

    # a plain (non-background) stream has no resumable session
    code, _, raw = client.request(
        "POST",
        "/v1/responses",
        {
            "model": model,
            "input": "hi",
            "max_output_tokens": 8,
            "stream": True,
            **extra,
        },
        stream=True,
    )
    plain = parse_sse(raw) if code == 200 else []
    pid = _first_response_id(plain)
    if pid:
        code_r, _ = client.get_json(f"/v1/responses/{pid}?stream=true")
        report.add(
            cat,
            "background_stream.non_background_resume_404",
            "PASS" if code_r == 404 else "FAIL",
            f"HTTP {code_r}",
        )
    else:
        report.add(cat, "background_stream.non_background_resume_404", "SKIP", f"no id (HTTP {code})")


def run_output_logprobs_stream_checks(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    """Streaming logprobs (local contract: include or top_logprobs>0 emits them)."""
    cat = "output_logprobs"

    def stream_events(body: dict[str, Any]) -> tuple[int, list[tuple[str | None, dict[str, Any]]]]:
        code, _, raw = client.request("POST", "/v1/responses", body, stream=True)
        return code, parse_sse(raw) if code == 200 else []

    include_body = {
        "model": model,
        "input": "Reply with exactly: LP_OK",
        "max_output_tokens": 16,
        "temperature": 0,
        "stream": True,
        "include": ["message.output_text.logprobs"],
        **extra,
    }
    code, events = stream_events(include_body)
    deltas = [obj for t, obj in events if t == "response.output_text.delta"]
    lp_entries = [e for obj in deltas for e in _as_list(obj.get("logprobs"))]
    lp_ok = len(deltas) > 0 and len(lp_entries) > 0 and all(_lp_entry_ok(e) for e in lp_entries)
    report.add(
        cat,
        "stream_delta_logprobs.local",
        "PASS" if lp_ok else "FAIL",
        f"local: include or top_logprobs>0 yields logprobs. HTTP {code} deltas={len(deltas)} "
        f"entries={len(lp_entries)} sample={lp_entries[:1]}",
    )
    done = next((obj for t, obj in events if t == "response.output_text.done"), None)
    done_lp = _as_list(_as_dict(done).get("logprobs"))
    done_ok = len(done_lp) > 0 and all(_lp_entry_ok(e) for e in done_lp)
    report.add(
        cat,
        "stream_done_logprobs.local",
        "PASS" if done_ok else "FAIL",
        f"local: entries={len(done_lp)} sample={done_lp[:1]}",
    )

    top_body = {
        "model": model,
        "input": "Reply with exactly: LP_OK",
        "max_output_tokens": 16,
        "temperature": 0,
        "stream": True,
        "top_logprobs": 2,
        **extra,
    }
    code_t, events_t = stream_events(top_body)
    deltas_t = [obj for t, obj in events_t if t == "response.output_text.delta"]
    lp_t = [e for obj in deltas_t for e in _as_list(obj.get("logprobs"))]
    top_ok = len(lp_t) > 0 and all(_lp_top_ok(e) for e in lp_t)
    report.add(
        cat,
        "stream_delta_top_logprobs.local",
        "PASS" if top_ok else "FAIL",
        f"local: top_logprobs>0 yields logprobs. HTTP {code_t} deltas={len(deltas_t)} "
        f"entries={len(lp_t)} top0={_as_list(_as_dict(lp_t[0]).get('top_logprobs')) if lp_t else None}",
    )

    plain_body = {
        "model": model,
        "input": "Reply with exactly: LP_OK",
        "max_output_tokens": 16,
        "temperature": 0,
        "stream": True,
        **extra,
    }
    code_p, events_p = stream_events(plain_body)
    deltas_p = [obj for t, obj in events_p if t == "response.output_text.delta"]
    plain_ok = len(deltas_p) > 0 and all(not (obj.get("logprobs") or []) for obj in deltas_p)
    report.add(
        cat,
        "stream_logprobs_omitted_by_default",
        "PASS" if plain_ok else "FAIL",
        f"HTTP {code_p} deltas={len(deltas_p)}",
    )


def run_web_search_stream_checks(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    """Official streaming web_search: call lifecycle events + url_citation annotations.

    Under a web-search fixture the server is deterministic; when no source shows up
    (e.g. provider=none), the probes are not judgeable and report SKIP.
    """
    cat = "web_search_stream"
    body = {
        "model": model,
        "input": "What is example.com used for? Reply briefly and cite the sources.",
        "max_output_tokens": 128,
        "temperature": 0,
        "stream": True,
        "tools": [
            {
                "type": "web_search",
                "search_context_size": "medium",
                "filters": {"allowed_domains": ["example.com", "iana.org", "rfc-editor.org"]},
            }
        ],
        "include": ["web_search_call.action.sources"],
        "reasoning": {"effort": "none"},
        **extra,
    }
    code, _, raw = client.request("POST", "/v1/responses", body, stream=True)
    if code != 200:
        report.add(cat, "web_search_stream.events", "FAIL", f"HTTP {code}")
        report.add(cat, "web_search_stream.annotation", "FAIL", f"HTTP {code}")
        return
    events = parse_sse(raw)

    def is_ws_call(obj: dict[str, Any]) -> bool:
        return _as_dict(obj.get("item")).get("type") == "web_search_call"

    ws_items: list[dict[str, Any]] = []
    for t, obj in events:
        if t in ("response.output_item.added", "response.output_item.done") and is_ws_call(obj):
            ws_items.append(_as_dict(obj.get("item")))
        if t in ("response.completed", "response.incomplete", "response.failed"):
            for it in _as_list(_as_dict(obj.get("response")).get("output")):
                if _as_dict(it).get("type") == "web_search_call":
                    ws_items.append(_as_dict(it))
    sources = [s for it in ws_items for s in _as_list(_as_dict(it.get("action")).get("sources"))]
    ws_events = [t for t, _ in events if t is not None and t.startswith("response.web_search_call.")]
    search_seen = bool(ws_items) or bool(ws_events)

    if not search_seen or not sources:
        reason = (
            "no web_search_call observed"
            if not search_seen
            else "search returned no sources (provider=none, or no LLAMA_WEB_SEARCH_FIXTURE on the server?)"
        )
        report.add(cat, "web_search_stream.events", "SKIP", f"{reason}; not judgeable")
        report.add(cat, "web_search_stream.annotation", "SKIP", f"{reason}; not judgeable")
        return

    def first_idx(pred: Callable[[str, dict[str, Any]], bool]) -> int | None:
        return next((i for i, (t, o) in enumerate(events) if t is not None and pred(t, o)), None)

    # call lifecycle: added -> in_progress -> searching -> completed -> done (others may interleave)
    idx = {
        "output_item.added": first_idx(lambda t, o: t == "response.output_item.added" and is_ws_call(o)),
        "in_progress": first_idx(lambda t, _o: t == "response.web_search_call.in_progress"),
        "searching": first_idx(lambda t, _o: t == "response.web_search_call.searching"),
        "completed": first_idx(lambda t, _o: t == "response.web_search_call.completed"),
        "output_item.done": first_idx(lambda t, o: t == "response.output_item.done" and is_ws_call(o)),
    }
    i_added = idx["output_item.added"]
    i_in_progress = idx["in_progress"]
    i_searching = idx["searching"]
    i_completed = idx["completed"]
    i_done = idx["output_item.done"]
    order_ok = (
        all(v is not None for v in idx.values())
        and i_added < i_in_progress < i_searching < i_completed < i_done
    )
    report.add(
        cat,
        "web_search_stream.events",
        "PASS" if order_ok else "FAIL",
        f"indices={idx} (want added<in_progress<searching<completed<done)",
    )

    # url_citation annotations arrive before output_text.done and echo in content_part.done
    ann_url = [
        o
        for t, o in events
        if t == "response.output_text.annotation.added"
        and _as_dict(o.get("annotation")).get("type") == "url_citation"
        and isinstance(_as_dict(o.get("annotation")).get("start_index"), int)
        and isinstance(_as_dict(o.get("annotation")).get("end_index"), int)
    ]
    i_first_ann = first_idx(
        lambda t, o: t == "response.output_text.annotation.added"
        and _as_dict(o.get("annotation")).get("type") == "url_citation"
    )
    i_text_done = first_idx(lambda t, _o: t == "response.output_text.done")
    before_done = (
        i_first_ann is not None and i_text_done is not None and i_first_ann < i_text_done
    )
    part_done_ok = any(
        _as_dict(o.get("part")).get("type") == "output_text"
        and any(
            _as_dict(a).get("type") == "url_citation"
            for a in _as_list(_as_dict(o.get("part")).get("annotations"))
        )
        for t, o in events
        if t == "response.content_part.done"
    )
    ann_ok = bool(ann_url) and before_done and part_done_ok
    report.add(
        cat,
        "web_search_stream.annotation",
        "PASS" if ann_ok else "FAIL",
        f"url_citations={len(ann_url)} before_text_done={before_done} "
        f"part_done_annotations={part_done_ok} sources={len(sources)}",
    )
