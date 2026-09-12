"""Official streaming event checks."""

from __future__ import annotations

from typing import Any

from .catalog import CONDITIONAL_STREAM_EVENTS, CORE_STREAM_EVENTS
from .http_client import ResponsesHttpClient, iter_sse, parse_sse
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

    # --- max_tool_calls cap: excess calls are ignored, the stream still completes ---
    mcode, _, mraw = client.request(
        "POST",
        "/v1/responses",
        {
            "model": model,
            "input": "call ping",
            "max_output_tokens": 64,
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
        "input": "Reply with exactly: RESUME_OK",
        "max_output_tokens": 64,
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
    created_resp = created.get("response") if isinstance(created, dict) else None
    immediate = (
        "response.created" in types
        and isinstance(created_resp, dict)
        and created_resp.get("status") == "in_progress"
        and created_resp.get("background") is True
    )
    report.add(
        cat,
        "background_stream.immediate",
        "PASS" if immediate else "FAIL",
        f"types={types[:4]} status={(created_resp or {}).get('status')!r}",
    )
    rid = _first_response_id(first)
    cursor = max((obj.get("sequence_number", -1) for _, obj in first), default=-1)
    if not rid or cursor < 0:
        report.add(cat, "background_stream.resume", "FAIL", f"no id/cursor in first events (rid={rid!r}, cursor={cursor})")
        report.add(cat, "background_stream.retrieve_after_disconnect", "FAIL", "no id")
    else:
        # GET /v1/responses/{id}?stream=true&starting_after=N
        # must replay only events after the cursor and finish with response.completed
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
            ok_resume = (
                len(rest) > 0
                and all(isinstance(s, int) and s > cursor for s in rest_seqs)
                and "response.completed" in rest_types
            )
            report.add(
                cat,
                "background_stream.resume",
                "PASS" if ok_resume else "FAIL",
                f"cursor={cursor} n={len(rest)} seq head={rest_seqs[:2]} completed={'response.completed' in rest_types}",
            )
        code, data = client.get_json(f"/v1/responses/{rid}")
        ok_get = code == 200 and isinstance(data, dict) and data.get("status") == "completed"
        report.add(
            cat,
            "background_stream.retrieve_after_disconnect",
            "PASS" if ok_get else "FAIL",
            f"HTTP {code} status={(data or {}).get('status')!r}",
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
    try:
        resp2 = client.open_stream("/v1/responses", cancel_body, timeout=60.0)
        head2: list[tuple[str | None, dict[str, Any]]] = []
        for ev in iter_sse(resp2, max_events=2):
            head2.append(ev)
        rid2 = _first_response_id(head2)
        if not rid2:
            resp2.close()
            report.add(cat, "background_stream.cancel", "FAIL", "no response id in first events")
        else:
            code_c, _ = client.post_json(f"/v1/responses/{rid2}/cancel", {})
            ended = False
            try:
                for _ in iter_sse(resp2):  # the cancelled stream must terminate
                    pass
                ended = True
            except Exception:
                ended = False
            finally:
                resp2.close()
            code_g, data_g = client.get_json(f"/v1/responses/{rid2}")
            status = (data_g or {}).get("status")
            ok_cancel = code_c == 200 and ended and status == "cancelled"
            report.add(
                cat,
                "background_stream.cancel",
                "PASS" if ok_cancel else "FAIL",
                f"cancel_http={code_c} stream_ended={ended} status={status!r}",
            )
    except Exception as e:
        report.add(cat, "background_stream.cancel", "FAIL", f"{e!r}")

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
