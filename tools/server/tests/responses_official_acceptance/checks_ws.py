"""Official Responses WebSocket checks (transport, mid-turn steering, beta inject).

Normative source: the official "WebSocket mode" guide (per-connection lane FIFO,
16 in-flight cap, 32 named-stream cap, 60-minute connection cap), the
"WebSocket events" reference (stream_id echo rules, error envelope shape,
response.steer.* / response.inject.* shapes) and the "Mid-turn steering" guide.
store=false continuation reuse follows the official same-WebSocket-connection rule.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable
from urllib.parse import urlparse

from .http_client import ResponsesHttpClient
from .report import Report

# events that end a response on any transport
_TERMINAL = ("response.completed", "response.incomplete", "response.failed")

_PROMPT = "Reply with exactly: OK"

_CHECK_BUDGET_S = 60.0  # wall-clock budget for one check
_RECV_CAP_S = 30.0      # upper bound for a single recv call
_MAX_FRAMES = 400       # frame cap for a single-response drain

_STEER_ID_RE = re.compile(r"^steer_[0-9a-f]{32}$")
_STEER_STOP = ("response.steer.accepted", "response.steer.failed", "error")
_START_STOP = ("response.created", "error", "response.failed")

# Output long enough that a just-created response is still running when the
# steering event lands; short enough to end below the token cap and complete.
_LONG_PROMPT = "Count from 1 to 40, separated by spaces."

_RUNNABLE_NAMES = (
    "connect/create default lane",
    "stream_id echo (named lane)",
    "stream_id validation",
    "error shape: unknown event type",
    "error shape: invalid json",
    "previous_response_not_found",
    "lane FIFO",
    "two lanes concurrent",
    "named lane limit (32)",
    "steer.accepted shape (default + named lane)",
    "steer.error: invalid_input cases",
    "steer.error: response_not_found",
    "steer.error: steering_not_supported (conversation)",
    "steer.error: steering_not_supported (compaction)",
    "steer.error: too_many_pending_steers (33x)",
    "steer interrupt -> successor",
    "steer.accepted on successor (re-steer)",
    "steer.error: response_already_completed",
    "steer.pending: waiting for required input",
    "inject.created shape",
    "inject.error: response_not_found",
    "inject.error: response_already_completed",
    "inject schema violation closes connection",
    "store=false continuation over WS (connection-local cache)",
    "store=false continuation rejected outside the connection",
    "steer successor carries store=false target",
    "store=false: failed same-lane continuation evicts the parent (cross-lane fork keeps it)",
    "generate=false warmup: created+completed, empty output, chainable (store=false)",
    "generate=false warmup does not consume queued steers",
)


def _ws_url(base_url: str) -> str:
    """Derive the ws(s):// responses endpoint from the HTTP base URL."""
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{host}:{port}/v1/responses"


def _connect(ws_connect: Callable[..., Any], client: ResponsesHttpClient, beta: bool = False):
    headers = {"Authorization": f"Bearer {client.api_key}"}
    if beta:
        # official beta WebSocket clients opt in to the beta Responses surface
        headers["OpenAI-Beta"] = "responses_multi_agent=v1"
    return ws_connect(
        _ws_url(client.base_url),
        additional_headers=headers,
        open_timeout=15,
        close_timeout=5,
    )


def _create(model: str, extra: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """response.create envelope with the official top-level fields plus overrides."""
    ev: dict[str, Any] = {
        "type": "response.create",
        "model": model,
        "input": _PROMPT,
        "max_output_tokens": 32,
        "temperature": 0,
    }
    ev.update(overrides)
    ev.update(extra)
    return ev


def _recv_frame(ws: Any, deadline: float) -> dict[str, Any] | None:
    """One JSON frame; None when the absolute deadline is exhausted."""
    remain = deadline - time.monotonic()
    if remain <= 0:
        return None
    try:
        raw = ws.recv(timeout=min(_RECV_CAP_S, remain))
    except TimeoutError:
        return None
    try:
        obj = json.loads(raw)
    except Exception:
        return {"_unparsed": str(raw)[:160]}
    return obj if isinstance(obj, dict) else {"_unparsed": str(obj)[:160]}


def _drain_one(ws: Any, deadline: float, max_frames: int = _MAX_FRAMES):
    """Read frames until an error or a terminal event: (frames, outcome, event)."""
    frames: list[dict[str, Any]] = []
    while len(frames) < max_frames:
        obj = _recv_frame(ws, deadline)
        if obj is None:
            return frames, "timeout", None
        frames.append(obj)
        et = obj.get("type")
        if et == "error":
            return frames, "error", obj
        if et in _TERMINAL:
            return frames, et, obj
    return frames, "max_frames", None


def _err_of(ev: Any) -> dict[str, Any]:
    """The error object of an error frame, {} otherwise."""
    if isinstance(ev, dict) and isinstance(ev.get("error"), dict):
        return ev["error"]
    return {}


def _has_key(obj: Any, key: str) -> bool:
    """True when key appears anywhere in the JSON tree (lists included)."""
    if isinstance(obj, dict):
        return key in obj or any(_has_key(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_key(v, key) for v in obj)
    return False


def _read_until(
    ws: Any,
    deadline: float,
    pred: Callable[[dict[str, Any], list[dict[str, Any]]], bool],
    max_frames: int = _MAX_FRAMES,
) -> list[dict[str, Any]]:
    """Read frames until pred(frame, frames) turns true or the budget runs out."""
    frames: list[dict[str, Any]] = []
    while len(frames) < max_frames:
        obj = _recv_frame(ws, deadline)
        if obj is None:
            break
        frames.append(obj)
        if pred(obj, frames):
            break
    return frames


def _resp_id(frame: Any) -> Any:
    """The response.id carried by a response-scoped frame, None when absent."""
    resp = frame.get("response") if isinstance(frame, dict) else None
    return resp.get("id") if isinstance(resp, dict) else None


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _types_of(frames: list[dict[str, Any]]) -> str:
    return ",".join(sorted({str(f.get("type")) for f in frames}))


def run_ws_checks(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    """WebSocket transport checks over /v1/responses."""
    cat = "ws"

    try:
        from websockets.sync.client import connect as ws_connect
    except Exception as err:  # noqa: BLE001 - report and stop the group
        report.add(cat, "dependency (websockets)", "FAIL",
                   f"websockets package required for ws checks: {err}"[:300])
        return

    def add(name: str, ok: bool, detail: str) -> None:
        report.add(cat, name, "PASS" if ok else "FAIL", detail[:300])

    def guarded(name: str, body: Callable[[float], None]) -> None:
        deadline = time.monotonic() + _CHECK_BUDGET_S
        try:
            body(deadline)
        except Exception as err:  # noqa: BLE001 - one broken check must not hide the rest
            add(name, False, f"crash {type(err).__name__}: {err}")

    def add_skips() -> None:
        report.add(cat, "60-minute connection limit", "SKIP",
                   "not exercised in acceptance runs (would need 60 minutes)")
        report.add(cat, "16 in-flight cap", "SKIP",
                   "not exercised in acceptance runs (hard to reproduce reliably "
                   "against a local server; revisit in phase 2)")

    # Fast preflight: when the endpoint does not speak WebSocket at all, fail the
    # group quickly instead of burning the per-check budget once per check.
    try:
        with _connect(ws_connect, client):
            pass
    except Exception as err:  # noqa: BLE001 - recorded per check below
        why = f"ws endpoint unavailable: {type(err).__name__}: {err}"
        for name in _RUNNABLE_NAMES:
            add(name, False, why)
        add_skips()
        return

    def default_lane(deadline: float) -> None:
        # default lane: no stream_id on the request, no stream_id on any frame
        with _connect(ws_connect, client) as ws:
            ws.send(json.dumps(_create(model, extra)))
            frames, outcome, ev = _drain_one(ws, deadline)
            created = any(f.get("type") == "response.created" for f in frames)
            leaked = _has_key(frames, "stream_id")
            ok = outcome == "response.completed" and created and not leaked
            detail = f"outcome={outcome} frames={len(frames)} created={created} stream_id_leak={leaked}"
            if outcome == "error":
                detail += f" error={json.dumps(_err_of(ev))[:120]}"
            add("connect/create default lane", ok, detail)

    def named_lane_echo(deadline: float) -> None:
        sid = "lane_a"
        with _connect(ws_connect, client) as ws:
            ws.send(json.dumps(_create(model, extra, stream_id=sid)))
            frames, outcome, ev = _drain_one(ws, deadline)
            start = next(
                (i for i, f in enumerate(frames)
                 if f.get("type") == "response.created" and f.get("stream_id") == sid),
                None,
            )
            tail = frames[start:] if start is not None else []
            echoed = sum(1 for f in tail if f.get("stream_id") == sid)
            terminal_sid = ev.get("stream_id") if isinstance(ev, dict) else None
            ok = (
                outcome == "response.completed"
                and start is not None
                and echoed == len(tail)
                and terminal_sid == sid
            )
            detail = (
                f"outcome={outcome} frames={len(frames)} created_at={start} "
                f"echo={echoed}/{len(tail)} terminal_stream_id={terminal_sid!r}"
            )
            if outcome == "error":
                detail += f" error={json.dumps(_err_of(ev))[:120]}"
            add("stream_id echo (named lane)", ok, detail)

    def stream_id_validation(deadline: float) -> None:
        # official rule: 1-256 chars, letters/digits/underscore/hyphen/period only
        cases: tuple[tuple[str, Any], ...] = (
            ("empty", ""),
            ("non_string", 123),
            ("too_long", "x" * 257),
            ("bad_chars", "bad name!"),
        )
        all_ok = True
        details: list[str] = []
        for label, sid in cases:
            case_deadline = min(deadline, time.monotonic() + 15.0)
            ok = False
            detail = f"{label}: no frame"
            try:
                with _connect(ws_connect, client) as ws:
                    ws.send(json.dumps(_create(model, extra, stream_id=sid)))
                    frames, outcome, ev = _drain_one(ws, case_deadline, max_frames=64)
                    err = _err_of(ev)
                    st = ev.get("status") if isinstance(ev, dict) else None
                    echo = isinstance(ev, dict) and "stream_id" in ev
                    ok = (
                        outcome == "error"
                        and st == 400
                        and err.get("code") == "invalid_stream_id"
                        and err.get("param") == "stream_id"
                        and not echo
                    )
                    if outcome == "error":
                        detail = (f"{label}: status={st} code={err.get('code')!r} "
                                  f"param={err.get('param')!r} echo={echo}")
                    else:
                        detail = f"{label}: {outcome} (want error frame)"
            except Exception as err2:  # noqa: BLE001 - report the case, keep going
                detail = f"{label}: {type(err2).__name__}: {err2}"
            all_ok = all_ok and ok
            details.append(detail)
        add("stream_id validation", all_ok, "; ".join(details))

    def unknown_event_type(deadline: float) -> None:
        with _connect(ws_connect, client) as ws:
            ws.send(json.dumps({"type": "bogus"}))
            frames, outcome, ev = _drain_one(ws, deadline, max_frames=32)
            err = _err_of(ev)
            st = ev.get("status") if isinstance(ev, dict) else None
            ok = (
                outcome == "error"
                and st == 400
                and isinstance(err.get("type"), str) and bool(err.get("type"))
                and err.get("code") == "unsupported_event_type"
                and isinstance(err.get("message"), str) and bool(err.get("message"))
                and err.get("param") == "type"
            )
            add("error shape: unknown event type", ok,
                f"outcome={outcome} status={st} error={json.dumps(err)[:200]}")

    def invalid_json(deadline: float) -> None:
        with _connect(ws_connect, client) as ws:
            ws.send("not-a-json")
            frames, outcome, ev = _drain_one(ws, deadline, max_frames=32)
            err = _err_of(ev)
            st = ev.get("status") if isinstance(ev, dict) else None
            ok = (
                outcome == "error"
                and st == 400
                and isinstance(err.get("type"), str) and bool(err.get("type"))
                and err.get("code") == "invalid_json"
                and isinstance(err.get("message"), str) and bool(err.get("message"))
                and "param" in err
            )
            add("error shape: invalid json", ok,
                f"outcome={outcome} status={st} error={json.dumps(err)[:200]}")

    def previous_response_missing(deadline: float) -> None:
        sid = "gone"
        with _connect(ws_connect, client) as ws:
            ws.send(json.dumps(_create(model, extra, stream_id=sid, previous_response_id="resp_missing")))
            frames, outcome, ev = _drain_one(ws, deadline, max_frames=64)
            err = _err_of(ev)
            st = ev.get("status") if isinstance(ev, dict) else None
            st_sid = ev.get("stream_id") if isinstance(ev, dict) else None
            ok = (
                outcome == "error"
                and st == 400
                and err.get("code") == "previous_response_not_found"
                and err.get("param") == "previous_response_id"
                and st_sid == sid
            )
            add("previous_response_not_found", ok,
                f"outcome={outcome} status={st} stream_id={st_sid!r} error={json.dumps(err)[:180]}")

    def lane_fifo(deadline: float) -> None:
        sid = "fifo"
        with _connect(ws_connect, client) as ws:
            ws.send(json.dumps(_create(model, extra, stream_id=sid, max_output_tokens=8)))
            ws.send(json.dumps(_create(model, extra, stream_id=sid, max_output_tokens=8)))
            frames: list[dict[str, Any]] = []
            created_at: list[int] = []
            terminal_at: list[int] = []
            errors: list[dict[str, Any]] = []
            while len(frames) < _MAX_FRAMES:
                obj = _recv_frame(ws, deadline)
                if obj is None:
                    break
                frames.append(obj)
                et = obj.get("type")
                if et == "response.created":
                    created_at.append(len(frames))
                elif et in _TERMINAL:
                    terminal_at.append(len(frames))
                elif et == "error":
                    errors.append(obj)
                if len(terminal_at) >= 2:
                    break
            echo = sum(1 for f in frames if f.get("stream_id") == sid)
            ok = (
                len(created_at) == 2
                and len(terminal_at) == 2
                and created_at[1] > terminal_at[0]
                and echo == len(frames)
                and not errors
            )
            detail = (
                f"created@{created_at} terminal@{terminal_at} frames={len(frames)} "
                f"lane_frames={echo} errors={len(errors)}"
            )
            if errors:
                detail += f" error={json.dumps(_err_of(errors[0]))[:120]}"
            add("lane FIFO", ok, detail)

    def two_lanes(deadline: float) -> None:
        lanes = ("lane_x", "lane_y")
        with _connect(ws_connect, client) as ws:
            for sid in lanes:
                ws.send(json.dumps(_create(model, extra, stream_id=sid, max_output_tokens=8)))
            frames: list[dict[str, Any]] = []
            while len(frames) < _MAX_FRAMES:
                obj = _recv_frame(ws, deadline)
                if obj is None:
                    break
                frames.append(obj)
                if sum(1 for f in frames if f.get("type") in _TERMINAL) >= 2:
                    break
            created = {f.get("stream_id") for f in frames if f.get("type") == "response.created"}
            terminals = [f for f in frames if f.get("type") in _TERMINAL]
            term_lanes = {f.get("stream_id") for f in terminals}
            stray = {f.get("stream_id") for f in frames} - set(lanes)
            ok = (
                len(terminals) == 2
                and len(term_lanes) == 2
                and len(created) == 2
                and term_lanes == created
                and not stray
            )
            add("two lanes concurrent", ok,
                f"created={sorted(map(str, created))} terminals={sorted(map(str, term_lanes))} "
                f"frames={len(frames)} stray={sorted(map(str, stray)) or 'none'}")

    def named_lane_limit(deadline: float) -> None:
        ids = [f"limit_{i:02d}" for i in range(33)]
        last = ids[-1]
        with _connect(ws_connect, client) as ws:
            for sid in ids:
                ws.send(json.dumps(_create(model, extra, stream_id=sid, max_output_tokens=1, input="hi")))
            limit_ev: dict[str, Any] | None = None
            early_ev: dict[str, Any] | None = None
            created_lanes: set[Any] = set()
            n = 0
            while n < 4000:
                obj = _recv_frame(ws, deadline)
                if obj is None:
                    break
                n += 1
                et = obj.get("type")
                if et == "response.created":
                    created_lanes.add(obj.get("stream_id"))
                elif et == "error":
                    err = _err_of(obj)
                    if err.get("code") == "websocket_stream_limit_reached":
                        if obj.get("stream_id") == last:
                            limit_ev = obj
                            break
                        early_ev = obj
                        break
            err = _err_of(limit_ev)
            lsid = limit_ev.get("stream_id") if isinstance(limit_ev, dict) else None
            lst = limit_ev.get("status") if isinstance(limit_ev, dict) else None
            accepted = len(created_lanes & set(ids[:32]))
            ok = (
                limit_ev is not None
                and lst == 400
                and err.get("param") == "stream_id"
                and early_ev is None
            )
            detail = (
                f"limit_stream_id={lsid!r} status={lst} code={err.get('code')!r} "
                f"param={err.get('param')!r} created_seen={accepted}/32 frames={n}"
            )
            if early_ev is not None:
                detail += f" early_limit_stream_id={early_ev.get('stream_id')!r}"
            add("named lane limit (32)", ok, detail)

    def steer_accepted_shape(deadline: float) -> None:
        name = "steer.accepted shape (default + named lane)"

        def run_case(lane: str | None) -> tuple[dict[str, Any] | None, str | None, str]:
            with _connect(ws_connect, client) as ws:
                ev = _create(model, extra, max_output_tokens=200, input=_LONG_PROMPT)
                if lane is not None:
                    ev["stream_id"] = lane
                ws.send(json.dumps(ev))
                pre = _read_until(ws, deadline, lambda o, _fs: o.get("type") in _START_STOP)
                rid = next((_resp_id(f) for f in pre if f.get("type") == "response.created"), None)
                if not isinstance(rid, str):
                    return None, None, f"no response.created (types={_types_of(pre)})"
                ws.send(json.dumps({"type": "response.steer", "previous_response_id": rid, "input": "Keep it brief."}))
                frames = _read_until(ws, deadline, lambda o, _fs: o.get("type") in _STEER_STOP)
                acc = next((f for f in frames if f.get("type") == "response.steer.accepted"), None)
                if acc is None:
                    last = frames[-1] if frames else {}
                    return None, rid, f"no steer.accepted; types={_types_of(frames)} last={json.dumps(last)[:110]}"
                return acc, rid, ""

        results: list[str] = []
        all_ok = True
        for label, lane in (("default", None), ("named", "steer_lane")):
            acc, rid, err = run_case(lane)
            if acc is None:
                all_ok = False
                results.append(f"{label}: {err}")
                continue
            steer = acc.get("steer") if isinstance(acc.get("steer"), dict) else {}
            id_ok = isinstance(steer.get("id"), str) and bool(_STEER_ID_RE.match(steer.get("id") or ""))
            prev_ok = steer.get("previous_response_id") == rid
            seq_ok = _is_number(acc.get("sequence_number"))
            lane_ok = ("stream_id" not in acc) if lane is None else (acc.get("stream_id") == lane)
            all_ok = all_ok and id_ok and prev_ok and seq_ok and lane_ok
            results.append(f"{label}: id_ok={id_ok} prev_ok={prev_ok} seq_ok={seq_ok} lane_ok={lane_ok}")
        add(name, all_ok, "; ".join(results))

    def steer_invalid_input(deadline: float) -> None:
        name = "steer.error: invalid_input cases"
        with _connect(ws_connect, client) as ws:
            ws.send(json.dumps(_create(model, extra, max_output_tokens=200, input=_LONG_PROMPT)))
            pre = _read_until(ws, deadline, lambda o, _fs: o.get("type") in _START_STOP)
            rid = next((_resp_id(f) for f in pre if f.get("type") == "response.created"), None)
            if not isinstance(rid, str):
                add(name, False, f"no response.created (types={_types_of(pre)})")
                return
            cases: tuple[tuple[str, dict[str, Any]], ...] = (
                ("with_stream_id", {"type": "response.steer", "previous_response_id": rid,
                                    "input": "hi", "stream_id": "lane_a"}),
                ("extra_model", {"type": "response.steer", "previous_response_id": rid,
                                 "input": "hi", "model": "gpt-4.1"}),
                ("assistant_role", {"type": "response.steer", "previous_response_id": rid,
                                    "input": [{"type": "message", "role": "assistant", "content": "hi"}]}),
                ("empty_input", {"type": "response.steer", "previous_response_id": rid, "input": []}),
                ("numeric_input", {"type": "response.steer", "previous_response_id": rid, "input": 123}),
            )
            results: list[str] = []
            all_ok = True
            for label, ev in cases:
                ws.send(json.dumps(ev))
                frames = _read_until(ws, min(deadline, time.monotonic() + 10.0),
                                     lambda o, _fs: o.get("type") in ("response.steer.failed", "error"))
                fail = next((f for f in frames if f.get("type") == "response.steer.failed"), None)
                if fail is None:
                    all_ok = False
                    results.append(f"{label}: no steer.failed (types={_types_of(frames)})")
                    continue
                err = _err_of(fail)
                steer = fail.get("steer") if isinstance(fail.get("steer"), dict) else {}
                prev_ok = steer.get("previous_response_id") == rid
                no_id = "id" not in steer
                ok = (err.get("type") == "invalid_request_error" and err.get("code") == "invalid_input"
                      and prev_ok and no_id)
                all_ok = all_ok and ok
                results.append(f"{label}: code={err.get('code')!r} prev_ok={prev_ok} no_id={no_id}")
            add(name, all_ok, "; ".join(results))

    def steer_response_not_found(deadline: float) -> None:
        name = "steer.error: response_not_found"
        with _connect(ws_connect, client) as ws:
            ws.send(json.dumps({"type": "response.steer", "previous_response_id": "resp_missing", "input": "hello"}))
            frames = _read_until(ws, deadline, lambda o, _fs: o.get("type") in ("response.steer.failed", "error"))
            fail = next((f for f in frames if f.get("type") == "response.steer.failed"), None)
            if fail is None:
                add(name, False, f"no steer.failed (types={_types_of(frames)})")
                return
            err = _err_of(fail)
            steer = fail.get("steer") if isinstance(fail.get("steer"), dict) else {}
            prev_ok = steer.get("previous_response_id") == "resp_missing"
            no_id = "id" not in steer
            ok = (err.get("type") == "invalid_request_error" and err.get("code") == "response_not_found"
                  and prev_ok and no_id)
            add(name, ok, f"code={err.get('code')!r} type={err.get('type')!r} prev_ok={prev_ok} no_id={no_id}")

    def _steer_not_supported_case(name: str, create_ev: dict[str, Any], deadline: float) -> None:
        with _connect(ws_connect, client) as ws:
            ws.send(json.dumps(create_ev))
            pre = _read_until(ws, deadline, lambda o, _fs: o.get("type") in _START_STOP)
            rid = next((_resp_id(f) for f in pre if f.get("type") == "response.created"), None)
            if not isinstance(rid, str):
                add(name, False, f"no response.created (types={_types_of(pre)})")
                return
            ws.send(json.dumps({"type": "response.steer", "previous_response_id": rid, "input": "Keep it brief."}))
            frames = _read_until(ws, deadline, lambda o, _fs: o.get("type") in ("response.steer.failed", "error"))
            fail = next((f for f in frames if f.get("type") == "response.steer.failed"), None)
            if fail is None:
                add(name, False, f"no steer.failed (types={_types_of(frames)})")
                return
            err = _err_of(fail)
            steer = fail.get("steer") if isinstance(fail.get("steer"), dict) else {}
            prev_ok = steer.get("previous_response_id") == rid
            ok = err.get("type") == "invalid_request_error" and err.get("code") == "steering_not_supported" and prev_ok
            add(name, ok, f"code={err.get('code')!r} type={err.get('type')!r} prev_ok={prev_ok}")

    def steer_not_supported_conversation(deadline: float) -> None:
        name = "steer.error: steering_not_supported (conversation)"
        code, conv = client.post_json("/v1/conversations", {})
        cid = conv.get("id") if isinstance(conv, dict) else None
        if code != 200 or not isinstance(cid, str):
            add(name, False, f"conversation create HTTP {code} body={json.dumps(conv)[:120]}")
            return
        ev = _create(model, extra, max_output_tokens=200, input=_LONG_PROMPT)
        ev["conversation"] = cid
        _steer_not_supported_case(name, ev, deadline)

    def steer_not_supported_compaction(deadline: float) -> None:
        name = "steer.error: steering_not_supported (compaction)"
        ev = _create(model, extra, max_output_tokens=200, input=_LONG_PROMPT)
        ev["context_management"] = [{"type": "compaction"}]
        _steer_not_supported_case(name, ev, deadline)

    def steer_too_many(deadline: float) -> None:
        name = "steer.error: too_many_pending_steers (33x)"
        with _connect(ws_connect, client) as ws:
            ws.send(json.dumps(_create(model, extra, max_output_tokens=200, input=_LONG_PROMPT)))
            pre = _read_until(ws, deadline, lambda o, _fs: o.get("type") in _START_STOP)
            rid = next((_resp_id(f) for f in pre if f.get("type") == "response.created"), None)
            if not isinstance(rid, str):
                add(name, False, f"no response.created (types={_types_of(pre)})")
                return
            for i in range(33):
                ev = {"type": "response.steer", "previous_response_id": rid, "input": f"keep going ({i})"}
                ws.send(json.dumps(ev))
            n_resp = 0
            accepted = 0
            first_too_many: int | None = None
            other: Any = None
            while n_resp < 33:
                obj = _recv_frame(ws, deadline)
                if obj is None:
                    break
                t = obj.get("type")
                if t == "response.steer.accepted":
                    n_resp += 1
                    accepted += 1
                elif t == "response.steer.failed":
                    n_resp += 1
                    code = _err_of(obj).get("code")
                    if code == "too_many_pending_steers":
                        first_too_many = n_resp
                        break
                    other = code
                    break
                elif t == "error":
                    other = "generic_error"
                    break
            ok = first_too_many is not None and accepted >= 1 and first_too_many >= 33
            add(name, ok, f"accepted={accepted} responses={n_resp} first_too_many_at={first_too_many} other={other!r}")

    def steer_interrupt_successor(deadline: float) -> None:
        name = "steer interrupt -> successor"
        with _connect(ws_connect, client) as ws:
            ws.send(json.dumps(_create(model, extra, max_output_tokens=200, input=_LONG_PROMPT)))
            pre = _read_until(ws, deadline, lambda o, _fs: o.get("type") in _START_STOP)
            rid = next((_resp_id(f) for f in pre if f.get("type") == "response.created"), None)
            if not isinstance(rid, str):
                add(name, False, f"no response.created (types={_types_of(pre)})")
                return
            ws.send(json.dumps({"type": "response.steer", "previous_response_id": rid, "input": "Wrap up briefly."}))
            fa = _read_until(
                ws, deadline,
                lambda o, _fs: o.get("type") in ("response.steer.failed", "error")
                or (o.get("type") in _TERMINAL and _resp_id(o) == rid),
                max_frames=2000,
            )
            term = next((f for f in fa if f.get("type") in _TERMINAL and _resp_id(f) == rid), None)
            sbad = next((f for f in fa if f.get("type") in ("response.steer.failed", "error")), None)
            if sbad is not None:
                add(name, False, f"steer failed before commit: {json.dumps(sbad)[:150]}")
                return
            if term is None:
                add(name, False, f"no target terminal (types={_types_of(fa)})")
                return
            term_type = term.get("type")
            if term_type == "response.incomplete":
                resp = term.get("response") if isinstance(term.get("response"), dict) else {}
                details = resp.get("incomplete_details") if isinstance(resp.get("incomplete_details"), dict) else {}
                term_ok = details.get("reason") == "steered"
            else:
                term_ok = term_type == "response.completed"
            fb = _read_until(
                ws, deadline,
                lambda o, _fs: o.get("type") in ("response.steer.failed", "error", "response.failed")
                or (o.get("type") == "response.created" and _resp_id(o) != rid),
                max_frames=2000,
            )
            succ = next((f for f in fb if f.get("type") == "response.created" and _resp_id(f) != rid), None)
            fbad = next((f for f in fb if f.get("type") in ("response.steer.failed", "error", "response.failed")), None)
            if succ is None:
                add(name, False, f"terminal={term_type} term_ok={term_ok}; no successor (types={_types_of(fb)})")
                return
            sid = _resp_id(succ)
            fc = _read_until(
                ws, deadline,
                lambda o, _fs: o.get("type") in ("response.steer.failed", "error")
                or (o.get("type") in _TERMINAL and _resp_id(o) == sid),
                max_frames=2000,
            )
            sterm = next((f for f in fc if f.get("type") in _TERMINAL and _resp_id(f) == sid), None)
            cbad = next((f for f in fc if f.get("type") in ("response.steer.failed", "error")), None)
            sterm_ok = sterm is not None and sterm.get("type") == "response.completed"
            ok = term_ok and fbad is None and cbad is None and sterm_ok
            add(name, ok, f"terminal={term_type} term_ok={term_ok} successor_id={sid!r} "
                          f"successor_completed={sterm_ok} steer_failed={fbad is not None or cbad is not None}")

    def steer_successor_resteer(deadline: float) -> None:
        name = "steer.accepted on successor (re-steer)"
        with _connect(ws_connect, client) as ws:
            ws.send(json.dumps(_create(model, extra, max_output_tokens=200, input=_LONG_PROMPT)))
            pre = _read_until(ws, deadline, lambda o, _fs: o.get("type") in _START_STOP)
            rid = next((_resp_id(f) for f in pre if f.get("type") == "response.created"), None)
            if not isinstance(rid, str):
                add(name, False, f"no response.created (types={_types_of(pre)})")
                return
            ws.send(json.dumps({"type": "response.steer", "previous_response_id": rid, "input": "Wrap up briefly."}))
            fb = _read_until(
                ws, deadline,
                lambda o, _fs: o.get("type") in ("response.steer.failed", "error", "response.failed")
                or (o.get("type") == "response.created" and _resp_id(o) != rid),
                max_frames=2000,
            )
            succ = next((f for f in fb if f.get("type") == "response.created" and _resp_id(f) != rid), None)
            fbad = next((f for f in fb if f.get("type") in ("response.steer.failed", "error", "response.failed")), None)
            if succ is None:
                bad = json.dumps(fbad)[:90] if fbad else None
                add(name, False, f"no successor created (types={_types_of(fb)} bad={bad})")
                return
            sid = _resp_id(succ)
            ws.send(json.dumps({"type": "response.steer", "previous_response_id": sid, "input": "One more tweak."}))
            frames = _read_until(
                ws, deadline,
                lambda o, _fs: o.get("type") in ("response.steer.failed", "error")
                or (o.get("type") == "response.steer.accepted"
                    and (o.get("steer") or {}).get("previous_response_id") == sid),
            )
            acc = next((f for f in frames if f.get("type") == "response.steer.accepted"
                        and (f.get("steer") or {}).get("previous_response_id") == sid), None)
            if acc is None:
                last = frames[-1] if frames else {}
                last_s = json.dumps(last)[:110]
                add(name, False, f"no steer.accepted for successor; types={_types_of(frames)} last={last_s}")
                return
            steer = acc.get("steer") if isinstance(acc.get("steer"), dict) else {}
            id_ok = isinstance(steer.get("id"), str) and bool(_STEER_ID_RE.match(steer.get("id") or ""))
            prev_ok = steer.get("previous_response_id") == sid
            add(name, id_ok and prev_ok, f"successor_id={sid!r} prev_ok={prev_ok} id_ok={id_ok}")

    def steer_already_completed(deadline: float) -> None:
        name = "steer.error: response_already_completed"
        with _connect(ws_connect, client) as ws:
            ws.send(json.dumps(_create(model, extra, max_output_tokens=16)))
            pre = _read_until(ws, deadline, lambda o, _fs: o.get("type") in _TERMINAL + ("error",))
            term = next((f for f in pre if f.get("type") in _TERMINAL), None)
            if term is None or term.get("type") != "response.completed":
                ttype = term.get("type") if isinstance(term, dict) else None
                add(name, False, f"setup: terminal={ttype!r} (want response.completed)")
                return
            rid = _resp_id(term)
            ws.send(json.dumps({"type": "response.steer", "previous_response_id": rid, "input": "one more"}))
            frames = _read_until(ws, deadline, lambda o, _fs: o.get("type") in ("response.steer.failed", "error"))
            fail = next((f for f in frames if f.get("type") == "response.steer.failed"), None)
            if fail is None:
                add(name, False, f"no steer.failed (types={_types_of(frames)})")
                return
            err = _err_of(fail)
            steer = fail.get("steer") if isinstance(fail.get("steer"), dict) else {}
            prev_ok = steer.get("previous_response_id") == rid
            ok = (err.get("type") == "invalid_request_error" and err.get("code") == "response_already_completed"
                  and prev_ok)
            add(name, ok, f"code={err.get('code')!r} type={err.get('type')!r} prev_ok={prev_ok}")

    def steer_pending(deadline: float) -> None:
        name = "steer.pending: waiting for required input"
        tools = [
            {
                "type": "function",
                "name": "get_project_status",
                "description": "Get the status of a project",
                "parameters": {"type": "object", "properties": {"project": {"type": "string"}},
                               "required": ["project"]},
            }
        ]
        with _connect(ws_connect, client) as ws:
            prompt = "Use the get_project_status tool to check the task tracker project."
            ev = _create(model, extra, tools=tools, tool_choice={"type": "function", "name": "get_project_status"},
                         input=prompt, max_output_tokens=64)
            ws.send(json.dumps(ev))
            pre = _read_until(ws, deadline, lambda o, _fs: o.get("type") in _START_STOP)
            rid = next((_resp_id(f) for f in pre if f.get("type") == "response.created"), None)
            if not isinstance(rid, str):
                add(name, False, f"no response.created (types={_types_of(pre)})")
                return
            ws.send(json.dumps({"type": "response.steer", "previous_response_id": rid,
                                "input": "Also mention the deadline."}))
            frames = _read_until(
                ws, deadline,
                lambda o, _fs: o.get("type") in ("response.steer.pending", "response.steer.failed",
                                                 "error", "response.failed"),
                max_frames=2000,
            )
            pend = next((f for f in frames if f.get("type") == "response.steer.pending"), None)
            fail = next((f for f in frames
                         if f.get("type") in ("response.steer.failed", "error", "response.failed")), None)
            term = next((f for f in frames if f.get("type") in _TERMINAL and _resp_id(f) == rid), None)
            resp = term.get("response") if isinstance(term, dict) and isinstance(term.get("response"), dict) else {}
            out = resp.get("output") if isinstance(resp.get("output"), list) else []
            fcalls = [it for it in out if isinstance(it, dict) and it.get("type") == "function_call"]
            pend_steer = pend.get("steer") if isinstance(pend, dict) and isinstance(pend.get("steer"), dict) else {}
            req = pend.get("required_input") if isinstance(pend, dict) else None
            if not isinstance(req, list):
                req = []
            req_ok = bool(req) and all(
                isinstance(it, dict) and it.get("type") == "function_call_output"
                and isinstance(it.get("call_id"), str) and isinstance(it.get("name"), str)
                for it in req
            )
            fc_by_call = {it.get("call_id"): it.get("name") for it in fcalls}
            match_ok = req_ok and all(fc_by_call.get(it.get("call_id")) == it.get("name") for it in req)
            order_ok = False
            if pend is not None and term is not None:
                pos = {id(f): i for i, f in enumerate(frames)}
                order_ok = pos.get(id(term), 10**9) < pos.get(id(pend), -1)
            id_ok = isinstance(pend_steer.get("id"), str) and bool(_STEER_ID_RE.match(pend_steer.get("id") or ""))
            pend_ok = (
                pend is not None
                and pend.get("reason") == "waiting_for_required_input"
                and pend_steer.get("previous_response_id") == rid
                and id_ok
                and match_ok
                and order_ok
                and fail is None
            )
            if not fcalls:
                add(name, False, f"pending_ok={pend_ok} then no function_call in target output "
                                 f"(term={'yes' if term else 'no'} types={_types_of(frames)})")
                return
            cont_input = [
                {"type": "function_call_output", "call_id": it.get("call_id"), "output": "{\"status\":\"on track\"}"}
                for it in fcalls
            ]
            cont = _create(model, extra, tools=tools, input=cont_input, max_output_tokens=64)
            cont["previous_response_id"] = rid
            ws.send(json.dumps(cont))
            fc2 = _read_until(
                ws, deadline,
                lambda o, _fs: o.get("type") in ("response.steer.failed", "error", "response.failed")
                or (o.get("type") == "response.created" and _resp_id(o) != rid),
            )
            cre = next((f for f in fc2 if f.get("type") == "response.created" and _resp_id(f) != rid), None)
            f2bad = next((f for f in fc2
                          if f.get("type") in ("response.steer.failed", "error", "response.failed")), None)
            if cre is None:
                add(name, False, f"pending_ok={pend_ok}; continuation: no response.created "
                                 f"(types={_types_of(fc2)} bad={json.dumps(f2bad)[:90] if f2bad else None})")
                return
            sid = _resp_id(cre)
            fc3 = _read_until(
                ws, deadline,
                lambda o, _fs: o.get("type") in ("response.steer.failed", "error")
                or (o.get("type") in _TERMINAL and _resp_id(o) == sid),
                max_frames=2000,
            )
            cterm = next((f for f in fc3 if f.get("type") in _TERMINAL and _resp_id(f) == sid), None)
            cbad = next((f for f in fc3 if f.get("type") in ("response.steer.failed", "error")), None)
            cont_ok = cterm is not None and cterm.get("type") == "response.completed" and cbad is None and f2bad is None
            ok = pend_ok and cont_ok
            p_reason = pend.get("reason") if isinstance(pend, dict) else None
            ctype = cterm.get("type") if isinstance(cterm, dict) else None
            add(name, ok, f"pending_ok={pend_ok} reason={p_reason!r} "
                          f"req={len(req)} fc={len(fcalls)} cont_term={ctype} "
                          f"steer_failed={fail is not None or f2bad is not None or cbad is not None}")

    def inject_created(deadline: float) -> None:
        name = "inject.created shape"
        sent = [{"type": "function_call_output", "call_id": "call_x", "output": "ok"}]
        with _connect(ws_connect, client, beta=True) as ws:
            ws.send(json.dumps(_create(model, extra, max_output_tokens=200, input=_LONG_PROMPT)))
            pre = _read_until(ws, deadline, lambda o, _fs: o.get("type") in _START_STOP)
            rid = next((_resp_id(f) for f in pre if f.get("type") == "response.created"), None)
            if not isinstance(rid, str):
                add(name, False, f"no response.created (types={_types_of(pre)})")
                return
            ws.send(json.dumps({"type": "response.inject", "response_id": rid, "input": sent}))
            frames = _read_until(ws, deadline,
                                 lambda o, _fs: o.get("type") in ("response.inject.created", "response.inject.failed",
                                                                 "error"))
            cre = next((f for f in frames if f.get("type") == "response.inject.created"), None)
            fail = next((f for f in frames if f.get("type") in ("response.inject.failed", "error")), None)
            cre_rid = cre.get("response_id") if isinstance(cre, dict) else None
            cre_seq = cre.get("sequence_number") if isinstance(cre, dict) else None
            ok = cre is not None and cre_rid == rid and _is_number(cre_seq)
            detail = f"response_id={cre_rid!r} seq_ok={_is_number(cre_seq)}"
            if fail is not None:
                detail += f" failed={json.dumps(fail)[:120]}"
            add(name, ok, detail)

    def inject_not_found(deadline: float) -> None:
        name = "inject.error: response_not_found"
        sent = [{"type": "function_call_output", "call_id": "call_x", "output": "ok"}]
        with _connect(ws_connect, client, beta=True) as ws:
            ws.send(json.dumps({"type": "response.inject", "response_id": "resp_missing", "input": sent}))
            frames = _read_until(ws, deadline,
                                 lambda o, _fs: o.get("type") in ("response.inject.failed", "response.inject.created",
                                                                 "error"))
            fail = next((f for f in frames if f.get("type") == "response.inject.failed"), None)
            if fail is None:
                add(name, False, f"no inject.failed (types={_types_of(frames)})")
                return
            err = _err_of(fail)
            rid_ok = fail.get("response_id") == "resp_missing"
            echo_ok = fail.get("input") == sent
            ok = rid_ok and echo_ok and err.get("code") == "response_not_found"
            add(name, ok, f"response_id={fail.get('response_id')!r} input_echo={echo_ok} code={err.get('code')!r}")

    def inject_already_completed(deadline: float) -> None:
        name = "inject.error: response_already_completed"
        sent = [{"type": "function_call_output", "call_id": "call_x", "output": "ok"}]
        with _connect(ws_connect, client, beta=True) as ws:
            ws.send(json.dumps(_create(model, extra, max_output_tokens=16)))
            pre = _read_until(ws, deadline, lambda o, _fs: o.get("type") in _TERMINAL + ("error",))
            term = next((f for f in pre if f.get("type") in _TERMINAL), None)
            if term is None or term.get("type") != "response.completed":
                ttype = term.get("type") if isinstance(term, dict) else None
                add(name, False, f"setup: terminal={ttype!r} (want response.completed)")
                return
            rid = _resp_id(term)
            ws.send(json.dumps({"type": "response.inject", "response_id": rid, "input": sent}))
            frames = _read_until(ws, deadline,
                                 lambda o, _fs: o.get("type") in ("response.inject.failed", "response.inject.created",
                                                                 "error"))
            fail = next((f for f in frames if f.get("type") == "response.inject.failed"), None)
            if fail is None:
                add(name, False, f"no inject.failed (types={_types_of(frames)})")
                return
            err = _err_of(fail)
            rid_ok = fail.get("response_id") == rid
            echo_ok = fail.get("input") == sent
            ok = rid_ok and echo_ok and err.get("code") == "response_already_completed"
            add(name, ok, f"response_id={fail.get('response_id')!r} code={err.get('code')!r} input_echo={echo_ok}")

    def inject_schema_close(deadline: float) -> None:
        name = "inject schema violation closes connection"
        from websockets.exceptions import ConnectionClosed

        with _connect(ws_connect, client, beta=True) as ws:
            ws.send(json.dumps({"type": "response.inject", "input": []}))
            frames = _read_until(ws, deadline, lambda o, _fs: o.get("type") == "error")
            err = next((f for f in frames if f.get("type") == "error"), None)
            eobj = _err_of(err)
            shape_ok = (
                err is not None
                and err.get("status") == 400
                and isinstance(eobj.get("type"), str) and bool(eobj.get("type"))
                and isinstance(eobj.get("message"), str) and bool(eobj.get("message"))
            )
            closed = False
            close_budget = time.monotonic() + 10.0
            while True:
                remain = min(deadline, close_budget) - time.monotonic()
                if remain <= 0:
                    break
                try:
                    ws.recv(timeout=min(2.0, remain))
                except ConnectionClosed:
                    closed = True
                    break
                except TimeoutError:
                    continue
            add(name, shape_ok and closed,
                f"status={err.get('status') if isinstance(err, dict) else None} "
                f"error={json.dumps(eobj)[:120]} closed={closed}")

    def store_false_continuation(deadline: float) -> None:
        name = "store=false continuation over WS (connection-local cache)"
        with _connect(ws_connect, client) as ws:
            ws.send(json.dumps(_create(model, extra, store=False, max_output_tokens=16)))
            f1 = _read_until(
                ws, deadline,
                lambda o, _fs: o.get("type") in _TERMINAL + ("error",),
                max_frames=2000,
            )
            created1 = next((f for f in f1 if f.get("type") == "response.created"), None)
            term1 = next((f for f in f1 if f.get("type") in _TERMINAL), None)
            err1 = next((f for f in f1 if f.get("type") == "error"), None)
            rid1 = _resp_id(created1)
            if (not isinstance(rid1, str) or err1 is not None or term1 is None
                    or term1.get("type") != "response.completed"):
                ttype = term1.get("type") if isinstance(term1, dict) else None
                err_s = json.dumps(_err_of(err1))[:110] if err1 is not None else None
                add(name, False, f"setup: id={rid1!r} terminal={ttype!r} error={err_s}")
                return
            cont = _create(model, extra, max_output_tokens=16)
            cont["previous_response_id"] = rid1
            ws.send(json.dumps(cont))
            f2 = _read_until(
                ws, deadline,
                lambda o, _fs: o.get("type") in _TERMINAL + ("error",),
                max_frames=2000,
            )
            created2 = next((f for f in f2 if f.get("type") == "response.created"), None)
            term2 = next((f for f in f2 if f.get("type") in _TERMINAL), None)
            err2 = next((f for f in f2 if f.get("type") == "error"), None)
            rid2 = _resp_id(created2)
            new_created = isinstance(rid2, str) and rid2 != rid1
            cont_term = term2.get("type") if isinstance(term2, dict) else None
            ok = new_created and cont_term == "response.completed" and err2 is None
            detail = (f"first={rid1!r} second={rid2!r} new_created={new_created} "
                      f"cont_terminal={cont_term} error={err2 is not None}")
            if err2 is not None:
                detail += f" err={json.dumps(_err_of(err2))[:110]}"
            add(name, ok, detail)

    def store_false_outside_connection(deadline: float) -> None:
        name = "store=false continuation rejected outside the connection"
        with _connect(ws_connect, client) as ws:
            ws.send(json.dumps(_create(model, extra, store=False, max_output_tokens=16)))
            f1 = _read_until(
                ws, deadline,
                lambda o, _fs: o.get("type") in _TERMINAL + ("error",),
                max_frames=2000,
            )
            created1 = next((f for f in f1 if f.get("type") == "response.created"), None)
            term1 = next((f for f in f1 if f.get("type") in _TERMINAL), None)
            rid = _resp_id(created1)
            if (not isinstance(rid, str) or term1 is None
                    or term1.get("type") != "response.completed"):
                ttype = term1.get("type") if isinstance(term1, dict) else None
                add(name, False, f"setup: id={rid!r} terminal={ttype!r} (types={_types_of(f1)})")
                return
            # a) a different WS connection must not see the connection-local cache
            aerr: dict[str, Any] | None = None
            a_started = False
            with _connect(ws_connect, client) as ws2:
                ev_a = _create(model, extra, max_output_tokens=16)
                ev_a["previous_response_id"] = rid
                ws2.send(json.dumps(ev_a))
                fa = _read_until(
                    ws2, min(deadline, time.monotonic() + 15.0),
                    lambda o, _fs: o.get("type") in ("error", "response.created", "response.failed"),
                )
                aerr = next((f for f in fa if f.get("type") == "error"), None)
                a_started = any(f.get("type") == "response.created" for f in fa)
            a_ok = (
                aerr is not None
                and aerr.get("status") == 400
                and _err_of(aerr).get("code") == "previous_response_not_found"
                and not a_started
            )
            # b) plain HTTP POST must also be rejected; the flat HTTP envelope keeps its
            #    numeric status + invalid_request_error type and the same string code
            body_b = {"model": model, "input": "x", "previous_response_id": rid,
                      "max_output_tokens": 16, **extra}
            code_b, data_b = client.post_json("/v1/responses", body_b)
            eb = _err_of(data_b)
            b_ok = (code_b == 400 and eb.get("type") == "invalid_request_error"
                    and eb.get("code") == "previous_response_not_found")
            # c) forging the internal WS marker must not unlock the cache
            body_c = {"model": model, "input": "x", "previous_response_id": rid,
                      "max_output_tokens": 16, "__oai_ws_local": "x", **extra}
            code_c, data_c = client.post_json("/v1/responses", body_c)
            ec = _err_of(data_c)
            c_ok = (code_c == 400 and ec.get("type") == "invalid_request_error"
                    and ec.get("code") == "previous_response_not_found")
            a_st = aerr.get("status") if isinstance(aerr, dict) else None
            msg_b = str(eb.get("message"))[:40]
            msg_c = str(ec.get("message"))[:40]
            detail = (f"other_ws: status={a_st} code={_err_of(aerr).get('code')!r} start={a_started}; "
                      f"http: {code_b} code={eb.get('code')!r} type={eb.get('type')!r} msg={msg_b!r}; "
                      f"forge: {code_c} code={ec.get('code')!r} type={ec.get('type')!r} msg={msg_c!r}")
            add(name, a_ok and b_ok and c_ok, detail)

    def steer_successor_store_false(deadline: float) -> None:
        name = "steer successor carries store=false target"
        with _connect(ws_connect, client) as ws:
            ws.send(json.dumps(_create(model, extra, store=False, max_output_tokens=200, input=_LONG_PROMPT)))
            pre = _read_until(ws, deadline, lambda o, _fs: o.get("type") in _START_STOP)
            rid = next((_resp_id(f) for f in pre if f.get("type") == "response.created"), None)
            if not isinstance(rid, str):
                add(name, False, f"no response.created (types={_types_of(pre)})")
                return
            ws.send(json.dumps({"type": "response.steer", "previous_response_id": rid, "input": "Wrap up briefly."}))
            fa = _read_until(
                ws, deadline,
                lambda o, _fs: o.get("type") in ("response.steer.failed", "error")
                or (o.get("type") in _TERMINAL and _resp_id(o) == rid),
                max_frames=2000,
            )
            term = next((f for f in fa if f.get("type") in _TERMINAL and _resp_id(f) == rid), None)
            sbad = next((f for f in fa if f.get("type") in ("response.steer.failed", "error")), None)
            if sbad is not None:
                add(name, False, f"steer failed: {json.dumps(sbad)[:140]}")
                return
            if term is None:
                add(name, False, f"no target terminal (types={_types_of(fa)})")
                return
            fb = _read_until(
                ws, deadline,
                lambda o, _fs: o.get("type") in ("response.steer.failed", "error", "response.failed")
                or (o.get("type") == "response.created" and _resp_id(o) != rid),
                max_frames=2000,
            )
            succ = next((f for f in fb if f.get("type") == "response.created" and _resp_id(f) != rid), None)
            fbad = next((f for f in fb
                         if f.get("type") in ("response.steer.failed", "error", "response.failed")), None)
            if succ is None:
                bad = json.dumps(fbad)[:100] if fbad else None
                add(name, False, f"no successor created (types={_types_of(fb)} bad={bad})")
                return
            sid = _resp_id(succ)
            fc = _read_until(
                ws, deadline,
                lambda o, _fs: o.get("type") in ("response.steer.failed", "error")
                or (o.get("type") in _TERMINAL and _resp_id(o) == sid),
                max_frames=2000,
            )
            sterm = next((f for f in fc if f.get("type") in _TERMINAL and _resp_id(f) == sid), None)
            cbad = next((f for f in fc if f.get("type") in ("response.steer.failed", "error")), None)
            sterm_ok = sterm is not None and sterm.get("type") == "response.completed"
            ok = sterm_ok and fbad is None and cbad is None
            add(name, ok, f"target_terminal={term.get('type')} successor_id={sid!r} "
                          f"successor_completed={sterm_ok} steer_failed={fbad is not None or cbad is not None}")

    def store_false_failed_same_lane_eviction(deadline: float) -> None:
        name = "store=false: failed same-lane continuation evicts the parent (cross-lane fork keeps it)"
        # (b) pick a continuation field that reliably yields a 4xx error frame
        candidates: tuple[tuple[str, dict[str, Any]], ...] = (
            ("temperature:string", {"temperature": "hot"}),
            ("max_output_tokens:string", {"max_output_tokens": "tiny"}),
            ("input:number", {"input": 123}),
            ("store:string", {"store": "yes"}),
        )
        vehicle: str | None = None
        vehicle_ev: dict[str, Any] = {}
        probe_obs: list[str] = []
        with _connect(ws_connect, client) as ws_p:
            for label, bad in candidates:
                ev_p = _create(model, extra, store=False, max_output_tokens=16)
                ev_p.update(bad)
                ws_p.send(json.dumps(ev_p))
                pf = _read_until(
                    ws_p, min(deadline, time.monotonic() + 10.0),
                    lambda o, _fs: o.get("type") == "error" or o.get("type") in _TERMINAL,
                )
                perr = next((f for f in pf if f.get("type") == "error"), None)
                pst = perr.get("status") if isinstance(perr, dict) else None
                if perr is not None and isinstance(pst, int) and 400 <= pst < 500:
                    vehicle, vehicle_ev = label, bad
                    break
                obs = "accepted" if any(f.get("type") in _TERMINAL for f in pf) else "no 4xx/terminal"
                probe_obs.append(f"{label}->{obs}")
        if vehicle is None:
            probes_s = "; ".join(probe_obs)
            add(name, False, f"no 4xx vehicle field; probes: {probes_s}")
            return

        with _connect(ws_connect, client) as ws:
            # (a) long store=false target X on the default lane; wait for completion
            ws.send(json.dumps(_create(model, extra, store=False, max_output_tokens=128, input=_LONG_PROMPT)))
            f0 = _read_until(
                ws, deadline,
                lambda o, _fs: o.get("type") in _TERMINAL + ("error",),
                max_frames=2000,
            )
            created0 = next((f for f in f0 if f.get("type") == "response.created"), None)
            term0 = next((f for f in f0 if f.get("type") in _TERMINAL), None)
            err0 = next((f for f in f0 if f.get("type") == "error"), None)
            x_id = _resp_id(created0)
            if (not isinstance(x_id, str) or err0 is not None or term0 is None
                    or term0.get("type") != "response.completed"):
                ttype0 = term0.get("type") if isinstance(term0, dict) else None
                add(name, False, f"setup: X={x_id!r} terminal={ttype0!r} vehicle={vehicle}")
                return
            # (c) a failing cross-lane fork must keep the shared parent alive
            fork_ev = _create(model, extra, store=False, max_output_tokens=16, stream_id="fork")
            fork_ev.update(vehicle_ev)
            fork_ev["previous_response_id"] = x_id
            ws.send(json.dumps(fork_ev))
            fc = _read_until(
                ws, deadline,
                lambda o, _fs: o.get("type") == "error"
                or ((o.get("type") == "response.created" or o.get("type") in _TERMINAL)
                    and o.get("stream_id") == "fork"),
                max_frames=2000,
            )
            ferr = next((f for f in fc if f.get("type") == "error"), None)
            fst = ferr.get("status") if isinstance(ferr, dict) else None
            f_code = _err_of(ferr).get("code")
            c_ok = (ferr is not None and ferr.get("stream_id") == "fork"
                    and isinstance(fst, int) and 400 <= fst < 500
                    and f_code != "previous_response_not_found")
            if not c_ok:
                fsid = ferr.get("stream_id") if isinstance(ferr, dict) else None
                add(name, False, f"fork: error={ferr is not None} status={fst!r} code={f_code!r} "
                                 f"stream_id={fsid!r} vehicle={vehicle}")
                return
            # (d) the source lane must still resolve X
            cont1 = _create(model, extra, store=False, max_output_tokens=16)
            cont1["previous_response_id"] = x_id
            ws.send(json.dumps(cont1))
            fd = _read_until(
                ws, deadline,
                lambda o, _fs: o.get("type") in _TERMINAL + ("error",),
                max_frames=2000,
            )
            created1 = next((f for f in fd if f.get("type") == "response.created"), None)
            term1 = next((f for f in fd if f.get("type") in _TERMINAL), None)
            err1 = next((f for f in fd if f.get("type") == "error"), None)
            x2_id = _resp_id(created1)
            d_ok = (isinstance(x2_id, str) and x2_id != x_id and err1 is None
                    and term1 is not None and term1.get("type") == "response.completed")
            if not d_ok:
                t1 = term1.get("type") if isinstance(term1, dict) else None
                add(name, False, f"fork ok; X2={x2_id!r} terminal={t1!r} err={err1 is not None}")
                return
            # (e) a failing same-lane continuation from X2 must evict it
            bad2 = _create(model, extra, store=False, max_output_tokens=16)
            bad2.update(vehicle_ev)
            bad2["previous_response_id"] = x2_id
            ws.send(json.dumps(bad2))
            fe = _read_until(
                ws, deadline,
                lambda o, _fs: o.get("type") == "error" or o.get("type") == "response.created",
                max_frames=2000,
            )
            eerr = next((f for f in fe if f.get("type") == "error"), None)
            est = eerr.get("status") if isinstance(eerr, dict) else None
            e_code = _err_of(eerr).get("code")
            e_ok = (eerr is not None and isinstance(est, int) and 400 <= est < 500
                    and e_code != "previous_response_not_found")
            if not e_ok:
                estarted = any(f.get("type") == "response.created" for f in fe)
                add(name, False, f"X={x_id!r} X2={x2_id!r} same-lane fail: error={eerr is not None} "
                                 f"status={est!r} code={e_code!r} started={estarted}")
                return
            # (f) after the failed same-lane continuation X2 must be gone
            cont2 = _create(model, extra, store=False, max_output_tokens=16)
            cont2["previous_response_id"] = x2_id
            ws.send(json.dumps(cont2))
            ff = _read_until(
                ws, deadline,
                lambda o, _fs: o.get("type") in ("error", "response.created", "response.failed"),
                max_frames=2000,
            )
            ferr2 = next((f for f in ff if f.get("type") == "error"), None)
            f_created = next((f for f in ff if f.get("type") == "response.created"), None)
            fst2 = ferr2.get("status") if isinstance(ferr2, dict) else None
            f2_code = _err_of(ferr2).get("code")
            f_ok = (ferr2 is not None and fst2 == 400 and f2_code == "previous_response_not_found"
                    and f_created is None)
            ok = c_ok and d_ok and e_ok and f_ok
            add(name, ok, f"vehicle={vehicle}; X={x_id!r} X2={x2_id!r} fork_err={fst!r} same_err={est!r} "
                          f"after_evict: status={fst2!r} code={f2_code!r} created={f_created is not None}")

    def gen_false_warmup_shape(deadline: float) -> None:
        name = "generate=false warmup: created+completed, empty output, chainable (store=false)"
        with _connect(ws_connect, client) as ws:
            # a) warmup returns [created, completed] only: no in_progress, no deltas, empty output
            ws.send(json.dumps(_create(model, extra, store=False, generate=False, input=_PROMPT)))
            f0 = _read_until(ws, deadline,
                             lambda o, _fs: o.get("type") in _TERMINAL + ("error",),
                             max_frames=2000)
            created0 = next((f for f in f0 if f.get("type") == "response.created"), None)
            term0 = next((f for f in f0 if f.get("type") in _TERMINAL), None)
            err0 = next((f for f in f0 if f.get("type") == "error"), None)
            in_progress0 = [f for f in f0 if f.get("type") == "response.in_progress"]
            wid = _resp_id(created0)
            deltas0 = sum(1 for f in f0 if "response.output_text.delta" in str(f.get("type")))
            resp0 = term0.get("response") if isinstance(term0, dict) and isinstance(term0.get("response"), dict) else {}
            out0 = resp0.get("output")
            out0_len = len(out0) if isinstance(out0, list) else None
            t0 = term0.get("type") if isinstance(term0, dict) else None
            a_ok = (err0 is None and isinstance(wid, str) and t0 == "response.completed"
                    and deltas0 == 0 and out0 == [] and not in_progress0
                    and ("status" not in resp0 or resp0.get("status") == "completed"))
            if not a_ok:
                add(name, False, f"warmup: id={wid!r} terminal={t0!r} err={err0 is not None} "
                                 f"deltas={deltas0} out={out0_len} in_progress={len(in_progress0)}")
                return
            # b) the warmup id chains on the same connection
            cont = _create(model, extra, store=False, input=_PROMPT)
            cont["previous_response_id"] = wid
            ws.send(json.dumps(cont))
            f1 = _read_until(ws, deadline,
                             lambda o, _fs: o.get("type") in _TERMINAL + ("error",),
                             max_frames=2000)
            created1 = next((f for f in f1 if f.get("type") == "response.created"), None)
            term1 = next((f for f in f1 if f.get("type") in _TERMINAL), None)
            err1 = next((f for f in f1 if f.get("type") == "error"), None)
            cid = _resp_id(created1)
            deltas1 = sum(1 for f in f1 if "response.output_text.delta" in str(f.get("type")))
            t1 = term1.get("type") if isinstance(term1, dict) else None
            b_ok = (isinstance(cid, str) and cid != wid and err1 is None and t1 == "response.completed")
            add(name, a_ok and b_ok, f"warmup_id={wid!r} chain_id={cid!r} deltas={deltas0}/{deltas1} "
                                     f"out_len={out0_len} terminals={t0!r}/{t1!r}")

    def gen_false_warmup_steers(deadline: float) -> None:
        name = "generate=false warmup does not consume queued steers"
        with _connect(ws_connect, client) as ws:
            # a) long store=false target A on the default lane
            ws.send(json.dumps(_create(model, extra, store=False, max_output_tokens=128, input=_LONG_PROMPT)))
            pre = _read_until(ws, deadline, lambda o, _fs: o.get("type") in _START_STOP, max_frames=2000)
            a_id = _resp_id(next((f for f in pre if f.get("type") == "response.created"), None))
            if not isinstance(a_id, str):
                add(name, False, f"no response.created (types={_types_of(pre)})")
                return
            # b) queue a steer on A and wait for the accepted ack
            ws.send(json.dumps({"type": "response.steer", "previous_response_id": a_id,
                                "input": "Wrap up briefly."}))
            sa = _read_until(ws, deadline, lambda o, _fs: o.get("type") in _STEER_STOP, max_frames=2000)
            acc = next((f for f in sa if f.get("type") == "response.steer.accepted"), None)
            sfail0 = next((f for f in sa if f.get("type") in ("response.steer.failed", "error")), None)
            if acc is None or sfail0 is not None:
                st0 = sfail0.get("type") if isinstance(sfail0, dict) else None
                add(name, False, f"steer not accepted: found={st0!r} types={_types_of(sa)}")
                return
            # c) fire the warmup continuation; it must not swallow the queued steer
            warm = _create(model, extra, store=False, generate=False, input=_PROMPT)
            warm["previous_response_id"] = a_id
            ws.send(json.dumps(warm))
            # d) a response other than A must stream deltas (warmup emits none); the
            #    owner is the latest created response, since delta frames carry no response id
            state: dict[str, Any] = {"cur": None, "hit": False, "ndelta": 0,
                                     "sfail": None, "err": None, "ids": []}

            def _pred(o: dict[str, Any], _fs: list[dict[str, Any]]) -> bool:
                t = str(o.get("type"))
                if t == "response.created":
                    state["ids"].append(_resp_id(o))
                    state["cur"] = _resp_id(o)
                elif t in _TERMINAL:
                    state["cur"] = None
                elif t == "response.steer.failed":
                    state["sfail"] = o
                elif t == "error":
                    state["err"] = o
                if "response.output_text.delta" in t:
                    state["ndelta"] += 1
                    if state["cur"] is not None and state["cur"] != a_id:
                        state["hit"] = True
                return bool(state["hit"] or state["sfail"] is not None or state["err"] is not None)

            frames = _read_until(ws, deadline, _pred, max_frames=2000)
            ok = bool(state["hit"]) and state["sfail"] is None and state["err"] is None
            add(name, ok, f"A={a_id!r} seen={state['ids']} deltas={state['ndelta']} hit={state['hit']} "
                          f"steer_failed={state['sfail'] is not None} err={state['err'] is not None}; "
                          f"tail={_types_of(frames[-8:])}")

    guarded("connect/create default lane", default_lane)
    guarded("stream_id echo (named lane)", named_lane_echo)
    guarded("stream_id validation", stream_id_validation)
    guarded("error shape: unknown event type", unknown_event_type)
    guarded("error shape: invalid json", invalid_json)
    guarded("previous_response_not_found", previous_response_missing)
    guarded("lane FIFO", lane_fifo)
    guarded("two lanes concurrent", two_lanes)
    guarded("named lane limit (32)", named_lane_limit)
    guarded("steer.accepted shape (default + named lane)", steer_accepted_shape)
    guarded("steer.error: invalid_input cases", steer_invalid_input)
    guarded("steer.error: response_not_found", steer_response_not_found)
    guarded("steer.error: steering_not_supported (conversation)", steer_not_supported_conversation)
    guarded("steer.error: steering_not_supported (compaction)", steer_not_supported_compaction)
    guarded("steer.error: too_many_pending_steers (33x)", steer_too_many)
    guarded("steer interrupt -> successor", steer_interrupt_successor)
    guarded("steer.accepted on successor (re-steer)", steer_successor_resteer)
    guarded("steer.error: response_already_completed", steer_already_completed)
    guarded("steer.pending: waiting for required input", steer_pending)
    guarded("inject.created shape", inject_created)
    guarded("inject.error: response_not_found", inject_not_found)
    guarded("inject.error: response_already_completed", inject_already_completed)
    guarded("inject schema violation closes connection", inject_schema_close)
    guarded("store=false continuation over WS (connection-local cache)", store_false_continuation)
    guarded("store=false continuation rejected outside the connection", store_false_outside_connection)
    guarded("steer successor carries store=false target", steer_successor_store_false)
    guarded("store=false: failed same-lane continuation evicts the parent (cross-lane fork keeps it)",
            store_false_failed_same_lane_eviction)
    guarded("generate=false warmup: created+completed, empty output, chainable (store=false)",
            gen_false_warmup_shape)
    guarded("generate=false warmup does not consume queued steers", gen_false_warmup_steers)
    add_skips()
