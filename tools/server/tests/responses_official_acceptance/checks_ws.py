"""Official Responses WebSocket checks (transport shape: lanes, stream_id, errors).

Normative source: the official "WebSocket mode" guide (per-connection lane FIFO,
16 in-flight cap, 32 named-stream cap, 60-minute connection cap) and the
"WebSocket events" reference (stream_id echo rules, error envelope shape).
"""

from __future__ import annotations

import json
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
)


def _ws_url(base_url: str) -> str:
    """Derive the ws(s):// responses endpoint from the HTTP base URL."""
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{host}:{port}/v1/responses"


def _connect(ws_connect: Callable[..., Any], client: ResponsesHttpClient):
    return ws_connect(
        _ws_url(client.base_url),
        additional_headers={"Authorization": f"Bearer {client.api_key}"},
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

    guarded("connect/create default lane", default_lane)
    guarded("stream_id echo (named lane)", named_lane_echo)
    guarded("stream_id validation", stream_id_validation)
    guarded("error shape: unknown event type", unknown_event_type)
    guarded("error shape: invalid json", invalid_json)
    guarded("previous_response_not_found", previous_response_missing)
    guarded("lane FIFO", lane_fifo)
    guarded("two lanes concurrent", two_lanes)
    guarded("named lane limit (32)", named_lane_limit)
    add_skips()
