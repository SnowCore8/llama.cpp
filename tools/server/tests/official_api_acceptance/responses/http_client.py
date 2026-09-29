"""Minimal HTTP client for Responses acceptance (stdlib only)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def openai_base(base_url: str) -> str:
    """OpenAI Python SDK expects the versioned root (.../v1)."""
    u = base_url.rstrip("/")
    return u if u.endswith("/v1") else f"{u}/v1"


def as_dict(v: Any) -> dict[str, Any]:
    """Normalize a JSON value to a dict; anything else becomes {}."""
    return v if isinstance(v, dict) else {}


def as_list(v: Any) -> list[Any]:
    """Normalize a JSON value to a list; anything else becomes []."""
    return v if isinstance(v, list) else []


class ResponsesHttpClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> tuple[int, dict[str, str], bytes]:
        data = None if body is None else json.dumps(body).encode()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream" if stream else "application/json",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def post_json(self, path: str, body: dict[str, Any]) -> tuple[int, Any]:
        code, _, raw = self.request("POST", path, body)
        try:
            return code, json.loads(raw.decode() or "null")
        except Exception:
            return code, {"_raw": raw.decode(errors="replace")[:500]}

    def get_json(self, path: str) -> tuple[int, Any]:
        code, _, raw = self.request("GET", path)
        try:
            return code, json.loads(raw.decode() or "null")
        except Exception:
            return code, {"_raw": raw.decode(errors="replace")[:500]}

    def delete_json(self, path: str) -> tuple[int, Any]:
        code, _, raw = self.request("DELETE", path)
        try:
            return code, json.loads(raw.decode() or "null")
        except Exception:
            return code, {"_raw": raw.decode(errors="replace")[:500]}

    def open_stream(
        self,
        path: str,
        body: dict[str, Any] | None = None,
        timeout: float | None = None,
    ):
        """Open a streaming request and return the live response (caller must close()).

        Uses POST when a body is given, GET otherwise. Non-2xx raises HTTPError,
        like urlopen; callers that expect error statuses should catch it.
        """
        data = None if body is None else json.dumps(body).encode()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            method="POST" if body is not None else "GET",
            headers=headers,
        )
        return urllib.request.urlopen(req, timeout=timeout or self.timeout)


def parse_sse_block(block: str) -> tuple[str | None, dict[str, Any]] | None:
    """Parse one SSE event block; None when it carries no JSON data."""
    ev = None
    data = None
    for line in block.splitlines():
        if line.startswith("event:"):
            ev = line[6:].strip()
        elif line.startswith("data:"):
            data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    try:
        obj = json.loads(data)
    except Exception:
        return None
    return (ev or obj.get("type"), obj)


def parse_sse(raw: bytes) -> list[tuple[str | None, dict[str, Any]]]:
    events: list[tuple[str | None, dict[str, Any]]] = []
    for block in raw.decode(errors="replace").split("\n\n"):
        parsed = parse_sse_block(block)
        if parsed is not None:
            events.append(parsed)
    return events


def iter_sse(resp, max_events: int = 0, state: dict[str, Any] | None = None):
    """Yield (event, obj) incrementally from an open HTTP response.

    Reads in fixed-size chunks until EOF; stops early after max_events (>0).
    Pass the same state dict to continue later without losing read-ahead bytes;
    it also records EOF so a second call returns at once. The caller keeps
    ownership of the response.
    """
    if state is None:
        state = {}
    buf: str = state.pop("_buf", "")
    n = 0
    while True:
        while "\n\n" in buf:
            block, buf = buf.split("\n\n", 1)
            parsed = parse_sse_block(block)
            if parsed is not None:
                yield parsed
                n += 1
                if max_events and n >= max_events:
                    state["_buf"] = buf
                    return
        if state.get("_eof"):
            return
        chunk = resp.read(8192)
        if not chunk:
            state["_eof"] = True
            # flush a trailing block that arrived without the final blank line
            parsed = parse_sse_block(buf)
            if parsed is not None:
                yield parsed
            return
        buf += chunk.decode(errors="replace")


def output_text(d: dict[str, Any]) -> str:
    if isinstance(d.get("output_text"), str) and d["output_text"]:
        return d["output_text"]
    parts: list[str] = []
    for o in d.get("output") or []:
        if o.get("type") != "message":
            continue
        for c in o.get("content") or []:
            if c.get("text"):
                parts.append(c["text"])
    return "\n".join(parts)
