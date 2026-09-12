"""Minimal HTTP client for Responses acceptance (stdlib only)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


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


def parse_sse(raw: bytes) -> list[tuple[str | None, dict[str, Any]]]:
    events: list[tuple[str | None, dict[str, Any]]] = []
    for block in raw.decode(errors="replace").split("\n\n"):
        ev = None
        data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except Exception:
            continue
        events.append((ev or obj.get("type"), obj))
    return events


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
