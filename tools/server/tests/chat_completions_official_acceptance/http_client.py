"""Minimal HTTP client for Chat Completions acceptance (stdlib only)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class ChatHttpClient:
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


def parse_sse(raw: bytes) -> list[tuple[str | None, dict[str, Any]]]:
    """Parse SSE blocks; event type is None for OpenAI chat chunks (data-only)."""
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
            if data == "[DONE]":
                events.append(("[DONE]", {}))
            continue
        try:
            obj = json.loads(data)
        except Exception:
            continue
        events.append((ev or obj.get("object"), obj))
    return events


def choice_text(data: dict[str, Any]) -> str:
    """Extract assistant text from a ChatCompletion response dict."""
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("text"):
                parts.append(part["text"])
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text") or "")
        return "".join(parts)
    return ""


def choice_reasoning(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    msg = choices[0].get("message")
    if isinstance(msg, dict):
        rc = msg.get("reasoning_content")
        return rc if isinstance(rc, str) else ""
    return ""


def tool_calls(data: dict[str, Any]) -> list[dict[str, Any]]:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return []
    msg = choices[0].get("message")
    if not isinstance(msg, dict):
        return []
    tcs = msg.get("tool_calls")
    return [x for x in (tcs or []) if isinstance(x, dict)]
