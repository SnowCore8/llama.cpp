"""Local discovery: GET /v1/tools and GET /props slot_save_path."""

from __future__ import annotations

from typing import Any

from .http_client import HttpClient
from .report import Report


def run_discovery_checks(client: HttpClient, report: Report) -> None:
    code, tools = client.get_json("/v1/tools")
    ok = code == 200 and isinstance(tools, list)
    report.add(
        "discovery",
        "GET /v1/tools",
        "PASS" if ok else "FAIL",
        f"HTTP {code} type={type(tools).__name__} n={len(tools) if isinstance(tools, list) else '-'}",
    )

    code, oai = client.get_json("/v1/tools?format=openai")
    data = oai.get("data") if isinstance(oai, dict) else None
    ok = (
        code == 200
        and isinstance(oai, dict)
        and oai.get("object") == "list"
        and isinstance(data, list)
    )
    report.add(
        "discovery",
        "GET /v1/tools?format=openai",
        "PASS" if ok else "FAIL",
        f"HTTP {code} object={oai.get('object') if isinstance(oai, dict) else None} "
        f"n={len(data) if isinstance(data, list) else '-'}",
    )

    code, props = client.get_json("/props")
    path = props.get("slot_save_path") if isinstance(props, dict) else None
    ok = code == 200 and isinstance(path, str) and len(path) > 0
    report.add(
        "discovery",
        "GET /props slot_save_path",
        "PASS" if ok else "FAIL",
        f"HTTP {code} slot_save_path={path!r}",
    )
