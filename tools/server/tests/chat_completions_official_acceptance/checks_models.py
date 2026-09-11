"""OpenAI Models list/retrieve checks (Model schema)."""

from __future__ import annotations

from openai.types import Model

from .http_client import ChatHttpClient
from .report import Report


def run_models_checks(
    client: ChatHttpClient,
    report: Report,
    model: str,
) -> None:
    code, data = client.get_json("/v1/models")
    if code != 200 or not isinstance(data, dict):
        report.add("endpoint", "GET /v1/models", "FAIL", f"HTTP {code}")
        return

    rows = data.get("data")
    if not isinstance(rows, list) or not rows:
        report.add("endpoint", "GET /v1/models", "FAIL", f"empty data={data!r}"[:200])
        return

    errs = []
    found = False
    for row in rows:
        if not isinstance(row, dict):
            errs.append("non-object row")
            continue
        try:
            Model.model_validate(
                {
                    "id": row.get("id"),
                    "object": row.get("object", "model"),
                    "created": row.get("created", 0),
                    "owned_by": row.get("owned_by", "llamacpp"),
                }
            )
        except Exception as e:
            errs.append(str(e)[:120])
            continue
        if row.get("id") == model:
            found = True

    report.add(
        "endpoint",
        "GET /v1/models",
        "PASS" if not errs and found else "FAIL",
        f"n={len(rows)} found_model={found} errs={errs[:1]}",
    )
