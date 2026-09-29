"""OpenAI Models list checks (Model schema)."""

from __future__ import annotations

import json
from urllib.parse import quote

from openai.types import Model

from .http_client import ResponsesHttpClient
from .report import Report


def run_models_checks(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
) -> None:
    # OpenAI-shaped list
    code, data = client.get_json("/v1/models")
    if code != 200 or not isinstance(data, dict):
        report.add("endpoint", "GET /v1/models", "FAIL", f"HTTP {code}")
        report.add("endpoint", "GET /v1/models/{model}", "FAIL", f"list HTTP {code}")
        report.add("endpoint", "GET /v1/models/{model} (missing)", "FAIL", f"list HTTP {code}")
        return

    rows = data.get("data")
    if not isinstance(rows, list) or not rows:
        report.add("endpoint", "GET /v1/models", "FAIL", f"empty data={data!r}"[:200])
        report.add("endpoint", "GET /v1/models/{model}", "FAIL", "empty models list")
        report.add("endpoint", "GET /v1/models/{model} (missing)", "FAIL", "empty models list")
        return

    errs = []
    found = False
    for row in rows:
        if not isinstance(row, dict):
            errs.append("non-object row")
            continue
        # strict Model shape: object, non-empty string id and owned_by, integer created
        if row.get("object") != "model":
            errs.append(f"object={row.get('object')!r}")
        if not (isinstance(row.get("id"), str) and row["id"]):
            errs.append(f"id={row.get('id')!r}")
        if not isinstance(row.get("created"), int) or isinstance(row.get("created"), bool):
            errs.append(f"created={row.get('created')!r}")
        if not (isinstance(row.get("owned_by"), str) and row["owned_by"]):
            errs.append(f"owned_by={row.get('owned_by')!r}")
        # shutdown_date is optional: absent or null ok, non-empty string ok
        shutdown_date = row.get("shutdown_date")
        if not (shutdown_date is None or (isinstance(shutdown_date, str) and shutdown_date)):
            errs.append(f"shutdown_date={shutdown_date!r}")
        try:
            Model.model_validate(
                {
                    "id": row.get("id"),
                    "object": row.get("object", "model"),
                    "created": row.get("created", 0),
                    "owned_by": row.get("owned_by"),
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

    # retrieve by id (official Models.retrieve) returns the Model shape
    code_one, one = client.get_json(f"/v1/models/{quote(model, safe='')}")
    one_d = one if isinstance(one, dict) else {}
    one_sd = one_d.get("shutdown_date")
    one_ok = (
        code_one == 200
        and one_d.get("id") == model
        and one_d.get("object") == "model"
        and isinstance(one_d.get("created"), int)
        and isinstance(one_d.get("owned_by"), str)
        and bool(one_d.get("owned_by"))
        and (one_sd is None or (isinstance(one_sd, str) and bool(one_sd)))
    )
    report.add(
        "endpoint",
        "GET /v1/models/{model}",
        "PASS" if one_ok else "FAIL",
        f"HTTP {code_one} body={json.dumps(one)[:160]}",
    )

    # unknown id: 404 + OpenAI-style error envelope (at least a non-empty message)
    code_404, missing = client.get_json("/v1/models/model_does_not_exist")
    miss_d = missing if isinstance(missing, dict) else {}
    err = miss_d.get("error") if isinstance(miss_d.get("error"), dict) else {}
    miss_ok = (
        code_404 == 404
        and isinstance(err.get("message"), str)
        and bool(err["message"])
        and isinstance(err.get("type"), str)
        and bool(err["type"])
    )
    report.add(
        "endpoint",
        "GET /v1/models/{model} (missing)",
        "PASS" if miss_ok else "FAIL",
        f"HTTP {code_404} body={json.dumps(missing)[:200]}",
    )
