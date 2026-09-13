"""OpenAI Embeddings API checks (POST /v1/embeddings)."""

from __future__ import annotations

import base64
import struct
from typing import Any

from .http_client import ResponsesHttpClient
from .report import Report

# Fixed ASCII probes: any tokenizer can embed them, and results stay reproducible.
_TEXT = "The quick brown fox jumped over the lazy dog"
_BATCH = ["batch item alpha", "batch item beta", "batch item gamma"]

# Row plan, emitted up-front as SKIP when the server runs without --embeddings.
_ROWS: tuple[tuple[str, str], ...] = (
    ("endpoint", "embeddings.create"),
    ("endpoint", "embeddings.batch"),
    ("endpoint", "embeddings.deterministic"),
    ("endpoint", "embeddings.distinct_inputs"),
    ("create_param", "embeddings.encoding_format_base64"),
    ("create_param", "embeddings.encoding_format_invalid"),
    ("create_param", "embeddings.encoding_format_type_rejected"),
    ("create_param", "embeddings.input_null_rejected"),
    ("create_param", "embeddings.input_empty_list_rejected"),
    ("scenario", "embeddings.empty_input"),
    ("create_param", "embeddings.dimensions"),
)


def _error_message(body: Any) -> str:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return err["message"]
    return ""


def _error_envelope(body: Any) -> bool:
    """OpenAI error shape: {"error": {"message": <non-empty str>, ...}}."""
    return isinstance(body, dict) and isinstance(body.get("error"), dict) and bool(_error_message(body))


def _item_of(body: Any, i: int = 0) -> dict[str, Any]:
    data = body.get("data") if isinstance(body, dict) else None
    if isinstance(data, list) and len(data) > i and isinstance(data[i], dict):
        return data[i]
    return {}


def _embedding_of(body: Any, i: int = 0) -> Any:
    return _item_of(body, i).get("embedding")


def _vec_ok(vec: Any) -> bool:
    return (
        isinstance(vec, list)
        and len(vec) > 1
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vec)
    )


def _dim(vec: Any) -> int:
    return len(vec) if isinstance(vec, list) else -1


def run_embeddings_checks(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
) -> None:
    # A server started without --embeddings answers "does not support embeddings";
    # that is an inapplicable probe (SKIP), not a failure. Match on the message,
    # not the status code (observed: HTTP 501 not_supported_error).
    code, probe = client.post_json("/v1/embeddings", {"model": model, "input": "embedding probe"})
    if code != 200 and "does not support embeddings" in _error_message(probe):
        detail = f"server does not support embeddings (HTTP {code}): {_error_message(probe)!r}"[:200]
        for area, name in _ROWS:
            report.add(area, name, "SKIP", detail)
        return

    # Official CreateEmbeddingResponse for a single string input: object "list",
    # data[0] object "embedding" + index 0 + non-empty float array, model echo,
    # usage.prompt_tokens > 0 and total_tokens == prompt_tokens.
    code, data = client.post_json("/v1/embeddings", {"input": _TEXT, "model": model})
    d = data if isinstance(data, dict) else {}
    rows = d.get("data") if isinstance(d.get("data"), list) else []
    item = rows[0] if rows and isinstance(rows[0], dict) else {}
    vec = item.get("embedding")
    usage = d.get("usage") if isinstance(d.get("usage"), dict) else {}
    create_ok = (
        code == 200
        and d.get("object") == "list"
        and d.get("model") == model
        and len(rows) == 1
        and item.get("object") == "embedding"
        and item.get("index") == 0
        and _vec_ok(vec)
        and isinstance(usage.get("prompt_tokens"), int)
        and not isinstance(usage.get("prompt_tokens"), bool)
        and usage["prompt_tokens"] > 0
        and usage.get("total_tokens") == usage.get("prompt_tokens")
    )
    report.add(
        "endpoint",
        "embeddings.create",
        "PASS" if create_ok else "FAIL",
        f"HTTP {code} object={d.get('object')!r} model={d.get('model')!r} n={len(rows)} "
        f"d0_object={item.get('object')!r} index={item.get('index')!r} dim={_dim(vec)} usage={usage!r}"[:240],
    )

    # Array input: one data entry per input, index in request order, vectors non-empty.
    code, data = client.post_json("/v1/embeddings", {"input": _BATCH, "model": model})
    d = data if isinstance(data, dict) else {}
    rows = d.get("data") if isinstance(d.get("data"), list) else []
    items_ok = all(
        isinstance(r, dict) and r.get("object") == "embedding" and _vec_ok(r.get("embedding"))
        for r in rows
    )
    batch_ok = (
        code == 200
        and d.get("object") == "list"
        and len(rows) == len(_BATCH)
        and [r.get("index") for r in rows if isinstance(r, dict)] == list(range(len(_BATCH)))
        and items_ok
    )
    report.add(
        "endpoint",
        "embeddings.batch",
        "PASS" if batch_ok else "FAIL",
        f"HTTP {code} n={len(rows)} indices={[r.get('index') for r in rows if isinstance(r, dict)]} "
        f"dims={[_dim(r.get('embedding')) for r in rows if isinstance(r, dict)]}"[:200],
    )

    # Same input twice: observed bitwise identical locally, so assert exact equality
    # (no tolerance); maxdiff is reported for the record.
    code1, data1 = client.post_json("/v1/embeddings", {"input": "determinism probe text", "model": model})
    code2, data2 = client.post_json("/v1/embeddings", {"input": "determinism probe text", "model": model})
    e1, e2 = _embedding_of(data1), _embedding_of(data2)
    same_len = isinstance(e1, list) and isinstance(e2, list) and len(e1) == len(e2)
    exact = same_len and e1 == e2
    maxdiff = max((abs(x - y) for x, y in zip(e1, e2)), default=None) if same_len else None
    det_ok = code1 == 200 and code2 == 200 and _vec_ok(e1) and exact
    report.add(
        "endpoint",
        "embeddings.deterministic",
        "PASS" if det_ok else "FAIL",
        f"HTTP {code1}/{code2} dim={_dim(e1)} exact_equal={exact} maxdiff={maxdiff}"[:200],
    )

    # Different inputs must not collapse to the same vector (guard against constant output).
    code, data = client.post_json("/v1/embeddings", {"input": ["distinct alpha", "distinct beta"], "model": model})
    ea, eb = _embedding_of(data, 0), _embedding_of(data, 1)
    spread = None
    if code == 200 and _vec_ok(ea) and _vec_ok(eb) and len(ea) == len(eb):
        spread = max(abs(x - y) for x, y in zip(ea, eb))
    distinct_ok = code == 200 and spread is not None and spread > 1e-9
    n_rows = len(data.get("data") or []) if isinstance(data, dict) else 0
    report.add(
        "endpoint",
        "embeddings.distinct_inputs",
        "PASS" if distinct_ok else "FAIL",
        f"HTTP {code} n={n_rows} maxdiff={spread}"[:200],
    )

    # encoding_format "base64": embedding is a decodable base64 string; decoded float
    # count and values match the explicit "float" request for the same input.
    code_f, data_f = client.post_json(
        "/v1/embeddings", {"input": "base64 probe text", "model": model, "encoding_format": "float"}
    )
    code_b, data_b = client.post_json(
        "/v1/embeddings", {"input": "base64 probe text", "model": model, "encoding_format": "base64"}
    )
    vec_f = _embedding_of(data_f)
    item_b = _item_of(data_b)
    enc = item_b.get("embedding")
    n_floats = -1
    b64_diff: float | None = None
    decode_err = ""
    if isinstance(enc, str):
        try:
            raw = base64.b64decode(enc, validate=True)
            floats = struct.unpack(f"{len(raw) // 4}f", raw)
            n_floats = len(floats)
            if isinstance(vec_f, list) and len(vec_f) == n_floats:
                b64_diff = max((abs(x - y) for x, y in zip(floats, vec_f)), default=0.0)
        except Exception as e:  # noqa: BLE001 - record the decode failure in the row
            decode_err = str(e)[:80]
    # Official Embedding item is {embedding, index, object} only; extras (if any)
    # are tolerated and reported.
    extra_keys = sorted(k for k in item_b if k not in ("embedding", "index", "object"))
    b64_ok = (
        code_f == 200
        and code_b == 200
        and _vec_ok(vec_f)
        and item_b.get("object") == "embedding"
        and item_b.get("index") == 0
        and isinstance(enc, str)
        and not decode_err
        and n_floats == len(vec_f)
        and b64_diff is not None
        and b64_diff <= 1e-6
    )
    report.add(
        "create_param",
        "embeddings.encoding_format_base64",
        "PASS" if b64_ok else "FAIL",
        f"HTTP {code_f}/{code_b} n_floats={n_floats} float_n={_dim(vec_f)} maxdiff={b64_diff} "
        f"extra_keys={extra_keys} decode_err={decode_err!r}"[:240],
    )

    # Invalid encoding_format: 400 with an OpenAI error envelope; a control request
    # (same body, "float") must still succeed, otherwise the 400 is not attributable
    # to the format field (e.g. a server-wide pooling misconfig).
    code_bad, data_bad = client.post_json(
        "/v1/embeddings", {"input": "x", "model": model, "encoding_format": "cbor"}
    )
    code_ctl, data_ctl = client.post_json(
        "/v1/embeddings", {"input": "x", "model": model, "encoding_format": "float"}
    )
    bad_ok = (
        code_bad == 400
        and _error_envelope(data_bad)
        and code_ctl == 200
        and _vec_ok(_embedding_of(data_ctl))
    )
    report.add(
        "create_param",
        "embeddings.encoding_format_invalid",
        "PASS" if bad_ok else "FAIL",
        f"HTTP {code_bad} msg={_error_message(data_bad)!r} control_http={code_ctl}"[:240],
    )

    # encoding_format must be a string ("float" | "base64"); a wrong JSON type must
    # yield 400 with an OpenAI error envelope, never a 5xx.
    code_ft, data_ft = client.post_json(
        "/v1/embeddings", {"input": "x", "model": model, "encoding_format": 123}
    )
    ft_ok = code_ft == 400 and _error_envelope(data_ft)
    report.add(
        "create_param",
        "embeddings.encoding_format_type_rejected",
        "PASS" if ft_ok else "FAIL",
        f"HTTP {code_ft} msg={_error_message(data_ft)!r}"[:240],
    )

    # input: null is neither a string nor an array; official rejects it with a 4xx
    # error envelope (this used to surface as a local 500).
    code_null, data_null = client.post_json("/v1/embeddings", {"input": None, "model": model})
    null_ok = code_null == 400 and _error_envelope(data_null)
    report.add(
        "create_param",
        "embeddings.input_null_rejected",
        "PASS" if null_ok else "FAIL",
        f"HTTP {code_null} msg={_error_message(data_null)!r}"[:240],
    )

    # input: [] is not accepted officially (empty input); must be 400, not a 5xx.
    code_el, data_el = client.post_json("/v1/embeddings", {"input": [], "model": model})
    el_ok = code_el == 400 and _error_envelope(data_el)
    report.add(
        "create_param",
        "embeddings.input_empty_list_rejected",
        "PASS" if el_ok else "FAIL",
        f"HTTP {code_el} msg={_error_message(data_el)!r}"[:240],
    )

    # Empty string: official says input cannot be an empty string. Locally it is
    # rejected only when the tokenizer produces no tokens (Qwen: 400); BERT-style
    # tokenizers may turn "" into special tokens and accept it - that case is
    # recorded as SKIP, not asserted as failure.
    code_e, data_e = client.post_json("/v1/embeddings", {"input": "", "model": model})
    code_c, data_c = client.post_json("/v1/embeddings", {"input": "control probe text", "model": model})
    ctl_ok = code_c == 200 and _vec_ok(_embedding_of(data_c))
    if code_e == 400 and _error_envelope(data_e) and ctl_ok:
        status, detail = "PASS", f"HTTP 400 rejected: {_error_message(data_e)!r} (official: empty string not allowed)"
    elif code_e == 200 and _vec_ok(_embedding_of(data_e)):
        status = "SKIP"
        detail = (
            f"local accepts empty input (HTTP 200, dim={_dim(_embedding_of(data_e))}); official says input "
            f"cannot be an empty string - recorded, not asserted"
        )
    else:
        status = "FAIL"
        detail = f"HTTP {code_e} body={str(data_e)[:120]} control_http={code_c}"[:200]
    report.add("scenario", "embeddings.empty_input", status, detail)

    # dimensions: official field, only supported by text-embedding-3 and later.
    # Probe how the local server handles it, then assert only when unambiguous:
    # honored -> PASS; accepted-but-ignored or 4xx -> recorded as SKIP; 5xx -> FAIL.
    code_n, data_n = client.post_json("/v1/embeddings", {"input": "dimensions probe", "model": model})
    vec_n = _embedding_of(data_n)
    full = _dim(vec_n)
    code_d, data_d = client.post_json(
        "/v1/embeddings", {"input": "dimensions probe", "model": model, "dimensions": 128}
    )
    dim_d = _dim(_embedding_of(data_d))
    if code_n != 200 or not _vec_ok(vec_n):
        status = "FAIL"
        detail = f"precondition failed: plain embeddings HTTP {code_n}"[:200]
    elif code_d == 200 and dim_d == 128:
        status, detail = "PASS", f"dimensions honored: dim={dim_d}"
    elif code_d == 200 and dim_d == full:
        status = "SKIP"
        detail = (
            f"accepted but ignored: dim stays {full}; official: dimensions only supported in "
            f"text-embedding-3 and later models - recorded, not asserted"
        )
    elif 400 <= code_d < 500:
        status = "SKIP"
        detail = f"HTTP {code_d} rejected: {_error_message(data_d)!r} - recorded, not asserted"
    else:
        status = "FAIL"
        detail = f"HTTP {code_d} dim={dim_d} body={str(data_d)[:120]}"[:200]
    report.add("create_param", "embeddings.dimensions", status, detail)
