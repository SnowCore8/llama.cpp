"""Official HTTP endpoint checks (Chat Completions core)."""

from __future__ import annotations

import json
import time
from typing import Any

from .http_client import ChatHttpClient, choice_text
from .report import Report
from .validators import validate_completion


def json_preview(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)[:160]
    except Exception:
        return repr(obj)[:160]


def run_endpoint_checks(
    client: ChatHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    body = {
        "model": model,
        "max_tokens": 64,
        "temperature": 0,
        "messages": [{"role": "user", "content": "Reply with exactly: CHAT_CREATE"}],
        **extra,
    }
    code, data = client.post_json("/v1/chat/completions", body)
    if code != 200 or not isinstance(data, dict):
        report.add("endpoint", "POST /v1/chat/completions (create)", "FAIL", f"HTTP {code}")
    else:
        ok, detail = validate_completion(data)
        text = choice_text(data)
        if ok and data.get("object") == "chat.completion" and "CHAT_CREATE" in text:
            report.add(
                "endpoint",
                "POST /v1/chat/completions (create)",
                "PASS",
                f"id={data.get('id')}",
            )
        else:
            report.add(
                "endpoint",
                "POST /v1/chat/completions (create)",
                "FAIL",
                f"sdk={detail} text={text!r}",
            )

    # input_tokens adjunct
    tok_body = {
        "model": model,
        "messages": [{"role": "user", "content": "token probe hello"}],
        **{k: v for k, v in extra.items() if k != "chat_template_kwargs"},
    }
    code, tok = client.post_json("/v1/chat/completions/input_tokens", tok_body)
    if code == 404:
        report.add(
            "endpoint",
            "POST /v1/chat/completions/input_tokens",
            "NOT_IMPLEMENTED",
            "HTTP 404",
        )
    elif code == 200 and isinstance(tok, dict) and isinstance(tok.get("input_tokens"), int):
        report.add(
            "endpoint",
            "POST /v1/chat/completions/input_tokens",
            "PASS",
            json_preview(tok),
        )
    else:
        report.add(
            "endpoint",
            "POST /v1/chat/completions/input_tokens",
            "FAIL" if code >= 400 else "PARTIAL",
            f"HTTP {code} {json_preview(tok)}",
        )

    # stream endpoint (HTTP-level smoke)
    scode, headers, raw = client.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 32,
            "stream": True,
            "temperature": 0,
            "messages": [{"role": "user", "content": "Reply with exactly: CHAT_STREAM"}],
            **extra,
        },
        stream=True,
    )
    ct = headers.get("Content-Type", "") or headers.get("content-type", "")
    stream_ok = scode == 200 and "text/event-stream" in ct and b"[DONE]" in raw
    report.add(
        "endpoint",
        "POST /v1/chat/completions (stream)",
        "PASS" if stream_ok else "FAIL",
        f"HTTP {scode} ct={ct}",
    )

    # GET /v1/chat/completions — official query contract: `order` enum, default `asc`
    def _store_probe(marker: str):
        return client.post_json(
            "/v1/chat/completions",
            {
                "model": model,
                "store": True,
                "max_tokens": 4,
                "temperature": 0,
                "messages": [{"role": "user", "content": f"Reply with exactly: {marker}"}],
                **extra,
            },
        )

    first_code, first_data = _store_probe("LIST_A")
    time.sleep(1.1)  # `created` is unix seconds; keep the two entries apart
    second_code, second_data = _store_probe("LIST_B")
    first_id = first_data.get("id") if first_code == 200 and isinstance(first_data, dict) else ""
    second_id = second_data.get("id") if second_code == 200 and isinstance(second_data, dict) else ""
    if not (first_id and second_id):
        report.add(
            "endpoint",
            "GET /v1/chat/completions (list)",
            "FAIL",
            f"store setup failed: {first_code}/{second_code}",
        )
        return

    code, page = client.get_json("/v1/chat/completions?limit=100")
    ids = [c.get("id") for c in (page.get("data") or [])] if isinstance(page, dict) else []
    asc_ok = first_id in ids and second_id in ids and ids.index(first_id) < ids.index(second_id)
    report.add(
        "endpoint",
        "GET /v1/chat/completions (order default asc)",
        "PASS" if code == 200 and asc_ok else "FAIL",
        f"HTTP {code} first<second={asc_ok} n={len(ids)}",
    )

    code, page = client.get_json("/v1/chat/completions?order=desc&limit=100")
    ids = [c.get("id") for c in (page.get("data") or [])] if isinstance(page, dict) else []
    desc_ok = first_id in ids and second_id in ids and ids.index(second_id) < ids.index(first_id)
    report.add(
        "endpoint",
        "GET /v1/chat/completions (order=desc)",
        "PASS" if code == 200 and desc_ok else "FAIL",
        f"HTTP {code} second<first={desc_ok}",
    )

    code, data = client.get_json("/v1/chat/completions?order=bogus")
    report.add(
        "endpoint",
        "GET /v1/chat/completions (order=bogus)",
        "PASS" if code == 400 else "FAIL",
        f"HTTP {code} {json_preview(data)}",
    )

    # Official list query extras: `after` cursor, `model` / `metadata` filters, empty shape.
    time.sleep(1.1)  # keep LIST_C strictly after LIST_B in `created` order
    third_code, third_data = _store_probe("LIST_C")
    third_id = third_data.get("id") if third_code == 200 and isinstance(third_data, dict) else ""
    if not third_id:
        report.add(
            "endpoint",
            "GET /v1/chat/completions (after + limit)",
            "FAIL",
            f"store setup failed: {third_code}",
        )
    else:
        acode, adata = client.get_json(f"/v1/chat/completions?after={first_id}&order=asc&limit=1")
        aids = [c.get("id") for c in (adata.get("data") or [])] if isinstance(adata, dict) else []
        a_first = adata.get("first_id") if isinstance(adata, dict) else "?"
        a_hm = adata.get("has_more") if isinstance(adata, dict) else "?"
        after_ok = (
            acode == 200
            and isinstance(adata, dict)
            and aids == [second_id]
            and a_first == second_id
            and adata.get("last_id") == second_id
            and a_hm is True
        )
        report.add(
            "endpoint",
            "GET /v1/chat/completions (after + limit)",
            "PASS" if after_ok else "FAIL",
            f"HTTP {acode} data={aids!r} first_id={a_first!r} has_more={a_hm!r}",
        )
        tcode, tdata = client.get_json(f"/v1/chat/completions?after={third_id}&order=asc&limit=100")
        tids = [c.get("id") for c in (tdata.get("data") or [])] if isinstance(tdata, dict) else None
        t_first = tdata.get("first_id") if isinstance(tdata, dict) else "?"
        t_hm = tdata.get("has_more") if isinstance(tdata, dict) else "?"
        tail_ok = (
            tcode == 200
            and isinstance(tdata, dict)
            and tids == []
            and t_hm is False
            and not t_first
            and not tdata.get("last_id")
        )
        report.add(
            "endpoint",
            "GET /v1/chat/completions (after tail)",
            "PASS" if tail_ok else "FAIL",
            f"HTTP {tcode} data={tids!r} first_id={t_first!r} has_more={t_hm!r}",
        )

    fcode, fdata = client.get_json(f"/v1/chat/completions?model={model}&limit=100")
    fitems = fdata.get("data") if isinstance(fdata, dict) else None
    fids = [c.get("id") for c in (fitems or [])]
    model_ok = (
        fcode == 200
        and isinstance(fitems, list)
        and first_id in fids
        and second_id in fids
        and all(isinstance(c, dict) and c.get("model") == model for c in fitems)
    )
    report.add(
        "endpoint",
        "GET /v1/chat/completions (model filter)",
        "PASS" if model_ok else "FAIL",
        f"HTTP {fcode} n={len(fids)} first={first_id in fids} second={second_id in fids}",
    )

    md_code, md_data = client.post_json(
        "/v1/chat/completions",
        {
            "model": model,
            "store": True,
            "max_tokens": 4,
            "temperature": 0,
            "metadata": {"suite": "endpoint-filter", "kind": "meta"},
            "messages": [{"role": "user", "content": "Reply with exactly: LIST_MD"}],
            **extra,
        },
    )
    md_id = md_data.get("id") if md_code == 200 and isinstance(md_data, dict) else ""
    if not md_id:
        report.add(
            "endpoint",
            "GET /v1/chat/completions (metadata filter)",
            "FAIL",
            f"store setup failed: {md_code}",
        )
    else:
        qcode, qdata = client.get_json("/v1/chat/completions?metadata[suite]=endpoint-filter&limit=100")
        qitems = qdata.get("data") if isinstance(qdata, dict) else None
        qids = [c.get("id") for c in (qitems or [])]
        echo_ok = all(
            isinstance(c, dict) and (c.get("metadata") or {}).get("suite") == "endpoint-filter"
            for c in (qitems or [])
        )
        md_ok = (
            qcode == 200
            and isinstance(qitems, list)
            and bool(qids)
            and md_id in qids
            and echo_ok
        )
        report.add(
            "endpoint",
            "GET /v1/chat/completions (metadata filter)",
            "PASS" if md_ok else "FAIL",
            f"HTTP {qcode} n={len(qids)} hit={md_id in qids} echo={echo_ok}",
        )

    ecode, edata = client.get_json("/v1/chat/completions?model=no-such-model-xyz")
    eids = [c.get("id") for c in (edata.get("data") or [])] if isinstance(edata, dict) else None
    e_first = edata.get("first_id") if isinstance(edata, dict) else "?"
    e_hm = edata.get("has_more") if isinstance(edata, dict) else "?"
    empty_ok = (
        ecode == 200
        and isinstance(edata, dict)
        and eids == []
        and e_hm is False
        and not e_first
        and not edata.get("last_id")
    )
    report.add(
        "endpoint",
        "GET /v1/chat/completions (empty list shape)",
        "PASS" if empty_ok else "FAIL",
        f"HTTP {ecode} data={eids!r} first_id={e_first!r} has_more={e_hm!r}",
    )

    # GET /v1/chat/completions/{id}/messages — official stored-messages contract
    mcode, mdata = client.get_json(f"/v1/chat/completions/{first_id}/messages")
    msgs = mdata.get("data") if isinstance(mdata, dict) else None
    msgs_ok = (
        mcode == 200
        and isinstance(mdata, dict)
        and mdata.get("object") == "list"
        and isinstance(mdata.get("has_more"), bool)
        and isinstance(msgs, list)
        and len(msgs) >= 1
        and mdata.get("first_id") == msgs[0].get("id")
        and mdata.get("last_id") == msgs[-1].get("id")
    )
    msg0_ok = (
        msgs_ok
        and isinstance(msgs[0], dict)
        and isinstance(msgs[0].get("id"), str)
        and msgs[0]["id"].startswith("msg_")
        and msgs[0].get("role") == "assistant"
        and "LIST_A" in str(msgs[0].get("content"))
        and "content_parts" in msgs[0]
    )
    report.add(
        "endpoint",
        "GET /v1/chat/completions/{id}/messages",
        "PASS" if msg0_ok else "FAIL",
        f"HTTP {mcode} n={len(msgs) if isinstance(msgs, list) else None}",
    )

    rcode, rdata = client.get_json(f"/v1/chat/completions/{first_id}/messages?order=desc")
    rmsgs = rdata.get("data") if isinstance(rdata, dict) else []
    asc_ids = [m.get("id") for m in (msgs or [])]
    desc_ids = [m.get("id") for m in (rmsgs or [])]
    report.add(
        "endpoint",
        "GET /v1/chat/completions/{id}/messages (order=desc)",
        "PASS" if rcode == 200 and desc_ids == list(reversed(asc_ids)) else "FAIL",
        f"HTTP {rcode} asc={asc_ids} desc={desc_ids}",
    )

    last_msg_id = asc_ids[-1] if asc_ids else "none"
    acode, adata = client.get_json(
        f"/v1/chat/completions/{first_id}/messages?after={last_msg_id}"
    )
    atail = adata.get("data") if isinstance(adata, dict) else None
    report.add(
        "endpoint",
        "GET /v1/chat/completions/{id}/messages (after)",
        "PASS" if acode == 200 and atail == [] else "FAIL",
        f"HTTP {acode} tail={json_preview(atail)}",
    )

    bcode, bdata = client.get_json(f"/v1/chat/completions/{first_id}/messages?order=bogus")
    report.add(
        "endpoint",
        "GET /v1/chat/completions/{id}/messages (order=bogus)",
        "PASS" if bcode == 400 else "FAIL",
        f"HTTP {bcode} {json_preview(bdata)}",
    )

    ucode, _ = client.get_json("/v1/chat/completions/chatcmpl-not-stored/messages")
    report.add(
        "endpoint",
        "GET /v1/chat/completions/{id}/messages (unknown id)",
        "PASS" if ucode == 404 else "FAIL",
        f"HTTP {ucode}",
    )

    # GET /v1/models/{model} — official retrieve; shutdown_date is null locally
    list_code, list_data = client.get_json("/v1/models")
    entry = None
    if isinstance(list_data, dict):
        for c in list_data.get("data") or []:
            if c.get("id") == model:
                entry = c
                break
    mcode, mdata = client.get_json(f"/v1/models/{model}")
    retrieve_ok = (
        mcode == 200
        and isinstance(mdata, dict)
        and mdata.get("id") == model
        and mdata.get("object") == "model"
        and "shutdown_date" in mdata
        and mdata.get("shutdown_date") is None
        and entry is not None
        and entry.get("object") == "model"
        and "shutdown_date" in entry
        and entry.get("shutdown_date") is None
    )
    report.add(
        "endpoint",
        "GET /v1/models/{model}",
        "PASS" if retrieve_ok else "FAIL",
        f"retrieve_HTTP={mcode} list_HTTP={list_code} "
        f"shutdown_date={mdata.get('shutdown_date') if isinstance(mdata, dict) else None}",
    )

    umcode, _ = client.get_json("/v1/models/no-such-model")
    report.add(
        "endpoint",
        "GET /v1/models/{model} (unknown id)",
        "PASS" if umcode == 404 else "FAIL",
        f"HTTP {umcode}",
    )

    # OpenAI: an unknown model on create is 404 with the official model_not_found body
    nfcode, nfdata = client.post_json(
        "/v1/chat/completions",
        {
            "model": "no-such-model-xyz",
            "max_tokens": 4,
            "temperature": 0,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    expected_not_found = {
        "code": "model_not_found",
        "message": "The model `no-such-model-xyz` does not exist or you do not have access to it.",
        "param": None,
        "type": "invalid_request_error",
    }
    nf_ok = nfcode == 404 and isinstance(nfdata, dict) and nfdata.get("error") == expected_not_found
    report.add(
        "endpoint",
        "POST /v1/chat/completions (unknown model)",
        "PASS" if nf_ok else "FAIL",
        f"HTTP {nfcode} body={json_preview(nfdata)}",
    )
