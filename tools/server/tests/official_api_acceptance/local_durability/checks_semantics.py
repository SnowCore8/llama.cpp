"""Local semantics: compact expand + chat store."""

from __future__ import annotations

import json
from typing import Any

from .http_client import HttpClient, output_text
from .report import Report


def _create(client: HttpClient, model: str, extra: dict[str, Any], body: dict[str, Any]):
    payload = {"model": model, "max_output_tokens": 48, **extra, **body}
    if "max_output_tokens" in body:
        payload["max_output_tokens"] = body["max_output_tokens"]
    return client.post_json("/v1/responses", payload)


def _json_contains(obj: Any, needle: str) -> bool:
    try:
        return needle in json.dumps(obj, ensure_ascii=False)
    except Exception:
        return False


def run_semantic_checks(
    client: HttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    secret = "LOCAL_COMPACT_SECRET_7f3a"
    try:
        code, seed = _create(
            client,
            model,
            extra,
            {
                "input": f"Remember the secret code {secret}. Reply with exactly: COMPACT_SEED_OK",
                "temperature": 0,
                "store": True,
            },
        )
    except Exception as e:  # noqa: BLE001
        report.add("semantic", "compact_expand_via_previous_response_id", "FAIL", f"seed exception: {e}")
        code, seed = 0, {}

    if code != 200 or not isinstance(seed, dict) or not seed.get("id"):
        report.add("semantic", "compact_expand_via_previous_response_id", "FAIL", f"seed HTTP {code}")
    else:
        try:
            ccode, compacted = client.post_json(
                "/v1/responses/compact",
                {
                    "model": model,
                    "previous_response_id": seed["id"],
                    "input": "fold history please",
                    "store": True,
                },
            )
        except Exception as e:  # noqa: BLE001
            report.add(
                "semantic",
                "compact_expand_via_previous_response_id",
                "FAIL",
                f"compact exception: {e}",
            )
            ccode, compacted = 0, {}
        cid = compacted.get("id") if isinstance(compacted, dict) else None
        out = compacted.get("output") if isinstance(compacted, dict) else None
        has_comp = isinstance(out, list) and any(
            isinstance(x, dict)
            and x.get("type") == "compaction"
            and isinstance(x.get("encrypted_content"), str)
            and str(x.get("encrypted_content")).startswith("local.")
            for x in out
        )
        if ccode != 200 or not cid or not has_comp:
            report.add(
                "semantic",
                "compact_expand_via_previous_response_id",
                "FAIL",
                f"compact HTTP {ccode} id={cid!r} has_comp={has_comp}",
            )
        else:
            rcode, got = client.get_json(f"/v1/responses/{cid}")
            retrieved = rcode == 200 and isinstance(got, dict) and got.get("id") == cid
            try:
                fcode, follow = _create(
                    client,
                    model,
                    extra,
                    {
                        "previous_response_id": cid,
                        "input": "What was the secret code? Reply with the code only.",
                        "temperature": 0,
                        "store": True,
                        "max_output_tokens": 32,
                    },
                )
            except Exception as e:  # noqa: BLE001
                report.add(
                    "semantic",
                    "compact_expand_via_previous_response_id",
                    "FAIL",
                    f"follow exception: {e}",
                )
                fcode, follow = 0, {}
            expanded = False
            if fcode == 200 and isinstance(follow, dict) and follow.get("id"):
                icode, items = client.get_json(f"/v1/responses/{follow['id']}/input_items")
                expanded = icode == 200 and _json_contains(items, secret)
            text = output_text(follow) if isinstance(follow, dict) else ""
            ok = retrieved and expanded
            report.add(
                "semantic",
                "compact_expand_via_previous_response_id",
                "PASS" if ok else "FAIL",
                f"retrieve={rcode} follow={fcode} expanded={expanded} text={text!r}",
            )

    # Chat store persists under the openai-files-path root
    ch_code, chat = client.post_json(
        "/v1/chat/completions",
        {
            "model": model,
            "store": True,
            "metadata": {"src": "local_durability"},
            "messages": [{"role": "user", "content": "Reply with exactly: CHAT_STORE_OK"}],
            "temperature": 0,
            "max_tokens": 32,
            **extra,
        },
    )
    chat_id = chat.get("id") if isinstance(chat, dict) else None
    if ch_code == 200 and chat_id:
        gcode, got = client.get_json(f"/v1/chat/completions/{chat_id}")
        ok_chat = gcode == 200 and isinstance(got, dict) and got.get("id") == chat_id
    else:
        ok_chat = False
        gcode = ch_code

    report.add(
        "semantic",
        "chat_store_persists",
        "PASS" if ok_chat else "FAIL",
        f"chat_create={ch_code} chat_get={gcode}",
    )
