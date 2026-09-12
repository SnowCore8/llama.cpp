"""Official Conversations API checks (conversation objects + items)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from .http_client import ResponsesHttpClient
from .report import Report

ITEM_TEXT = {"type": "message", "role": "user", "content": "hello"}
ITEM_TEXT2 = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "world"}]}
ITEM_FC = {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"}
ITEM_FCO = {"type": "function_call_output", "call_id": "call_1", "output": "42"}
ITEM_ASSISTANT_LP = {
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "output_text",
            "text": "LP_TEXT",
            "annotations": [],
            "logprobs": [
                {
                    "token": "LP_TEXT",
                    "bytes": [76, 80],
                    "logprob": -0.5,
                    "top_logprobs": [{"token": "LP_TEXT", "bytes": [76, 80], "logprob": -0.5}],
                }
            ],
        }
    ],
}


def _openai_base(base_url: str) -> str:
    u = base_url.rstrip("/")
    return u if u.endswith("/v1") else f"{u}/v1"


def _content_text(item: dict[str, Any]) -> str:
    for part in item.get("content") or []:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            return part["text"]
    return ""


def _sdk_message_text(item: Any) -> str:
    for part in getattr(item, "content", None) or []:
        text = getattr(part, "text", None)
        if isinstance(text, str):
            return text
    return ""


def run_conversation_checks(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    """Conversations API: create/retrieve/update/delete plus the item sub-resource."""
    cat = "conversation"

    def add(name: str, cond: bool, detail: str) -> None:
        report.add(cat, name, "PASS" if cond else "FAIL", detail)

    # create with metadata and initial items
    code, conv = client.post_json("/v1/conversations", {"metadata": {"topic": "demo"}, "items": [ITEM_TEXT, ITEM_TEXT2]})
    cid = conv.get("id", "") if isinstance(conv, dict) else ""
    shape_ok = (
        code == 200
        and isinstance(conv, dict)
        and conv.get("object") == "conversation"
        and isinstance(conv.get("id"), str)
        and conv["id"].startswith("conv_")
        and isinstance(conv.get("created_at"), int)
        and conv.get("metadata") == {"topic": "demo"}
    )
    add("create", shape_ok, f"HTTP {code} id={cid} body={json.dumps(conv)[:200]}")

    code, empty = client.post_json("/v1/conversations", {})
    add(
        "create.empty_body",
        code == 200
        and isinstance(empty, dict)
        and empty.get("object") == "conversation"
        and isinstance(empty.get("created_at"), int)
        and (empty.get("metadata") in ({}, None)),
        f"HTTP {code} body={json.dumps(empty)[:160]}",
    )

    code, too_many = client.post_json("/v1/conversations", {"items": [dict(ITEM_TEXT) for _ in range(21)]})
    add("create.items_limit", code == 400, f"HTTP {code} (want 400)")

    if shape_ok:
        code, page = client.get_json(f"/v1/conversations/{cid}/items")
        data = page.get("data") if isinstance(page, dict) else None
        stored_ok = (
            code == 200
            and isinstance(page, dict)
            and page.get("object") == "list"
            and isinstance(data, list)
            and len(data) == 2
            and data[0].get("id", "").startswith("msg_")
            and data[1].get("id", "").startswith("msg_")
            and _content_text(data[0]) == "world"
            and _content_text(data[1]) == "hello"
            and data[0].get("role") == "user"
            and data[0].get("status") == "completed"
            and data[0].get("type") == "message"
            and page.get("has_more") is False
            and page.get("first_id") == data[0].get("id")
            and page.get("last_id") == data[1].get("id")
        )
        add("create.items_stored", stored_ok, f"HTTP {code} page={json.dumps(page)[:240]}")

        code, got = client.get_json(f"/v1/conversations/{cid}")
        add(
            "get",
            code == 200
            and isinstance(got, dict)
            and got.get("id") == cid
            and got.get("object") == "conversation"
            and got.get("metadata") == {"topic": "demo"},
            f"HTTP {code} body={json.dumps(got)[:160]}",
        )

        code, updated = client.post_json(f"/v1/conversations/{cid}", {"metadata": {"topic": "demo2", "k": "v"}})
        code_get, after_update = client.get_json(f"/v1/conversations/{cid}")
        add(
            "update",
            code == 200
            and isinstance(updated, dict)
            and updated.get("metadata") == {"topic": "demo2", "k": "v"}
            and code_get == 200
            and after_update.get("metadata") == {"topic": "demo2", "k": "v"},
            f"HTTP {code} metadata={json.dumps(updated.get('metadata'))[:120]}",
        )

        code, _ = client.post_json(f"/v1/conversations/{cid}", {"metadata": {f"k{i}": "v" for i in range(17)}})
        code_bad_value, _ = client.post_json(f"/v1/conversations/{cid}", {"metadata": {"k": 1}})
        add(
            "update.invalid_metadata",
            code == 400 and code_bad_value == 400,
            f"17_pairs={code} non_string={code_bad_value} (want 400/400)",
        )
    else:
        report.add(cat, "create.items_stored", "FAIL", "create failed, cannot verify stored items")
        report.add(cat, "get", "FAIL", "create failed")
        report.add(cat, "update", "FAIL", "create failed")
        report.add(cat, "update.invalid_metadata", "FAIL", "create failed")

    code, _ = client.get_json("/v1/conversations/conv_does_not_exist")
    add("get.missing", code == 404, f"HTTP {code} (want 404)")
    code, _ = client.post_json("/v1/conversations/conv_does_not_exist", {"metadata": {}})
    add("update.missing", code == 404, f"HTTP {code} (want 404)")

    # item sub-resource on a fresh conversation
    code, c2 = client.post_json("/v1/conversations", {})
    cid2 = c2.get("id", "") if isinstance(c2, dict) else ""
    if not cid2:
        for name in (
            "items.add",
            "items.add.too_many",
            "items.list.default_desc",
            "items.list.asc",
            "items.list.limit_after",
            "items.list.invalid",
            "items.get",
            "items.get.missing",
            "items.delete",
            "items.delete.missing",
        ):
            report.add(cat, name, "FAIL", f"second create failed (HTTP {code})")
    else:
        code, added = client.post_json(f"/v1/conversations/{cid2}/items", {"items": [ITEM_TEXT, ITEM_FC, ITEM_FCO]})
        adata = added.get("data") if isinstance(added, dict) else None
        add(
            "items.add",
            code == 200
            and isinstance(added, dict)
            and added.get("object") == "list"
            and isinstance(adata, list)
            and len(adata) == 3
            and adata[0].get("id", "").startswith("msg_")
            and adata[1].get("id", "").startswith("fc_")
            and adata[2].get("id", "").startswith("fco_")
            and adata[1].get("call_id") == "call_1"
            and adata[2].get("output") == "42"
            and added.get("has_more") is False
            and added.get("first_id") == adata[0].get("id")
            and added.get("last_id") == adata[2].get("id"),
            f"HTTP {code} ids={[i.get('id') for i in adata] if isinstance(adata, list) else None}",
        )

        code, _ = client.post_json(f"/v1/conversations/{cid2}/items", {"items": [dict(ITEM_TEXT) for _ in range(21)]})
        add("items.add.too_many", code == 400, f"HTTP {code} (want 400)")

        code, desc = client.get_json(f"/v1/conversations/{cid2}/items")
        ddata = desc.get("data") if isinstance(desc, dict) else None
        ids_desc = [i.get("id") for i in ddata] if isinstance(ddata, list) else []
        add(
            "items.list.default_desc",
            code == 200
            and ids_desc == [adata[2]["id"], adata[1]["id"], adata[0]["id"]]
            and desc.get("first_id") == ids_desc[0]
            and desc.get("last_id") == ids_desc[-1],
            f"HTTP {code} ids={ids_desc}",
        )

        code, asc = client.get_json(f"/v1/conversations/{cid2}/items?order=asc")
        adata2 = asc.get("data") if isinstance(asc, dict) else None
        ids_asc = [i.get("id") for i in adata2] if isinstance(adata2, list) else []
        add(
            "items.list.asc",
            code == 200 and ids_asc == [adata[0]["id"], adata[1]["id"], adata[2]["id"]],
            f"HTTP {code} ids={ids_asc}",
        )

        code, page1 = client.get_json(f"/v1/conversations/{cid2}/items?order=asc&limit=2")
        p1 = page1.get("data") if isinstance(page1, dict) else None
        code2, page2 = client.get_json(
            f"/v1/conversations/{cid2}/items?order=asc&limit=2&after={adata[1]['id']}"
        )
        p2 = page2.get("data") if isinstance(page2, dict) else None
        add(
            "items.list.limit_after",
            code == 200
            and isinstance(p1, list)
            and [i.get("id") for i in p1] == [adata[0]["id"], adata[1]["id"]]
            and page1.get("has_more") is True
            and code2 == 200
            and isinstance(p2, list)
            and [i.get("id") for i in p2] == [adata[2]["id"]]
            and page2.get("has_more") is False,
            f"HTTP {code}/{code2} first={len(p1) if isinstance(p1, list) else None} "
            f"more={page1.get('has_more')} second={len(p2) if isinstance(p2, list) else None}",
        )

        code_order, _ = client.get_json(f"/v1/conversations/{cid2}/items?order=bogus")
        code_zero, _ = client.get_json(f"/v1/conversations/{cid2}/items?limit=0")
        code_big, _ = client.get_json(f"/v1/conversations/{cid2}/items?limit=101")
        code_after, _ = client.get_json(f"/v1/conversations/{cid2}/items?after=item_bogus")
        add(
            "items.list.invalid",
            code_order == 400 and code_zero == 400 and code_big == 400 and code_after == 400,
            f"order={code_order} limit0={code_zero} limit101={code_big} after={code_after} (want 400s)",
        )

        code, item = client.get_json(f"/v1/conversations/{cid2}/items/{adata[1]['id']}")
        add(
            "items.get",
            code == 200
            and isinstance(item, dict)
            and item.get("id") == adata[1]["id"]
            and item.get("type") == "function_call"
            and item.get("name") == "lookup",
            f"HTTP {code} body={json.dumps(item)[:160]}",
        )

        code, _ = client.get_json(f"/v1/conversations/{cid2}/items/msg_does_not_exist")
        add("items.get.missing", code == 404, f"HTTP {code} (want 404)")

        code, conv_after_delete = client.delete_json(f"/v1/conversations/{cid2}/items/{adata[1]['id']}")
        code_items, left = client.get_json(f"/v1/conversations/{cid2}/items?order=asc")
        left_ids = [i.get("id") for i in left.get("data", [])] if isinstance(left, dict) else []
        code_gone, _ = client.get_json(f"/v1/conversations/{cid2}/items/{adata[1]['id']}")
        add(
            "items.delete",
            code == 200
            and isinstance(conv_after_delete, dict)
            and conv_after_delete.get("id") == cid2
            and conv_after_delete.get("object") == "conversation"
            and isinstance(conv_after_delete.get("created_at"), int)
            and code_items == 200
            and left_ids == [adata[0]["id"], adata[2]["id"]]
            and code_gone == 404,
            f"HTTP {code} object={conv_after_delete.get('object') if isinstance(conv_after_delete, dict) else None} "
            f"left={left_ids} gone={code_gone}",
        )

        code, _ = client.delete_json(f"/v1/conversations/{cid2}/items/{adata[1]['id']}")
        add("items.delete.missing", code == 404, f"HTTP {code} (want 404)")

    # include gating: assistant output_text logprobs are stored but hidden by default
    code, c3 = client.post_json("/v1/conversations", {"items": [ITEM_ASSISTANT_LP]})
    cid3 = c3.get("id", "") if isinstance(c3, dict) else ""
    if not cid3:
        report.add(cat, "items.include.logprobs", "FAIL", f"create failed (HTTP {code})")
    else:
        code_list, listing = client.get_json(f"/v1/conversations/{cid3}/items")
        ldata = listing.get("data", []) if isinstance(listing, dict) else []
        lp_item = ldata[0] if ldata else {}
        lp_part = (lp_item.get("content") or [{}])[0] if isinstance(lp_item.get("content"), list) else {}
        code_inc, included = client.get_json(
            f"/v1/conversations/{cid3}/items?include=message.output_text.logprobs"
        )
        idata = included.get("data", []) if isinstance(included, dict) else []
        inc_part = (idata[0].get("content") or [{}])[0] if idata and isinstance(idata[0].get("content"), list) else {}
        code_one, one = client.get_json(
            f"/v1/conversations/{cid3}/items/{lp_item.get('id', '')}?include=message.output_text.logprobs"
        )
        one_part = (one.get("content") or [{}])[0] if isinstance(one, dict) and isinstance(one.get("content"), list) else {}
        add(
            "items.include.logprobs",
            code_list == 200
            and _content_text(lp_item) == "LP_TEXT"
            and "logprobs" not in lp_part
            and code_inc == 200
            and inc_part.get("logprobs") == ITEM_ASSISTANT_LP["content"][0]["logprobs"]
            and code_one == 200
            and one_part.get("logprobs") == ITEM_ASSISTANT_LP["content"][0]["logprobs"],
            f"HTTP {code_list}/{code_inc}/{code_one} default_has_lp={'logprobs' in lp_part} "
            f"included_n={len(inc_part.get('logprobs') or [])}",
        )

    # delete conversation returns the deleted resource and removes it
    if cid:
        code, deleted = client.delete_json(f"/v1/conversations/{cid}")
        code_get, _ = client.get_json(f"/v1/conversations/{cid}")
        code_again, _ = client.delete_json(f"/v1/conversations/{cid}")
        add(
            "delete",
            code == 200
            and isinstance(deleted, dict)
            and deleted.get("id") == cid
            and deleted.get("object") == "conversation.deleted"
            and deleted.get("deleted") is True
            and code_get == 404
            and code_again == 404,
            f"HTTP {code} body={json.dumps(deleted)[:160]} get={code_get} again={code_again}",
        )
    else:
        report.add(cat, "delete", "FAIL", "create failed")

    # disk persistence behind --openai-files-path
    root = os.environ.get("LLAMA_OPENAI_FILES_PATH", "")
    conv_dir = Path(root, "conversations") if root else None
    if conv_dir is None or not conv_dir.is_dir():
        report.add(cat, "persistence.disk", "SKIP", "no LLAMA_OPENAI_FILES_PATH/conversations dir")
    else:
        code, conv_p = client.post_json("/v1/conversations", {"metadata": {"persist": "1"}, "items": [ITEM_TEXT]})
        pid = conv_p.get("id", "") if isinstance(conv_p, dict) else ""
        found = False
        for path in conv_dir.glob("*.json"):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (
                doc.get("id") == pid
                and doc.get("metadata") == {"persist": "1"}
                and isinstance(doc.get("items"), list)
                and len(doc["items"]) == 1
                and doc["items"][0].get("role") == "user"
            ):
                found = True
                break
        add("persistence.disk", code == 200 and found, f"HTTP {code} id={pid} dir={conv_dir}")

    # official Python SDK surfaces the same resource tree
    try:
        sdk = OpenAI(api_key=client.api_key, base_url=_openai_base(client.base_url), timeout=60.0)
        sdk_conv = sdk.conversations.create(
            metadata={"sdk": "1"},
            items=[{"type": "message", "role": "user", "content": "sdk hello"}],
        )
        sdk_got = sdk.conversations.retrieve(sdk_conv.id)
        sdk_added = sdk.conversations.items.create(
            sdk_conv.id,
            items=[{"type": "message", "role": "user", "content": "sdk second"}],
        )
        sdk_page = sdk.conversations.items.list(sdk_conv.id, order="asc", limit=20)
        sdk_item = sdk.conversations.items.retrieve(sdk_page.data[0].id, conversation_id=sdk_conv.id)
        sdk_updated = sdk.conversations.update(sdk_conv.id, metadata={"sdk": "2"})
        sdk_item_del = sdk.conversations.items.delete(sdk_page.data[0].id, conversation_id=sdk_conv.id)
        sdk_deleted = sdk.conversations.delete(sdk_conv.id)
        ok = (
            sdk_conv.object == "conversation"
            and sdk_conv.id.startswith("conv_")
            and sdk_got.id == sdk_conv.id
            and len(sdk_added.data) == 1
            and [_sdk_message_text(i) for i in sdk_page.data] == ["sdk hello", "sdk second"]
            and sdk_item.id == sdk_page.data[0].id
            and sdk_updated.metadata == {"sdk": "2"}
            and sdk_item_del.object == "conversation"
            and sdk_deleted.object == "conversation.deleted"
            and sdk_deleted.deleted is True
        )
        add(
            "sdk.conversations",
            ok,
            f"id={sdk_conv.id} page={[_sdk_message_text(i) for i in sdk_page.data]}",
        )
    except Exception as err:  # noqa: BLE001 - surface any SDK failure as a gap
        report.add(cat, "sdk.conversations", "FAIL", f"{type(err).__name__}: {err}")
