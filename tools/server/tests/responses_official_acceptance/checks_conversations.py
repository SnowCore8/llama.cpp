"""Official Conversations API checks (conversation objects + items)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from .http_client import (
    ResponsesHttpClient,
    as_dict as _as_dict,
    as_list as _as_list,
    openai_base,
    output_text,
    parse_sse,
)
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
                    "bytes": list(b"LP_TEXT"),
                    "logprob": -0.5,
                    "top_logprobs": [{"token": "LP_TEXT", "bytes": list(b"LP_TEXT"), "logprob": -0.5}],
                }
            ],
        }
    ],
}


def _first_content_part(item: dict[str, Any]) -> dict[str, Any]:
    parts = item.get("content")
    return _as_dict(parts[0]) if isinstance(parts, list) and parts else {}


def _content_text(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    for part in item.get("content") or []:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            return part["text"]
    return ""


def _conversation_items(client: ResponsesHttpClient, cid: str) -> tuple[int, list[dict[str, Any]]]:
    code, page = client.get_json(f"/v1/conversations/{cid}/items?order=asc")
    data = page.get("data") if isinstance(page, dict) else None
    return code, data if isinstance(data, list) else []


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

    code_empty, empty = client.post_json("/v1/conversations", {})
    add(
        "create.empty_body",
        code_empty == 200
        and isinstance(empty, dict)
        and empty.get("object") == "conversation"
        and isinstance(empty.get("created_at"), int)
        and (empty.get("metadata") in ({}, None)),
        f"HTTP {code_empty} body={json.dumps(empty)[:160]}",
    )

    code, too_many = client.post_json("/v1/conversations", {"items": [dict(ITEM_TEXT) for _ in range(21)]})
    add("create.items_limit", code == 400, f"HTTP {code} (want 400)")

    if shape_ok:
        code, page = client.get_json(f"/v1/conversations/{cid}/items")
        page_d = _as_dict(page)
        data = _as_list(page_d.get("data"))
        stored_ok = (
            code == 200
            and page_d.get("object") == "list"
            and len(data) == 2
            and all(isinstance(x, dict) for x in data)
            and data[0].get("id", "").startswith("msg_")
            and data[1].get("id", "").startswith("msg_")
            and _content_text(data[0]) == "world"
            and _content_text(data[1]) == "hello"
            and data[0].get("role") == "user"
            and data[0].get("status") == "completed"
            and data[0].get("type") == "message"
            and page_d.get("has_more") is False
            and page_d.get("first_id") == data[0].get("id")
            and page_d.get("last_id") == data[1].get("id")
        )
        add("create.items_stored", stored_ok, f"HTTP {code} page={json.dumps(page)[:240]}")

        code, got = client.get_json(f"/v1/conversations/{cid}")
        got_d = _as_dict(got)
        add(
            "get",
            code == 200
            and got_d.get("id") == cid
            and got_d.get("object") == "conversation"
            and got_d.get("metadata") == {"topic": "demo"},
            f"HTTP {code} body={json.dumps(got)[:160]}",
        )

        code, updated = client.post_json(f"/v1/conversations/{cid}", {"metadata": {"topic": "demo2", "k": "v"}})
        code_get, after_update = client.get_json(f"/v1/conversations/{cid}")
        updated_d = _as_dict(updated)
        add(
            "update",
            code == 200
            and updated_d.get("metadata") == {"topic": "demo2", "k": "v"}
            and code_get == 200
            and _as_dict(after_update).get("metadata") == {"topic": "demo2", "k": "v"},
            f"HTTP {code} metadata={json.dumps(updated_d.get('metadata'))[:120]}",
        )

        code, _ = client.post_json(f"/v1/conversations/{cid}", {"metadata": {f"k{i}": "v" for i in range(17)}})
        code_bad_value, _ = client.post_json(f"/v1/conversations/{cid}", {"metadata": {"k": 1}})
        add(
            "update.invalid_metadata",
            code == 400 and code_bad_value == 400,
            f"17_pairs={code} non_string={code_bad_value} (want 400/400)",
        )

        code_no_md, _ = client.post_json(f"/v1/conversations/{cid}", {})
        add("update.no_metadata", code_no_md == 400, f"HTTP {code_no_md} (want 400)")
    else:
        report.add(cat, "create.items_stored", "FAIL", "create failed, cannot verify stored items")
        report.add(cat, "get", "FAIL", "create failed")
        report.add(cat, "update", "FAIL", "create failed")
        report.add(cat, "update.invalid_metadata", "FAIL", "create failed")
        report.add(cat, "update.no_metadata", "FAIL", "create failed")

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
            "items.add.invalid",
            "items.list.default_desc",
            "items.list.asc",
            "items.list.desc_after",
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
        added_d = _as_dict(added)
        adata = _as_list(added_d.get("data"))
        ids = [i.get("id") if isinstance(i, dict) else None for i in adata]
        add_ok = (
            code == 200
            and added_d.get("object") == "list"
            and len(adata) == 3
            and all(isinstance(i, dict) for i in adata)
            and all(isinstance(x, str) and x for x in ids)
            and ids[0].startswith("msg_")
            and ids[1].startswith("fc_")
            and ids[2].startswith("fco_")
            and adata[1].get("call_id") == "call_1"
            and adata[2].get("output") == "42"
            and added_d.get("has_more") is False
            and added_d.get("first_id") == ids[0]
            and added_d.get("last_id") == ids[2]
        )
        add("items.add", add_ok, f"HTTP {code} ids={ids}")

        code, _ = client.post_json(f"/v1/conversations/{cid2}/items", {"items": [dict(ITEM_TEXT) for _ in range(21)]})
        add("items.add.too_many", code == 400, f"HTTP {code} (want 400)")

        code_missing, _ = client.post_json(f"/v1/conversations/{cid2}/items", {})
        code_bad_role, _ = client.post_json(
            f"/v1/conversations/{cid2}/items",
            {"items": [{"type": "message", "role": "bogus", "content": "x"}]},
        )
        code_no_content, _ = client.post_json(
            f"/v1/conversations/{cid2}/items", {"items": [{"type": "message", "role": "user"}]}
        )
        add(
            "items.add.invalid",
            code_missing == 400 and code_bad_role == 400 and code_no_content == 400,
            f"missing_items={code_missing} bad_role={code_bad_role} no_content={code_no_content} (want 400s)",
        )

        if not add_ok:
            for name in (
                "items.list.default_desc",
                "items.list.asc",
                "items.list.desc_after",
                "items.list.limit_after",
                "items.get",
                "items.delete",
                "items.delete.missing",
            ):
                report.add(cat, name, "FAIL", "items.add failed; cannot verify")
        else:
            code, desc = client.get_json(f"/v1/conversations/{cid2}/items")
            desc_d = _as_dict(desc)
            ddata = _as_list(desc_d.get("data"))
            ids_desc = [i.get("id") if isinstance(i, dict) else None for i in ddata]
            add(
                "items.list.default_desc",
                code == 200
                and ids_desc == [ids[2], ids[1], ids[0]]
                and desc_d.get("first_id") == ids_desc[0]
                and desc_d.get("last_id") == ids_desc[-1],
                f"HTTP {code} ids={ids_desc}",
            )

            code, asc = client.get_json(f"/v1/conversations/{cid2}/items?order=asc")
            adata2 = _as_list(_as_dict(asc).get("data"))
            ids_asc = [i.get("id") if isinstance(i, dict) else None for i in adata2]
            add(
                "items.list.asc",
                code == 200 and ids_asc == [ids[0], ids[1], ids[2]],
                f"HTTP {code} ids={ids_asc}",
            )

            code_da, desc_after = client.get_json(
                f"/v1/conversations/{cid2}/items?order=desc&after={ids[1]}"
            )
            da_data = _as_list(_as_dict(desc_after).get("data"))
            ids_da = [i.get("id") if isinstance(i, dict) else None for i in da_data]
            add(
                "items.list.desc_after",
                code_da == 200 and ids_da == [ids[0]],
                f"HTTP {code_da} after={ids[1]} ids={ids_da}",
            )

            code, page1 = client.get_json(f"/v1/conversations/{cid2}/items?order=asc&limit=2")
            page1_d = _as_dict(page1)
            p1 = _as_list(page1_d.get("data"))
            code2, page2 = client.get_json(
                f"/v1/conversations/{cid2}/items?order=asc&limit=2&after={ids[1]}"
            )
            page2_d = _as_dict(page2)
            p2 = _as_list(page2_d.get("data"))
            add(
                "items.list.limit_after",
                code == 200
                and [i.get("id") if isinstance(i, dict) else None for i in p1] == [ids[0], ids[1]]
                and page1_d.get("has_more") is True
                and code2 == 200
                and [i.get("id") if isinstance(i, dict) else None for i in p2] == [ids[2]]
                and page2_d.get("has_more") is False,
                f"HTTP {code}/{code2} first={len(p1)} more={page1_d.get('has_more')} second={len(p2)}",
            )

            code, item = client.get_json(f"/v1/conversations/{cid2}/items/{ids[1]}")
            item_d = _as_dict(item)
            add(
                "items.get",
                code == 200
                and item_d.get("id") == ids[1]
                and item_d.get("type") == "function_call"
                and item_d.get("name") == "lookup",
                f"HTTP {code} body={json.dumps(item)[:160]}",
            )

            code, conv_after_delete = client.delete_json(f"/v1/conversations/{cid2}/items/{ids[1]}")
            cad = _as_dict(conv_after_delete)
            code_items, left = client.get_json(f"/v1/conversations/{cid2}/items?order=asc")
            left_ids = [
                i.get("id") if isinstance(i, dict) else None
                for i in _as_list(_as_dict(left).get("data"))
            ]
            code_gone, _ = client.get_json(f"/v1/conversations/{cid2}/items/{ids[1]}")
            add(
                "items.delete",
                code == 200
                and cad.get("id") == cid2
                and cad.get("object") == "conversation"
                and isinstance(cad.get("created_at"), int)
                and code_items == 200
                and left_ids == [ids[0], ids[2]]
                and code_gone == 404,
                f"HTTP {code} object={cad.get('object')} left={left_ids} gone={code_gone}",
            )

            code, _ = client.delete_json(f"/v1/conversations/{cid2}/items/{ids[1]}")
            add("items.delete.missing", code == 404, f"HTTP {code} (want 404)")

        code_order, _ = client.get_json(f"/v1/conversations/{cid2}/items?order=bogus")
        code_zero, _ = client.get_json(f"/v1/conversations/{cid2}/items?limit=0")
        code_big, _ = client.get_json(f"/v1/conversations/{cid2}/items?limit=101")
        code_after, _ = client.get_json(f"/v1/conversations/{cid2}/items?after=item_bogus")
        add(
            "items.list.invalid",
            code_order == 400 and code_zero == 400 and code_big == 400 and code_after == 400,
            f"order={code_order} limit0={code_zero} limit101={code_big} after={code_after} (want 400s) (local contract)",
        )

        code, _ = client.get_json(f"/v1/conversations/{cid2}/items/msg_does_not_exist")
        add("items.get.missing", code == 404, f"HTTP {code} (want 404)")

    # empty conversation item list has null cursors
    eid = _as_dict(empty).get("id", "")
    if not eid:
        report.add(cat, "items.list.empty", "FAIL", f"no empty conversation id (HTTP {code_empty})")
    else:
        code, page_e = client.get_json(f"/v1/conversations/{eid}/items")
        page_e_d = _as_dict(page_e)
        add(
            "items.list.empty",
            code == 200
            and page_e_d.get("object") == "list"
            and _as_list(page_e_d.get("data")) == []
            and page_e_d.get("first_id") is None
            and page_e_d.get("last_id") is None
            and page_e_d.get("has_more") is False,
            f"HTTP {code} body={json.dumps(page_e)[:200]}",
        )

    # official default page size is 20
    code, conv20 = client.post_json(
        "/v1/conversations",
        {"items": [{"type": "message", "role": "user", "content": f"m{i}"} for i in range(20)]},
    )
    lid = _as_dict(conv20).get("id", "")
    if not lid:
        report.add(cat, "items.list.default_limit", "FAIL", f"create failed (HTTP {code})")
    else:
        code_add1, _ = client.post_json(
            f"/v1/conversations/{lid}/items",
            {"items": [{"type": "message", "role": "user", "content": "m20"}]},
        )
        code_def, page_def = client.get_json(f"/v1/conversations/{lid}/items")
        page_def_d = _as_dict(page_def)
        code_all, page_all = client.get_json(f"/v1/conversations/{lid}/items?limit=100")
        page_all_d = _as_dict(page_all)
        add(
            "items.list.default_limit",
            code_add1 == 200
            and code_def == 200
            and len(_as_list(page_def_d.get("data"))) == 20
            and page_def_d.get("has_more") is True
            and code_all == 200
            and len(_as_list(page_all_d.get("data"))) == 21
            and page_all_d.get("has_more") is False,
            f"HTTP {code_def}/{code_all} default_n={len(_as_list(page_def_d.get('data')))} "
            f"more={page_def_d.get('has_more')} limit100_n={len(_as_list(page_all_d.get('data')))}",
        )

    # include gating: assistant output_text logprobs are stored but hidden by default
    code, c3 = client.post_json("/v1/conversations", {"items": [ITEM_ASSISTANT_LP]})
    cid3 = c3.get("id", "") if isinstance(c3, dict) else ""
    if not cid3:
        report.add(cat, "items.include.logprobs", "FAIL", f"create failed (HTTP {code})")
    else:
        code_list, listing = client.get_json(f"/v1/conversations/{cid3}/items")
        ldata = _as_list(_as_dict(listing).get("data"))
        lp_item = _as_dict(ldata[0]) if ldata else {}
        lp_part = _first_content_part(lp_item)
        lp_id = lp_item.get("id") if isinstance(lp_item.get("id"), str) else ""
        code_inc, included = client.get_json(
            f"/v1/conversations/{cid3}/items?include=message.output_text.logprobs"
        )
        idata = _as_list(_as_dict(included).get("data"))
        inc_part = _first_content_part(_as_dict(idata[0]) if idata else {})
        code_one, one = client.get_json(
            f"/v1/conversations/{cid3}/items/{lp_id}?include=message.output_text.logprobs"
        )
        one_part = _first_content_part(_as_dict(one))
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
            f"included_n={len(_as_list(inc_part.get('logprobs')))}",
        )

    # delete conversation returns the deleted resource and removes it
    if cid:
        code, deleted = client.delete_json(f"/v1/conversations/{cid}")
        deleted_d = _as_dict(deleted)
        code_get, _ = client.get_json(f"/v1/conversations/{cid}")
        code_again, _ = client.delete_json(f"/v1/conversations/{cid}")
        add(
            "delete",
            code == 200
            and deleted_d.get("id") == cid
            and deleted_d.get("object") == "conversation.deleted"
            and deleted_d.get("deleted") is True
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
        pid = _as_dict(conv_p).get("id", "")
        found = False
        for path in conv_dir.glob("*.json"):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(doc, dict):
                continue
            items_p = _as_list(doc.get("items"))
            if (
                doc.get("id") == pid
                and doc.get("metadata") == {"persist": "1"}
                and len(items_p) == 1
                and isinstance(items_p[0], dict)
                and items_p[0].get("role") == "user"
            ):
                found = True
                break
        add("persistence.disk", code == 200 and found, f"HTTP {code} id={pid} dir={conv_dir}")

    # official Python SDK surfaces the same resource tree
    try:
        sdk = OpenAI(api_key=client.api_key, base_url=openai_base(client.base_url), timeout=60.0)
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


def run_conversation_response_checks(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    """Official conversation membership: history is prepended, the finished turn appended."""
    cat = "conversation"

    def add(name: str, cond: bool, detail: str) -> None:
        report.add(cat, name, "PASS" if cond else "FAIL", detail)

    def body_for(cid: Any, text: str, **more: Any) -> dict[str, Any]:
        # explicit probe fields win over vendor extra; thinking off keeps exact-text probes observable
        return {
            **extra,
            "model": model,
            "input": text,
            "max_output_tokens": 24,
            "temperature": 0,
            "reasoning": {"effort": "none"},
            "conversation": cid,
            **more,
        }

    # string form: existing items reach the model, input + output join the conversation
    code, conv = client.post_json(
        "/v1/conversations", {"items": [{"type": "message", "role": "user", "content": "hello"}]}
    )
    cid = conv.get("id", "") if isinstance(conv, dict) else ""
    if not cid:
        add("responses.conversation.turn", False, f"create failed (HTTP {code})")
    else:
        code_r, resp = client.post_json("/v1/responses", body_for(cid, "Reply with exactly: TURN_OK"))
        text = output_text(resp) if isinstance(resp, dict) else ""
        code_l, items = _conversation_items(client, cid)
        texts = [_content_text(i) for i in items]
        add(
            "responses.conversation.turn",
            code_r == 200
            and isinstance(resp, dict)
            and isinstance(resp.get("conversation"), dict)
            and resp["conversation"].get("id") == cid
            and "TURN_OK" in text
            and code_l == 200
            and len(items) == 3
            and all(isinstance(i, dict) for i in items)
            and items[0].get("role") == "user"
            and texts[0] == "hello"
            and items[1].get("role") == "user"
            and texts[1] == "Reply with exactly: TURN_OK"
            and items[2].get("role") == "assistant"
            and items[2].get("id", "").startswith("msg_")
            and "TURN_OK" in texts[2],
            f"HTTP {code_r} echo={resp.get('conversation') if isinstance(resp, dict) else None} "
            f"items={len(items)} texts={texts}",
        )

        # object form {"id": ...} behaves the same and is echoed as an object
        code, cid2_conv = client.post_json("/v1/conversations", {})
        cid2 = cid2_conv.get("id", "") if isinstance(cid2_conv, dict) else ""
        code_o, resp_o = client.post_json(
            "/v1/responses", body_for({"id": cid2}, "Reply with exactly: OBJ_OK")
        )
        code_ol, items_o = _conversation_items(client, cid2)
        add(
            "responses.conversation.object_form",
            code_o == 200
            and isinstance(resp_o, dict)
            and isinstance(resp_o.get("conversation"), dict)
            and resp_o["conversation"].get("id") == cid2
            and code_ol == 200
            and len(items_o) == 2
            and all(isinstance(i, dict) for i in items_o)
            and items_o[0].get("role") == "user"
            and items_o[1].get("role") == "assistant",
            f"HTTP {code_o} items={len(items_o)}",
        )

        # store=false only controls previous_response_id retention, not the conversation
        code, cid3_conv = client.post_json("/v1/conversations", {})
        cid3 = cid3_conv.get("id", "") if isinstance(cid3_conv, dict) else ""
        code_s, resp_s = client.post_json(
            "/v1/responses", body_for(cid3, "Reply with exactly: STORE_OK", store=False)
        )
        code_sl, items_s = _conversation_items(client, cid3)
        add(
            "responses.conversation.store_false",
            code_s == 200
            and code_sl == 200
            and len(items_s) == 2
            and all(isinstance(i, dict) for i in items_s)
            and items_s[0].get("role") == "user"
            and items_s[1].get("role") == "assistant",
            f"HTTP {code_s} items={len(items_s)} (local contract)",
        )

        # streaming create appends the turn too (separate remember call site)
        code, cid4_conv = client.post_json("/v1/conversations", {})
        cid4 = cid4_conv.get("id", "") if isinstance(cid4_conv, dict) else ""
        body_s = body_for(cid4, "Reply with exactly: STREAM_OK")
        body_s["stream"] = True
        code_st, _, raw = client.request("POST", "/v1/responses", body_s, stream=True)
        events = parse_sse(raw) if code_st == 200 else []
        done = next((obj for t, obj in events if t == "response.completed"), None)
        done_resp = _as_dict(_as_dict(done).get("response"))
        code_stl, items_st = _conversation_items(client, cid4)
        add(
            "responses.conversation.stream",
            code_st == 200
            and isinstance(done, dict)
            and isinstance(done_resp.get("conversation"), dict)
            and done_resp["conversation"].get("id") == cid4
            and code_stl == 200
            and len(items_st) == 2
            and all(isinstance(i, dict) for i in items_st)
            and items_st[1].get("role") == "assistant",
            f"HTTP {code_st} items={len(items_st)}",
        )

        # the model actually sees the conversation history
        code, cid5_conv = client.post_json(
            "/v1/conversations",
            {
                "items": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": "The secret code is ZEBRA-42. Just acknowledge.",
                    }
                ]
            },
        )
        cid5 = cid5_conv.get("id", "") if isinstance(cid5_conv, dict) else ""
        code_h, resp_h = client.post_json(
            "/v1/responses", body_for(cid5, "What is the secret code? Reply with exactly the code.")
        )
        text_h = output_text(resp_h) if isinstance(resp_h, dict) else ""
        add(
            "responses.conversation.context",
            code_h == 200 and "ZEBRA-42" in text_h,
            f"HTTP {code_h} text={text_h!r}",
        )

    # validation: mutually exclusive with previous_response_id, 400 on missing or bad shape
    code_prev, first = client.post_json("/v1/responses", {"model": model, "input": "Reply with exactly: X", **extra})
    prev_id = _as_dict(first).get("id", "")
    code_me, me_conv = client.post_json("/v1/conversations", {})
    me_cid = _as_dict(me_conv).get("id", "")
    if code_prev != 200 or not prev_id or code_me != 200 or not me_cid:
        report.add(
            cat,
            "responses.conversation.mutual_exclusion",
            "FAIL",
            f"prereq failed: first_http={code_prev} previous_response_id={prev_id or None} "
            f"conv_http={code_me} conv_id={me_cid or None}",
        )
    else:
        code_pe, _ = client.post_json(
            "/v1/responses",
            {"model": model, "input": "hi", "previous_response_id": prev_id, "conversation": me_cid, **extra},
        )
        add(
            "responses.conversation.mutual_exclusion",
            code_pe == 400,
            f"HTTP {code_pe} (want 400) first_http={code_prev} previous_response_id={prev_id} conversation={me_cid}",
        )

    code_missing, _ = client.post_json(
        "/v1/responses", {"model": model, "input": "hi", "conversation": "conv_does_not_exist", **extra}
    )
    add(
        "responses.conversation.missing",
        code_missing == 400,
        f"HTTP {code_missing} (want 400) (local contract)",
    )

    code_shape, _ = client.post_json("/v1/responses", {"model": model, "input": "hi", "conversation": 123, **extra})
    add("responses.conversation.bad_shape", code_shape == 400, f"HTTP {code_shape} (want 400)")
