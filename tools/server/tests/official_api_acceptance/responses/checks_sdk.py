"""Live OpenAI Python SDK checks (responses / models / completions)."""

from __future__ import annotations

import json
from typing import Any

from openai import NotFoundError, OpenAI
from pydantic import BaseModel

from .http_client import openai_base
from .report import Report
from .validators import validate_response


def run_sdk_checks(
    report: Report,
    *,
    base_url: str,
    api_key: str,
    model: str,
    extra: dict[str, Any],
) -> None:
    client = OpenAI(api_key=api_key, base_url=openai_base(base_url), timeout=180.0)
    extra_body = dict(extra) if extra else {}
    # Keep exact-text SDK probes observable when the server runs with --reasoning-preserve.
    extra_body.setdefault("reasoning", {"effort": "none"})
    extra_body = extra_body or None

    try:
        resp = client.responses.create(
            model=model,
            input="Reply with exactly: SDK_RESP_OK",
            max_output_tokens=64,
            temperature=0,
            extra_body=extra_body,
        )
        ok, detail = validate_response(resp)
        text = getattr(resp, "output_text", None) or ""
        report.add(
            "sdk",
            "responses.create",
            "PASS" if ok and "SDK_RESP_OK" in text else "FAIL",
            f"status={getattr(resp,'status',None)!r} text={text!r} schema={detail}",
        )
    except Exception as e:
        report.add("sdk", "responses.create", "FAIL", f"{type(e).__name__}: {e}"[:240])

    stored_id: str | None = None
    try:
        stored = client.responses.create(
            model=model,
            input="Reply with exactly: SDK_STORED",
            max_output_tokens=32,
            temperature=0,
            store=True,
            extra_body=extra_body,
        )
        stored_id = getattr(stored, "id", None)
        ok, detail = validate_response(stored)
        report.add(
            "sdk",
            "responses.create.store",
            "PASS" if ok and stored_id else "FAIL",
            f"id={stored_id!r} schema={detail}",
        )
    except Exception as e:
        report.add(
            "sdk",
            "responses.create.store",
            "FAIL",
            f"{type(e).__name__}: {e}"[:240],
        )

    if stored_id:
        try:
            got = client.responses.retrieve(stored_id)
            ok, detail = validate_response(got)
            text = getattr(got, "output_text", None) or ""
            report.add(
                "sdk",
                "responses.retrieve",
                "PASS" if ok and got.id == stored_id and "SDK_STORED" in text else "FAIL",
                f"id={got.id!r} text={text!r} schema={detail}",
            )
        except Exception as e:
            report.add(
                "sdk",
                "responses.retrieve",
                "FAIL",
                f"{type(e).__name__}: {e}"[:240],
            )
    else:
        report.add("sdk", "responses.retrieve", "SKIP", "no stored response id")

    try:
        from openai.types.responses import ResponseCustomToolCall

        tool_resp = client.responses.create(
            model=model,
            input="Call the dj_play tool now.",
            tools=[{"type": "custom", "name": "dj_play", "format": {"type": "text"}}],
            tool_choice={"type": "custom", "name": "dj_play"},
            max_output_tokens=128,
            temperature=0,
            extra_body=extra_body,
        )
        tool_items = [
            item
            for item in (getattr(tool_resp, "output", None) or [])
            if getattr(item, "type", None) == "custom_tool_call"
        ]
        tool_item = tool_items[0] if tool_items else None
        fields_ok = (
            tool_item is not None
            and getattr(tool_item, "name", None) == "dj_play"
            and isinstance(getattr(tool_item, "input", None), str)
            and isinstance(getattr(tool_item, "call_id", None), str)
            and isinstance(getattr(tool_item, "id", None), str)
        )
        typed = isinstance(tool_item, ResponseCustomToolCall)
        detail = (
            f"n={len(tool_items)} typed={typed} name={getattr(tool_item, 'name', None)!r} "
            f"input={str(getattr(tool_item, 'input', None))[:60]!r} "
            f"id={str(getattr(tool_item, 'id', None))[:20]!r}"
        )
        if tool_item is not None and not typed:
            # Item present but the SDK union did not type it; field checks decide the verdict.
            detail += " (fields-only: SDK union missed custom_tool_call)"
        report.add("sdk", "responses.create.custom_tool", "PASS" if fields_ok else "FAIL", detail)
    except ImportError as e:
        report.add(
            "sdk",
            "responses.create.custom_tool",
            "SKIP",
            f"ResponseCustomToolCall missing: {e}"[:200],
        )
    except Exception as e:
        report.add("sdk", "responses.create.custom_tool", "FAIL", f"{type(e).__name__}: {e}"[:240])

    try:
        types: list[str] = []
        with client.responses.stream(
            model=model,
            input="Reply with exactly: SDK_RESP_STREAM",
            max_output_tokens=48,
            temperature=0,
            extra_body=extra_body,
        ) as stream:
            for ev in stream:
                et = getattr(ev, "type", None)
                if isinstance(et, str):
                    types.append(et)
            final = stream.get_final_response()
        text = getattr(final, "output_text", None) or ""
        ok = "response.completed" in types and "SDK_RESP_STREAM" in text
        report.add(
            "sdk",
            "responses.stream",
            "PASS" if ok else "FAIL",
            f"n_events={len(types)} text={text!r}",
        )
    except Exception as e:
        report.add("sdk", "responses.stream", "FAIL", f"{type(e).__name__}: {e}"[:240])

    if hasattr(client.responses, "parse"):
        class _Box(BaseModel):
            ok: bool

        try:
            parsed = client.responses.parse(
                model=model,
                input='Return only JSON: {"ok":true}',
                max_output_tokens=64,
                temperature=0,
                text_format=_Box,
                extra_body=extra_body,
            )
            out = getattr(parsed, "output_parsed", None)
            report.add(
                "sdk",
                "responses.parse",
                "PASS" if out is not None and getattr(out, "ok", None) is True else "FAIL",
                f"parsed={out!r}",
            )
        except TypeError as e:
            report.add("sdk", "responses.parse", "SKIP", f"signature unsupported: {e}"[:200])
        except Exception as e:
            report.add("sdk", "responses.parse", "FAIL", f"{type(e).__name__}: {e}"[:240])
    else:
        report.add("sdk", "responses.parse", "SKIP", "not present in installed openai SDK")

    if hasattr(client.responses, "compact"):
        try:
            prev_id = stored_id
            if not prev_id:
                prev = client.responses.create(
                    model=model,
                    input="compact seed",
                    max_output_tokens=8,
                    store=True,
                    extra_body=extra_body,
                )
                prev_id = getattr(prev, "id", None)
            if prev_id:
                compacted = client.responses.compact(
                    model=model,
                    input="compact me",
                    previous_response_id=prev_id,
                )
                ok = (
                    getattr(compacted, "object", None) == "response.compaction"
                    and isinstance(getattr(compacted, "output", None), list)
                )
                report.add(
                    "sdk",
                    "responses.compact",
                    "PASS" if ok else "FAIL",
                    f"id={getattr(compacted, 'id', None)!r}",
                )
            else:
                report.add("sdk", "responses.compact", "SKIP", "no previous_response_id")
        except Exception as e:
            report.add("sdk", "responses.compact", "FAIL", f"{type(e).__name__}: {e}"[:240])
    else:
        report.add("sdk", "responses.compact", "SKIP", "not present in installed openai SDK")

    if hasattr(client.responses, "input_tokens"):
        try:
            tok = client.responses.input_tokens.count(
                model=model,
                input="token probe for sdk",
            )
            n = getattr(tok, "input_tokens", None)
            report.add(
                "sdk",
                "responses.input_tokens",
                "PASS" if isinstance(n, int) and n > 0 else "FAIL",
                f"input_tokens={n!r}",
            )
        except Exception as e:
            report.add(
                "sdk",
                "responses.input_tokens",
                "FAIL",
                f"{type(e).__name__}: {e}"[:240],
            )

    if hasattr(client.responses, "input_items") and stored_id:
        try:
            page = client.responses.input_items.list(stored_id)
            data = getattr(page, "data", None) or []
            report.add(
                "sdk",
                "responses.input_items",
                "PASS" if isinstance(data, list) else "FAIL",
                f"n={len(data)}",
            )
        except Exception as e:
            report.add(
                "sdk",
                "responses.input_items",
                "FAIL",
                f"{type(e).__name__}: {e}"[:240],
            )
    elif hasattr(client.responses, "input_items"):
        report.add("sdk", "responses.input_items", "SKIP", "no stored response id")

    if hasattr(client.responses, "cancel"):
        try:
            bg = client.responses.create(
                model=model,
                input="Write a long poem about the ocean, at least twenty lines.",
                max_output_tokens=256,
                background=True,
                extra_body=extra_body,
            )
            bg_id = getattr(bg, "id", None)
            if not bg_id:
                report.add("sdk", "responses.cancel", "SKIP", "background create returned no id")
            elif getattr(bg, "status", None) in ("completed", "cancelled", "failed", "incomplete"):
                report.add(
                    "sdk",
                    "responses.cancel",
                    "SKIP",
                    f"background finished before cancel status={getattr(bg, 'status', None)!r}",
                )
            else:
                cancelled = client.responses.cancel(bg_id)
                ok = getattr(cancelled, "status", None) == "cancelled"
                report.add(
                    "sdk",
                    "responses.cancel",
                    "PASS" if ok else "FAIL",
                    f"id={bg_id} status={getattr(cancelled, 'status', None)!r}",
                )
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"[:240]
            if "background" in msg.lower() or "unsupported" in msg.lower():
                report.add("sdk", "responses.cancel", "SKIP", msg)
            else:
                report.add("sdk", "responses.cancel", "FAIL", msg)

    delete_id: str | None = None
    try:
        victim = client.responses.create(
            model=model,
            input="Reply with exactly: SDK_DELETE",
            max_output_tokens=32,
            store=True,
            extra_body=extra_body,
        )
        delete_id = getattr(victim, "id", None)
        if delete_id:
            # openai>=2.54: responses.delete returns None (Accept: */*), not ResponseDeleted
            deleted = client.responses.delete(delete_id)
            ok = deleted is None or getattr(deleted, "deleted", None) is True
            report.add(
                "sdk",
                "responses.delete",
                "PASS" if ok else "FAIL",
                f"id={delete_id} deleted={deleted!r}",
            )
        else:
            report.add("sdk", "responses.delete", "FAIL", "create returned no id")
    except Exception as e:
        report.add("sdk", "responses.delete", "FAIL", f"{type(e).__name__}: {e}"[:240])

    if delete_id:
        try:
            client.responses.retrieve(delete_id)
            report.add(
                "sdk",
                "responses.retrieve_after_delete",
                "FAIL",
                "retrieve succeeded after delete",
            )
        except NotFoundError as e:
            report.add(
                "sdk",
                "responses.retrieve_after_delete",
                "PASS",
                f"NotFoundError: {e}"[:160],
            )
        except Exception as e:
            report.add(
                "sdk",
                "responses.retrieve_after_delete",
                "FAIL",
                f"{type(e).__name__}: {e}"[:240],
            )

    if hasattr(client.responses, "connect"):
        try:
            saw_terminal = False
            terminal = ""
            detail = ""
            # Include reasoning.none: without it, --reasoning-preserve often ends as
            # response.incomplete; SDK recv() has no timeout and blocks forever after.
            send_body: dict[str, Any] = {
                "type": "response.create",
                "model": model,
                "input": "Reply with exactly: SDK_WS_OK",
                "max_output_tokens": 48,
                "temperature": 0,
                "reasoning": {"effort": "none"},
            }
            if extra:
                send_body.update(extra)
            with client.responses.connect() as conn:
                conn.send(send_body)
                for _ in range(200):
                    ev = conn.recv()
                    et = getattr(ev, "type", None)
                    if et == "error":
                        detail = str(ev)[:200]
                        terminal = "error"
                        saw_terminal = True
                        break
                    if et in ("response.completed", "response.incomplete", "response.failed"):
                        saw_terminal = True
                        terminal = et or ""
                        resp_obj = getattr(ev, "response", None)
                        detail = f"id={getattr(resp_obj, 'id', None)} event={et}"
                        break
            ok = saw_terminal and terminal == "response.completed"
            report.add(
                "sdk",
                "responses.connect",
                "PASS" if ok else "FAIL",
                detail or "no terminal WS event",
            )
        except ImportError as e:
            report.add(
                "sdk",
                "responses.connect",
                "SKIP",
                f"websockets dependency missing: {e}"[:200],
            )
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"[:240]
            if "websocket" in msg.lower() or "connect" in msg.lower():
                report.add("sdk", "responses.connect", "SKIP", msg)
            else:
                report.add("sdk", "responses.connect", "FAIL", msg)
    else:
        report.add("sdk", "responses.connect", "SKIP", "not present in installed openai SDK")

    try:
        page = client.models.list()
        ids = [m.id for m in page.data]
        report.add(
            "sdk",
            "models.list",
            "PASS" if model in ids else "FAIL",
            f"ids={ids[:5]}",
        )
    except Exception as e:
        report.add("sdk", "models.list", "FAIL", f"{type(e).__name__}: {e}"[:200])

    try:
        emb = client.embeddings.create(model=model, input="SDK embeddings roundtrip probe")
        vec = emb.data[0].embedding
        ok = (
            emb.object == "list"
            and emb.model == model
            and emb.data[0].object == "embedding"
            and emb.data[0].index == 0
            and isinstance(vec, list)
            and len(vec) > 1
            and all(isinstance(x, float) for x in vec)
            and emb.usage.prompt_tokens > 0
            and emb.usage.total_tokens == emb.usage.prompt_tokens
        )
        report.add(
            "sdk",
            "embeddings.create",
            "PASS" if ok else "FAIL",
            f"n={len(emb.data)} dim={len(vec)} model={emb.model!r} usage={emb.usage!r}",
        )
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"[:240]
        # Server started without --embeddings answers with this message; not a failure.
        if "does not support embeddings" in msg:
            report.add("sdk", "embeddings.create", "SKIP", msg)
        else:
            report.add("sdk", "embeddings.create", "FAIL", msg)

    try:
        # /no_think: raw Completions has no chat template; reasoning models otherwise
        # spend the token budget inside <think> and miss the exact reply.
        cmpl = client.completions.create(
            model=model,
            prompt="/no_think\nReply with exactly: SDK_OAI_CMPL",
            max_tokens=32,
            temperature=0,
            extra_body=extra_body,
        )
        text = "".join(ch.text or "" for ch in (cmpl.choices or []))
        report.add(
            "sdk",
            "completions.create",
            "PASS" if "SDK_OAI_CMPL" in text else "FAIL",
            f"text={text!r}",
        )
    except Exception as e:
        report.add(
            "sdk",
            "completions.create",
            "FAIL",
            f"{type(e).__name__}: {e}"[:240],
        )

    try:
        chunks: list[str] = []
        # Raw Completions continues the prompt (no chat template). Prefer a
        # continuation-friendly prefix so local instruct models emit the marker.
        with client.completions.create(
            model=model,
            # Instruct-style marker (same idea as non-stream SDK_OAI_CMPL). Avoid
            # "The answer is SDK_CMPL_S1" - models often continue S2/S3... without S1.
            prompt="/no_think\nReply with exactly: SDK_CMPL_STREAM",
            max_tokens=32,
            temperature=0,
            stream=True,
            extra_body=extra_body,
        ) as stream:
            for chunk in stream:
                for ch in chunk.choices or []:
                    chunks.append(ch.text or "")
        text = "".join(chunks)
        report.add(
            "sdk",
            "completions.create.stream",
            "PASS" if "SDK_CMPL_STREAM" in text and len(text) > 0 else "FAIL",
            f"text={text!r}",
        )
    except Exception as e:
        report.add(
            "sdk",
            "completions.create.stream",
            "FAIL",
            f"{type(e).__name__}: {e}"[:240],
        )
