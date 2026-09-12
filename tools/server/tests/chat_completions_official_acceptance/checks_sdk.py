"""Live OpenAI Python SDK checks (chat.completions / models)."""

from __future__ import annotations

from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from .report import Report
from .validators import validate_completion


def _openai_base(base_url: str) -> str:
    """OpenAI Python SDK expects the versioned root (…/v1)."""
    u = base_url.rstrip("/")
    return u if u.endswith("/v1") else f"{u}/v1"


def run_sdk_checks(
    report: Report,
    *,
    base_url: str,
    api_key: str,
    model: str,
    extra: dict[str, Any],
) -> None:
    client = OpenAI(api_key=api_key, base_url=_openai_base(base_url), timeout=180.0)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: SDK_CHAT_OK"}],
            max_tokens=64,
            temperature=0,
            extra_body=extra or None,
        )
        ok, detail = validate_completion(resp)
        text = resp.choices[0].message.content or "" if resp.choices else ""
        report.add(
            "sdk",
            "chat.completions.create",
            "PASS" if ok and "SDK_CHAT_OK" in text else "FAIL",
            f"finish={getattr(resp.choices[0], 'finish_reason', None)!r} text={text!r} schema={detail}",
        )
    except Exception as e:
        report.add(
            "sdk",
            "chat.completions.create",
            "FAIL",
            f"{type(e).__name__}: {e}"[:240],
        )

    try:
        parts: list[str] = []
        with client.chat.completions.stream(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: SDK_CHAT_STREAM"}],
            max_tokens=48,
            temperature=0,
            extra_body=extra or None,
        ) as stream:
            for ev in stream:
                if getattr(ev, "type", None) == "content.delta":
                    parts.append(ev.delta)
            final = stream.get_final_completion()
        text = "".join(parts)
        fr = final.choices[0].finish_reason if final.choices else None
        ok = fr in ("stop", "length") and "SDK_CHAT_STREAM" in text
        report.add(
            "sdk",
            "chat.completions.stream",
            "PASS" if ok else "FAIL",
            f"finish={fr!r} text={text!r}",
        )
    except Exception as e:
        report.add(
            "sdk",
            "chat.completions.stream",
            "FAIL",
            f"{type(e).__name__}: {e}"[:240],
        )

    if hasattr(client.chat.completions, "parse"):
        class _Box(BaseModel):
            ok: bool

        try:
            parsed = client.chat.completions.parse(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": 'Output exactly {"ok":true} with no markdown fences or prose.',
                    }
                ],
                max_tokens=64,
                temperature=0,
                response_format=_Box,
                extra_body=extra or None,
            )
            msg = parsed.choices[0].message if parsed.choices else None
            out = getattr(msg, "parsed", None) if msg else None
            report.add(
                "sdk",
                "chat.completions.parse",
                "PASS" if out is not None and getattr(out, "ok", None) is True else "FAIL",
                f"parsed={out!r}",
            )
        except TypeError as e:
            report.add(
                "sdk",
                "chat.completions.parse",
                "SKIP",
                f"signature unsupported: {e}"[:200],
            )
        except Exception as e:
            report.add(
                "sdk",
                "chat.completions.parse",
                "FAIL",
                f"{type(e).__name__}: {e}"[:240],
            )
    else:
        report.add(
            "sdk",
            "chat.completions.parse",
            "SKIP",
            "not present in installed openai SDK",
        )

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

    # Stored Chat Completions CRUD (store=true)
    stored_id: str | None = None
    try:
        stored = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: SDK_STORE_OK"}],
            max_tokens=32,
            temperature=0,
            store=True,
            metadata={"suite": "sdk", "k": "v"},
            extra_body=extra or None,
        )
        stored_id = stored.id
        text = stored.choices[0].message.content or "" if stored.choices else ""
        report.add(
            "sdk",
            "chat.completions.create.store",
            "PASS" if stored_id and "SDK_STORE_OK" in text else "FAIL",
            f"id={stored_id!r} text={text!r}",
        )
    except Exception as e:
        report.add(
            "sdk",
            "chat.completions.create.store",
            "FAIL",
            f"{type(e).__name__}: {e}"[:240],
        )

    if stored_id:
        try:
            got = client.chat.completions.retrieve(stored_id)
            text = got.choices[0].message.content or "" if got.choices else ""
            report.add(
                "sdk",
                "chat.completions.retrieve",
                "PASS" if got.id == stored_id and "SDK_STORE_OK" in text else "FAIL",
                f"id={got.id!r} text={text!r}",
            )
        except Exception as e:
            report.add(
                "sdk",
                "chat.completions.retrieve",
                "FAIL",
                f"{type(e).__name__}: {e}"[:240],
            )

        try:
            # list defaults to asc; the entry we just stored is the newest, so ask desc
            page = client.chat.completions.list(limit=5, order="desc")
            ids = [c.id for c in page.data]
            report.add(
                "sdk",
                "chat.completions.list",
                "PASS" if stored_id in ids else "FAIL",
                f"ids={ids[:5]}",
            )
        except Exception as e:
            report.add(
                "sdk",
                "chat.completions.list",
                "FAIL",
                f"{type(e).__name__}: {e}"[:240],
            )

        try:
            updated = client.chat.completions.update(
                stored_id, metadata={"suite": "sdk", "k": "updated"}
            )
            md = getattr(updated, "metadata", None) or {}
            ok = (md.get("k") if isinstance(md, dict) else None) == "updated"
            report.add(
                "sdk",
                "chat.completions.update",
                "PASS" if ok else "FAIL",
                f"metadata={md!r}",
            )
        except Exception as e:
            report.add(
                "sdk",
                "chat.completions.update",
                "FAIL",
                f"{type(e).__name__}: {e}"[:240],
            )

        try:
            deleted = client.chat.completions.delete(stored_id)
            ok = getattr(deleted, "deleted", None) is True
            report.add(
                "sdk",
                "chat.completions.delete",
                "PASS" if ok else "FAIL",
                f"deleted={deleted!r}",
            )
        except Exception as e:
            report.add(
                "sdk",
                "chat.completions.delete",
                "FAIL",
                f"{type(e).__name__}: {e}"[:240],
            )

        try:
            client.chat.completions.retrieve(stored_id)
            report.add(
                "sdk",
                "chat.completions.retrieve_after_delete",
                "FAIL",
                "retrieve succeeded after delete",
            )
        except Exception as e:
            report.add(
                "sdk",
                "chat.completions.retrieve_after_delete",
                "PASS" if "404" in str(e) or "NotFound" in type(e).__name__ else "FAIL",
                f"{type(e).__name__}: {e}"[:160],
            )
    else:
        for name in (
            "chat.completions.retrieve",
            "chat.completions.list",
            "chat.completions.update",
            "chat.completions.delete",
            "chat.completions.retrieve_after_delete",
        ):
            report.add("sdk", name, "SKIP", "no stored id")
