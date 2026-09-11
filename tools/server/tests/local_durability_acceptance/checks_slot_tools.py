"""Local Slot KV checks."""

from __future__ import annotations

from .http_client import HttpClient
from .report import Report


def run_slot_and_tools_checks(
    client: HttpClient,
    report: Report,
    model: str,
) -> None:
    # --- Slot KV save / restore / erase ---
    code, props = client.get_json("/props")
    slot_path = props.get("slot_save_path") if isinstance(props, dict) else None
    endpoint_slots = props.get("endpoint_slots") if isinstance(props, dict) else None
    if not (code == 200 and isinstance(slot_path, str) and slot_path and endpoint_slots):
        report.add(
            "discovery",
            "slots_save_restore_erase",
            "SKIP",
            f"slot_save_path/endpoint_slots unavailable path={slot_path!r} endpoint_slots={endpoint_slots!r}",
        )
        return

    fname = "local_durability_slot0.bin"
    # Warm slot 0 with a short chat turn (id_slot is llama-server extension).
    wcode, _ = client.post_json(
        "/v1/chat/completions",
        {
            "model": model,
            "id_slot": 0,
            "cache_prompt": True,
            "max_tokens": 8,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": "You are a slot-kv warm prefix. Remember SLOT_WARM_OK."},
                {"role": "user", "content": "Reply with exactly: WARM"},
            ],
        },
    )
    if wcode != 200:
        report.add(
            "discovery",
            "slots_save_restore_erase",
            "FAIL",
            f"warm HTTP {wcode}",
        )
        return

    scode, saved = client.post_json("/slots/0?action=save", {"filename": fname})
    save_ok = scode == 200 and isinstance(saved, dict) and (
        "n_saved" in saved or "n_written" in saved or "filename" in saved
    )
    rcode, restored = client.post_json("/slots/0?action=restore", {"filename": fname})
    restore_ok = rcode == 200 and isinstance(restored, dict) and (
        "n_restored" in restored or "n_read" in restored or "filename" in restored
    )
    ecode, erased = client.post_json("/slots/0?action=erase", {})
    erase_ok = ecode == 200 and isinstance(erased, dict)

    ok = save_ok and restore_ok and erase_ok
    report.add(
        "discovery",
        "slots_save_restore_erase",
        "PASS" if ok else "FAIL",
        f"save={scode} restore={rcode} erase={ecode} saved={saved if isinstance(saved, dict) else None}",
    )
