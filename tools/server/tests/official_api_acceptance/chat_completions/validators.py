"""SDK schema validators for ChatCompletion / stream chunks."""

from __future__ import annotations

from typing import Any

from openai.types.chat import ChatCompletion, ChatCompletionChunk


def validate_completion(obj: Any) -> tuple[bool, str]:
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    elif not isinstance(obj, dict):
        return False, f"expected dict or ChatCompletion, got {type(obj).__name__}"
    required = [k for k, v in ChatCompletion.model_fields.items() if v.is_required()]
    missing = [k for k in required if k not in obj]
    try:
        ChatCompletion.model_validate(obj)
        sdk_ok, sdk_err = True, ""
    except Exception as e:
        sdk_ok, sdk_err = False, str(e)
    if missing or not sdk_ok:
        return False, f"missing={missing}; sdk={sdk_err[:240]}"
    return True, "ok"


def validate_chunk(obj: dict[str, Any]) -> tuple[bool, str]:
    try:
        ChatCompletionChunk.model_validate(obj)
        return True, "ok"
    except Exception as e:
        return False, str(e)[:240]
