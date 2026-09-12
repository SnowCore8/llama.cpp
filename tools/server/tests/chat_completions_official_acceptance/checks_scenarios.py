"""End-to-end Chat Completions scenarios."""

from __future__ import annotations

from typing import Any

from .http_client import ChatHttpClient, choice_text, parse_sse, tool_calls
from .report import Report
from .validators import validate_completion


def _create(
    client: ChatHttpClient,
    model: str,
    extra: dict[str, Any],
    body: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    payload = {
        "model": model,
        "max_tokens": 64,
        "temperature": 0,
        **extra,
        **body,
    }
    if "max_tokens" in body:
        payload["max_tokens"] = body["max_tokens"]
    code, data = client.post_json("/v1/chat/completions", payload)
    return code, data if isinstance(data, dict) else {"_data": data}


def _ok_completion(code: int, data: dict[str, Any]) -> tuple[bool, str]:
    if code != 200:
        return False, f"HTTP {code}"
    ok, detail = validate_completion(data)
    if not ok:
        return False, detail
    return True, "ok"


def run_scenario_checks(
    client: ChatHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    # --- basic user message ---
    code, data = _create(
        client,
        model,
        extra,
        {"messages": [{"role": "user", "content": "Reply with exactly: SCENARIO_BASIC"}]},
    )
    ok, detail = _ok_completion(code, data)
    text = choice_text(data)
    report.add(
        "scenario",
        "basic_user_message",
        "PASS" if ok and "SCENARIO_BASIC" in text else ("PARTIAL" if ok else "FAIL"),
        detail if not ok else f"text={text!r}",
    )

    # --- usage tokens ---
    usage = data.get("usage") if code == 200 else None
    usage_ok = (
        isinstance(usage, dict)
        and isinstance(usage.get("prompt_tokens"), int)
        and isinstance(usage.get("completion_tokens"), int)
        and isinstance(usage.get("total_tokens"), int)
    )
    report.add(
        "scenario",
        "completion_usage_tokens",
        "PASS" if usage_ok else "FAIL",
        repr(usage)[:160],
    )

    # --- system + user messages ---
    code, data = _create(
        client,
        model,
        extra,
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Reply with exactly: SCENARIO_SYS"},
            ]
        },
    )
    ok, detail = _ok_completion(code, data)
    text = choice_text(data)
    report.add(
        "scenario",
        "system_and_user_messages",
        "PASS" if ok and "SCENARIO_SYS" in text else ("PARTIAL" if ok else "FAIL"),
        detail if not ok else f"text={text!r}",
    )

    # --- multi-turn messages array ---
    code, data = _create(
        client,
        model,
        extra,
        {
            "messages": [
                {"role": "user", "content": "The password is ALFA_NINE. Reply OK."},
                {"role": "assistant", "content": "OK"},
                {
                    "role": "user",
                    "content": "What is the password? Reply with only the password.",
                },
            ],
            "max_tokens": 48,
        },
    )
    ok, detail = _ok_completion(code, data)
    text = choice_text(data)
    if not ok:
        report.add("scenario", "multi_turn_messages_replay", "FAIL", detail)
    elif "ALFA_NINE" in text:
        report.add("scenario", "multi_turn_messages_replay", "PASS", f"text={text!r}")
    else:
        report.add(
            "scenario",
            "multi_turn_messages_replay",
            "PARTIAL",
            f"completed but recall missed text={text!r}",
        )

    # --- missing messages error ---
    code, err = client.post_json(
        "/v1/chat/completions",
        {"model": model, "max_tokens": 16, **extra},
    )
    report.add(
        "scenario",
        "missing_messages_error",
        "PASS" if code >= 400 else "FAIL",
        f"HTTP {code} body={str(err)[:120]}",
    )

    # --- missing model error ---
    code, err = client.post_json(
        "/v1/chat/completions",
        {
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            **extra,
        },
    )
    if code >= 400:
        report.add("scenario", "missing_model_error", "PASS", f"HTTP {code}")
    elif code == 200 and isinstance(err, dict) and err.get("model"):
        report.add(
            "scenario",
            "missing_model_error",
            "FAIL",
            f"HTTP 200 defaulted model={err.get('model')!r} (public API requires model)",
        )
    else:
        report.add("scenario", "missing_model_error", "FAIL", f"HTTP {code}")

    # --- forced function tool call ---
    tools_fc = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    code, data = _create(
        client,
        model,
        extra,
        {
            "tools": tools_fc,
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "max_tokens": 128,
        },
    )
    tcs = tool_calls(data) if code == 200 else []
    if code != 200:
        report.add("scenario", "forced_function_tool_call", "FAIL", f"HTTP {code}")
    elif tcs:
        fn = (tcs[0].get("function") or {}).get("name")
        report.add(
            "scenario",
            "forced_function_tool_call",
            "PASS",
            f"name={fn} id={tcs[0].get('id')}",
        )
    else:
        report.add(
            "scenario",
            "forced_function_tool_call",
            "PARTIAL",
            f"no tool_calls finish={(data.get('choices') or [{}])[0].get('finish_reason')!r} "
            f"text={choice_text(data)!r}",
        )

    # --- tool message continuation ---
    if tcs and tcs[0].get("id"):
        tc_id = tcs[0]["id"]
        code2, data2 = _create(
            client,
            model,
            extra,
            {
                "tools": tools_fc,
                "messages": [
                    {"role": "user", "content": "What is the weather in Paris?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tcs,
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": '{"temp_c": 21, "condition": "sunny"}',
                    },
                ],
                "max_tokens": 64,
            },
        )
        ok2, detail2 = _ok_completion(code2, data2)
        text2 = choice_text(data2)
        if not ok2:
            report.add("scenario", "tool_message_continue", "FAIL", detail2)
        elif any(x in text2.lower() for x in ("21", "sunny", "paris")):
            report.add("scenario", "tool_message_continue", "PASS", f"text={text2!r}")
        else:
            report.add(
                "scenario",
                "tool_message_continue",
                "PARTIAL",
                f"completed text={text2!r}",
            )
    else:
        report.add("scenario", "tool_message_continue", "SKIP", "no prior tool_call")

    # --- stream happy path ---
    scode, headers, raw = client.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 48,
            "stream": True,
            "temperature": 0,
            "messages": [{"role": "user", "content": "Reply with exactly: SCEN_STREAM"}],
            **extra,
        },
        stream=True,
    )
    events = parse_sse(raw) if scode == 200 else []
    chunks = [obj for t, obj in events if obj.get("object") == "chat.completion.chunk"]
    saw_done = any(t == "[DONE]" for t, _ in events)
    content = ""
    for c in chunks:
        delta = (c.get("choices") or [{}])[0].get("delta", {})
        if isinstance(delta, dict) and delta.get("content"):
            content += delta["content"]
    ok_stream = (
        scode == 200
        and "text/event-stream" in (headers.get("Content-Type", "") or "")
        and saw_done
        and "SCEN_STREAM" in content
    )
    report.add(
        "scenario",
        "stream_text_completion",
        "PASS" if ok_stream else "FAIL",
        f"HTTP {scode} content={content!r}",
    )
