"""Simple end-to-end scenarios against the public Responses API.

These are behavioral checks (not field-by-field catalog probes). Normative
behavior follows OpenAI Responses docs + SDK shapes.
"""

from __future__ import annotations

from typing import Any

from .http_client import ResponsesHttpClient, output_text, parse_sse
from .report import Report
from .validators import validate_response


def _create(
    client: ResponsesHttpClient,
    model: str,
    extra: dict[str, Any],
    body: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    payload = {"model": model, "max_output_tokens": 64, **extra, **body}
    # body keys win over defaults for max_output_tokens if provided
    if "max_output_tokens" in body:
        payload["max_output_tokens"] = body["max_output_tokens"]
    code, data = client.post_json("/v1/responses", payload)
    return code, data if isinstance(data, dict) else {"_data": data}


def _ok_completed(code: int, data: dict[str, Any]) -> tuple[bool, str]:
    if code != 200:
        return False, f"HTTP {code}"
    ok, detail = validate_response(data)
    if not ok:
        return False, detail
    if data.get("status") != "completed":
        return False, f"status={data.get('status')!r}"
    return True, "ok"


def _function_calls(data: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in data.get("output") or []:
        if isinstance(item, dict) and item.get("type") == "function_call":
            out.append(item)
    return out


def run_scenario_checks(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    # --- 1) basic string input ---
    code, data = _create(
        client,
        model,
        extra,
        {"input": "Reply with exactly: SCENARIO_BASIC", "temperature": 0},
    )
    ok, detail = _ok_completed(code, data)
    text = output_text(data)
    if ok and "SCENARIO_BASIC" in text:
        report.add("scenario", "basic_string_input", "PASS", f"id={data.get('id')}")
    elif ok:
        report.add(
            "scenario",
            "basic_string_input",
            "PARTIAL",
            f"completed but text={text!r}",
        )
    else:
        report.add("scenario", "basic_string_input", "FAIL", detail)

    # --- 2) completed response carries usage ---
    usage = data.get("usage") if code == 200 else None
    usage_ok = (
        isinstance(usage, dict)
        and isinstance(usage.get("input_tokens"), int)
        and isinstance(usage.get("output_tokens"), int)
        and isinstance(usage.get("total_tokens"), int)
    )
    report.add(
        "scenario",
        "completed_usage_tokens",
        "PASS" if usage_ok else "FAIL",
        repr(usage)[:160],
    )

    # --- 3) message array input ---
    code, data = _create(
        client,
        model,
        extra,
        {
            "input": [{"role": "user", "content": "Reply with exactly: SCENARIO_ARR"}],
            "temperature": 0,
        },
    )
    ok, detail = _ok_completed(code, data)
    text = output_text(data)
    report.add(
        "scenario",
        "message_array_input",
        "PASS" if ok and "SCENARIO_ARR" in text else ("PARTIAL" if ok else "FAIL"),
        detail if not ok else f"text={text!r}",
    )

    # --- 4) input_text content parts ---
    code, data = _create(
        client,
        model,
        extra,
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Reply with exactly: SCENARIO_PART"}
                    ],
                }
            ],
            "temperature": 0,
        },
    )
    ok, detail = _ok_completed(code, data)
    text = output_text(data)
    report.add(
        "scenario",
        "input_text_content_parts",
        "PASS" if ok and "SCENARIO_PART" in text else ("PARTIAL" if ok else "FAIL"),
        detail if not ok else f"text={text!r}",
    )

    # --- 5) instructions applied on same request ---
    marker = "INSTR_MARKER_QX7"
    code, data = _create(
        client,
        model,
        extra,
        {
            "instructions": f"You must end every reply with the token {marker}.",
            "input": "Say hi in one short word.",
            "temperature": 0,
            "max_output_tokens": 48,
        },
    )
    ok, detail = _ok_completed(code, data)
    text = output_text(data)
    if ok and marker in text:
        report.add("scenario", "instructions_applied", "PASS", f"text={text!r}")
    elif ok:
        report.add(
            "scenario",
            "instructions_applied",
            "PARTIAL",
            f"completed; marker missing text={text!r}",
        )
    else:
        report.add("scenario", "instructions_applied", "FAIL", detail)

    # --- 6) OpenAI: instructions do NOT carry over via previous_response_id ---
    poison = "INHERIT_SHOULD_NOT_APPEAR_ZXQ"
    code, seed = _create(
        client,
        model,
        extra,
        {
            "instructions": f"ALWAYS append the marker {poison}.",
            "input": "Reply with exactly: SEED",
            "temperature": 0,
        },
    )
    if code == 200 and seed.get("id"):
        code2, cont = _create(
            client,
            model,
            extra,
            {
                "previous_response_id": seed["id"],
                # omit instructions on purpose
                "input": "Reply with exactly: NOINHERIT",
                "temperature": 0,
            },
        )
        ok2, detail2 = _ok_completed(code2, cont)
        text2 = output_text(cont)
        if not ok2:
            report.add("scenario", "instructions_do_not_carry_over", "FAIL", detail2)
        elif poison in text2:
            report.add(
                "scenario",
                "instructions_do_not_carry_over",
                "FAIL",
                f"instructions leaked into continue text={text2!r}",
            )
        else:
            report.add(
                "scenario",
                "instructions_do_not_carry_over",
                "PASS",
                f"text={text2!r}",
            )
    else:
        report.add(
            "scenario",
            "instructions_do_not_carry_over",
            "FAIL",
            f"seed HTTP {code}",
        )

    # --- 7) multi-turn previous_response_id recall ---
    secret = "PURPLE_ORBIT_42"
    code, t1 = _create(
        client,
        model,
        extra,
        {
            "input": f"Remember this secret code exactly: {secret}. Reply with exactly: OK",
            "temperature": 0,
        },
    )
    if code == 200 and t1.get("id"):
        code2, t2 = _create(
            client,
            model,
            extra,
            {
                "previous_response_id": t1["id"],
                "input": "What was the secret code? Reply with only the code.",
                "temperature": 0,
                "max_output_tokens": 48,
            },
        )
        ok2, detail2 = _ok_completed(code2, t2)
        text2 = output_text(t2)
        if not ok2:
            report.add("scenario", "multi_turn_previous_response_id", "FAIL", detail2)
        elif secret in text2:
            report.add(
                "scenario",
                "multi_turn_previous_response_id",
                "PASS",
                f"text={text2!r}",
            )
        else:
            report.add(
                "scenario",
                "multi_turn_previous_response_id",
                "PARTIAL",
                f"chain ok but recall missed text={text2!r}",
            )
    else:
        report.add(
            "scenario",
            "multi_turn_previous_response_id",
            "FAIL",
            f"turn1 HTTP {code}",
        )

    # --- 8) retrieve matches create ---
    code, created = _create(
        client,
        model,
        extra,
        {"input": "Reply with exactly: RETRIEVE_ME", "temperature": 0},
    )
    if code == 200 and created.get("id"):
        rid = created["id"]
        gcode, got = client.get_json(f"/v1/responses/{rid}")
        if gcode == 404:
            report.add(
                "scenario",
                "retrieve_matches_create",
                "NOT_IMPLEMENTED",
                "GET retrieve 404",
            )
        elif gcode != 200:
            report.add(
                "scenario",
                "retrieve_matches_create",
                "FAIL",
                f"HTTP {gcode}",
            )
        else:
            same = (
                got.get("id") == rid
                and output_text(got) == output_text(created)
                and got.get("status") == created.get("status")
            )
            report.add(
                "scenario",
                "retrieve_matches_create",
                "PASS" if same else "FAIL",
                f"created_text={output_text(created)!r} got_text={output_text(got)!r}",
            )
    else:
        report.add("scenario", "retrieve_matches_create", "FAIL", f"create HTTP {code}")

    # --- 9) store=false cannot continue ---
    code, ns = _create(
        client,
        model,
        extra,
        {
            "store": False,
            "input": "Reply with exactly: NOSTORE_SCEN",
            "temperature": 0,
        },
    )
    if code == 200 and ns.get("id"):
        code2, err = _create(
            client,
            model,
            extra,
            {"previous_response_id": ns["id"], "input": "x"},
        )
        report.add(
            "scenario",
            "store_false_blocks_previous",
            "PASS" if code2 == 400 else "FAIL",
            f"continue HTTP {code2} body={str(err)[:120]}",
        )
    else:
        report.add("scenario", "store_false_blocks_previous", "FAIL", f"HTTP {code}")

    # --- 10) missing previous_response_id → 400 ---
    code, err = _create(
        client,
        model,
        extra,
        {"previous_response_id": "resp_does_not_exist_zzz", "input": "x"},
    )
    report.add(
        "scenario",
        "missing_previous_response_id_400",
        "PASS" if code == 400 else "FAIL",
        f"HTTP {code}",
    )

    # --- 11) missing model → error (OpenAI: model is required on create) ---
    payload = {"input": "hi", "max_output_tokens": 16, **extra}
    code, err = client.post_json("/v1/responses", payload)
    if code >= 400:
        report.add("scenario", "missing_model_error", "PASS", f"HTTP {code}")
    elif code == 200 and isinstance(err, dict) and err.get("model"):
        # Some local servers default model; still non-compliant vs public API.
        report.add(
            "scenario",
            "missing_model_error",
            "FAIL",
            f"HTTP 200 defaulted model={err.get('model')!r} (public API requires model)",
        )
    else:
        report.add("scenario", "missing_model_error", "FAIL", f"HTTP {code}")

    # --- 12) input-array multi-turn replay (official message items, no previous_id) ---
    code, data = _create(
        client,
        model,
        extra,
        {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "The password is ALFA_NINE. Reply OK.",
                        }
                    ],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "OK"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "What is the password? Reply with only the password.",
                        }
                    ],
                },
            ],
            "temperature": 0,
            "max_output_tokens": 48,
        },
    )
    ok, detail = _ok_completed(code, data)
    text = output_text(data)
    if not ok:
        report.add("scenario", "input_array_multi_turn_replay", "FAIL", detail)
    elif "ALFA_NINE" in text:
        report.add(
            "scenario",
            "input_array_multi_turn_replay",
            "PASS",
            f"text={text!r}",
        )
    else:
        report.add(
            "scenario",
            "input_array_multi_turn_replay",
            "PARTIAL",
            f"completed but recall missed text={text!r}",
        )

    # --- 13) forced function tool call ---
    code, data = _create(
        client,
        model,
        extra,
        {
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            "tool_choice": {"type": "function", "name": "get_weather"},
            "input": "What is the weather in Paris?",
            "temperature": 0,
            "max_output_tokens": 128,
        },
    )
    fcs = _function_calls(data) if code == 200 else []
    if code != 200:
        report.add("scenario", "forced_function_tool_call", "FAIL", f"HTTP {code}")
    elif fcs:
        fc = fcs[0]
        report.add(
            "scenario",
            "forced_function_tool_call",
            "PASS",
            f"name={fc.get('name')} call_id={fc.get('call_id')}",
        )
    else:
        report.add(
            "scenario",
            "forced_function_tool_call",
            "PARTIAL",
            f"no function_call in output status={data.get('status')} text={output_text(data)!r}",
        )

    # --- 14) function_call_output continuation ---
    if fcs and fcs[0].get("call_id"):
        call_id = fcs[0]["call_id"]
        code2, data2 = _create(
            client,
            model,
            extra,
            {
                "previous_response_id": data.get("id"),
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": '{"temp_c": 21, "condition": "sunny"}',
                    }
                ],
                "temperature": 0,
                "max_output_tokens": 64,
            },
        )
        ok2, detail2 = _ok_completed(code2, data2)
        text2 = output_text(data2)
        if not ok2:
            report.add("scenario", "function_call_output_continue", "FAIL", detail2)
        elif any(x in text2.lower() for x in ("21", "sunny", "paris")):
            report.add(
                "scenario",
                "function_call_output_continue",
                "PASS",
                f"text={text2!r}",
            )
        else:
            report.add(
                "scenario",
                "function_call_output_continue",
                "PARTIAL",
                f"completed text={text2!r}",
            )
    else:
        report.add(
            "scenario",
            "function_call_output_continue",
            "SKIP",
            "no prior function_call",
        )

    # --- 15) stream + previous_response_id ---
    code, seed = _create(
        client,
        model,
        extra,
        {"input": "Reply with exactly: STREAM_SEED", "temperature": 0},
    )
    if code == 200 and seed.get("id"):
        scode, headers, raw = client.request(
            "POST",
            "/v1/responses",
            {
                "model": model,
                "previous_response_id": seed["id"],
                "input": "Reply with exactly: STREAM_CONT",
                "max_output_tokens": 48,
                "stream": True,
                "temperature": 0,
                **extra,
            },
            stream=True,
        )
        events = parse_sse(raw) if scode == 200 else []
        types = [t for t, _ in events]
        ok_stream = (
            scode == 200
            and "text/event-stream" in headers.get("Content-Type", "")
            and "response.completed" in types
        )
        report.add(
            "scenario",
            "stream_with_previous_response_id",
            "PASS" if ok_stream else "FAIL",
            f"HTTP {scode} events={len(events)}",
        )
    else:
        report.add(
            "scenario",
            "stream_with_previous_response_id",
            "FAIL",
            f"seed HTTP {code}",
        )

    # --- 16) delete then previous_response_id fails ---
    code, victim = _create(
        client,
        model,
        extra,
        {"input": "Reply with exactly: DEL_SCEN", "temperature": 0},
    )
    if code == 200 and victim.get("id"):
        did = victim["id"]
        dcode, deleted = client.delete_json(f"/v1/responses/{did}")
        if dcode == 404:
            report.add(
                "scenario",
                "delete_then_previous_fails",
                "NOT_IMPLEMENTED",
                "DELETE 404",
            )
        else:
            ccode, _ = _create(
                client,
                model,
                extra,
                {"previous_response_id": did, "input": "x"},
            )
            report.add(
                "scenario",
                "delete_then_previous_fails",
                "PASS" if ccode == 400 else "FAIL",
                f"delete HTTP {dcode} continue HTTP {ccode} deleted={deleted!r}"[:160],
            )
    else:
        report.add("scenario", "delete_then_previous_fails", "FAIL", f"create HTTP {code}")
