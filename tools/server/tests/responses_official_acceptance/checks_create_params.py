"""Official create-parameter probes (behavior where observable, else retrieve round-trip)."""

from __future__ import annotations

import json
from typing import Any

from .catalog import create_param_names
from .http_client import ResponsesHttpClient, output_text
from .report import Report


def run_create_param_checks(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
    seed_response_id: str | None,
) -> None:
    def create(extra_fields: dict[str, Any], input_text: str = "Reply with exactly: P") -> tuple[int, dict]:
        body = {
            "model": model,
            "input": input_text,
            "max_output_tokens": 48,
            # Prefer no reasoning so exact-text probes stay observable under --reasoning-preserve.
            "reasoning": {"effort": "none"},
            **extra,
            **extra_fields,
        }
        code, data = client.post_json("/v1/responses", body)
        return code, data if isinstance(data, dict) else {"_data": data}

    # --- model: required; missing must 400; present echoed ---
    code_missing, err = client.post_json(
        "/v1/responses",
        {"input": "x", "max_output_tokens": 8, **extra},
    )
    code_ok, data_ok = create({})
    model_ok = (
        code_missing >= 400
        and code_ok == 200
        and data_ok.get("model") == model
    )
    report.add(
        "create_param",
        "model",
        "PASS" if model_ok else "FAIL",
        f"missing_http={code_missing} create_model={data_ok.get('model')!r}",
    )

    code, data = create({})
    report.add("create_param", "input", "PASS" if code == 200 else "FAIL", f"HTTP {code}")

    code, data = client.post_json(
        "/v1/responses",
        {
            "model": model,
            "input": [{"role": "user", "content": "Reply with exactly: ARR"}],
            "max_output_tokens": 32,
            **extra,
        },
    )
    report.add(
        "create_param",
        "input.array_easy_message",
        "PASS" if code == 200 and "ARR" in output_text(data if isinstance(data, dict) else {}) else "FAIL",
        f"HTTP {code}",
    )

    code, data = client.post_json(
        "/v1/responses",
        {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Reply with exactly: PART"}],
                }
            ],
            "max_output_tokens": 32,
            **extra,
        },
    )
    report.add(
        "create_param",
        "input.input_text_parts",
        "PASS" if code == 200 and "PART" in output_text(data if isinstance(data, dict) else {}) else "FAIL",
        f"HTTP {code}",
    )

    # --- input: assistant message phase labels are accepted and round-trip ---
    code, data = create(
        {
            "input": [
                {"role": "user", "content": "Reply with exactly: PH1"},
                {"role": "assistant", "content": "PH1", "phase": "commentary"},
                {"role": "user", "content": "Reply with exactly: PH2"},
            ],
            "store": True,
            "max_output_tokens": 32,
        }
    )
    phase_kept = False
    if code == 200 and data.get("id"):
        icode, items = client.get_json(f"/v1/responses/{data['id']}/input_items")
        if icode == 200 and isinstance(items, dict):
            phase_kept = any(
                isinstance(x, dict) and x.get("phase") == "commentary"
                for x in (items.get("data") or [])
            )
    report.add(
        "create_param",
        "input.assistant_phase_roundtrip",
        "PASS" if code == 200 and phase_kept else "FAIL",
        f"HTTP {code} phase_kept={phase_kept}",
    )
    if code == 200 and data.get("id"):
        code2, data2 = create(
            {
                "previous_response_id": data["id"],
                "input": "Reply with exactly: PH3",
                "max_output_tokens": 32,
            }
        )
        report.add(
            "create_param",
            "input.assistant_phase_previous",
            "PASS" if code2 == 200 and "PH3" in output_text(data2) else "FAIL",
            f"HTTP {code2}",
        )
    else:
        report.add("create_param", "input.assistant_phase_previous", "SKIP", "no stored id")

    code_bad, _ = create(
        {
            "input": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "x", "phase": "bogus"},
                {"role": "user", "content": "again"},
            ]
        }
    )
    report.add(
        "create_param",
        "input.phase.invalid",
        "PASS" if code_bad == 400 else "FAIL",
        f"HTTP {code_bad}",
    )

    code, data = create(
        {
            "instructions": "Your entire reply must be exactly: INSTR_CREATE_OK",
            "input": "Begin.",
            "temperature": 0,
            "max_output_tokens": 16,
        }
    )
    instr_ok = code == 200 and "INSTR_CREATE_OK" in output_text(data)
    report.add(
        "create_param",
        "instructions",
        "PASS" if instr_ok else "FAIL",
        f"HTTP {code} text={output_text(data)!r}",
    )

    if seed_response_id:
        code, data = create({"previous_response_id": seed_response_id})
        report.add(
            "create_param",
            "previous_response_id",
            "PASS" if code == 200 else "FAIL",
            f"HTTP {code}",
        )
    else:
        report.add("create_param", "previous_response_id", "SKIP", "no seed id")

    code, data = create({"previous_response_id": "resp_missing_official"})
    report.add(
        "create_param",
        "previous_response_id.missing",
        "PASS" if code == 400 else "FAIL",
        f"HTTP {code}",
    )

    # store semantics (OpenAI: store=false must not be usable as previous)
    code, data = create({"store": True})
    report.add(
        "create_param",
        "store",
        "PASS" if code == 200 and data.get("store") is True else "FAIL",
        f"HTTP {code} store={data.get('store')!r}",
    )
    code, data = create({"store": False, "input": "Reply with exactly: NOSTORE"})
    rid_ns = data.get("id") if code == 200 else None
    report.add(
        "create_param",
        "store.false",
        "PASS" if code == 200 and data.get("store") is False else "FAIL",
        f"HTTP {code}",
    )
    if rid_ns:
        code2, _ = create({"previous_response_id": rid_ns, "input": "x"})
        report.add(
            "create_param",
            "store.false_blocks_previous",
            "PASS" if code2 == 400 else "FAIL",
            f"HTTP {code2}",
        )

    # temperature: echo + temp=0 determinism (two identical creates)
    outs = []
    temp_echo = None
    for _ in range(2):
        code, data = create(
            {"temperature": 0, "input": "Reply with exactly: TEMP0_DET", "max_output_tokens": 16}
        )
        temp_echo = data.get("temperature")
        outs.append(output_text(data))
    temp_ok = code == 200 and temp_echo == 0 and len(outs) == 2 and outs[0] == outs[1] and "TEMP0_DET" in outs[0]
    report.add(
        "create_param",
        "temperature",
        "PASS" if temp_ok else "FAIL",
        f"echoed={temp_echo!r} outs={outs!r}",
    )

    # top_p: echo + retrieve round-trip
    code, data = create({"top_p": 0.5, "input": "Reply with exactly: TOPP"})
    rid = data.get("id")
    gcode, got = client.get_json(f"/v1/responses/{rid}") if rid else (0, {})
    topp_ok = (
        code == 200
        and data.get("top_p") == 0.5
        and gcode == 200
        and isinstance(got, dict)
        and got.get("top_p") == 0.5
    )
    report.add(
        "create_param",
        "top_p",
        "PASS" if topp_ok else "FAIL",
        f"HTTP {code}/{gcode} create={data.get('top_p')!r} get={got.get('top_p') if isinstance(got, dict) else None!r}",
    )

    # max_output_tokens: echo + usage.output_tokens <= max + incomplete on tight cap
    code, data = create(
        {
            "max_output_tokens": 4,
            "temperature": 0,
            "input": "Write a long detailed essay about the history of mathematics.",
        }
    )
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    out_tok = int(usage.get("output_tokens") or 0)
    max_ok = (
        code == 200
        and data.get("max_output_tokens") == 4
        and out_tok <= 4
        and data.get("status") == "incomplete"
        and (data.get("incomplete_details") or {}).get("reason") == "max_output_tokens"
    )
    report.add(
        "create_param",
        "max_output_tokens",
        "PASS" if max_ok else "FAIL",
        f"status={data.get('status')!r} out_tok={out_tok} details={data.get('incomplete_details')!r}",
    )

    code, data = create({"top_logprobs": 2, "include": ["message.output_text.logprobs"], "input": "Reply with exactly: TLP"})
    has_lp = False
    for item in data.get("output") or []:
        for part in item.get("content") or []:
            if isinstance(part.get("logprobs"), list) and part["logprobs"]:
                has_lp = True
    report.add(
        "create_param",
        "top_logprobs",
        "PASS" if code == 200 and has_lp else "FAIL",
        f"HTTP {code} logprobs_nonempty={has_lp}",
    )

    # tools triad (required on Response object)
    code, data = create(
        {
            "tools": [
                {
                    "type": "function",
                    "name": "echo",
                    "description": "echo",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "input": "Reply with exactly: TOOLS",
        }
    )
    ok = (
        code == 200
        and data.get("tool_choice") == "auto"
        and data.get("parallel_tool_calls") is True
        and isinstance(data.get("tools"), list)
    )
    report.add("create_param", "tools", "PASS" if ok else "FAIL", f"HTTP {code}")
    report.add("create_param", "tool_choice", "PASS" if ok else "FAIL", f"HTTP {code} (auto)")
    report.add(
        "create_param",
        "parallel_tool_calls",
        "PASS" if ok else "FAIL",
        f"HTTP {code}",
    )

    # Local web_search deepen: emit web_search_call; include sources; message url_citation.
    code_ws, data_ws = create(
        {
            "tools": [
                {
                    "type": "web_search",
                    "search_context_size": "medium",
                    "filters": {"allowed_domains": ["example.com", "iana.org", "rfc-editor.org"]},
                }
            ],
            "include": ["web_search_call.action.sources"],
            "reasoning": {"effort": "none"},
            "input": "What is example.com used for? Reply briefly.",
            "max_output_tokens": 128,
        }
    )
    ws_calls = [
        o
        for o in (data_ws.get("output") or [])
        if isinstance(o, dict) and o.get("type") == "web_search_call"
    ]
    sources_ok = False
    if ws_calls:
        action = ws_calls[0].get("action") or {}
        sources = action.get("sources") or []
        sources_ok = isinstance(sources, list) and len(sources) >= 1
    cites = []
    for o in data_ws.get("output") or []:
        if not isinstance(o, dict) or o.get("type") != "message":
            continue
        for part in o.get("content") or []:
            if isinstance(part, dict):
                cites.extend(part.get("annotations") or [])
    cite_ok = any(isinstance(a, dict) and a.get("type") == "url_citation" for a in cites)
    report.add(
        "create_param",
        "tools.web_search.local",
        "PASS" if code_ws == 200 and ws_calls and sources_ok and cite_ok else "FAIL",
        f"HTTP {code_ws} ws_calls={len(ws_calls)} sources={sources_ok} cites={cite_ok} body={str(data_ws)[:200]}",
    )

    code_mcp, data_mcp = create(
        {
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "example",
                    "server_url": "https://example.invalid/mcp",
                }
            ],
            "input": "Reply with exactly: MCP",
        }
    )
    report.add(
        "create_param",
        "tools.mcp.reject",
        "PASS" if code_mcp == 400 else "FAIL",
        f"HTTP {code_mcp} body={str(data_mcp)[:160]}",
    )

    # allowed_tools constraint: web_search only stays when listed in the allowed set.
    # Each row needs a control run (same payload minus tool_choice) proving the model
    # actually searches for this prompt; otherwise the row is not judgeable -> SKIP.
    ws_prompt = "What is example.com used for? Reply briefly."
    ws_echo_tool = {
        "type": "function",
        "name": "echo_tool",
        "description": "echo",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
    }

    def ws_control(tools: list[dict[str, Any]]) -> tuple[bool, str]:
        ccode, cdata = create(
            {
                "tools": tools,
                "reasoning": {"effort": "none"},
                "input": ws_prompt,
                "max_output_tokens": 96,
            }
        )
        calls = [
            o
            for o in (cdata.get("output") or [])
            if isinstance(o, dict) and o.get("type") == "web_search_call"
        ]
        return ccode == 200 and bool(calls), f"control HTTP {ccode} ws_calls={len(calls)}"

    code_deny, data_deny = create(
        {
            "tools": [{"type": "web_search", "search_context_size": "medium"}, ws_echo_tool],
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": "function", "name": "echo_tool"}],
            },
            "reasoning": {"effort": "none"},
            "input": ws_prompt,
            "max_output_tokens": 96,
        }
    )
    ws_deny = [
        o
        for o in (data_deny.get("output") or [])
        if isinstance(o, dict) and o.get("type") == "web_search_call"
    ]
    ctrl_deny_ok, ctrl_deny_detail = ws_control(
        [{"type": "web_search", "search_context_size": "medium"}, ws_echo_tool]
    )
    if not ctrl_deny_ok:
        report.add(
            "create_param",
            "tools.web_search.denied_by_allowed_tools",
            "SKIP",
            f"control run saw no web_search_call; not judgeable ({ctrl_deny_detail})",
        )
    else:
        report.add(
            "create_param",
            "tools.web_search.denied_by_allowed_tools",
            "PASS" if code_deny == 200 and not ws_deny else "FAIL",
            f"HTTP {code_deny} ws_calls={len(ws_deny)}; basis={ctrl_deny_detail}",
        )

    code_allow, data_allow = create(
        {
            "tools": [{"type": "web_search", "search_context_size": "medium"}],
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": "web_search"}],
            },
            "reasoning": {"effort": "none"},
            "input": ws_prompt,
            "max_output_tokens": 96,
        }
    )
    ws_allow = [
        o
        for o in (data_allow.get("output") or [])
        if isinstance(o, dict) and o.get("type") == "web_search_call"
    ]
    ctrl_allow_ok, ctrl_allow_detail = ws_control([{"type": "web_search", "search_context_size": "medium"}])
    if not ctrl_allow_ok:
        report.add(
            "create_param",
            "tools.web_search.allowed_by_allowed_tools",
            "SKIP",
            f"control run saw no web_search_call; not judgeable ({ctrl_allow_detail})",
        )
    else:
        report.add(
            "create_param",
            "tools.web_search.allowed_by_allowed_tools",
            "PASS" if code_allow == 200 and ws_allow else "FAIL",
            f"HTTP {code_allow} ws_calls={len(ws_allow)}; basis={ctrl_allow_detail}",
        )

    tools_fc = [
        {
            "type": "function",
            "name": "echo_tool",
            "description": "echo",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
        }
    ]
    code, data = create(
        {
            "tools": tools_fc,
            "tool_choice": "required",
            "input": "Use a tool now.",
            "max_output_tokens": 128,
        }
    )
    fcs = [
        x
        for x in (data.get("output") or [])
        if isinstance(x, dict) and x.get("type") == "function_call"
    ]
    report.add(
        "create_param",
        "tool_choice.required",
        "PASS" if code == 200 and fcs else "FAIL",
        f"HTTP {code} n_fc={len(fcs)} status={data.get('status')!r}",
    )

    code, data = create(
        {
            "tools": tools_fc,
            "tool_choice": {"type": "function", "name": "echo_tool"},
            "input": "call echo_tool with x=hi",
            "max_output_tokens": 128,
        }
    )
    fcs = [
        x
        for x in (data.get("output") or [])
        if isinstance(x, dict) and x.get("type") == "function_call"
    ]
    report.add(
        "create_param",
        "tool_choice.function",
        "PASS"
        if code == 200 and fcs and fcs[0].get("name") == "echo_tool"
        else "FAIL",
        f"HTTP {code} n_fc={len(fcs)}",
    )

    code, data = create(
        {
            "tools": tools_fc,
            "tool_choice": "none",
            "input": "Reply with exactly: NONE_OK and do not call tools.",
            "max_output_tokens": 64,
        }
    )
    fcs = [
        x
        for x in (data.get("output") or [])
        if isinstance(x, dict) and x.get("type") == "function_call"
    ]
    report.add(
        "create_param",
        "tool_choice.none",
        "PASS" if code == 200 and not fcs and "NONE_OK" in output_text(data) else "FAIL",
        f"HTTP {code} n_fc={len(fcs)} text={output_text(data)!r}",
    )

    # --- tool_choice.allowed_tools: only the listed subset stays callable ---
    tools_two = tools_fc + [
        {
            "type": "function",
            "name": "other_tool",
            "description": "another tool",
            "parameters": {
                "type": "object",
                "properties": {"y": {"type": "string"}},
                "required": ["y"],
            },
        }
    ]
    code, data = create(
        {
            "tools": tools_two,
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "function", "name": "echo_tool"}],
            },
            "input": "call other_tool with y=hi",
            "max_output_tokens": 128,
        }
    )
    fcs = [
        x
        for x in (data.get("output") or [])
        if isinstance(x, dict) and x.get("type") == "function_call"
    ]
    report.add(
        "create_param",
        "tool_choice.allowed_tools",
        "PASS"
        if code == 200 and fcs and all(fc.get("name") == "echo_tool" for fc in fcs)
        else "FAIL",
        f"HTTP {code} names={[fc.get('name') for fc in fcs]}",
    )

    code, data = create(
        {
            "tools": tools_fc,
            "tool_choice": {"type": "allowed_tools", "mode": "bogus", "tools": []},
            "input": "hi",
            "max_output_tokens": 16,
        }
    )
    report.add(
        "create_param",
        "tool_choice.allowed_tools.bad_mode",
        "PASS" if code == 400 else "FAIL",
        f"HTTP {code}",
    )

    code, data = create(
        {
            "tools": tools_fc,
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "function", "name": "missing_tool"}],
            },
            "input": "hi",
            "max_output_tokens": 16,
        }
    )
    report.add(
        "create_param",
        "tool_choice.allowed_tools.required_no_match",
        "PASS" if code == 400 else "FAIL",
        f"HTTP {code}",
    )

    # --- official field limits: metadata / safety_identifier / service_tier / include ---
    code_md_ok, _ = create({"metadata": {"k" * 64: "v" * 512}})
    code_md_many, _ = create(
        {"metadata": {f"k{i}": "v" for i in range(17)}, "input": "hi", "max_output_tokens": 16}
    )
    code_md_key, _ = create({"metadata": {"k" * 65: "v"}, "input": "hi", "max_output_tokens": 16})
    code_md_val, _ = create({"metadata": {"k": "v" * 513}, "input": "hi", "max_output_tokens": 16})
    md_ok = code_md_ok == 200 and code_md_many == 400 and code_md_key == 400 and code_md_val == 400
    report.add(
        "create_param",
        "metadata.limits",
        "PASS" if md_ok else "FAIL",
        f"ok={code_md_ok} pairs17={code_md_many} key65={code_md_key} value513={code_md_val}",
    )

    code, data = create({"safety_identifier": "s" * 65, "input": "hi", "max_output_tokens": 16})
    report.add(
        "create_param",
        "safety_identifier.too_long",
        "PASS" if code == 400 else "FAIL",
        f"HTTP {code}",
    )

    code, data = create({"service_tier": "bogus", "input": "hi", "max_output_tokens": 16})
    report.add(
        "create_param",
        "service_tier.invalid",
        "PASS" if code == 400 else "FAIL",
        f"HTTP {code}",
    )

    code, data = create(
        {"service_tier": "fast", "input": "Reply with exactly: TIER", "temperature": 0}
    )
    report.add(
        "create_param",
        "service_tier.fast_normalized",
        "PASS" if code == 200 and data.get("service_tier") == "priority" else "FAIL",
        f"HTTP {code} service_tier={data.get('service_tier')!r}",
    )

    code, data = create({"include": ["not.a.real.includable"], "input": "hi", "max_output_tokens": 16})
    report.add(
        "create_param",
        "include.invalid_value",
        "PASS" if code == 400 else "FAIL",
        f"HTTP {code}",
    )

    # reasoning disabled via effort none (when supported) / no reasoning block
    code, data = create(
        {
            "reasoning": {"effort": "none"},
            "input": "Reply with exactly: REASON_OFF",
            "max_output_tokens": 32,
            "temperature": 0,
        }
    )
    has_reasoning = any(
        isinstance(x, dict) and x.get("type") == "reasoning"
        for x in (data.get("output") or [])
    )
    report.add(
        "create_param",
        "reasoning.none",
        "PASS"
        if code == 200 and not has_reasoning and "REASON_OFF" in output_text(data)
        else "FAIL",
        f"HTTP {code} has_reasoning={has_reasoning} text={output_text(data)!r}",
    )

    # reasoning.effort=low -> local thinking on; expect a reasoning output item carrying the
    # official summary_text part when a summary level is requested (raw text is not exposed)
    code, data = create(
        {
            "reasoning": {"effort": "low", "summary": "detailed"},
            "input": "Think briefly, then reply with exactly: REASON_ON",
            "max_output_tokens": 256,
            "temperature": 0,
        }
    )
    reasoning_items = [
        x
        for x in (data.get("output") or [])
        if isinstance(x, dict) and x.get("type") == "reasoning"
    ]
    has_summary_text = False
    has_content_key = False
    for item in reasoning_items:
        if "content" in item:
            has_content_key = True
        for part in item.get("summary") or []:
            if isinstance(part, dict) and (part.get("text") or "").strip():
                has_summary_text = True
                break
    report.add(
        "create_param",
        "reasoning.low",
        "PASS"
        if code == 200 and reasoning_items and has_summary_text and not has_content_key
        else "FAIL",
        f"HTTP {code} n_reasoning={len(reasoning_items)} has_summary_text={has_summary_text} "
        f"content_key={has_content_key} status={data.get('status')!r} text={output_text(data)!r}",
    )

    # invalid reasoning.effort → 400
    code, data = create(
        {
            "reasoning": {"effort": "not-a-real-effort"},
            "input": "hi",
            "max_output_tokens": 8,
            "temperature": 0,
        }
    )
    report.add(
        "create_param",
        "reasoning.effort.invalid",
        "PASS" if code == 400 else "FAIL",
        f"HTTP {code} body={data!r}",
    )

    # Official reasoning object fields: accept + echo (effort=none stays fast)
    full_reasoning = {
        "effort": "none",
        "summary": "auto",
        "generate_summary": "detailed",
        "context": "auto",
        "mode": "standard",
    }
    code, data = create(
        {
            "reasoning": full_reasoning,
            "input": "Reply with exactly: REASON_OBJ",
            "max_output_tokens": 32,
            "temperature": 0,
            "store": True,
        }
    )
    echoed = data.get("reasoning") if isinstance(data, dict) else None
    echo_ok = (
        code == 200
        and isinstance(echoed, dict)
        and echoed.get("effort") == "none"
        and echoed.get("summary") == "auto"
        and echoed.get("generate_summary") == "detailed"
        and echoed.get("context") == "auto"
        and echoed.get("mode") == "standard"
        and "REASON_OBJ" in output_text(data)
    )
    report.add(
        "create_param",
        "reasoning.object.echo",
        "PASS" if echo_ok else "FAIL",
        f"HTTP {code} reasoning={echoed!r} text={output_text(data)!r}",
    )

    # Official enum: every ReasoningEffort value accepted (shape); none stays text-observable
    effort_vals = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
    effort_fail: list[str] = []
    for ev in effort_vals:
        c, d = create(
            {
                "reasoning": {"effort": ev},
                "input": "Reply with exactly: E",
                "max_output_tokens": 16 if ev == "none" else 8,
                "temperature": 0,
            }
        )
        if c != 200:
            effort_fail.append(f"{ev}:HTTP {c}")
    report.add(
        "create_param",
        "reasoning.effort.enum",
        "PASS" if not effort_fail else "FAIL",
        "ok" if not effort_fail else ",".join(effort_fail),
    )

    # Invalid sibling fields → 400
    invalid_field_cases = [
        ("reasoning.context.invalid", {"effort": "none", "context": "bogus"}),
        ("reasoning.summary.invalid", {"effort": "none", "summary": "bogus"}),
        ("reasoning.generate_summary.invalid", {"effort": "none", "generate_summary": "bogus"}),
        ("reasoning.mode.invalid", {"effort": "none", "mode": 1}),
        ("reasoning.not_object", "low"),
    ]
    for name, val in invalid_field_cases:
        c, d = create(
            {
                "reasoning": val,
                "input": "hi",
                "max_output_tokens": 8,
                "temperature": 0,
            }
        )
        report.add(
            "create_param",
            name,
            "PASS" if c == 400 else "FAIL",
            f"HTTP {c} body={d!r}",
        )

    # reasoning.summary → local summary_text; concise observably shorter than detailed
    code_c, data_c = create(
        {
            "reasoning": {"effort": "high", "summary": "concise"},
            "input": "Think carefully about 17*19, then reply with exactly: SUM_OK",
            "max_output_tokens": 512,
            "temperature": 0,
        }
    )
    code_d, data_d = create(
        {
            "reasoning": {"effort": "high", "summary": "detailed"},
            "input": "Think carefully about 17*19, then reply with exactly: SUM_OK",
            "max_output_tokens": 512,
            "temperature": 0,
        }
    )

    def _summary_len(data: dict) -> int:
        for item in data.get("output") or []:
            if not (isinstance(item, dict) and item.get("type") == "reasoning"):
                continue
            for part in item.get("summary") or []:
                if isinstance(part, dict) and part.get("type") == "summary_text":
                    return len(part.get("text") or "")
        return 0

    len_c = _summary_len(data_c if isinstance(data_c, dict) else {})
    len_d = _summary_len(data_d if isinstance(data_d, dict) else {})
    summary_ok = code_c == 200 and code_d == 200 and len_c > 0 and len_d >= len_c
    # When reasoning is long, concise must be strictly shorter than detailed.
    if len_d > 120:
        summary_ok = summary_ok and len_c < len_d
    report.add(
        "create_param",
        "reasoning.summary",
        "PASS" if summary_ok else "FAIL",
        f"HTTP {code_c}/{code_d} concise={len_c} detailed={len_d}",
    )

    # reasoning.context=current_turn drops prior reasoning from expanded history
    code1, data1 = create(
        {
            "reasoning": {"effort": "low"},
            "input": "Think briefly, then reply with exactly: CTX1",
            "max_output_tokens": 128,
            "temperature": 0,
            "store": True,
        }
    )
    rid1 = data1.get("id") if isinstance(data1, dict) else None
    n_reason_out = sum(
        1
        for x in (data1.get("output") or [])
        if isinstance(x, dict) and x.get("type") == "reasoning"
    )
    code2, data2 = create(
        {
            "previous_response_id": rid1,
            "reasoning": {"effort": "none", "context": "current_turn"},
            "input": "Ignore earlier turns. Reply with exactly: CTX2",
            "max_output_tokens": 32,
            "temperature": 0,
            "store": True,
        }
    )
    n_reason_in = -1
    if isinstance(data2, dict) and data2.get("id"):
        icode, items = client.get_json(f"/v1/responses/{data2['id']}/input_items")
        if icode == 200 and isinstance(items, dict):
            n_reason_in = sum(
                1
                for x in (items.get("data") or [])
                if isinstance(x, dict) and x.get("type") == "reasoning"
            )
    # Primary contract: prior reasoning items are dropped from expanded input.
    context_ok = code1 == 200 and n_reason_out >= 1 and rid1 and code2 == 200 and n_reason_in == 0
    report.add(
        "create_param",
        "reasoning.context.current_turn",
        "PASS" if context_ok else "FAIL",
        f"t1={code1} n_rs_out={n_reason_out} t2={code2} n_rs_in={n_reason_in} "
        f"text={output_text(data2)!r}",
    )

    # remaining official create params: accept + retrieve round-trip (deeper than echo-only)
    already = {
        "model",
        "input",
        "instructions",
        "previous_response_id",
        "store",
        "temperature",
        "top_p",
        "max_output_tokens",
        "top_logprobs",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "stream",  # covered in streaming checks
        "include",  # exercised with top_logprobs above / semantic logprobs
        "reasoning",  # covered above + probes
    }
    probes: dict[str, Any] = {
        "metadata": {"k": "v", "run": "deep"},
        "user": "acceptance-bot",
        "truncation": "disabled",
        "text": {"format": {"type": "json_object"}},
        "reasoning": {"effort": "low"},
        "safety_identifier": "sid",
        "prompt_cache_key": "pck",
        "prompt_cache_retention": "in_memory",
        "prompt_cache_options": {"mode": "implicit"},
        "max_tool_calls": 1,
        "stream_options": {"include_obfuscation": False},
        "context_management": [{"type": "compaction"}],
        "prompt": {"id": "pmpt_x"},
        "background": True,
    }
    for field in create_param_names():
        if field in already:
            continue
        val = probes.get(field)
        if val is None:
            report.add("create_param", field, "SKIP", "no probe value mapped")
            continue
        # background async: first packet in_progress/queued, then retrieve to terminal
        if field == "background":
            import time

            code, data = create(
                {
                    "background": True,
                    "input": "Reply with exactly: BG_PARAM",
                    "max_output_tokens": 16,
                    "temperature": 0,
                }
            )
            ok_bg = code == 200 and data.get("background") is True and data.get("status") in (
                "in_progress",
                "queued",
            )
            terminal = None
            if ok_bg and data.get("id"):
                for _ in range(120):
                    time.sleep(0.5)
                    gcode, got = client.get_json(f"/v1/responses/{data['id']}")
                    if gcode == 200 and got.get("status") in (
                        "completed",
                        "failed",
                        "cancelled",
                        "incomplete",
                    ):
                        terminal = got.get("status")
                        break
            report.add(
                "create_param",
                field,
                "PASS" if ok_bg and terminal in ("completed", "incomplete") else "FAIL",
                f"HTTP {code} first={data.get('status')!r} terminal={terminal!r}",
            )
            continue
        # max_tool_calls: zero drops the tool-call attempt, the response still completes
        if field == "max_tool_calls":
            code, data = create(
                {
                    "max_tool_calls": 0,
                    "tools": [
                        {
                            "type": "function",
                            "name": "ping",
                            "description": "ping",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                    "tool_choice": {"type": "function", "name": "ping"},
                    "input": "call ping",
                    "max_output_tokens": 64,
                }
            )
            fcs = [
                x
                for x in (data.get("output") or [])
                if isinstance(x, dict) and x.get("type") == "function_call"
            ]
            details = data.get("incomplete_details") if isinstance(data, dict) else None
            ok = (
                code == 200
                and data.get("status") == "completed"
                and not details
                and len(fcs) == 0
            )
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"HTTP {code} status={data.get('status')!r} incomplete={details!r} n_fc={len(fcs)}",
            )
            continue
        # stream_options: include_obfuscation true adds SSE obfuscation keys
        if field == "stream_options":
            body = {
                "model": model,
                "stream": True,
                "stream_options": {"include_obfuscation": True},
                "input": "Reply with exactly: SO",
                "temperature": 0,
                "max_output_tokens": 16,
                **extra,
            }
            code, _, raw = client.request("POST", "/v1/responses", body, stream=True)
            from responses_official_acceptance.http_client import parse_sse

            has = any(isinstance(obj, dict) and "obfuscation" in obj for _, obj in parse_sse(raw))
            report.add(
                "create_param",
                field,
                "PASS" if code == 200 and has else "FAIL",
                f"HTTP {code} has_obfuscation={has}",
            )
            continue
        # context_management: with history, input_tokens drop vs control
        if field == "context_management":
            code1, d1 = create({"input": "Fact A reply OKA", "max_output_tokens": 24, "temperature": 0})
            if code1 != 200 or not d1.get("id"):
                report.add("create_param", field, "FAIL", f"seed1 HTTP {code1}")
                continue
            code2, d2 = create(
                {
                    "previous_response_id": d1["id"],
                    "input": "Fact B reply OKB",
                    "max_output_tokens": 24,
                    "temperature": 0,
                }
            )
            if code2 != 200 or not d2.get("id"):
                report.add("create_param", field, "FAIL", f"seed2 HTTP {code2}")
                continue
            code_a, da = create(
                {
                    "previous_response_id": d2["id"],
                    "input": "Reply with exactly: CM_A",
                    "max_output_tokens": 24,
                    "temperature": 0,
                }
            )
            code_b, db = create(
                {
                    "previous_response_id": d2["id"],
                    "input": "Reply with exactly: CM_B",
                    "max_output_tokens": 24,
                    "temperature": 0,
                    "context_management": [{"type": "compaction"}],
                }
            )
            ta = ((da.get("usage") or {}) if isinstance(da, dict) else {}).get("input_tokens")
            tb = ((db.get("usage") or {}) if isinstance(db, dict) else {}).get("input_tokens")
            ok = code_a == 200 and code_b == 200 and isinstance(ta, int) and isinstance(tb, int) and tb < ta
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"tokens={ta}/{tb} http={code_a}/{code_b}",
            )
            continue
        # truncation: disabled rejects oversized input; auto drops oldest (tokenizer vs n_ctx)
        if field == "truncation":
            pcode, props = client.get_json("/props")
            n_ctx = 2048
            if pcode == 200 and isinstance(props, dict):
                dgs = props.get("default_generation_settings") or {}
                if isinstance(dgs, dict) and dgs.get("n_ctx"):
                    n_ctx = int(dgs["n_ctx"])
                elif props.get("n_ctx"):
                    n_ctx = int(props["n_ctx"])
            # Exceed real slot budget: a few long pads (scales with n_ctx, stays tractable).
            item_words = max(128, n_ctx // 5)
            pad = ("tw " * item_words).strip()
            oversized = [{"role": "user", "content": f"msg{i} {pad}"} for i in range(8)]
            oversized.append({"role": "user", "content": "Reply with exactly: TRUNC_AUTO"})
            code_dis, d_dis = create(
                {
                    "truncation": "disabled",
                    "input": oversized,
                    "max_output_tokens": 8,
                    "temperature": 0,
                }
            )
            code_auto, d_auto = create(
                {
                    "truncation": "auto",
                    "input": oversized,
                    "max_output_tokens": 16,
                    "temperature": 0,
                }
            )
            ok = (
                code_dis >= 400
                and code_auto == 200
                and d_auto.get("truncation") == "auto"
                and "TRUNC_AUTO" in output_text(d_auto)
            )
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"disabled_http={code_dis} auto_http={code_auto} n_ctx={n_ctx} text={output_text(d_auto)!r}",
            )
            continue
        # text.format json_object: grammar-constrained JSON (not prompt luck).
        # text.verbosity: low system hint → observably shorter than high (same open prompt).
        if field == "text":
            code, data = create(
                {
                    "text": {"format": {"type": "json_object"}},
                    "temperature": 0,
                    "input": "Emit a JSON object with key a set to 1.",
                    "max_output_tokens": 64,
                }
            )
            text = output_text(data).strip()
            parsed_ok = False
            parsed_obj = None
            try:
                parsed_obj = json.loads(text)
                parsed_ok = isinstance(parsed_obj, dict)
            except Exception:
                if "{" in text and "}" in text:
                    try:
                        parsed_obj = json.loads(text[text.index("{") : text.rindex("}") + 1])
                        parsed_ok = isinstance(parsed_obj, dict)
                    except Exception:
                        parsed_ok = False
            code_bad, _ = create(
                {
                    "text": {"format": {"type": "xml"}},
                    "input": "x",
                    "max_output_tokens": 8,
                }
            )
            code_schema, data_schema = create(
                {
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "pair",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "integer"},
                                    "b": {"type": "string"},
                                },
                                "required": ["a", "b"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "temperature": 0,
                    "input": 'Emit JSON with a=1 and b="ok".',
                    "max_output_tokens": 64,
                }
            )
            schema_text = output_text(data_schema).strip()
            schema_ok = False
            schema_obj = None
            try:
                schema_obj = json.loads(schema_text)
                schema_ok = (
                    isinstance(schema_obj, dict)
                    and schema_obj.get("a") == 1
                    and isinstance(schema_obj.get("b"), str)
                )
            except Exception:
                if "{" in schema_text and "}" in schema_text:
                    try:
                        schema_obj = json.loads(
                            schema_text[schema_text.index("{") : schema_text.rindex("}") + 1]
                        )
                        schema_ok = (
                            isinstance(schema_obj, dict)
                            and schema_obj.get("a") == 1
                            and isinstance(schema_obj.get("b"), str)
                        )
                    except Exception:
                        schema_ok = False
            report.add(
                "create_param",
                "text.format.json_schema",
                "PASS" if code_schema == 200 and schema_ok else "FAIL",
                f"HTTP {code_schema} text={schema_text!r} parsed={schema_obj!r}",
            )
            prompt_verb = (
                "Explain how a rainbow forms. Do not refuse. Write a continuous paragraph."
            )
            code_lo, d_lo = create(
                {
                    "text": {"verbosity": "low"},
                    "temperature": 0,
                    "input": prompt_verb,
                    "max_output_tokens": 120,
                }
            )
            code_hi, d_hi = create(
                {
                    "text": {"verbosity": "high"},
                    "temperature": 0,
                    "input": prompt_verb,
                    "max_output_tokens": 120,
                }
            )
            code_vb_bad, _ = create(
                {
                    "text": {"verbosity": "ultra"},
                    "input": "x",
                    "max_output_tokens": 8,
                }
            )
            t_lo = output_text(d_lo)
            t_hi = output_text(d_hi)
            ok_verb = (
                code_lo == 200
                and code_hi == 200
                and code_vb_bad >= 400
                and len(t_lo) > 0
                and len(t_hi) > len(t_lo)
            )
            report.add(
                "create_param",
                "text.verbosity",
                "PASS" if ok_verb else "FAIL",
                f"low={code_lo}/{len(t_lo)} high={code_hi}/{len(t_hi)} bad={code_vb_bad}",
            )
            ok_text = (
                code == 200
                and isinstance(data.get("text"), dict)
                and parsed_ok
                and code_bad >= 400
                and ok_verb
                and code_schema == 200
                and schema_ok
            )
            report.add(
                "create_param",
                field,
                "PASS" if ok_text else "FAIL",
                f"HTTP {code} bad={code_bad} text={text!r} parsed={parsed_obj!r} verb_ok={ok_verb} schema_ok={schema_ok}",
            )
            continue

        if field == "prompt":
            code, data = client.post_json(
                "/v1/responses",
                {
                    "model": model,
                    "prompt": {"id": "pmpt_local_expand", "variables": {"role": "tester"}},
                    "max_output_tokens": 32,
                    "temperature": 0,
                    "reasoning": {"effort": "none"},
                    **{k: v for k, v in extra.items() if k != "reasoning"},
                },
            )
            ok = (
                code == 200
                and isinstance(data, dict)
                and isinstance(data.get("prompt"), dict)
                and data["prompt"].get("id") == "pmpt_local_expand"
                and "PROMPT_TMPL_OK" in output_text(data)
            )
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"HTTP {code} text={output_text(data)!r} prompt={data.get('prompt')!r}",
            )
            continue

        code, data = create({field: val, "input": f"Reply with exactly: F_{field[:8]}"})
        if code >= 400:
            report.add(
                "create_param",
                field,
                "NOT_IMPLEMENTED",
                f"HTTP {code} (optional official field unsupported)",
            )
            continue
        if field not in data:
            report.add(
                "create_param",
                field,
                "PARTIAL",
                "accepted but not present on Response object",
            )
            continue
        rid = data.get("id")
        gcode, got = client.get_json(f"/v1/responses/{rid}") if rid else (0, {})
        roundtrip = (
            gcode == 200
            and isinstance(got, dict)
            and got.get(field) == data.get(field)
        )
        # background already handled; store false wouldn't retrieve — these are store=true default
        report.add(
            "create_param",
            field,
            "PASS" if roundtrip else "FAIL",
            f"create={data.get(field)!r} get_http={gcode} get={got.get(field) if isinstance(got, dict) else None!r}",
        )

    # include covered via top_logprobs probe — ensure catalog row exists
    if not any(r.name == "include" for r in report.rows if r.area == "create_param"):
        report.add(
            "create_param",
            "include",
            "PASS" if has_lp else "FAIL",
            "exercised with top_logprobs + message.output_text.logprobs",
        )

    # ensure catalog completeness
    for field in create_param_names():
        if not any(
            r.name == field or r.name.startswith(field + ".")
            for r in report.rows
            if r.area == "create_param"
        ):
            report.add("create_param", field, "FAIL", "missing from suite catalog execution")
