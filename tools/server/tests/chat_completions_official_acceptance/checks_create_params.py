"""Official Chat Completions create-parameter probes."""

from __future__ import annotations

import json
from typing import Any

from .catalog import create_param_names
from .http_client import ChatHttpClient, choice_reasoning, choice_text, tool_calls
from .report import Report


# Out-of-scope optional cloud fields: explicit 400 reject is the desired local contract.
EXPECT_HOSTED_REJECT = frozenset({"audio"})
# web_search_options: local deepen (search inject); expect HTTP 200 when present.


def run_create_param_checks(
    client: ChatHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    def create(
        extra_fields: dict[str, Any],
        *,
        user: str = "Reply with exactly: P",
        max_tokens: int = 48,
    ) -> tuple[int, dict]:
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [{"role": "user", "content": user}],
            **extra,
            **extra_fields,
        }
        if "max_tokens" in extra_fields:
            body["max_tokens"] = extra_fields["max_tokens"]
        if "messages" in extra_fields:
            body["messages"] = extra_fields["messages"]
        if "model" in extra_fields:
            body["model"] = extra_fields["model"]
        code, data = client.post_json("/v1/chat/completions", body)
        return code, data if isinstance(data, dict) else {"_data": data}

    # --- model ---
    code_missing, _ = client.post_json(
        "/v1/chat/completions",
        {
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "x"}],
            **extra,
        },
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

    # --- messages ---
    code, data = create({})
    report.add(
        "create_param",
        "messages",
        "PASS" if code == 200 else "FAIL",
        f"HTTP {code}",
    )
    code, data = create(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Reply with exactly: PART"}],
                }
            ]
        }
    )
    report.add(
        "create_param",
        "messages.text_parts",
        "PASS" if code == 200 and "PART" in choice_text(data) else "FAIL",
        f"HTTP {code} text={choice_text(data)!r}",
    )

    # --- max_tokens / max_completion_tokens ---
    # OpenAI: max_tokens is optional; missing must still succeed.
    code_missing_mt, data_missing_mt = client.post_json(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly: NOMAX"}],
            "temperature": 0,
            **extra,
        },
    )
    code, data = create(
        {
            "max_tokens": 4,
            "messages": [
                {
                    "role": "user",
                    "content": "Write a long detailed essay about mathematics history.",
                }
            ],
        },
        max_tokens=4,
    )
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    out_tok = int(usage.get("completion_tokens") or 0)
    fr = (data.get("choices") or [{}])[0].get("finish_reason") if code == 200 else None
    max_ok = (
        code_missing_mt == 200
        and "NOMAX" in choice_text(data_missing_mt if isinstance(data_missing_mt, dict) else {})
        and code == 200
        and fr == "length"
        and out_tok <= 4
    )
    report.add(
        "create_param",
        "max_tokens",
        "PASS" if max_ok else "FAIL",
        f"missing_http={code_missing_mt} finish={fr!r} out_tok={out_tok}",
    )

    code, data = create(
        {
            "max_completion_tokens": 4,
            "messages": [
                {
                    "role": "user",
                    "content": "Write a long detailed essay about mathematics history.",
                }
            ],
        },
        max_tokens=4,
    )
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    out_tok = int(usage.get("completion_tokens") or 0)
    fr = (data.get("choices") or [{}])[0].get("finish_reason") if code == 200 else None
    mct_ok = code == 200 and fr == "length" and out_tok <= 4
    report.add(
        "create_param",
        "max_completion_tokens",
        "PASS" if mct_ok else "FAIL",
        f"HTTP {code} finish={fr!r} out_tok={out_tok}",
    )

    # --- temperature determinism ---
    outs: list[str] = []
    for _ in range(2):
        code, data = create(
            {"messages": [{"role": "user", "content": "Reply with exactly: TEMP0_DET"}]},
            max_tokens=16,
        )
        outs.append(choice_text(data))
    temp_ok = code == 200 and len(outs) == 2 and outs[0] == outs[1] and "TEMP0_DET" in outs[0]
    report.add(
        "create_param",
        "temperature",
        "PASS" if temp_ok else "FAIL",
        f"outs={outs!r}",
    )

    # --- top_p ---
    code, data = create(
        {"top_p": 0.5, "messages": [{"role": "user", "content": "Reply with exactly: TOPP"}]}
    )
    code_bad_p, _ = create({"top_p": "nope", "messages": [{"role": "user", "content": "x"}]})
    ok_p = code == 200 and "TOPP" in choice_text(data) and code_bad_p >= 400
    report.add(
        "create_param",
        "top_p",
        "PASS" if ok_p else "FAIL",
        f"HTTP {code} bad={code_bad_p}",
    )

    # --- stop: control continues past marker; with stop=["7"], marker is not emitted ---
    prompt_seq = "Continue this sequence exactly with the next numbers only: 1 2 3 4 5 6"
    code_ctrl, d_ctrl = create(
        {"messages": [{"role": "user", "content": prompt_seq}]},
        max_tokens=24,
    )
    code_stop, d_stop = create(
        {
            "stop": ["7"],
            "messages": [{"role": "user", "content": prompt_seq}],
        },
        max_tokens=24,
    )
    text_ctrl = choice_text(d_ctrl)
    text_stop = choice_text(d_stop)
    fr_stop = (d_stop.get("choices") or [{}])[0].get("finish_reason") if code_stop == 200 else None
    stop_ok = (
        code_ctrl == 200
        and code_stop == 200
        and "7" in text_ctrl
        and "10" in text_ctrl
        and fr_stop == "stop"
        and "7" not in text_stop
        and "10" not in text_stop
    )
    report.add(
        "create_param",
        "stop",
        "PASS" if stop_ok else "FAIL",
        f"ctrl={text_ctrl!r} stop={text_stop!r} fr={fr_stop!r}",
    )

    # --- logprobs + top_logprobs ---
    code, data = create(
        {
            "logprobs": True,
            "top_logprobs": 2,
            "messages": [{"role": "user", "content": "Reply with exactly: TLP"}],
        }
    )
    choice = (data.get("choices") or [{}])[0] if code == 200 else {}
    lp = choice.get("logprobs") if isinstance(choice, dict) else None
    has_lp = isinstance(lp, dict) and isinstance(lp.get("content"), list) and lp["content"]
    report.add(
        "create_param",
        "logprobs",
        "PASS" if code == 200 and has_lp else "FAIL",
        f"HTTP {code} logprobs_nonempty={has_lp}",
    )
    code_bad_tlp, _ = create(
        {
            "logprobs": True,
            "top_logprobs": 21,
            "messages": [{"role": "user", "content": "Reply with exactly: TLP_BAD"}],
        }
    )
    report.add(
        "create_param",
        "top_logprobs",
        "PASS" if code == 200 and has_lp and code_bad_tlp >= 400 else "FAIL",
        f"HTTP {code} (with logprobs=true) bad_gt20={code_bad_tlp}",
    )

    # --- tools + tool_choice triad ---
    tools_fc = [
        {
            "type": "function",
            "function": {
                "name": "echo_tool",
                "description": "echo",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            },
        }
    ]
    code, data = create(
        {
            "tools": tools_fc,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "messages": [{"role": "user", "content": "Reply with exactly: TOOLS"}],
        }
    )
    auto_ok = code == 200 and "TOOLS" in choice_text(data)
    report.add("create_param", "tools", "PASS" if auto_ok else "FAIL", f"HTTP {code}")
    report.add(
        "create_param",
        "tool_choice",
        "PASS" if auto_ok else "FAIL",
        f"HTTP {code} (auto)",
    )
    code_par, data_par = create(
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "alpha",
                        "description": "A",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "beta",
                        "description": "B",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "messages": [{"role": "user", "content": "Call tools now."}],
        },
        max_tokens=128,
    )
    tcs_par = tool_calls(data_par)
    par_ok = code_par == 200 and 0 < len(tcs_par) <= 1
    report.add(
        "create_param",
        "parallel_tool_calls",
        "PASS" if par_ok else "FAIL",
        f"true_auto={code} false_required={code_par} n_tc={len(tcs_par)}",
    )

    code, data = create(
        {
            "tools": tools_fc,
            "tool_choice": "required",
            "messages": [{"role": "user", "content": "Use a tool now."}],
        },
        max_tokens=128,
    )
    tcs = tool_calls(data)
    report.add(
        "create_param",
        "tool_choice.required",
        "PASS" if code == 200 and tcs else "FAIL",
        f"HTTP {code} n_tc={len(tcs)} finish={(data.get('choices') or [{}])[0].get('finish_reason')!r}",
    )

    code, data = create(
        {
            "tools": tools_fc,
            "tool_choice": {"type": "function", "function": {"name": "echo_tool"}},
            "messages": [{"role": "user", "content": "call echo_tool with x=hi"}],
        },
        max_tokens=128,
    )
    tcs = tool_calls(data)
    names = [
        (tc.get("function") or {}).get("name")
        for tc in tcs
        if isinstance(tc.get("function"), dict)
    ]
    report.add(
        "create_param",
        "tool_choice.function",
        "PASS" if code == 200 and tcs and names[0] == "echo_tool" else "FAIL",
        f"HTTP {code} names={names}",
    )

    code, data = create(
        {
            "tools": tools_fc,
            "tool_choice": "none",
            "messages": [
                {"role": "user", "content": "Reply with exactly: NONE_OK and do not call tools."}
            ],
        },
        max_tokens=64,
    )
    report.add(
        "create_param",
        "tool_choice.none",
        "PASS"
        if code == 200 and not tool_calls(data) and "NONE_OK" in choice_text(data)
        else "FAIL",
        f"HTTP {code} n_tc={len(tool_calls(data))} text={choice_text(data)!r}",
    )

    # --- reasoning_effort=none ---
    code, data = create(
        {
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": "Reply with exactly: REASON_OFF"}],
        },
        max_tokens=32,
    )
    rc = choice_reasoning(data)
    reason_ok = code == 200 and not rc and "REASON_OFF" in choice_text(data)
    report.add(
        "create_param",
        "reasoning_effort.none",
        "PASS" if reason_ok else "FAIL",
        f"HTTP {code} reasoning_content={rc!r} text={choice_text(data)!r}",
    )

    # --- reasoning_effort=low → thinking on / reasoning_content present ---
    code, data = create(
        {
            "reasoning_effort": "low",
            "messages": [
                {
                    "role": "user",
                    "content": "Think briefly, then reply with exactly: REASON_ON",
                }
            ],
        },
        max_tokens=256,
    )
    rc = choice_reasoning(data)
    reason_low_ok = code == 200 and bool(rc and str(rc).strip())
    report.add(
        "create_param",
        "reasoning_effort.low",
        "PASS" if reason_low_ok else "FAIL",
        f"HTTP {code} reasoning_content={rc!r} text={choice_text(data)!r}",
    )

    # --- invalid reasoning_effort → 400 ---
    code_inv, data_inv = create(
        {
            "reasoning_effort": "not-a-real-effort",
            "messages": [{"role": "user", "content": "hi"}],
        },
        max_tokens=8,
    )
    reason_inv_ok = code_inv == 400
    report.add(
        "create_param",
        "reasoning_effort.invalid",
        "PASS" if reason_inv_ok else "FAIL",
        f"HTTP {code_inv} body={data_inv!r}",
    )

    # Official ReasoningEffort enum: every value accepted
    effort_vals = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
    effort_fail: list[str] = []
    for ev in effort_vals:
        c, _d = create(
            {
                "reasoning_effort": ev,
                "messages": [{"role": "user", "content": "Reply with exactly: E"}],
            },
            max_tokens=16 if ev == "none" else 8,
        )
        if c != 200:
            effort_fail.append(f"{ev}:HTTP {c}")
    reason_enum_ok = not effort_fail
    report.add(
        "create_param",
        "reasoning_effort.enum",
        "PASS" if reason_enum_ok else "FAIL",
        "ok" if reason_enum_ok else ",".join(effort_fail),
    )

    # Catalog key: aggregate none + low + invalid + enum
    report.add(
        "create_param",
        "reasoning_effort",
        "PASS" if reason_ok and reason_low_ok and reason_inv_ok and reason_enum_ok else "FAIL",
        "aggregated none+low+invalid+enum",
    )

    # --- response_format json_object ---
    code, data = create(
        {
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": 'Return only JSON: {"a":1}'}],
        },
        max_tokens=32,
    )
    text = choice_text(data).strip()
    parsed_ok = False
    try:
        json.loads(text)
        parsed_ok = True
    except Exception:
        if "{" in text and "}" in text:
            try:
                json.loads(text[text.index("{") : text.rindex("}") + 1])
                parsed_ok = True
            except Exception:
                parsed_ok = False
    report.add(
        "create_param",
        "response_format",
        "PASS" if code == 200 and parsed_ok else "FAIL",
        f"HTTP {code} text={text!r}",
    )

    # Remaining official create params: accept without hard error.
    # ChatCompletion objects do not echo request params (unlike OpenAI Response).
    already = {
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stop",
        "logprobs",
        "top_logprobs",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning_effort",
        "response_format",
        "stream",  # scored in checks_streaming
    }
    probes: dict[str, Any] = {
        "audio": {"voice": "alloy", "format": "wav"},
        "frequency_penalty": 0.1,
        "function_call": "none",
        "functions": [
            {
                "name": "legacy_fn",
                "description": "legacy",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        "logit_bias": {"50256": -100},
        "metadata": {"k": "v", "run": "acceptance"},
        "modalities": ["text"],
        "n": 1,
        "prediction": {"type": "content", "content": "Reply with exactly: PRED"},
        "presence_penalty": 0.1,
        "prompt_cache_key": "pck",
        "prompt_cache_options": {"mode": "implicit"},
        "prompt_cache_retention": "in_memory",
        "safety_identifier": "sid",
        "seed": 42,
        "store": True,
        "stream_options": {"include_usage": True},
        "user": "acceptance-bot",
        "verbosity": "low",
        "web_search_options": {},
    }
    for field in create_param_names():
        if field in already:
            continue
        val = probes.get(field)
        if val is None:
            report.add("create_param", field, "SKIP", "no probe value mapped")
            continue
        # Deepen observable local behavior for fields that already work.
        if field == "store":
            code, data = create(
                {
                    "store": True,
                    "metadata": {"suite": "create_param", "k": "store"},
                    "messages": [{"role": "user", "content": "Reply with exactly: STORE_PARAM"}],
                }
            )
            cid = data.get("id") if isinstance(data, dict) else None
            gcode, got = client.get_json(f"/v1/chat/completions/{cid}") if cid else (0, {})
            md = got.get("metadata") if isinstance(got, dict) else None
            ok = (
                code == 200
                and "STORE_PARAM" in choice_text(data)
                and gcode == 200
                and isinstance(md, dict)
                and md.get("k") == "store"
            )
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"create={code} get={gcode} metadata={md!r}",
            )
            continue
        if field == "n":
            code, data = create(
                {
                    "n": 2,
                    "seed": 7,
                    "messages": [{"role": "user", "content": "Reply with exactly: N2"}],
                }
            )
            n_ch = len(data.get("choices") or []) if isinstance(data, dict) else 0
            ok = code == 200 and n_ch == 2
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"HTTP {code} n_choices={n_ch}",
            )
            continue
        if field == "stream_options":
            from .http_client import parse_sse

            body = {
                "model": model,
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": 16,
                "temperature": 0,
                "messages": [{"role": "user", "content": "Reply with exactly: SOU"}],
                **extra,
            }
            code, _, raw = client.request("POST", "/v1/chat/completions", body, stream=True)
            events = parse_sse(raw) if code == 200 else []
            has_usage = any(
                isinstance(obj, dict) and isinstance(obj.get("usage"), dict) for _, obj in events
            )
            report.add(
                "create_param",
                field,
                "PASS" if code == 200 and has_usage else "FAIL",
                f"HTTP {code} has_usage={has_usage} n_events={len(events)}",
            )
            continue
        if field == "seed":
            outs: list[str] = []
            for _ in range(2):
                code, data = create(
                    {
                        "seed": 99,
                        "messages": [{"role": "user", "content": "Reply with exactly: SEED_P"}],
                    }
                )
                outs.append(choice_text(data))
            ok = code == 200 and len(outs) == 2 and outs[0] == outs[1] and "SEED_P" in outs[0]
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"outs={outs!r}",
            )
            continue
        if field == "metadata":
            code, data = create(
                {
                    "store": True,
                    "metadata": {"k": "v", "run": "acceptance"},
                    "messages": [{"role": "user", "content": "Reply with exactly: META_P"}],
                }
            )
            cid = data.get("id") if isinstance(data, dict) else None
            gcode, got = client.get_json(f"/v1/chat/completions/{cid}") if cid else (0, {})
            md = got.get("metadata") if isinstance(got, dict) else None
            ok = (
                code == 200
                and gcode == 200
                and isinstance(md, dict)
                and md.get("k") == "v"
                and md.get("run") == "acceptance"
            )
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"create={code} get={gcode} metadata={md!r}",
            )
            continue
        if field in ("frequency_penalty", "presence_penalty"):
            code, data = create(
                {
                    field: 0.1,
                    "messages": [{"role": "user", "content": f"Reply with exactly: F_{field[:8]}"}],
                }
            )
            code_bad, _ = create(
                {
                    field: 9.0,
                    "messages": [{"role": "user", "content": "x"}],
                }
            )
            if field == "frequency_penalty":
                # High frequency_penalty reduces repetition of the same token.
                prompt_rep = (
                    "Write the word banana as many times as you can, separated by spaces. "
                    "Do not write any other words."
                )
                code_lo, d_lo = create(
                    {
                        field: 0.0,
                        "temperature": 0.8,
                        "messages": [{"role": "user", "content": prompt_rep}],
                    },
                    max_tokens=48,
                )
                code_hi, d_hi = create(
                    {
                        field: 1.8,
                        "temperature": 0.8,
                        "messages": [{"role": "user", "content": prompt_rep}],
                    },
                    max_tokens=48,
                )
                n_lo = choice_text(d_lo).lower().count("banana")
                n_hi = choice_text(d_hi).lower().count("banana")
                ok_beh = (
                    code_lo == 200
                    and code_hi == 200
                    and n_lo >= 2
                    and n_hi < n_lo
                )
                detail_beh = f"banana_lo={n_lo} banana_hi={n_hi}"
            else:
                # High presence_penalty at temperature=0 diversifies away from a
                # degenerate dog/cat loop (observable unique-token / max-freq shift).
                from collections import Counter

                prompt_rep = (
                    "Write a long list of animals, repeating names is allowed, separated by spaces."
                )
                code_lo, d_lo = create(
                    {
                        field: 0.0,
                        "temperature": 0,
                        "messages": [{"role": "user", "content": prompt_rep}],
                    },
                    max_tokens=80,
                )
                code_hi, d_hi = create(
                    {
                        field: 2.0,
                        "temperature": 0,
                        "messages": [{"role": "user", "content": prompt_rep}],
                    },
                    max_tokens=80,
                )

                def _stats(t: str) -> tuple[int, int]:
                    toks = [w for w in t.lower().replace(",", " ").split() if w]
                    if not toks:
                        return 0, 0
                    return len(set(toks)), max(Counter(toks).values())

                uniq_lo, mf_lo = _stats(choice_text(d_lo))
                uniq_hi, mf_hi = _stats(choice_text(d_hi))
                ok_beh = (
                    code_lo == 200
                    and code_hi == 200
                    and mf_lo >= 10
                    and uniq_hi > uniq_lo
                    and mf_hi < mf_lo
                )
                detail_beh = f"uniq=({uniq_lo},{uniq_hi}) maxfreq=({mf_lo},{mf_hi})"
            ok = (
                code == 200
                and "F_" in choice_text(data)
                and code_bad >= 400
                and ok_beh
            )
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"ok_http={code} bad_http={code_bad} {detail_beh}",
            )
            continue
        if field == "verbosity":
            prompt_v = "Explain how a rainbow forms. Do not refuse. Write a continuous paragraph."
            code, data = create(
                {
                    "verbosity": "low",
                    "messages": [{"role": "user", "content": prompt_v}],
                },
                max_tokens=120,
            )
            code_hi, data_hi = create(
                {
                    "verbosity": "high",
                    "messages": [{"role": "user", "content": prompt_v}],
                },
                max_tokens=120,
            )
            code_bad, _ = create({"verbosity": "ultra", "messages": [{"role": "user", "content": "x"}]})
            t_lo = choice_text(data)
            t_hi = choice_text(data_hi)
            ok = (
                code == 200
                and code_hi == 200
                and code_bad >= 400
                and len(t_lo) > 0
                and len(t_hi) > len(t_lo)
            )
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"ok_http={code} hi_http={code_hi} bad_http={code_bad} low_len={len(t_lo)} high_len={len(t_hi)}",
            )
            continue
        if field == "modalities":
            code, data = create(
                {
                    "modalities": ["text"],
                    "messages": [{"role": "user", "content": "Reply with exactly: MOD_OK"}],
                }
            )
            code_bad, _ = create(
                {
                    "modalities": ["text", "audio"],
                    "messages": [{"role": "user", "content": "x"}],
                }
            )
            ok = code == 200 and "MOD_OK" in choice_text(data) and code_bad >= 400
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"ok_http={code} bad_http={code_bad}",
            )
            continue
        if field == "prediction":
            code, data = create(
                {
                    "prediction": {"type": "content", "content": "PRED_PREFIX "},
                    "messages": [
                        {
                            "role": "user",
                            "content": "Continue the assistant draft. End with exactly: PRED_OK",
                        }
                    ],
                },
                max_tokens=64,
            )
            code_bad, _ = create(
                {
                    "prediction": {"type": "other", "content": "x"},
                    "messages": [{"role": "user", "content": "x"}],
                }
            )
            text = choice_text(data)
            # Local: prediction prefills assistant; output should include the prefix.
            ok = code == 200 and code_bad >= 400 and "PRED_PREFIX" in text
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"ok_http={code} bad_http={code_bad} text={text!r}",
            )
            continue
        if field in ("user", "safety_identifier"):
            body = {
                "store": True,
                "user": "acceptance-bot",
                "safety_identifier": "sid-chat",
                "messages": [{"role": "user", "content": "Reply with exactly: USR_OK"}],
            }
            code, data = create(body)
            cid = data.get("id") if isinstance(data, dict) else None
            gcode, got = client.get_json(f"/v1/chat/completions/{cid}") if cid else (0, {})
            ok = (
                code == 200
                and gcode == 200
                and got.get("user") == "acceptance-bot"
                and got.get("safety_identifier") == "sid-chat"
                and "USR_OK" in choice_text(data)
            )
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"create={code} get={gcode} user={got.get('user')!r} sid={got.get('safety_identifier')!r}",
            )
            continue
        if field == "logit_bias":
            # Suppress tokens for BANANA via /tokenize; control must contain BANANA, biased must not.
            tcode, tdata = client.post_json("/tokenize", {"content": "BANANA", "add_special": False})
            toks = (tdata.get("tokens") if isinstance(tdata, dict) else None) or []
            bias = {str(t): -100 for t in toks if isinstance(t, int)}
            code_ctrl, data_ctrl = create(
                {"messages": [{"role": "user", "content": "Reply with exactly: BANANA"}], "temperature": 0}
            )
            code, data = create(
                {
                    "logit_bias": bias or {"0": -100},
                    "temperature": 0,
                    "messages": [{"role": "user", "content": "Reply with exactly: BANANA"}],
                }
            )
            code_bad, _ = create(
                {"logit_bias": "nope", "messages": [{"role": "user", "content": "x"}]}
            )
            ctrl = choice_text(data_ctrl)
            biased = choice_text(data)
            ok = (
                tcode == 200
                and bool(bias)
                and code_ctrl == 200
                and "BANANA" in ctrl
                and code == 200
                and "BANANA" not in biased
                and code_bad >= 400
            )
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"tok={tcode}/{toks!r} ctrl={ctrl!r} biased={biased!r} bad={code_bad}",
            )
            continue
        if field in ("prompt_cache_key", "prompt_cache_retention", "prompt_cache_options"):
            good = {
                "prompt_cache_key": "pck",
                "prompt_cache_retention": "in_memory",
                "prompt_cache_options": {"mode": "implicit"},
            }[field]
            bad = {
                "prompt_cache_key": 123,
                "prompt_cache_retention": "forever",
                "prompt_cache_options": {"mode": "magic"},
            }[field]
            code, data = create(
                {field: good, "messages": [{"role": "user", "content": f"Reply with exactly: F_{field[:8]}"}]}
            )
            code_bad, _ = create({field: bad, "messages": [{"role": "user", "content": "x"}]})
            ok = code == 200 and code_bad >= 400 and f"F_{field[:8]}" in choice_text(data)
            if field == "prompt_cache_key" and ok:
                import time as _time

                # Warm hit on identical prefix + key (unique per run to avoid cross-run contamination).
                uniq = f"pck{_time.time_ns()}"
                pad = f"Chat cache pad {uniq}: " + ("delta-echo-foxtrot " * 40)
                body_cache = {
                    "prompt_cache_key": f"chat-acceptance-cache-{uniq}",
                    "prompt_cache_retention": "in_memory",
                    "messages": [
                        {"role": "user", "content": f"{pad}\nReply with exactly: CHAT_CACHE"}
                    ],
                }
                c_cold, d_cold = create(body_cache, max_tokens=24)
                c_warm, d_warm = create(body_cache, max_tokens=24)

                def _cached2(d: dict) -> int:
                    u = d.get("usage") if isinstance(d, dict) else None
                    if not isinstance(u, dict):
                        return -1
                    details = u.get("prompt_tokens_details") or {}
                    if isinstance(details, dict):
                        return int(details.get("cached_tokens") or 0)
                    return 0

                cold_c = _cached2(d_cold if isinstance(d_cold, dict) else {})
                warm_c = _cached2(d_warm if isinstance(d_warm, dict) else {})
                ok_cache = c_cold == 200 and c_warm == 200 and cold_c == 0 and warm_c > cold_c
                report.add(
                    "create_param",
                    "prompt_cache_key.warm_hit",
                    "PASS" if ok_cache else "FAIL",
                    f"cold={cold_c} warm={warm_c} http={c_cold}/{c_warm} uniq={uniq}",
                )
                ok = ok and ok_cache
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"ok_http={code} bad_http={code_bad} text={choice_text(data)!r}",
            )
            continue
        if field == "function_call":
            # none: must not emit tool_calls; named: must force that legacy function.
            code_none, data_none = create(
                {
                    "function_call": "none",
                    "functions": [
                        {
                            "name": "legacy_fn",
                            "description": "legacy",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                    "messages": [
                        {"role": "user", "content": "Reply with exactly: FC_NONE and do not call tools."}
                    ],
                }
            )
            tcs_none = ((data_none.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
            code_named, data_named = create(
                {
                    "function_call": {"name": "legacy_fn"},
                    "functions": [
                        {
                            "name": "legacy_fn",
                            "description": "legacy",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                    "messages": [{"role": "user", "content": "Call legacy_fn now."}],
                }
            )
            tcs = ((data_named.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
            names = [
                (t.get("function") or {}).get("name")
                for t in tcs
                if isinstance(t, dict)
            ]
            ok = (
                code_none == 200
                and not tcs_none
                and "FC_NONE" in choice_text(data_none)
                and code_named == 200
                and names == ["legacy_fn"]
            )
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"none={code_none}/{choice_text(data_none)!r} named={code_named}/{names!r}",
            )
            continue
        if field == "functions":
            code, data = create(
                {
                    "functions": [
                        {
                            "name": "legacy_fn",
                            "description": "legacy",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                    "function_call": {"name": "legacy_fn"},
                    "messages": [{"role": "user", "content": "Call legacy_fn now."}],
                }
            )
            tcs = ((data.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
            names = [
                (t.get("function") or {}).get("name")
                for t in tcs
                if isinstance(t, dict)
            ]
            ok = code == 200 and names == ["legacy_fn"]
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"HTTP {code} names={names!r}",
            )
            continue
        if field == "web_search_options":
            # Local deepen: inject search context + message.annotations url_citation
            # (non-stream + stream terminal chunk).
            code, data = create(
                {
                    "web_search_options": {"search_context_size": "medium"},
                    "reasoning_effort": "none",
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is example.com used for? Reply briefly.",
                        }
                    ],
                }
            )
            msg = ((data.get("choices") or [{}])[0].get("message") or {}) if isinstance(data, dict) else {}
            anns = msg.get("annotations") if isinstance(msg, dict) else None
            cite_ok = isinstance(anns, list) and any(
                isinstance(a, dict) and a.get("type") == "url_citation" and a.get("url")
                for a in anns
            )
            # Stream: annotations must appear on a chunk delta (finish chunk).
            scode, _, sraw = client.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "stream": True,
                    "reasoning_effort": "none",
                    "max_tokens": 64,
                    "web_search_options": {"search_context_size": "low"},
                    "messages": [
                        {"role": "user", "content": "What is example.com used for? Reply briefly."}
                    ],
                },
                stream=True,
            )
            sbody = sraw.decode(errors="replace")
            stream_cite = scode == 200 and "url_citation" in sbody
            # filters.blocked_domains must drop matching fixture/SERP hits.
            code_blk, data_blk = create(
                {
                    "web_search_options": {
                        "search_context_size": "low",
                        "filters": {"blocked_domains": ["example.com"]},
                    },
                    "reasoning_effort": "none",
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is example.com used for? Reply briefly.",
                        }
                    ],
                }
            )
            msg_blk = (
                ((data_blk.get("choices") or [{}])[0].get("message") or {})
                if isinstance(data_blk, dict)
                else {}
            )
            anns_blk = msg_blk.get("annotations") if isinstance(msg_blk, dict) else None
            blocked_ok = code_blk == 200 and isinstance(anns_blk, list) and not any(
                isinstance(a, dict) and "example.com" in str(a.get("url") or "") for a in anns_blk
            )
            ok = code == 200 and cite_ok and stream_cite and blocked_ok
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                (
                    f"HTTP {code} cites={cite_ok} stream={scode}/{stream_cite} "
                    f"blocked={blocked_ok} anns={str(anns)[:80]}"
                ),
            )
            continue
        code, data = create(
            {field: val, "messages": [{"role": "user", "content": f"Reply with exactly: F_{field[:8]}"}]}
        )
        if code >= 400:
            if field in EXPECT_HOSTED_REJECT:
                report.add(
                    "create_param",
                    field,
                    "PASS" if code == 400 else "FAIL",
                    f"HTTP {code} (hosted cloud feature explicitly rejected)",
                )
            elif code in (404, 501):
                report.add(
                    "create_param",
                    field,
                    "NOT_IMPLEMENTED",
                    f"HTTP {code} (optional official field unsupported)",
                )
            else:
                report.add(
                    "create_param",
                    field,
                    "NOT_IMPLEMENTED",
                    f"HTTP {code} (optional official field rejected)",
                )
        else:
            report.add(
                "create_param",
                field,
                "PASS",
                "accepted (request-side; not echoed on ChatCompletion object)",
            )

    # catalog completeness (stream row owned by checks_streaming)
    for field in create_param_names():
        if field == "stream":
            continue
        if not any(
            r.name == field or r.name.startswith(field + ".")
            for r in report.rows
            if r.area == "create_param"
        ):
            report.add(
                "create_param",
                field,
                "FAIL",
                "missing from suite catalog execution",
            )
