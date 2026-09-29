"""Semantic-level Chat Completions checks."""

from __future__ import annotations

import json
import time
from typing import Any

from .http_client import ChatHttpClient, choice_reasoning, choice_text, tool_calls
from .report import Report


PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


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


def run_semantic_checks(
    client: ChatHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    # --- image_url content (1x1 PNG data URL) ---
    code, data = _create(
        client,
        model,
        extra,
        {
            "max_tokens": 32,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{PNG_1X1_B64}",
                            },
                        },
                        {"type": "text", "text": "Reply with exactly: IMAGE_OK"},
                    ],
                }
            ],
        },
    )
    report.add(
        "semantic",
        "image_url_content",
        "PASS" if code == 200 and "IMAGE_OK" in choice_text(data) else "FAIL",
        f"HTTP {code} text={choice_text(data)!r}",
    )

    # --- reasoning_effort=none: no reasoning_content ---
    code, data = _create(
        client,
        model,
        extra,
        {
            "reasoning_effort": "none",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Reply with exactly: NO_REASON"}],
        },
    )
    rc = choice_reasoning(data)
    report.add(
        "semantic",
        "reasoning_effort_none_no_reasoning_content",
        "PASS"
        if code == 200 and not rc and "NO_REASON" in choice_text(data)
        else "FAIL",
        f"HTTP {code} reasoning_content={rc!r} text={choice_text(data)!r}",
    )

    # --- response_format json_object ---
    code, data = _create(
        client,
        model,
        extra,
        {
            "response_format": {"type": "json_object"},
            "max_tokens": 32,
            "messages": [{"role": "user", "content": 'Return only JSON: {"ok":true}'}],
        },
    )
    text = choice_text(data).strip()
    parsed_ok = False
    try:
        obj = json.loads(text)
        parsed_ok = isinstance(obj, dict)
    except Exception:
        if "{" in text and "}" in text:
            try:
                obj = json.loads(text[text.index("{") : text.rindex("}") + 1])
                parsed_ok = isinstance(obj, dict)
            except Exception:
                parsed_ok = False
    report.add(
        "semantic",
        "response_format_json_object",
        "PASS" if code == 200 and parsed_ok else "FAIL",
        f"HTTP {code} text={text!r}",
    )

    # --- tool_choice auto vs required vs none ---
    tools_fc = [
        {
            "type": "function",
            "function": {
                "name": "ping",
                "description": "ping",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    code, data = _create(
        client,
        model,
        extra,
        {
            "tools": tools_fc,
            "tool_choice": "none",
            "messages": [
                {"role": "user", "content": "Reply with exactly: TC_NONE and do not call tools."}
            ],
        },
    )
    report.add(
        "semantic",
        "tool_choice_none_no_tools",
        "PASS"
        if code == 200 and not tool_calls(data) and "TC_NONE" in choice_text(data)
        else "FAIL",
        f"HTTP {code} n_tc={len(tool_calls(data))}",
    )

    code, data = _create(
        client,
        model,
        extra,
        {
            "tools": tools_fc,
            "tool_choice": "required",
            "messages": [{"role": "user", "content": "Use a tool."}],
            "max_tokens": 128,
        },
    )
    report.add(
        "semantic",
        "tool_choice_required_emits_tool_call",
        "PASS" if code == 200 and tool_calls(data) else "FAIL",
        f"HTTP {code} n_tc={len(tool_calls(data))}",
    )

    # --- input_tokens matches create usage ---
    # Explicit verbosity pins both probes to the shared explicit hint path: the
    # omitted-default injection applies to the create path only, so an omitted
    # probe would count the hint on one side and not the other.
    probe_msgs = [{"role": "user", "content": "token parity probe " + ("alpha " * 30)}]
    tcode, tok = client.post_json(
        "/v1/chat/completions/input_tokens",
        {"model": model, "messages": probe_msgs, "verbosity": "medium", **extra},
    )
    ccode, created = _create(
        client,
        model,
        extra,
        {"messages": probe_msgs, "max_tokens": 4, "verbosity": "medium"},
    )
    tin = tok.get("input_tokens") if isinstance(tok, dict) else None
    cin = (created.get("usage") or {}).get("prompt_tokens") if isinstance(created, dict) else None
    if tcode == 404:
        report.add(
            "semantic",
            "input_tokens_matches_create_usage",
            "SKIP",
            "input_tokens endpoint not implemented",
        )
    else:
        report.add(
            "semantic",
            "input_tokens_matches_create_usage",
            "PASS" if tcode == 200 and ccode == 200 and tin == cin and tin is not None else "FAIL",
            f"input_tokens={tin} create_usage={cin}",
        )

    # --- seed determinism (two identical creates) ---
    outs: list[str] = []
    for _ in range(2):
        code, data = _create(
            client,
            model,
            extra,
            {
                "seed": 12345,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Reply with exactly: SEED_DET"}],
            },
        )
        outs.append(choice_text(data))
    seed_ok = code == 200 and len(outs) == 2 and outs[0] == outs[1] and "SEED_DET" in outs[0]
    report.add(
        "semantic",
        "seed_determinism",
        "PASS" if seed_ok else "FAIL",
        f"outs={outs!r}",
    )

    # --- n=1 default shape ---
    code, data = _create(
        client,
        model,
        extra,
        {"messages": [{"role": "user", "content": "Reply with exactly: N1"}]},
    )
    n_ok = code == 200 and isinstance(data.get("choices"), list) and len(data["choices"]) == 1
    report.add(
        "semantic",
        "n_default_single_choice",
        "PASS" if n_ok else "FAIL",
        f"HTTP {code} n_choices={len(data.get('choices') or [])}",
    )

    # --- unsupported legacy functions with tools should not 500 ---
    code, _ = client.post_json(
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "functions": [
                {
                    "name": "legacy_only",
                    "description": "legacy",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            **extra,
        },
    )
    report.add(
        "semantic",
        "legacy_functions_accept_or_400",
        "PASS" if code != 500 else "FAIL",
        f"HTTP {code}",
    )

    # --- cloud-shaped fields: invalid enum/shape → 400; valid still accepted ---
    bad_cases = [
        ("prompt_cache_retention_invalid", {"prompt_cache_retention": "forever"}),
        ("prompt_cache_options_mode_invalid", {"prompt_cache_options": {"mode": "magic"}}),
        ("prompt_cache_key_not_string", {"prompt_cache_key": 123}),
    ]
    for name, extra_bad in bad_cases:
        code_bad, _ = client.post_json(
            "/v1/chat/completions",
            {
                "model": model,
                "max_tokens": 8,
                "temperature": 0,
                "messages": [{"role": "user", "content": "Reply with exactly: X"}],
                **extra_bad,
                **extra,
            },
        )
        report.add(
            "semantic",
            name,
            "PASS" if code_bad >= 400 else "FAIL",
            f"HTTP {code_bad}",
        )
    code_ok, data_ok = _create(
        client,
        model,
        extra,
        {
            "prompt_cache_key": "pck-valid",
            "prompt_cache_retention": "24h",
            "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Reply with exactly: CLOUD_SHAPED_OK"}],
        },
    )
    ok_valid = code_ok == 200 and "CLOUD_SHAPED_OK" in choice_text(data_ok)
    report.add(
        "semantic",
        "cloud_shaped_valid_accepted",
        "PASS" if ok_valid else "FAIL",
        f"HTTP {code_ok} text={choice_text(data_ok)!r}",
    )
    code_exp, _ = _create(
        client,
        model,
        extra,
        {
            "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    report.add(
        "semantic",
        "prompt_cache_options_explicit_requires_key",
        "PASS" if code_exp >= 400 else "FAIL",
        f"HTTP {code_exp}",
    )

    # --- prompt_cache_retention: in_memory → expires_at=0; 24h → future expires_at on disk ---
    import os
    from pathlib import Path

    root = Path(os.environ.get("LLAMA_OPENAI_FILES_PATH", "/tmp/llama-openai-files"))
    uniq = f"ret{time.time_ns()}"
    pad = f"retention pad {uniq}: " + ("golf-hotel-india " * 30)

    def _retention_probe(retention: str, key: str) -> tuple[int, int]:
        c, d = _create(
            client,
            model,
            extra,
            {
                "prompt_cache_key": key,
                "prompt_cache_retention": retention,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": f"{pad}\nReply with exactly: RET_OK"}],
            },
        )
        meta = root / "prompt_cache_keys" / f"{key}.json"
        exp = -1
        if meta.exists():
            try:
                exp = int(json.loads(meta.read_text()).get("expires_at") or 0)
            except Exception:
                exp = -1
        return c, exp

    c_mem, exp_mem = _retention_probe("in_memory", f"chat-ret-mem-{uniq}")
    c_24, exp_24 = _retention_probe("24h", f"chat-ret-24h-{uniq}")
    now = int(time.time())
    ok_ret = (
        c_mem == 200
        and c_24 == 200
        and exp_mem == 0
        and exp_24 > now + 23 * 3600
    )
    report.add(
        "semantic",
        "prompt_cache_retention_sets_ttl",
        "PASS" if ok_ret else "FAIL",
        f"mem_http={c_mem} exp_mem={exp_mem} h24_http={c_24} exp_24={exp_24} now={now}",
    )

    # --- prompt_cache_options.mode=implicit: auto key warms without explicit prompt_cache_key ---
    uniq_i = f"imp{time.time_ns()}"
    # Unique leading system message: avoids reusing the constant verbosity-hint
    # prefix written by earlier probes, so the first request starts cold.
    msg_i = {
        "role": "user",
        "content": f"implicit pad {uniq_i}: " + ("juliet-kilo " * 40) + "\nReply with exactly: IMP_OK",
    }
    body_imp = {
        "prompt_cache_options": {"mode": "implicit"},
        "prompt_cache_retention": "in_memory",
        "max_tokens": 24,
        "messages": [{"role": "system", "content": f"implicit cache probe {uniq_i}"}, msg_i],
    }
    c1, d1 = _create(client, model, extra, body_imp)
    c2, d2 = _create(client, model, extra, body_imp)

    def _cached(d: dict) -> int:
        u = d.get("usage") if isinstance(d, dict) else None
        if not isinstance(u, dict):
            return -1
        details = u.get("prompt_tokens_details") or {}
        if isinstance(details, dict):
            return int(details.get("cached_tokens") or 0)
        return 0

    ok_imp = c1 == 200 and c2 == 200 and _cached(d1) == 0 and _cached(d2) > 0
    report.add(
        "semantic",
        "prompt_cache_options_implicit_warms",
        "PASS" if ok_imp else "FAIL",
        f"http={c1}/{c2} cached=({_cached(d1)},{_cached(d2)})",
    )

    # --- prompt_cache_options.ttl: official documents 30m as the only supported value ---
    import os
    from pathlib import Path

    root = Path(os.environ.get("LLAMA_OPENAI_FILES_PATH", "/tmp/llama-openai-files"))
    uniq_t = f"ttl{time.time_ns()}"
    pad_t = f"chat ttl {uniq_t}: " + ("romeo-sierra " * 20)
    key_t = f"chat-ttl-30m-{uniq_t}"
    now_t = int(time.time())
    c30, _ = _create(
        client,
        model,
        extra,
        {
            "prompt_cache_key": key_t,
            "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            "max_tokens": 12,
            "messages": [{"role": "user", "content": f"{pad_t}\nReply with exactly: TTL_30M"}],
        },
    )
    meta = root / "prompt_cache_keys" / f"{key_t}.json"
    delta_30 = -1
    if meta.exists():
        try:
            exp = int(json.loads(meta.read_text()).get("expires_at") or 0)
            delta_30 = exp - now_t if exp > 0 else 0
        except Exception:
            delta_30 = -1
    rejected: dict[str, int] = {}
    for ttl in ("5m", "1h"):
        c, _ = _create(
            client,
            model,
            extra,
            {
                "prompt_cache_key": f"chat-ttl-{ttl}-{uniq_t}",
                "prompt_cache_options": {"mode": "explicit", "ttl": ttl},
                "max_tokens": 12,
                "messages": [{"role": "user", "content": f"{pad_t}\nReply with exactly: TTL_{ttl}"}],
            },
        )
        rejected[ttl] = c
    ok_ttl = c30 == 200 and abs(delta_30 - 30 * 60) <= 90 and all(c >= 400 for c in rejected.values())
    report.add(
        "semantic",
        "prompt_cache_options_ttl_only_30m",
        "PASS" if ok_ttl else "FAIL",
        f"http_30m={c30} delta_30m={delta_30} rejected={rejected}",
    )
