"""OpenAI Completions API checks (POST /v1/completions)."""

from __future__ import annotations

from typing import Any

from openai.types import Completion

from .catalog import completion_param_names
from .http_client import ResponsesHttpClient, parse_sse
from .report import Report

_PROMPT = "/no_think\nReply with exactly: OAI_CMPL_OK"
_PROMPT_PARAM = "/no_think\nReply with exactly: OAI_CMPL_PARAM"


def _completion_text(data: dict[str, Any]) -> str:
    text = ""
    for ch in data.get("choices") or []:
        if isinstance(ch, dict):
            text += ch.get("text") or ""
    return text


def _validate_shape(data: dict[str, Any]) -> tuple[bool, str]:
    try:
        Completion.model_validate(data)
        return data.get("object") == "text_completion", "ok"
    except Exception as e:
        return False, str(e)[:200]


def run_completions_checks(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    base = {
        "model": model,
        "prompt": _PROMPT,
        "max_tokens": 32,
        "temperature": 0,
    }
    if "chat_template_kwargs" in extra:
        base["chat_template_kwargs"] = extra["chat_template_kwargs"]

    code, data = client.post_json("/v1/completions", base)
    if code == 404:
        report.add("endpoint", "POST /v1/completions", "NOT_IMPLEMENTED", "HTTP 404")
        for field in completion_param_names():
            if field not in ("model", "prompt"):
                report.add("create_param", field, "NOT_IMPLEMENTED", "completions missing")
        return

    if code != 200 or not isinstance(data, dict):
        report.add("endpoint", "POST /v1/completions", "FAIL", f"HTTP {code}")
        return

    ok_shape, detail = _validate_shape(data)
    text = _completion_text(data)
    ok = ok_shape and "OAI_CMPL_OK" in text
    report.add(
        "endpoint",
        "POST /v1/completions",
        "PASS" if ok else "FAIL",
        f"text={text!r} schema={detail}",
    )

    code, headers, raw = client.request(
        "POST",
        "/v1/completions",
        {
            **base,
            "prompt": "/no_think\nReply with exactly: OAI_CMPL_STREAM",
            "stream": True,
        },
        stream=True,
    )
    ct = headers.get("Content-Type", "") or headers.get("content-type", "")
    events = parse_sse(raw) if code == 200 else []
    joined = ""
    for _, obj in events:
        if not isinstance(obj, dict):
            continue
        for ch in obj.get("choices") or []:
            if isinstance(ch, dict):
                joined += ch.get("text") or ""
    stream_ok = (
        code == 200
        and "text/event-stream" in ct
        and "OAI_CMPL_STREAM" in joined
    )
    report.add(
        "create_param",
        "stream",
        "PASS" if stream_ok else "FAIL",
        f"HTTP {code} text={joined!r}",
    )

    code, _ = client.post_json(
        "/v1/completions",
        {"model": model, "max_tokens": 8},
    )
    report.add(
        "scenario",
        "completions_missing_prompt_error",
        "PASS" if code >= 400 else "FAIL",
        f"HTTP {code}",
    )

    code_missing_model, _ = client.post_json(
        "/v1/completions",
        {"prompt": _PROMPT, "max_tokens": 8},
    )
    report.add(
        "create_param",
        "model",
        "PASS" if code_missing_model >= 400 else "FAIL",
        f"missing_model_http={code_missing_model}",
    )

    code_prompt, data_prompt = client.post_json(
        "/v1/completions",
        {"model": model, "prompt": _PROMPT_PARAM, "max_tokens": 32, "temperature": 0},
    )
    prompt_ok = (
        code_prompt == 200
        and isinstance(data_prompt, dict)
        and _validate_shape(data_prompt)[0]
        and "OAI_CMPL_PARAM" in _completion_text(data_prompt)
    )
    report.add(
        "create_param",
        "prompt",
        "PASS" if prompt_ok else "FAIL",
        f"HTTP {code_prompt}",
    )

    probes: dict[str, Any] = {
        "temperature": 0,
        "top_p": 0.9,
        "max_tokens": 24,
        "stop": ["\n\nSTOP_OAI_CMPL"],
        "n": 1,
        "echo": False,
        "suffix": "",
        "user": "acceptance-bot",
        "seed": 42,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "logit_bias": {},
        "logprobs": 0,
        "best_of": 1,
        "stream_options": {"include_usage": True},
    }
    covered = {"model", "prompt", "stream"}

    for field in completion_param_names():
        if field in covered:
            continue
        val = probes.get(field)
        if val is None:
            report.add("create_param", field, "SKIP", "no probe value mapped")
            continue

        body = {
            "model": model,
            "prompt": _PROMPT_PARAM,
            "max_tokens": 32,
            "temperature": 0,
            field: val,
        }
        if field == "stream_options":
            body = {
                "model": model,
                "prompt": "/no_think\nReply with exactly: OAI_CMPL_SO",
                "max_tokens": 24,
                "temperature": 0,
                "stream": True,
                "stream_options": val,
            }
            code, headers, raw = client.request("POST", "/v1/completions", body, stream=True)
            ct = headers.get("Content-Type", "") or headers.get("content-type", "")
            events = parse_sse(raw) if code == 200 else []
            joined = ""
            has_usage = False
            usage_null_ok = True
            n_events = 0
            n_obf_str = 0
            n_obf_nonempty = 0
            n_finish_obf = 0
            finish_obf_empty = True
            for _, obj in events:
                if not isinstance(obj, dict):
                    continue
                n_events += 1
                joined += _completion_text(obj)
                usage = obj.get("usage")
                is_usage_chunk = isinstance(usage, dict) and (obj.get("choices") or []) == []
                if isinstance(usage, dict):
                    has_usage = True
                if not is_usage_chunk and (usage is not None or "usage" not in obj):
                    usage_null_ok = False
                obf = obj.get("obfuscation")
                if isinstance(obf, str):
                    n_obf_str += 1
                    if obf:
                        n_obf_nonempty += 1
                    chs = obj.get("choices") or []
                    if chs and isinstance(chs[0], dict) and chs[0].get("finish_reason"):
                        n_finish_obf += 1
                        if obf != "":
                            finish_obf_empty = False
            # Official include_usage stream: "usage": null on every chunk except the usage
            # chunk; obfuscation defaults on ("" on the finish chunk, random elsewhere).
            usage_null_ok = usage_null_ok and n_events > 0
            obf_ok = (
                n_events > 0
                and n_obf_str == n_events
                and n_finish_obf >= 1
                and finish_obf_empty
                and n_obf_nonempty >= 1
            )
            ok = (
                code == 200
                and "text/event-stream" in ct
                and len(events) > 0
                and has_usage
                and usage_null_ok
                and obf_ok
            )
            report.add(
                "create_param",
                field,
                "PASS" if ok else ("NOT_IMPLEMENTED" if code == 404 else "FAIL"),
                f"HTTP {code} n_events={len(events)} has_usage={has_usage} "
                f"usage_null_ok={usage_null_ok} obf_ok={obf_ok} text={joined!r}"[:200],
            )

            # Official default (no stream_options): string "obfuscation" on every chunk,
            # "" on the finish chunk (include_obfuscation defaults to true).
            code_obf, _hdr_obf, raw_obf = client.request(
                "POST",
                "/v1/completions",
                {
                    "model": model,
                    "prompt": "/no_think\nReply with exactly: OAI_CMPL_OBF_DEF",
                    "max_tokens": 24,
                    "temperature": 0,
                    "stream": True,
                },
                stream=True,
            )
            obf_events = parse_sse(raw_obf) if code_obf == 200 else []
            default_all_str = bool(obf_events)
            n_default_nonempty = 0
            n_default_finish = 0
            default_finish_empty = True
            for _, obj in obf_events:
                if not isinstance(obj, dict):
                    default_all_str = False
                    continue
                v = obj.get("obfuscation")
                if not isinstance(v, str):
                    default_all_str = False
                    continue
                if v:
                    n_default_nonempty += 1
                chs = obj.get("choices") or []
                if chs and isinstance(chs[0], dict) and chs[0].get("finish_reason"):
                    n_default_finish += 1
                    if v != "":
                        default_finish_empty = False
            ok_obf_default = (
                code_obf == 200
                and default_all_str
                and n_default_finish >= 1
                and default_finish_empty
                and n_default_nonempty >= 1
            )
            report.add(
                "create_param",
                "stream_options_obfuscation_default",
                "PASS" if ok_obf_default else ("NOT_IMPLEMENTED" if code_obf == 404 else "FAIL"),
                f"HTTP {code_obf} n_events={len(obf_events)} all_str={default_all_str} "
                f"n_finish={n_default_finish} finish_empty={default_finish_empty} "
                f"nonempty={n_default_nonempty}"[:200],
            )

            # Official: stream_options.include_obfuscation=false omits the field entirely.
            code_off, _hdr_off, raw_off = client.request(
                "POST",
                "/v1/completions",
                {
                    "model": model,
                    "prompt": "/no_think\nReply with exactly: OAI_CMPL_OBF_OFF",
                    "max_tokens": 24,
                    "temperature": 0,
                    "stream": True,
                    "stream_options": {"include_obfuscation": False},
                },
                stream=True,
            )
            off_events = parse_sse(raw_off) if code_off == 200 else []
            n_off_with_obf = sum(
                1 for _, obj in off_events if isinstance(obj, dict) and "obfuscation" in obj
            )
            ok_obf_off = code_off == 200 and bool(off_events) and n_off_with_obf == 0
            report.add(
                "create_param",
                "stream_options_obfuscation_disabled",
                "PASS" if ok_obf_off else ("NOT_IMPLEMENTED" if code_off == 404 else "FAIL"),
                f"HTTP {code_off} n_events={len(off_events)} n_with_obfuscation={n_off_with_obf}",
            )
            continue

        if field == "n":
            code, data = client.post_json(
                "/v1/completions",
                {
                    "model": model,
                    "prompt": "/no_think\nReply with exactly: N2",
                    "max_tokens": 16,
                    "temperature": 0,
                    "n": 2,
                    "seed": 7,
                },
            )
            n_ch = len(data.get("choices") or []) if isinstance(data, dict) else 0
            shape_ok, detail = _validate_shape(data) if isinstance(data, dict) else (False, "bad")
            ok = code == 200 and shape_ok and n_ch == 2
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"HTTP {code} n_choices={n_ch} schema={detail}",
            )
            continue

        if field == "seed":
            outs: list[str] = []
            code = 0
            for _ in range(2):
                code, data = client.post_json(
                    "/v1/completions",
                    {
                        "model": model,
                        "prompt": "/no_think\nRepeat exactly: ZORRO42",
                        "max_tokens": 8,
                        "temperature": 0,
                        "seed": 99,
                    },
                )
                outs.append(_completion_text(data if isinstance(data, dict) else {}))
            code_bad, _ = client.post_json(
                "/v1/completions",
                {
                    "model": model,
                    "prompt": _PROMPT,
                    "max_tokens": 4,
                    "seed": "nope",
                },
            )
            det_ok = (
                code == 200
                and len(outs) == 2
                and outs[0] == outs[1]
                and "ZORRO42" in outs[0]
            )
            # Seed: int accepted, non-int 400, same seed → identical completion (1:1 behavior).
            ok = code == 200 and code_bad >= 400 and det_ok
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"bad_http={code_bad} det={det_ok} outs={outs!r}",
            )
            continue

        if field in ("frequency_penalty", "presence_penalty"):
            code, data = client.post_json(
                "/v1/completions",
                {
                    "model": model,
                    "prompt": _PROMPT_PARAM,
                    "max_tokens": 24,
                    "temperature": 0,
                    field: 0.1,
                },
            )
            code_bad, _ = client.post_json(
                "/v1/completions",
                {"model": model, "prompt": _PROMPT, "max_tokens": 4, field: 9.0},
            )
            shape_ok, detail = _validate_shape(data) if isinstance(data, dict) else (False, "bad")
            ok_base = (
                code == 200
                and shape_ok
                and "OAI_CMPL_PARAM" in _completion_text(data)
                and code_bad >= 400
            )
            if field == "frequency_penalty":
                prompt_rep = (
                    "/no_think\nWrite the word banana as many times as you can, separated by spaces. "
                    "Do not write any other words."
                )
                code_lo, d_lo = client.post_json(
                    "/v1/completions",
                    {
                        "model": model,
                        "prompt": prompt_rep,
                        "max_tokens": 48,
                        "temperature": 0.8,
                        "seed": 7,
                        field: 0.0,
                        **(
                            {"chat_template_kwargs": extra["chat_template_kwargs"]}
                            if "chat_template_kwargs" in extra
                            else {}
                        ),
                    },
                )
                code_hi, d_hi = client.post_json(
                    "/v1/completions",
                    {
                        "model": model,
                        "prompt": prompt_rep,
                        "max_tokens": 48,
                        "temperature": 0.8,
                        "seed": 7,
                        field: 1.8,
                        **(
                            {"chat_template_kwargs": extra["chat_template_kwargs"]}
                            if "chat_template_kwargs" in extra
                            else {}
                        ),
                    },
                )
                n_lo = _completion_text(d_lo if isinstance(d_lo, dict) else {}).lower().count("banana")
                n_hi = _completion_text(d_hi if isinstance(d_hi, dict) else {}).lower().count("banana")
                ok_beh = code_lo == 200 and code_hi == 200 and n_lo >= 2 and n_hi < n_lo
                detail_beh = f"banana=({n_lo},{n_hi})"
            else:
                # Match Chat deepen: animal list at temp=0; presence_penalty=2.0
                # raises unique tokens and lowers max frequency.
                from collections import Counter

                prompt_rep = (
                    "/no_think\nWrite a long list of animals, repeating names is allowed, "
                    "separated by spaces."
                )

                def _cmpl(pen: float) -> tuple[int, str]:
                    body_p = {
                        "model": model,
                        "prompt": prompt_rep,
                        "max_tokens": 80,
                        "temperature": 0,
                        field: pen,
                    }
                    if "chat_template_kwargs" in extra:
                        body_p["chat_template_kwargs"] = extra["chat_template_kwargs"]
                    c, d = client.post_json("/v1/completions", body_p)
                    return c, _completion_text(d if isinstance(d, dict) else {})

                def _stats(t: str) -> tuple[int, int]:
                    toks = [w for w in t.lower().replace(",", " ").split() if w]
                    if not toks:
                        return 0, 0
                    return len(set(toks)), max(Counter(toks).values())

                c0, t0 = _cmpl(0.0)
                c1, t1 = _cmpl(2.0)
                uniq0, maxf0 = _stats(t0)
                uniq1, maxf1 = _stats(t1)
                ok_beh = (
                    c0 == 200
                    and c1 == 200
                    and maxf0 >= 10
                    and uniq1 > uniq0
                    and maxf1 < maxf0
                )
                detail_beh = f"uniq=({uniq0},{uniq1}) maxfreq=({maxf0},{maxf1})"
            ok = ok_base and ok_beh
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"ok_http={code} bad_http={code_bad} {detail_beh} schema={detail}",
            )
            continue

        if field == "logit_bias":
            tcode, tdata = client.post_json("/tokenize", {"content": "BANANA", "add_special": False})
            toks = (tdata.get("tokens") if isinstance(tdata, dict) else None) or []
            bias = {str(t): -100 for t in toks if isinstance(t, int)}
            code_ctrl, data_ctrl = client.post_json(
                "/v1/completions",
                {
                    "model": model,
                    "prompt": "/no_think\nReply with exactly: BANANA",
                    "max_tokens": 16,
                    "temperature": 0,
                },
            )
            code, data = client.post_json(
                "/v1/completions",
                {
                    "model": model,
                    "prompt": "/no_think\nReply with exactly: BANANA",
                    "max_tokens": 16,
                    "temperature": 0,
                    "logit_bias": bias or {"0": -100},
                },
            )
            code_bad, _ = client.post_json(
                "/v1/completions",
                {"model": model, "prompt": _PROMPT, "max_tokens": 4, "logit_bias": "nope"},
            )
            ctrl = _completion_text(data_ctrl if isinstance(data_ctrl, dict) else {})
            biased = _completion_text(data if isinstance(data, dict) else {})
            shape_ok, detail = _validate_shape(data) if isinstance(data, dict) else (False, "bad")
            ok = (
                tcode == 200
                and bool(bias)
                and code_ctrl == 200
                and "BANANA" in ctrl
                and code == 200
                and shape_ok
                and "BANANA" not in biased
                and code_bad >= 400
            )
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"tok={toks!r} ctrl={ctrl!r} biased={biased!r} bad={code_bad} schema={detail}",
            )
            continue

        if field == "stop":
            body["prompt"] = "/no_think\nSay hello then write STOP_OAI_CMPL on its own line."
            body["max_tokens"] = 64

        code, data = client.post_json("/v1/completions", body)
        if code == 404:
            report.add("create_param", field, "NOT_IMPLEMENTED", "HTTP 404")
            continue
        if code >= 400:
            report.add("create_param", field, "NOT_IMPLEMENTED", f"HTTP {code}")
            continue
        if not isinstance(data, dict):
            report.add("create_param", field, "FAIL", f"HTTP {code} non-dict body")
            continue

        shape_ok, detail = _validate_shape(data)
        if field == "temperature":
            outs: list[str] = []
            codes: list[int] = []
            for _ in range(2):
                c2, d2 = client.post_json(
                    "/v1/completions",
                    {
                        "model": model,
                        "prompt": "/no_think\nReply with exactly: TEMP0_DET",
                        "max_tokens": 16,
                        "temperature": 0,
                    },
                )
                codes.append(c2)
                outs.append(_completion_text(d2 if isinstance(d2, dict) else {}))
            ok = (
                shape_ok
                and all(c == 200 for c in codes)
                and len(outs) == 2
                and outs[0] == outs[1]
                and "TEMP0_DET" in outs[0]
            )
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"codes={codes} outs={outs!r}",
            )
            continue
        elif field == "stop":
            prompt_seq = (
                "/no_think\nContinue this sequence exactly with the next numbers only: "
                "1 2 3 4 5 6"
            )
            body_ctrl = {
                "model": model,
                "prompt": prompt_seq,
                "max_tokens": 24,
                "temperature": 0,
                "reasoning_effort": "none",
            }
            body_stop = {**body_ctrl, "stop": [" 7"]}
            if "chat_template_kwargs" in extra:
                body_ctrl["chat_template_kwargs"] = extra["chat_template_kwargs"]
                body_stop["chat_template_kwargs"] = extra["chat_template_kwargs"]
            code_ctrl, data_ctrl = client.post_json("/v1/completions", body_ctrl)
            code_stop, data_stop = client.post_json("/v1/completions", body_stop)
            text_ctrl = _completion_text(data_ctrl if isinstance(data_ctrl, dict) else {})
            text_stop = _completion_text(data_stop if isinstance(data_stop, dict) else {})
            fr_stop = None
            if isinstance(data_stop, dict):
                ch0 = (data_stop.get("choices") or [{}])[0]
                if isinstance(ch0, dict):
                    fr_stop = ch0.get("finish_reason")
            ok = (
                code_ctrl == 200
                and code_stop == 200
                and _validate_shape(data_stop if isinstance(data_stop, dict) else {})[0]
                and "7" in text_ctrl
                and "10" in text_ctrl
                and fr_stop == "stop"
                and "7" not in text_stop
                and "10" not in text_stop
            )
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"ctrl={text_ctrl!r} stop={text_stop!r} fr={fr_stop!r}",
            )
            continue
        elif field == "n":
            choices = data.get("choices") or []
            ok = shape_ok and isinstance(choices, list) and len(choices) >= 1
        elif field == "logprobs":
            # Official Completions Logprobs: tokens/token_logprobs/top_logprobs/text_offset
            # (not Chat's content[] shape).
            body_lp = {
                "model": model,
                "prompt": "/no_think\nReply with exactly: LP",
                "max_tokens": 8,
                "temperature": 0,
                "logprobs": 2,
            }
            if "chat_template_kwargs" in extra:
                body_lp["chat_template_kwargs"] = extra["chat_template_kwargs"]
            code_lp, data_lp = client.post_json("/v1/completions", body_lp)
            ok = False
            detail_lp = "missing"
            if code_lp == 200 and isinstance(data_lp, dict):
                shape_ok_lp, detail_lp = _validate_shape(data_lp)
                for ch in data_lp.get("choices") or []:
                    if not isinstance(ch, dict):
                        continue
                    lp = ch.get("logprobs")
                    if not isinstance(lp, dict):
                        continue
                    toks = lp.get("tokens")
                    tlps = lp.get("token_logprobs")
                    tops = lp.get("top_logprobs")
                    offs = lp.get("text_offset")
                    ok = (
                        shape_ok_lp
                        and isinstance(toks, list)
                        and isinstance(tlps, list)
                        and isinstance(tops, list)
                        and isinstance(offs, list)
                        and len(toks) > 0
                        and len(toks) == len(tlps) == len(tops) == len(offs)
                        and "content" not in lp
                    )
                    detail_lp = f"keys={sorted(lp.keys())} n={len(toks)}"
                    break
            code_bad_lp, _ = client.post_json(
                "/v1/completions",
                {
                    "model": model,
                    "prompt": _PROMPT,
                    "max_tokens": 4,
                    "logprobs": 10,
                },
            )
            ok = ok and code_bad_lp >= 400
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"HTTP {code_lp} {detail_lp} bad_gt5={code_bad_lp}"[:200],
            )
            continue
        elif field == "user":
            text = _completion_text(data)
            ok = shape_ok and data.get("user") == "acceptance-bot"
        elif field == "top_p":
            code_bad, _ = client.post_json(
                "/v1/completions",
                {"model": model, "prompt": _PROMPT, "max_tokens": 4, "top_p": "nope"},
            )
            ok = shape_ok and "OAI_CMPL_PARAM" in _completion_text(data) and code_bad >= 400
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"ok_http={code} bad_http={code_bad} schema={detail}",
            )
            continue
        elif field == "max_tokens":
            code_cap, data_cap = client.post_json(
                "/v1/completions",
                {
                    "model": model,
                    "prompt": "/no_think\nWrite a long paragraph about rainbows.",
                    "max_tokens": 2,
                    "temperature": 0,
                    "reasoning_effort": "none",
                },
            )
            text_cap = _completion_text(data_cap if isinstance(data_cap, dict) else {})
            fr_cap = None
            n_tok = -1
            if isinstance(data_cap, dict):
                ch0 = (data_cap.get("choices") or [{}])[0]
                if isinstance(ch0, dict):
                    fr_cap = ch0.get("finish_reason")
                usage = data_cap.get("usage") or {}
                if isinstance(usage, dict):
                    n_tok = int(usage.get("completion_tokens") or -1)
            ok = (
                code_cap == 200
                and _validate_shape(data_cap if isinstance(data_cap, dict) else {})[0]
                and fr_cap == "length"
                and 0 < n_tok <= 2
                and len(text_cap.strip()) > 0
            )
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"HTTP {code_cap} fr={fr_cap!r} completion_tokens={n_tok} text={text_cap!r}",
            )
            continue
        elif field == "echo":
            echo_prompt = "ECHO_PARAM_UNIQUE "
            code_echo, data_echo = client.post_json(
                "/v1/completions",
                {
                    "model": model,
                    "prompt": echo_prompt,
                    "max_tokens": 8,
                    "temperature": 0,
                    "echo": True,
                    "reasoning_effort": "none",
                },
            )
            echo_text = _completion_text(data_echo if isinstance(data_echo, dict) else {})
            ok = (
                code_echo == 200
                and _validate_shape(data_echo if isinstance(data_echo, dict) else {})[0]
                and echo_text.startswith(echo_prompt)
            )
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"HTTP {code_echo} text={echo_text!r}",
            )
            continue
        else:
            ok = shape_ok

        report.add(
            "create_param",
            field,
            "PASS" if ok else "FAIL",
            f"HTTP {code} schema={detail} user={data.get('user')!r}"[:200]
            if field == "user"
            else f"HTTP {code} schema={detail}"[:200],
        )

    # Official Completions echo=true: choice text includes the prompt prefix.
    echo_prompt = "ECHO_PROMPT_UNIQUE "
    code_echo, data_echo = client.post_json(
        "/v1/completions",
        {
            "model": model,
            "prompt": echo_prompt,
            "max_tokens": 8,
            "temperature": 0,
            "echo": True,
            **extra,
        },
    )
    echo_text = ""
    if code_echo == 200 and isinstance(data_echo, dict):
        ch = (data_echo.get("choices") or [{}])[0]
        echo_text = ch.get("text") or ""
    report.add(
        "scenario",
        "completions_echo_true",
        "PASS" if code_echo == 200 and echo_text.startswith(echo_prompt) else "FAIL",
        f"HTTP {code_echo} text={echo_text!r}",
    )
    code_suf, data_suf = client.post_json(
        "/v1/completions",
        {
            "model": model,
            "prompt": "def add(a, b):\n    ",
            "suffix": "\n    return c\n",
            "max_tokens": 24,
            "temperature": 0,
            **extra,
        },
    )
    suf_text = ""
    if code_suf == 200 and isinstance(data_suf, dict):
        ch = (data_suf.get("choices") or [{}])[0]
        suf_text = ch.get("text") or ""
    report.add(
        "scenario",
        "completions_suffix_fim",
        "PASS" if code_suf == 200 and len(suf_text.strip()) > 0 else "FAIL",
        f"HTTP {code_suf} text={suf_text!r}",
    )
    # Soft/vocab FIM rewrite must observably enlarge the prompt vs bare prefix.
    prefix_only = "The color of the clear daytime sky is "
    code_bare, data_bare = client.post_json(
        "/v1/completions",
        {
            "model": model,
            "prompt": prefix_only,
            "max_tokens": 8,
            "temperature": 0,
            **extra,
        },
    )
    code_fim, data_fim = client.post_json(
        "/v1/completions",
        {
            "model": model,
            "prompt": prefix_only,
            "suffix": " and the grass is green.",
            "max_tokens": 8,
            "temperature": 0,
            **extra,
        },
    )

    def _prompt_tokens(d: dict) -> int:
        u = d.get("usage") if isinstance(d, dict) else None
        if not isinstance(u, dict):
            return -1
        return int(u.get("prompt_tokens") or 0)

    pt_bare = _prompt_tokens(data_bare if isinstance(data_bare, dict) else {})
    pt_fim = _prompt_tokens(data_fim if isinstance(data_fim, dict) else {})
    ok_fim_expand = (
        code_bare == 200
        and code_fim == 200
        and pt_bare > 0
        and pt_fim > pt_bare
    )
    report.add(
        "scenario",
        "completions_suffix_fim_expands_prompt",
        "PASS" if ok_fim_expand else "FAIL",
        f"bare_pt={pt_bare} fim_pt={pt_fim} http={code_bare}/{code_fim}",
    )
    code_bo, _ = client.post_json(
        "/v1/completions",
        {"model": model, "prompt": _PROMPT, "max_tokens": 4, "n": 2, "best_of": 1},
    )
    report.add(
        "scenario",
        "completions_best_of_lt_n_rejected",
        "PASS" if code_bo >= 400 else "FAIL",
        f"HTTP {code_bo}",
    )
    code_bo2, data_bo2 = client.post_json(
        "/v1/completions",
        {
            "model": model,
            "prompt": _PROMPT_PARAM,
            "max_tokens": 16,
            "temperature": 0.9,
            "n": 1,
            "best_of": 3,
            "logprobs": 1,
            **extra,
        },
    )
    n_choices = (
        len(data_bo2.get("choices") or [])
        if code_bo2 == 200 and isinstance(data_bo2, dict)
        else -1
    )
    has_lp = False
    if code_bo2 == 200 and isinstance(data_bo2, dict) and n_choices == 1:
        ch0 = (data_bo2.get("choices") or [{}])[0]
        has_lp = ch0.get("logprobs") is not None
    report.add(
        "scenario",
        "completions_best_of_gt_n_returns_n",
        "PASS" if code_bo2 == 200 and n_choices == 1 else "FAIL",
        f"HTTP {code_bo2} n_choices={n_choices}",
    )
    report.add(
        "scenario",
        "completions_best_of_ranked_logprobs",
        "PASS" if code_bo2 == 200 and n_choices == 1 and has_lp else "FAIL",
        f"HTTP {code_bo2} n_choices={n_choices} has_logprobs={has_lp}",
    )

    def _choice_logprob_sum(ch: dict) -> float | None:
        lp = ch.get("logprobs") if isinstance(ch, dict) else None
        if not isinstance(lp, dict):
            return None
        vals = lp.get("token_logprobs")
        if not isinstance(vals, list) or not vals:
            return None
        s = 0.0
        for v in vals:
            if isinstance(v, (int, float)):
                s += float(v)
            else:
                return None
        return s

    # best_of>n ranking: return top-n by sum(token_logprobs); choice[0] >= choice[1].
    code_rank, data_rank = client.post_json(
        "/v1/completions",
        {
            "model": model,
            "prompt": _PROMPT_PARAM + " /no_think Write two short different poems about rain.",
            "max_tokens": 24,
            "temperature": 1.2,
            "top_p": 0.95,
            "n": 2,
            "best_of": 4,
            "logprobs": 1,
            **extra,
        },
    )
    choices_rank = (
        data_rank.get("choices")
        if code_rank == 200 and isinstance(data_rank, dict)
        else None
    )
    scores: list[float] = []
    if isinstance(choices_rank, list) and len(choices_rank) == 2:
        for ch in choices_rank:
            s = _choice_logprob_sum(ch if isinstance(ch, dict) else {})
            if s is None:
                scores = []
                break
            scores.append(s)
    ok_rank = len(scores) == 2 and scores[0] >= scores[1]
    report.add(
        "scenario",
        "completions_best_of_rank_order",
        "PASS" if ok_rank else "FAIL",
        f"HTTP {code_rank} n={len(choices_rank) if isinstance(choices_rank, list) else -1} scores={scores!r}",
    )

    # OpenAI: stream + best_of > n is unsupported (needs all candidates to rank).
    code_bo_stream, _ = client.post_json(
        "/v1/completions",
        {
            "model": model,
            "prompt": _PROMPT_PARAM,
            "max_tokens": 8,
            "n": 1,
            "best_of": 3,
            "stream": True,
            **extra,
        },
    )
    report.add(
        "scenario",
        "completions_best_of_stream_rejected",
        "PASS" if code_bo_stream >= 400 else "FAIL",
        f"HTTP {code_bo_stream}",
    )
    code_ok_bo, data_ok_bo = client.post_json(
        "/v1/completions",
        {
            "model": model,
            "prompt": _PROMPT_PARAM,
            "max_tokens": 24,
            "temperature": 0,
            "best_of": 1,
            "suffix": "",
        },
    )
    ok_bo = (
        code_ok_bo == 200
        and isinstance(data_ok_bo, dict)
        and _validate_shape(data_ok_bo)[0]
        and "OAI_CMPL_PARAM" in _completion_text(data_ok_bo)
    )
    report.add(
        "scenario",
        "completions_best_of_one_suffix_empty_ok",
        "PASS" if ok_bo else "FAIL",
        f"HTTP {code_ok_bo}",
    )

    for field in completion_param_names():
        if any(r.name == field for r in report.rows if r.area == "create_param"):
            continue
        report.add("create_param", field, "FAIL", "missing from suite catalog execution")
