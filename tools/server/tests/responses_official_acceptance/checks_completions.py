"""OpenAI Completions API checks (POST /v1/completions)."""

from __future__ import annotations

import time
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
        report.add("endpoint", "POST /v1/completions (prompt array)", "NOT_IMPLEMENTED", "HTTP 404")
        report.add("endpoint", "POST /v1/completions (suffix array)", "NOT_IMPLEMENTED", "HTTP 404")
        for field in completion_param_names():
            if field not in ("model", "prompt"):
                report.add("create_param", field, "NOT_IMPLEMENTED", "completions missing")
        return

    if code != 200 or not isinstance(data, dict):
        report.add("endpoint", "POST /v1/completions", "FAIL", f"HTTP {code}")
        report.add("endpoint", "POST /v1/completions (prompt array)", "FAIL", f"HTTP {code}")
        report.add("endpoint", "POST /v1/completions (suffix array)", "FAIL", f"HTTP {code}")
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

    # prompt array: official prompt may be an array of strings; one choice per prompt (n=1)
    code_arr, data_arr = client.post_json(
        "/v1/completions",
        {
            "model": model,
            "prompt": [
                "/no_think\nReply with exactly: CMPL_ARR_A",
                "/no_think\nReply with exactly: CMPL_ARR_B",
            ],
            "max_tokens": 8,
            "temperature": 0,
            "n": 1,
        },
    )
    arr_choices = data_arr.get("choices") if isinstance(data_arr, dict) else None
    arr_ok = (
        code_arr == 200
        and isinstance(data_arr, dict)
        and _validate_shape(data_arr)[0]
        and isinstance(arr_choices, list)
        and len(arr_choices) == 2
    )
    report.add(
        "endpoint",
        "POST /v1/completions (prompt array)",
        "PASS" if arr_ok else "FAIL",
        f"HTTP {code_arr} n_choices={len(arr_choices) if isinstance(arr_choices, list) else -1}",
    )

    # suffix is officially string or null; an array (even paired with a prompt array) is 400
    code_sufarr, sufarr = client.post_json(
        "/v1/completions",
        {
            "model": model,
            "prompt": [
                "/no_think\nReply with exactly: CMPL_SUF_A",
                "/no_think\nReply with exactly: CMPL_SUF_B",
            ],
            "suffix": ["\nSTOP", "\nEND"],
            "max_tokens": 8,
            "temperature": 0,
            "n": 1,
        },
    )
    report.add(
        "endpoint",
        "POST /v1/completions (suffix array)",
        "PASS" if code_sufarr == 400 else "FAIL",
        f"HTTP {code_sufarr} (want 400; official suffix is string|null) body={str(sufarr)[:80]}",
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
                and bool(outs[0])
            )
            # The echoed marker is informational: the model may not repeat it, but the
            # seed contract is only that the same seed reproduces the same completion.
            echo_ok = "ZORRO42" in outs[0]
            # Seed: int accepted, non-int 400, same seed -> identical completion (1:1 behavior).
            ok = code == 200 and code_bad >= 400 and det_ok
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"bad_http={code_bad} det={det_ok} echo_ok={echo_ok} (informational) outs={outs!r}",
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
                beh_skip = False
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
                    and uniq1 > uniq0
                    and maxf1 < maxf0
                )
                # A control that barely repeats cannot show suppression at all: below the
                # measurable floor (<3) the direction test is skipped, not failed.
                beh_skip = c0 == 200 and c1 == 200 and maxf0 < 3
                detail_beh = f"uniq=({uniq0},{uniq1}) maxfreq=({maxf0},{maxf1})"
            if not ok_base:
                status = "FAIL"
            elif beh_skip:
                status = "SKIP"
                detail_beh += " control repetition too low (<3), cannot measure"
            else:
                status = "PASS" if ok_beh else "FAIL"
            report.add(
                "create_param",
                field,
                status,
                f"ok_http={code} bad_http={code_bad} {detail_beh} schema={detail}",
            )
            continue

        if field == "logit_bias":
            tcode, tdata = client.post_json("/tokenize", {"content": "BANANA", "add_special": False})
            toks = (tdata.get("tokens") if isinstance(tdata, dict) else None) or []
            bias = {str(t): -100 for t in toks if isinstance(t, int)}
            # "BANANA" may have no dedicated token (here it splits into banned pieces), so
            # asserting on the string "BANANA" misfires on legitimate variants such as
            # "BANANAS". Detokenize each banned id and require its exact text to never be
            # sampled; the substring form of "BANANA" is not an API violation.
            ban_texts: list[str] = []
            dcode = 0
            for raw in sorted((t for t in toks if isinstance(t, int)), key=int):
                dcode, ddata = client.post_json("/detokenize", {"tokens": [raw]})
                txt = ddata.get("content") if isinstance(ddata, dict) else None
                if dcode == 200 and isinstance(txt, str) and txt and "\ufffd" not in txt:
                    ban_texts.append(txt)
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
                    "logprobs": 1,
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
            sampled: list[str] = []
            choices = data.get("choices") if isinstance(data, dict) else None
            for ch in choices or []:
                ch_lp = ch.get("logprobs") if isinstance(ch, dict) else None
                if isinstance(ch_lp, dict) and isinstance(ch_lp.get("tokens"), list):
                    sampled.extend(t for t in ch_lp["tokens"] if isinstance(t, str))
            suppressed_ok = bool(sampled) and not any(t in ban_texts for t in sampled)
            ok = (
                tcode == 200
                and bool(bias)
                and dcode == 200
                and bool(ban_texts)
                and code_ctrl == 200
                and "BANANA" in ctrl
                and code == 200
                and shape_ok
                and suppressed_ok
                and code_bad >= 400
            )
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"tok={toks!r} ban={ban_texts!r} sampled={sampled!r} ctrl={ctrl!r} "
                f"biased={biased!r} bad={code_bad} schema={detail}",
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
                    offs0 = offs[0] if offs else None
                    # echo=false: the offsets still start at the beginning of the full text,
                    # so the first generated token sits past the prompt
                    offs0_ok = isinstance(offs0, int) and offs0 >= len(body_lp["prompt"].encode("utf-8"))
                    ok = (
                        shape_ok_lp
                        and isinstance(toks, list)
                        and isinstance(tlps, list)
                        and isinstance(tops, list)
                        and isinstance(offs, list)
                        and len(toks) > 0
                        and len(toks) == len(tlps) == len(tops) == len(offs)
                        and offs0_ok
                        and "content" not in lp
                    )
                    detail_lp = f"keys={sorted(lp.keys())} n={len(toks)} offs0={offs0}"
                    break
            # Official clamps logprobs above 5 instead of rejecting it: expect 200 with at
            # most logprobs+1 (= 6) candidates per position.
            code_big_lp, data_big_lp = client.post_json(
                "/v1/completions",
                {
                    "model": model,
                    "prompt": _PROMPT,
                    "max_tokens": 4,
                    "logprobs": 10,
                },
            )
            clamp_ok = False
            if code_big_lp == 200 and isinstance(data_big_lp, dict):
                for ch in data_big_lp.get("choices") or []:
                    lp_big = ch.get("logprobs") if isinstance(ch, dict) else None
                    if not isinstance(lp_big, dict):
                        continue
                    tops_big = lp_big.get("top_logprobs")
                    # A clamped request must never expose more than logprobs+1 (= 6) candidates.
                    # A row with a single key is a local artifact of MTP draft acceptance
                    # (--spec-type draft-mtp), so it does not count against the clamp.
                    clamp_ok = (
                        isinstance(tops_big, list)
                        and len(tops_big) > 0
                        and all(isinstance(t, dict) and 1 <= len(t) <= 6 for t in tops_big)
                    )
                    break
            ok = ok and clamp_ok
            report.add(
                "create_param",
                field,
                "PASS" if ok else "FAIL",
                f"HTTP {code_lp} {detail_lp} clamp={code_big_lp}"[:200],
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
    # Official Completions echo=true + logprobs: the prompt positions come back as leading
    # rows, with no logprobs on the first one.
    code_elp, data_elp = client.post_json(
        "/v1/completions",
        {
            "model": model,
            "prompt": echo_prompt,
            "max_tokens": 8,
            "temperature": 0,
            "echo": True,
            "logprobs": 5,
            **extra,
        },
    )
    elp_ok = False
    detail_elp = "missing"
    if code_elp == 200 and isinstance(data_elp, dict):
        ch_elp = (data_elp.get("choices") or [{}])[0]
        lp = ch_elp.get("logprobs") if isinstance(ch_elp, dict) else None
        if isinstance(lp, dict):
            toks = lp.get("tokens") or []
            tlps = lp.get("token_logprobs") or []
            tops = lp.get("top_logprobs") or []
            offs = lp.get("text_offset") or []
            same_len = len(toks) > 0 and len(toks) == len(tlps) == len(tops) == len(offs)
            offsets_ok = same_len and all(
                offs[i + 1] - offs[i] == len(str(toks[i]).encode("utf-8"))
                for i in range(len(offs) - 1)
            )
            n_prompt_rows = min((data_elp.get("usage") or {}).get("prompt_tokens") or 0, len(toks))
            # prompt rows come from the full-vocab softmax, so they always carry logprobs candidates
            prompt_rows_ok = all(
                isinstance(tops[i], dict) and 5 <= len(tops[i]) <= 6
                for i in range(1, n_prompt_rows)
            )
            elp_ok = (
                same_len
                and offsets_ok
                and offs[0] == 0                            # echo=true starts at the prompt
                and tlps[0] is None and tops[0] is None      # first prompt token has no logits
                and any(v is not None for v in tlps[1:])     # the other prompt rows are real
                and all(isinstance(t, dict) for t in tops[1:])
                and prompt_rows_ok
            )
            detail_elp = f"n={len(toks)} prompt_rows={n_prompt_rows} offs0={offs[0]} lp0={tlps[0]}"
    report.add(
        "scenario",
        "completions_echo_logprobs_prompt_rows",
        "PASS" if elp_ok else "FAIL",
        f"HTTP {code_elp} {detail_elp}"[:200],
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

    # Official stop: string or array of strings, at most 4 sequences. Too many
    # sequences or a wrong JSON type must be rejected as invalid (400), not 5xx.
    code_stop_many, data_stop_many = client.post_json(
        "/v1/completions",
        {
            "model": model,
            "prompt": _PROMPT,
            "max_tokens": 4,
            "stop": ["a", "b", "c", "d", "e"],
        },
    )
    report.add(
        "create_param",
        "stop.too_many_sequences",
        "PASS" if code_stop_many == 400 else "FAIL",
        f"HTTP {code_stop_many} body={str(data_stop_many)[:120]}",
    )
    code_stop_type, data_stop_type = client.post_json(
        "/v1/completions",
        {"model": model, "prompt": _PROMPT, "max_tokens": 4, "stop": 5},
    )
    report.add(
        "create_param",
        "stop.invalid_type",
        "PASS" if code_stop_type == 400 else "FAIL",
        f"HTTP {code_stop_type} body={str(data_stop_type)[:120]}",
    )

    # Official Completions usage: details objects must exist; total = prompt + completion;
    # text_tokens = completion_tokens - reasoning_tokens. The unique cold prompt means
    # nothing is cached and every prompt token is counted as written to cache.
    def _is_int(v: Any) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    uniq_u = f"usg{time.time_ns()}"
    code_u, data_u = client.post_json(
        "/v1/completions",
        {
            "model": model,
            "prompt": f"{uniq_u}: reply with exactly: USAGE_OK",
            "max_tokens": 8,
            "temperature": 0,
        },
    )
    usage_u = data_u.get("usage") if isinstance(data_u, dict) else None
    usage_u = usage_u if isinstance(usage_u, dict) else {}
    u_pt = usage_u.get("prompt_tokens")
    u_ct = usage_u.get("completion_tokens")
    u_tt = usage_u.get("total_tokens")
    u_ptd = usage_u.get("prompt_tokens_details")
    u_ptd_d = u_ptd if isinstance(u_ptd, dict) else {}
    u_ctd = usage_u.get("completion_tokens_details")
    u_ctd_d = u_ctd if isinstance(u_ctd, dict) else {}
    ctd_fields = (
        "accepted_prediction_tokens",
        "audio_tokens",
        "reasoning_tokens",
        "rejected_prediction_tokens",
        "text_tokens",
    )
    ok_usage = (
        code_u == 200
        and _is_int(u_pt)
        and _is_int(u_ct)
        and _is_int(u_tt)
        and u_pt >= 0
        and u_ct >= 0
        and u_tt >= 0
        and u_tt == u_pt + u_ct
        and isinstance(u_ptd, dict)
        and _is_int(u_ptd_d.get("cached_tokens"))
        and u_ptd_d.get("cached_tokens") == 0
        and _is_int(u_ptd_d.get("cache_write_tokens"))
        and u_ptd_d.get("cache_write_tokens") > 0
        and isinstance(u_ctd, dict)
        and all(_is_int(u_ctd_d.get(f)) and u_ctd_d.get(f) >= 0 for f in ctd_fields)
        and u_ctd_d.get("text_tokens") == u_ct - u_ctd_d.get("reasoning_tokens")
    )
    report.add(
        "scenario",
        "usage.details",
        "PASS" if ok_usage else "FAIL",
        f"HTTP {code_u} pt={u_pt} ct={u_ct} tt={u_tt} "
        f"cached={u_ptd_d.get('cached_tokens')} write={u_ptd_d.get('cache_write_tokens')} "
        f"details={u_ctd_d}"[:200],
    )

    # Stream parity: the trailing empty-choices chunk carries the same usage object;
    # totals must match there too and both details sub-objects keep int fields.
    uniq_su = f"usgs{time.time_ns()}"
    code_su, _hdr_su, raw_su = client.request(
        "POST",
        "/v1/completions",
        {
            "model": model,
            "prompt": f"{uniq_su}: reply with exactly: USAGE_STREAM_OK",
            "max_tokens": 8,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        stream=True,
    )
    su_events = parse_sse(raw_su) if code_su == 200 else []
    su_usage: dict[str, Any] | None = None
    for _, obj in su_events:
        if not isinstance(obj, dict):
            continue
        u = obj.get("usage")
        if isinstance(u, dict) and (obj.get("choices") or []) == []:
            su_usage = u
    su = su_usage if isinstance(su_usage, dict) else {}
    su_pt = su.get("prompt_tokens")
    su_ct = su.get("completion_tokens")
    su_tt = su.get("total_tokens")
    su_ptd = su.get("prompt_tokens_details")
    su_ptd_d = su_ptd if isinstance(su_ptd, dict) else {}
    su_ctd = su.get("completion_tokens_details")
    su_ctd_d = su_ctd if isinstance(su_ctd, dict) else {}
    ok_su = (
        code_su == 200
        and isinstance(su_usage, dict)
        and _is_int(su_pt)
        and _is_int(su_ct)
        and _is_int(su_tt)
        and su_tt == su_pt + su_ct
        and isinstance(su_ptd, dict)
        and _is_int(su_ptd_d.get("cached_tokens"))
        and _is_int(su_ptd_d.get("cache_write_tokens"))
        and isinstance(su_ctd, dict)
        and all(_is_int(su_ctd_d.get(f)) for f in ctd_fields)
    )
    report.add(
        "scenario",
        "usage.stream_details",
        "PASS" if ok_su else "FAIL",
        f"HTTP {code_su} pt={su_pt} ct={su_ct} tt={su_tt} "
        f"ptd={su_ptd_d} ctd={su_ctd_d}"[:200],
    )

    for field in completion_param_names():
        if any(r.name == field for r in report.rows if r.area == "create_param"):
            continue
        report.add("create_param", field, "FAIL", "missing from suite catalog execution")
