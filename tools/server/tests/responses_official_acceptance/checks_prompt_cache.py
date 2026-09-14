"""Prompt-cache hit monitoring from official Responses usage fields.

OpenAI does not expose a dedicated hit-rate API. Per docs / SDK:
  usage.input_tokens_details.cached_tokens
  usage.input_tokens_details.cache_write_tokens

Hit rate for a window is:
  sum(cached_tokens) / sum(input_tokens)

This module probes stable prefixes + prompt_cache_key and records that ratio,
then exercises cache-adverse traffic: key rotation, growing conversations
without a key, concurrent branches of one key, mid-prompt rewrites, aborted
streams, hostile key values, and cross-model isolation.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from .http_client import ResponsesHttpClient, output_text
from .report import Report


@dataclass
class CacheSample:
    index: int
    input_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    hit_rate: float  # cached / input for this request (0 if input==0)


def _usage_cache(data: dict[str, Any]) -> tuple[int, int, int]:
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return 0, 0, 0
    inp = int(usage.get("input_tokens") or 0)
    details = usage.get("input_tokens_details") or {}
    if not isinstance(details, dict):
        details = {}
    cached = int(details.get("cached_tokens") or 0)
    written = int(details.get("cache_write_tokens") or 0)
    return inp, cached, written


def measure_prompt_cache(
    client: ResponsesHttpClient,
    model: str,
    extra: dict[str, Any],
    *,
    rounds: int = 4,
    prompt_cache_key: str = "acceptance-cache-monitor",
    pause_s: float = 0.15,
) -> dict[str, Any]:
    """Run identical creates and return aggregate cache stats.

    Official guidance: stable identical prefixes maximize reuse. We keep the
    full input identical across rounds so usage.cached_tokens can rise after
    the cold request (OpenAI cloud usually needs >=1024 tokens; local KV may
    hit on shorter prompts).
    """
    prefix = (
        "You are a careful assistant. Keep answers short. "
        "Context block: " + ("alpha-bravo-charlie-delta " * 40)
    )
    # Identical every round — changing the suffix would bust the whole prefix.
    user_input = f"{prefix}\nReply with exactly: CACHE_HIT_PROBE"
    samples: list[CacheSample] = []
    for i in range(rounds):
        body = {
            "model": model,
            "input": user_input,
            "max_output_tokens": 24,
            "temperature": 0,
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_retention": "in_memory",
            **extra,
        }
        code, data = client.post_json("/v1/responses", body)
        if code != 200 or not isinstance(data, dict):
            raise RuntimeError(f"cache probe HTTP {code}: {data!r}"[:240])
        inp, cached, written = _usage_cache(data)
        rate = (cached / inp) if inp > 0 else 0.0
        samples.append(
            CacheSample(
                index=i,
                input_tokens=inp,
                cached_tokens=cached,
                cache_write_tokens=written,
                hit_rate=rate,
            )
        )
        if pause_s > 0:
            time.sleep(pause_s)

    sum_in = sum(s.input_tokens for s in samples)
    sum_cached = sum(s.cached_tokens for s in samples)
    sum_write = sum(s.cache_write_tokens for s in samples)
    # Exclude cold start (index 0) for "warm hit rate".
    warm = samples[1:] if len(samples) > 1 else samples
    warm_in = sum(s.input_tokens for s in warm)
    warm_cached = sum(s.cached_tokens for s in warm)

    # One intentional bust: mutated suffix should drop cached_tokens vs warm hit.
    bust_body = {
        "model": model,
        "input": f"{prefix}\nReply with exactly: CACHE_HIT_PROBE_BUSTED",
        "max_output_tokens": 24,
        "temperature": 0,
        "prompt_cache_key": prompt_cache_key,
        "prompt_cache_retention": "in_memory",
        **extra,
    }
    bcode, bdata = client.post_json("/v1/responses", bust_body)
    bust_inp, bust_cached, bust_write = (0, 0, 0)
    if bcode == 200 and isinstance(bdata, dict):
        bust_inp, bust_cached, bust_write = _usage_cache(bdata)

    return {
        "rounds": rounds,
        "prompt_cache_key": prompt_cache_key,
        "samples": [asdict(s) for s in samples],
        "totals": {
            "input_tokens": sum_in,
            "cached_tokens": sum_cached,
            "cache_write_tokens": sum_write,
            "hit_rate": (sum_cached / sum_in) if sum_in else 0.0,
        },
        "warm": {
            "input_tokens": warm_in,
            "cached_tokens": warm_cached,
            "hit_rate": (warm_cached / warm_in) if warm_in else 0.0,
        },
        "bust": {
            "input_tokens": bust_inp,
            "cached_tokens": bust_cached,
            "cache_write_tokens": bust_write,
            "hit_rate": (bust_cached / bust_inp) if bust_inp else 0.0,
        },
    }


def run_prompt_cache_checks(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> dict[str, Any] | None:
    try:
        stats = measure_prompt_cache(client, model, extra)
    except Exception as e:
        report.add("cache", "prompt_cache_hit_monitor", "FAIL", str(e)[:200])
        return None

    samples = stats["samples"]
    warm = stats["warm"]
    totals = stats["totals"]
    detail = (
        f"warm_hit_rate={warm['hit_rate']:.3f} "
        f"({warm['cached_tokens']}/{warm['input_tokens']}) "
        f"all_hit_rate={totals['hit_rate']:.3f} "
        f"samples={[s['cached_tokens'] for s in samples]} "
        f"bust_cached={stats['bust']['cached_tokens']}"
    )
    print(f"cache monitor: {detail}")

    # usage schema presence on every sample
    if any(s["input_tokens"] <= 0 for s in samples):
        report.add(
            "cache",
            "usage_input_tokens_present",
            "FAIL",
            "input_tokens missing/zero on a probe",
        )
    else:
        report.add("cache", "usage_input_tokens_present", "PASS", f"n={len(samples)}")

    # Official field: cached_tokens + cache_write_tokens must exist on each probe
    field_ok = True
    per_req = []
    for s in samples:
        # re-check via measure samples only carries ints; require non-negative ints
        if s["cached_tokens"] < 0 or s["cache_write_tokens"] < 0:
            field_ok = False
        per_req.append((s["cached_tokens"], s["cache_write_tokens"]))
    # Also require the keys exist on a live response (not just defaulted 0s in helper)
    probe_code, probe = client.post_json(
        "/v1/responses",
        {
            "model": model,
            "input": "Reply with exactly: CACHE_SCHEMA",
            "max_output_tokens": 8,
            "temperature": 0,
            **extra,
        },
    )
    usage = probe.get("usage") if isinstance(probe, dict) else None
    details = (usage or {}).get("input_tokens_details") if isinstance(usage, dict) else None
    keys_ok = (
        probe_code == 200
        and isinstance(details, dict)
        and "cached_tokens" in details
        and "cache_write_tokens" in details
    )
    report.add(
        "cache",
        "usage_cached_tokens_field",
        "PASS" if field_ok and keys_ok else "FAIL",
        f"per_request={per_req} schema_ok={keys_ok}",
    )

    # After warm-up, expect some cache reads (local KV / prompt cache).
    if warm["cached_tokens"] > 0 and warm["hit_rate"] > 0:
        report.add(
            "cache",
            "warm_hit_rate_positive",
            "PASS",
            f"hit_rate={warm['hit_rate']:.3f}",
        )
    else:
        report.add(
            "cache",
            "warm_hit_rate_positive",
            "FAIL",
            f"no warm cache reads: {detail}",
        )

    # Later rounds should not be worse than first for cached_tokens (monotonic-ish reuse)
    if len(samples) >= 2 and samples[-1]["cached_tokens"] >= samples[0]["cached_tokens"]:
        report.add(
            "cache",
            "repeat_prefix_reuses_cache",
            "PASS",
            f"first={samples[0]['cached_tokens']} last={samples[-1]['cached_tokens']}",
        )
    else:
        report.add(
            "cache",
            "repeat_prefix_reuses_cache",
            "FAIL",
            f"first={samples[0]['cached_tokens'] if samples else None} "
            f"last={samples[-1]['cached_tokens'] if samples else None}",
        )

    # Suffix change should reduce hits vs a warm identical request.
    bust = stats["bust"]
    avg_warm = warm["cached_tokens"] / max(len(samples) - 1, 1) if samples else 0.0
    if warm["cached_tokens"] <= 0:
        report.add(
            "cache",
            "suffix_change_busts_cache",
            "SKIP",
            "no warm hits to compare against",
        )
    elif bust["cached_tokens"] < avg_warm:
        report.add(
            "cache",
            "suffix_change_busts_cache",
            "PASS",
            f"warm_avg_cached={avg_warm:.1f} bust_cached={bust['cached_tokens']}",
        )
    else:
        report.add(
            "cache",
            "suffix_change_busts_cache",
            "FAIL",
            f"warm_avg_cached={avg_warm:.1f} bust_cached={bust['cached_tokens']}",
        )

    # Disk TTL expiry must clear slot KV (alive checked before touch refresh).
    _check_prompt_cache_ttl_expiry(client, report, model, extra)
    _check_retention_and_implicit(client, report, model, extra)
    _check_prompt_cache_options_ttl_only_30m(client, report, model, extra)

    # comparison_response_id diagnostics: the request must stay 200, the baseline
    # comes from the response store, an unknown id reports not-found.
    _check_comparison_diagnostics_not_found(client, report, model, extra)
    _check_comparison_diagnostics_hit(client, report, model, extra)
    _check_comparison_diagnostics_tools_changed(client, report, model, extra)
    _check_comparison_diagnostics_retrieve(client, report, model, extra)
    _check_comparison_response_id_not_string(client, report, model, extra)

    # Adverse agentic traffic: rotating keys, implicit anchors, concurrent
    # branches, mid-prompt edits, client aborts, hostile key values.
    _check_key_isolation(client, report, model, extra)
    _check_key_rotation_restore(client, report, model, extra)
    _check_append_reuse(client, report, model, extra)
    _check_single_text_append(client, report, model, extra)
    _check_same_key_parallel_branches(client, report, model, extra)
    _check_mid_prompt_rewrite(client, report, model, extra)
    _check_aborted_stream_retry(client, report, model, extra)
    _check_hostile_keys(client, report, model, extra)
    _check_cross_model_isolation(client, report, model, extra)

    # Aggregate monitor row: PASS only if blocking cache checks passed
    blocking = {
        r.name
        for r in report.rows
        if r.area == "cache"
        and r.status in ("FAIL", "PARTIAL", "NOT_IMPLEMENTED")
        and r.name != "prompt_cache_hit_monitor"
    }
    report.add(
        "cache",
        "prompt_cache_hit_monitor",
        "FAIL" if blocking else "PASS",
        detail if not blocking else f"blocked_by={sorted(blocking)} {detail}",
    )
    return stats


def _check_prompt_cache_ttl_expiry(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    import json
    import os
    from pathlib import Path

    key = "acceptance-cache-ttl-expiry"
    prefix = (
        "You are a careful assistant. Keep answers short. "
        "TTL block: " + ("echo-foxtrot-golf-hotel " * 40)
    )
    user_input = f"{prefix}\nReply with exactly: TTL_EXPIRY_PROBE"
    # Unique leading system message: keeps the injected verbosity hint out of the
    # prompt head, so an unrelated slot cannot answer the post-expiry request with
    # its cached hint prefix
    uniq = f"ttl{time.time_ns()}"
    body = {
        "model": model,
        "input": [
            _msg("system", f"ttl expiry probe {key} {uniq}"),
            _msg("user", user_input),
        ],
        "max_output_tokens": 24,
        "temperature": 0,
        "prompt_cache_key": key,
        "prompt_cache_retention": "in_memory",
        "prompt_cache_options": {"ttl": "30m"},
        **extra,
    }
    # Cold + warm
    c1, d1 = client.post_json("/v1/responses", body)
    c2, d2 = client.post_json("/v1/responses", body)
    if c1 != 200 or c2 != 200 or not isinstance(d2, dict):
        report.add(
            "cache",
            "prompt_cache_ttl_expiry_clears_kv",
            "FAIL",
            f"warm http={c1}/{c2}",
        )
        return
    _, warm_cached, _ = _usage_cache(d2)
    root = Path(os.environ.get("LLAMA_OPENAI_FILES_PATH", "/tmp/llama-openai-files"))
    # safe_id keeps alnum/_/- ; key has no dots
    meta = root / "prompt_cache_keys" / f"{key}.json"
    if not meta.exists():
        # Fallback: scan for matching key field
        meta = None
        for p in (root / "prompt_cache_keys").glob("*.json"):
            try:
                rec = json.loads(p.read_text())
                if rec.get("key") == key:
                    meta = p
                    break
            except Exception:
                continue
    if meta is None or not meta.exists():
        report.add(
            "cache",
            "prompt_cache_ttl_expiry_clears_kv",
            "FAIL",
            f"missing registry file under {root}/prompt_cache_keys for key={key!r}",
        )
        return
    rec = json.loads(meta.read_text())
    rec["expires_at"] = 1  # far past
    meta.write_text(json.dumps(rec))
    c3, d3 = client.post_json("/v1/responses", body)
    if c3 != 200 or not isinstance(d3, dict):
        report.add(
            "cache",
            "prompt_cache_ttl_expiry_clears_kv",
            "FAIL",
            f"post-expiry http={c3}",
        )
        return
    _, expired_cached, _ = _usage_cache(d3)
    # After forced disk expiry, expect no (or sharply reduced) cache reads vs warm.
    ok = warm_cached > 0 and expired_cached == 0
    report.add(
        "cache",
        "prompt_cache_ttl_expiry_clears_kv",
        "PASS" if ok else "FAIL",
        f"warm_cached={warm_cached} expired_cached={expired_cached} meta={meta.name}",
    )


def _check_retention_and_implicit(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    import json
    import os
    from pathlib import Path

    root = Path(os.environ.get("LLAMA_OPENAI_FILES_PATH", "/tmp/llama-openai-files"))
    uniq = f"ret{time.time_ns()}"
    pad = f"resp retention pad {uniq}: " + ("lima-mike " * 30)

    def _exp_for(key: str, retention: str) -> tuple[int, int]:
        body = {
            "model": model,
            "input": f"{pad}\nReply with exactly: RET_OK",
            "max_output_tokens": 16,
            "temperature": 0,
            "prompt_cache_key": key,
            "prompt_cache_retention": retention,
            **extra,
        }
        code, _ = client.post_json("/v1/responses", body)
        meta = root / "prompt_cache_keys" / f"{key}.json"
        exp = -1
        if meta.exists():
            try:
                exp = int(json.loads(meta.read_text()).get("expires_at") or 0)
            except Exception:
                exp = -1
        return code, exp

    c_mem, exp_mem = _exp_for(f"resp-ret-mem-{uniq}", "in_memory")
    c_24, exp_24 = _exp_for(f"resp-ret-24h-{uniq}", "24h")
    now = int(time.time())
    ok_ret = c_mem == 200 and c_24 == 200 and exp_mem == 0 and exp_24 > now + 23 * 3600
    report.add(
        "cache",
        "prompt_cache_retention_sets_ttl",
        "PASS" if ok_ret else "FAIL",
        f"mem={c_mem}/{exp_mem} h24={c_24}/{exp_24} now={now}",
    )

    uniq_i = f"imp{time.time_ns()}"
    inp = f"implicit pad {uniq_i}: " + ("november-oscar " * 40) + "\nReply with exactly: IMP_OK"
    # Unique leading system message: avoids reusing the constant verbosity-hint
    # prefix written by earlier probes, so the first request starts cold.
    body_imp = {
        "model": model,
        "input": [
            _msg("system", f"implicit cache probe {uniq_i}"),
            _msg("user", inp),
        ],
        "max_output_tokens": 24,
        "temperature": 0,
        "prompt_cache_options": {"mode": "implicit"},
        "prompt_cache_retention": "in_memory",
        **extra,
    }
    c1, d1 = client.post_json("/v1/responses", body_imp)
    c2, d2 = client.post_json("/v1/responses", body_imp)
    _, cold, _ = _usage_cache(d1 if isinstance(d1, dict) else {})
    _, warm, _ = _usage_cache(d2 if isinstance(d2, dict) else {})
    ok_imp = c1 == 200 and c2 == 200 and cold == 0 and warm > 0
    report.add(
        "cache",
        "prompt_cache_options_implicit_warms",
        "PASS" if ok_imp else "FAIL",
        f"http={c1}/{c2} cached=({cold},{warm})",
    )


def _check_prompt_cache_options_ttl_only_30m(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    """Official: 30m is the only supported prompt_cache_options.ttl, other values are rejected."""
    import json
    import os
    from pathlib import Path

    root = Path(os.environ.get("LLAMA_OPENAI_FILES_PATH", "/tmp/llama-openai-files"))
    uniq = f"ttl{time.time_ns()}"
    pad = f"ttl pad {uniq}: " + ("papa-quebec " * 20)
    key = f"resp-ttl-30m-{uniq}"
    body = {
        "model": model,
        "input": f"{pad}\nReply with exactly: TTL_30M",
        "max_output_tokens": 12,
        "temperature": 0,
        "prompt_cache_key": key,
        "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
        **extra,
    }
    now = int(time.time())
    code, _ = client.post_json("/v1/responses", body)
    meta = root / "prompt_cache_keys" / f"{key}.json"
    delta = -1
    if meta.exists():
        try:
            exp = int(json.loads(meta.read_text()).get("expires_at") or 0)
            delta = exp - now if exp > 0 else 0
        except Exception:
            delta = -1
    rejected: dict[str, int] = {}
    for ttl in ("5m", "1h"):
        other = dict(body)
        other["prompt_cache_key"] = f"resp-ttl-{ttl}-{uniq}"
        other["prompt_cache_options"] = {"mode": "explicit", "ttl": ttl}
        rejected[ttl], _ = client.post_json("/v1/responses", other)
    # Allow ±90s skew for request latency / clock.
    ok = code == 200 and abs(delta - 30 * 60) <= 90 and all(c >= 400 for c in rejected.values())
    report.add(
        "cache",
        "prompt_cache_options_ttl_only_30m",
        "PASS" if ok else "FAIL",
        f"http_30m={code} delta_30m={delta} rejected={rejected}",
    )


def _cache_body(
    model: str,
    extra: dict[str, Any],
    prompt: str | list[dict[str, Any]],
    *,
    key: str | None = None,
    retention: str | None = "in_memory",
    options: dict[str, Any] | None = None,
    max_output: int = 24,
) -> dict[str, Any]:
    """Build a cache probe body; extra wins over the defaults."""
    body: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output,
        "temperature": 0,
    }
    if key is not None:
        body["prompt_cache_key"] = key
    if retention is not None:
        body["prompt_cache_retention"] = retention
    if options is not None:
        body["prompt_cache_options"] = options
    body.update(extra)
    return body


def _msg(role: str, text: str, kind: str = "input_text") -> dict[str, Any]:
    """One official Responses message item, for array-typed inputs."""
    return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}


def _cached(data: Any) -> int:
    """cached_tokens of a response body, or -1 when the body is not usable."""
    return _usage_cache(data)[1] if isinstance(data, dict) else -1


def _post_stream_partial(
    client: ResponsesHttpClient,
    body: dict[str, Any],
    nbytes: int,
) -> int:
    """Open a streaming create, read part of the stream, then drop the connection."""
    import json
    import urllib.request

    req = urllib.request.Request(
        client.base_url + "/v1/responses",
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {client.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    status = 0
    try:
        resp = urllib.request.urlopen(req, timeout=client.timeout)
        status = int(resp.status)
        try:
            resp.read(nbytes)
        finally:
            resp.close()
    except Exception:
        # dropping the connection is the point of this probe
        pass
    return status


def _check_key_isolation(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    """Distinct explicit keys must not share KV, even for identical prompts."""
    uniq = f"iso{time.time_ns()}"
    pad = f"key isolation pad {uniq}: " + ("tango-uniform " * 40)
    text = f"{pad}\nReply with exactly: ISO_OK"
    keys = [f"resp-iso-{i}-{uniq}" for i in range(3)]

    codes: list[int] = []
    cold: list[int] = []
    for key in keys:
        code, data = client.post_json("/v1/responses", _cache_body(model, extra, text, key=key))
        codes.append(code)
        cold.append(_cached(data))
    # The first key must still hit on repeat, else the probe proves nothing.
    wcode, wdata = client.post_json("/v1/responses", _cache_body(model, extra, text, key=keys[0]))
    warm = _cached(wdata)

    ok = all(c == 200 for c in codes) and all(c == 0 for c in cold) and wcode == 200 and warm > 0
    report.add(
        "cache",
        "prompt_cache_key_isolation",
        "PASS" if ok else "FAIL",
        f"http={codes} cold_cached={cold} repeat_cached={warm}",
    )


def _check_key_rotation_restore(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    """More explicit keys than slots: a returning conversation must restore its KV.

    Rotating keys release the slot that held the previous conversation, so the
    parked level-2 state is the only way back for a returning one. The release
    must run before the level-2 load: when it ran after, every key switch
    counted a level-2 hit and then re-processed the whole prompt, so
    cached_tokens stayed 0 while hits advanced.

    Each conversation replays one identical prompt, so a working restore is a
    direct prefix reuse and a broken one is a full re-process.
    """
    name = "prompt_cache_key_rotation_restore"
    pcode, props = client.get_json("/props")
    slots = 0
    pc: dict[str, Any] = {}
    if isinstance(props, dict):
        slots = int(props.get("total_slots") or 0)
        if isinstance(props.get("prompt_cache"), dict):
            pc = props["prompt_cache"]
    if "hits" not in pc:
        # a router reports the cache config but keeps no counters; ask the child
        mcode, mprops = client.get_json(f"/props?model={model}")
        if mcode == 200 and isinstance(mprops, dict):
            if slots <= 0:
                slots = int(mprops.get("total_slots") or 0)
            if isinstance(mprops.get("prompt_cache"), dict):
                pc = mprops["prompt_cache"]
    if pcode != 200 or slots <= 0:
        report.add("cache", name, "SKIP", f"no slot count from /props (http={pcode}); cannot size the rotation")
        return
    if pc.get("enabled") is False:
        report.add("cache", name, "SKIP", f"level-2 prompt cache disabled: {pc}")
        return

    def hits() -> int:
        for path in ("/props", f"/props?model={model}"):
            code, body = client.get_json(path)
            if code != 200 or not isinstance(body, dict):
                continue
            block = body.get("prompt_cache")
            if isinstance(block, dict) and "hits" in block:
                return int(block["hits"] or 0)
        return -1

    uniq = f"rot{time.time_ns()}"
    tag = uniq[-5:]
    n = slots + 2
    convs: list[tuple[str, list[dict[str, Any]], str]] = []
    for i in range(n):
        convs.append((
            f"resp-rotation-{uniq}-{i}",
            [
                _msg("user", f"rotation pad {uniq}-{i}: " + ("uniform-tango " * 40)),
                _msg("assistant", "ack " * 20, "output_text"),
                _msg("user", f"Reply with exactly: ROT_{i}_{tag}"),
            ],
            f"ROT_{i}_{tag}",
        ))

    hits_before = hits()
    codes: list[int] = []
    cold: list[int] = []
    for key, hist, _ in convs:
        code, data = client.post_json(
            "/v1/responses",
            _cache_body(model, extra, [dict(m) for m in hist], key=key, max_output=32),
        )
        codes.append(code)
        cold.append(_cached(data))
    hits_mid = hits()

    warm: list[int] = []
    inputs: list[int] = []
    has_output = True
    answers_ok = True
    for key, hist, nonce in convs:
        code, data = client.post_json(
            "/v1/responses",
            _cache_body(model, extra, [dict(m) for m in hist], key=key, max_output=32),
        )
        codes.append(code)
        inp, c, _ = _usage_cache(data) if isinstance(data, dict) else (0, 0, 0)
        inputs.append(inp)
        warm.append(c)
        output = data.get("output") if isinstance(data, dict) else None
        # an empty output means the restore wedged the turn; text may legitimately be
        # empty when a reasoning model spends the whole budget on its thinking item
        if not (isinstance(output, list) and output):
            has_output = False
        if nonce not in (output_text(data) if isinstance(data, dict) else ""):
            # informational only: small models echo the nonce, reasoning models do not
            answers_ok = False
    hits_after = hits()

    restored = sum(1 for c in warm if c > 0)
    # A key switch can lose the tail of the prompt (checkpoint slack), not the prefix.
    weak = min(warm) < max(1, min(inputs) // 4)
    # Round 2 must load one parked state per conversation; round 1 is cold by design.
    loads = hits_after - hits_mid if hits_mid >= 0 and hits_after >= 0 else -1
    ok = (
        all(c == 200 for c in codes)
        and has_output
        and all(c == 0 for c in cold)
        and restored == n
        and not weak
        and (loads < 0 or loads >= n)
    )
    report.add(
        "cache",
        name,
        "PASS" if ok else "FAIL",
        f"slots={slots} n={n} http={codes} cold_cached={cold} warm_cached={warm} "
        f"warm_inputs={inputs} level2_loads={loads} hits={hits_before}->{hits_mid}->{hits_after} "
        f"has_output={has_output} answers_ok={answers_ok}",
    )


def _check_append_reuse(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    """Agentic shape: a growing message array, no key and no cache options.

    This is what coding agents send: history as message items, one new turn at a
    time. Every turn must reuse the conversation prefix (hybrid models reuse up
    to the last user message via a context checkpoint).
    """
    uniq = f"grow{time.time_ns()}"
    tag = uniq[-5:]
    # Unique leading system message: avoids reusing the constant verbosity-hint
    # prefix written by earlier probes, so the first turn starts cold.
    hist: list[dict[str, Any]] = [
        _msg("system", f"append cache probe {uniq}"),
        _msg("user", f"append reuse pad {uniq}: " + ("victor-whiskey " * 40)),
        _msg("assistant", "ack " * 20, "output_text"),
    ]
    codes: list[int] = []
    cached: list[int] = []
    inputs: list[int] = []
    has_output = True
    echo_ok = True
    for i in range(3):
        nonce = f"GROW_{i}_{tag}"
        hist.append(_msg("user", f"Reply with exactly: {nonce}"))
        code, data = client.post_json(
            "/v1/responses",
            _cache_body(model, extra, [dict(m) for m in hist], key=None, max_output=32),
        )
        inp, c, _ = _usage_cache(data) if isinstance(data, dict) else (0, 0, 0)
        codes.append(code)
        cached.append(c)
        inputs.append(inp)
        text = output_text(data) if isinstance(data, dict) else ""
        # a wedged restore would leave the turn empty; the exact echo is informational,
        # a reasoning model may spend the whole max_output_tokens budget on thinking
        if not text:
            has_output = False
        if nonce not in text:
            echo_ok = False
        hist.append(_msg("assistant", "ack " * 20, "output_text"))

    last = len(cached) - 1
    reuse_ok = cached[0] == 0 and cached[last] > 0 and cached[last] >= inputs[last] // 4
    ok = all(c == 200 for c in codes) and has_output and reuse_ok
    report.add(
        "cache",
        "prompt_cache_append_reuse",
        "PASS" if ok else "FAIL",
        f"http={codes} inputs={inputs} cached={cached} has_output={has_output} "
        f"echo_ok={echo_ok} (informational)",
    )


def _check_single_text_append(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    """Adverse shape: one text blob that grows, so the tail carries no message boundary.

    Correctness must hold on any model. Reuse needs a context checkpoint at or
    below the append boundary; hybrid/recurrent models only checkpoint at
    message boundaries and near the prompt end, so they replay the whole blob.
    """
    uniq = f"blob{time.time_ns()}"
    tag = uniq[-5:]
    text = f"single text pad {uniq}: " + ("zulu-yankee " * 40)
    codes: list[int] = []
    cached: list[int] = []
    inputs: list[int] = []
    has_output = True
    echo_ok = True
    for i in range(3):
        nonce = f"BLOB_{i}_{tag}"
        text = f"{text}\nuser: Reply with exactly: {nonce}"
        code, data = client.post_json("/v1/responses", _cache_body(model, extra, text, key=None, max_output=32))
        tin, c, _ = _usage_cache(data) if isinstance(data, dict) else (0, 0, 0)
        codes.append(code)
        cached.append(c)
        inputs.append(tin)
        # a wedged restore would leave the turn empty; the exact echo is informational,
        # a reasoning model may spend the whole max_output_tokens budget on thinking
        out = output_text(data) if isinstance(data, dict) else ""
        if not out:
            has_output = False
        if nonce not in out:
            echo_ok = False
        text = f"{text}\nassistant: ack"

    if not (all(c == 200 for c in codes) and has_output):
        report.add(
            "cache",
            "prompt_cache_single_text_append",
            "FAIL",
            f"http={codes} cached={cached} has_output={has_output} echo_ok={echo_ok} (informational)",
        )
    elif max(cached[1:]) >= max(1, inputs[-1] // 4):
        # hint-lead-only hits (constant 20-token default hint) are not growth reuse;
        # require reuse >= 25% of the prompt
        report.add(
            "cache",
            "prompt_cache_single_text_append",
            "PASS",
            f"cached={cached} single-message growth reused the prefix",
        )
    else:
        report.add(
            "cache",
            "prompt_cache_single_text_append",
            "SKIP",
            f"cached={cached}: no checkpoint at/below the append boundary "
            "(expected for hybrid/recurrent models, dense models reuse; growth needs >= inputs//4)",
        )


def _check_same_key_parallel_branches(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    """Concurrent branches of one conversation share a key; replies must not mix."""
    import threading

    uniq = f"conc{time.time_ns()}"
    tag = uniq[-5:]
    head = f"parallel branch pad {uniq}: " + ("xray-yankee " * 40)
    key = f"resp-conc-{uniq}"
    nonce_a = f"BRANCH_A_{tag}"
    nonce_b = f"BRANCH_B_{tag}"
    text_a = f"{head}\nuser: open alpha.txt\nassistant: ok\nuser: Reply with exactly: {nonce_a}"
    text_b = f"{head}\nuser: open beta.txt\nassistant: ok\nuser: Reply with exactly: {nonce_b}"

    out: dict[str, tuple[int, str]] = {}

    def probe(tag: str, text: str) -> None:
        code, data = client.post_json(
            "/v1/responses",
            _cache_body(model, extra, text, key=key, max_output=64),
        )
        out[tag] = (code, output_text(data) if isinstance(data, dict) else "")

    threads = [
        threading.Thread(target=probe, args=("a", text_a)),
        threading.Thread(target=probe, args=("b", text_b)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)

    code_a, text_out_a = out.get("a", (0, ""))
    code_b, text_out_b = out.get("b", (0, ""))
    own = nonce_a in text_out_a and nonce_b in text_out_b
    cross = nonce_b in text_out_a or nonce_a in text_out_b
    ok = code_a == 200 and code_b == 200 and own and not cross
    report.add(
        "cache",
        "prompt_cache_same_key_parallel_branches",
        "PASS" if ok else "FAIL",
        f"http=({code_a},{code_b}) own_nonce={own} cross_contamination={cross}",
    )


def _check_mid_prompt_rewrite(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    """Edits must not reuse KV past the edit or leak the other version's answer.

    Also probes the system-prompt churn pattern: rewriting the first message
    (timestamps, environment blocks) invalidates every later token.
    """
    uniq = f"rew{time.time_ns()}"
    tag = uniq[-5:]
    key = f"resp-rewrite-{uniq}"
    nonce_one = f"EDIT_ONE_{tag}"
    nonce_two = f"EDIT_TWO_{tag}"
    head = _msg("user", f"mid rewrite pad {uniq}: " + ("zulu-alpha " * 40))
    filler = _msg("assistant", "ack " * 20, "output_text")
    ask_one = _msg("user", f"open alpha.txt then Reply with exactly: {nonce_one}")
    ask_two = _msg("user", f"open beta.txt then Reply with exactly: {nonce_two}")

    def send(messages: list[dict[str, Any]]) -> tuple[int, int, int, str]:
        code, data = client.post_json(
            "/v1/responses",
            _cache_body(model, extra, [dict(m) for m in messages], key=key, max_output=48),
        )
        inp, cached, _ = _usage_cache(data) if isinstance(data, dict) else (0, 0, 0)
        return code, inp, cached, (output_text(data) if isinstance(data, dict) else "")

    base = [head, filler, ask_one]
    c1, _, cold, _ = send(base)
    c2, in2, warm, _ = send(base)
    c3, in3, edited, txt3 = send([head, filler, ask_two])
    c4, in4, again, txt4 = send(base)
    churned = _msg("user", f"mid rewrite pad {uniq} v2: " + ("zulu-alpha " * 40))
    c5, in5, churn, txt5 = send([churned, filler, ask_one])

    answers_ok = (
        nonce_two in txt3
        and nonce_one not in txt3
        and nonce_one in txt4
        and nonce_two not in txt4
        and nonce_one in txt5
    )
    ok = (
        c1 == 200
        and c2 == 200
        and c3 == 200
        and c4 == 200
        and c5 == 200
        and in2 > 0
        and warm > 0          # precondition: identical repeats do reuse
        and edited < in3      # never claim the edited prompt as fully cached
        and again > 0         # going back to the old text reuses again
        and churn < in5       # a rewritten head must not reuse past the edit
        and answers_ok
    )
    report.add(
        "cache",
        "prompt_cache_mid_prompt_rewrite",
        "PASS" if ok else "FAIL",
        f"http=({c1},{c2},{c3},{c4},{c5}) cached=({cold},{warm},{edited},{again},{churn}) "
        f"inputs=({in2},{in3},{in4},{in5}) answers_ok={answers_ok}",
    )


def _check_aborted_stream_retry(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    """A client that drops mid-stream must not poison the slot for the retry."""
    uniq = f"abrt{time.time_ns()}"
    pad = f"abort retry pad {uniq}: " + ("mike-november " * 40)
    nonce = f"ABORT_{uniq[-5:]}"
    text = f"{pad}\nReply with exactly: {nonce}"
    key = f"resp-abort-{uniq}"

    body = _cache_body(model, extra, text, key=key, max_output=64)
    body["stream"] = True
    stream_code = _post_stream_partial(client, body, 192)

    c1, d1 = client.post_json("/v1/responses", _cache_body(model, extra, text, key=key, max_output=64))
    c2, d2 = client.post_json("/v1/responses", _cache_body(model, extra, text, key=key, max_output=64))
    text_out = output_text(d1) if isinstance(d1, dict) else ""
    ok = (
        stream_code == 200
        and c1 == 200
        and c2 == 200
        and nonce in text_out
        and max(_cached(d1), _cached(d2)) > 0
    )
    report.add(
        "cache",
        "prompt_cache_aborted_stream_retry",
        "PASS" if ok else "FAIL",
        f"stream_http={stream_code} retry_http=({c1},{c2}) "
        f"cached=({_cached(d1)},{_cached(d2)}) nonce={nonce in text_out}",
    )


def _check_hostile_keys(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    """Weird prompt_cache_key values must not escape the registry dir or fail requests."""
    import json
    import os
    from pathlib import Path

    uniq = f"hkey{time.time_ns()}"
    pad = f"hostile key pad {uniq}: " + ("kilo-lima " * 20)
    keys = [
        "",                   # empty: accepted, treated as an unkeyed request
        "k" * 1024,           # very long id
        "h_" + "k" * 8192,    # very long id that already looks encoded
        "../../etc/passwd",   # path traversal shape
        "a/b\\c",             # path separators
        "key with space",     # whitespace
        "\u952e-\U0001f680",  # non-ASCII
    ]
    codes: list[int] = []
    for key in keys:
        code, _ = client.post_json(
            "/v1/responses",
            _cache_body(model, extra, f"{pad}\nReply with exactly: HKEY_OK", key=key, retention="24h"),
        )
        codes.append(code)

    root = Path(os.environ.get("LLAMA_OPENAI_FILES_PATH", "/tmp/llama-openai-files"))
    reg = root / "prompt_cache_keys"
    bad: list[str] = []
    stored: list[str] = []
    if reg.is_dir():
        for path in reg.glob("*.json"):
            name = path.name
            if "/" in name or "\\" in name or "\n" in name or name.startswith(".") or len(name) > 140:
                bad.append(name)
            try:
                stored.append(str(json.loads(path.read_text()).get("key") or ""))
            except Exception:
                bad.append(f"unreadable:{name[:40]}")
    else:
        bad.append(f"missing dir {reg}")
    # Every non-empty key asked for a TTL, so each must show up in the registry.
    missing = [k for k in keys if k and k not in stored]
    ok = all(c == 200 for c in codes) and not bad and not missing
    report.add(
        "cache",
        "prompt_cache_hostile_keys",
        "PASS" if ok else "FAIL",
        f"http={codes} bad_files={bad[:3]} missing={[k[:12] for k in missing]} registry={reg}",
    )


def _check_cross_model_isolation(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    """KV is per model instance: another model must not see this model's states."""
    code, listing = client.get_json("/v1/models")
    ids: list[str] = []
    if code == 200 and isinstance(listing, dict):
        ids = [m.get("id") for m in listing.get("data") or [] if isinstance(m, dict)]
    if model not in ids:
        # the client model string is not a served id; a second name here reaches the same
        # local model instance, so isolation between model names cannot be asserted
        report.add(
            "cache",
            "prompt_cache_cross_model_isolation",
            "SKIP",
            f"client model {model!r} is not a served id, ids={ids[:3]}",
        )
        return
    other = next((i for i in ids if i and i != model), None)
    if not other:
        report.add(
            "cache",
            "prompt_cache_cross_model_isolation",
            "SKIP",
            f"single-model server, ids={ids[:3]}",
        )
        return

    uniq = f"xmod{time.time_ns()}"
    pad = f"cross model pad {uniq}: " + ("sierra-tango " * 40)
    text = f"{pad}\nReply with exactly: XMODEL_OK"
    key = f"resp-xmodel-{uniq}"

    c1, _ = client.post_json("/v1/responses", _cache_body(model, extra, text, key=key))
    c2, d2 = client.post_json("/v1/responses", _cache_body(model, extra, text, key=key))
    c3, d3 = client.post_json("/v1/responses", _cache_body(other, extra, text, key=key))
    c4, d4 = client.post_json("/v1/responses", _cache_body(other, extra, text, key=key))

    if c3 != 200:
        report.add(
            "cache",
            "prompt_cache_cross_model_isolation",
            "SKIP",
            f"second model {other!r} unusable, http={c3}",
        )
        return

    cached = (_cached(d2), _cached(d3), _cached(d4))
    ok = c1 == 200 and c2 == 200 and c4 == 200 and cached[0] > 0 and cached[1] == 0 and cached[2] > 0
    report.add(
        "cache",
        "prompt_cache_cross_model_isolation",
        "PASS" if ok else "FAIL",
        f"model={model} other={other} http=({c1},{c2},{c3},{c4}) cached={cached}",
    )


def _diagnostics(data: Any) -> dict[str, Any] | None:
    """prompt_cache_diagnostics of a response body, or None when absent/not-a-dict."""
    diag = data.get("prompt_cache_diagnostics") if isinstance(data, dict) else None
    return diag if isinstance(diag, dict) else None


def _check_comparison_diagnostics_not_found(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    """A well-formed unknown comparison_response_id must 200 with a not-found diagnostic."""
    uniq = f"cmdnf{time.time_ns()}"
    missing = f"resp_nf_{uniq}"
    body = _cache_body(
        model,
        extra,
        f"comparison not found pad {uniq}\nReply with exactly: CMPNF_{uniq[-5:]}",
        key=f"resp-cmpnf-{uniq}",
        max_output=16,
    )
    body["prompt_cache_options"] = {"comparison_response_id": missing}
    code, data = client.post_json("/v1/responses", body)
    diag = _diagnostics(data)
    ok = code == 200 and diag is not None and diag.get("type") == "comparison_response_not_found"
    report.add(
        "cache",
        "prompt_cache_diagnostics_not_found",
        "PASS" if ok else "FAIL",
        f"http={code} id={missing!r} diagnostics={diag}",
    )


def _check_comparison_diagnostics_hit(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    """Repeating one prompt with the previous response id must report cache_hit."""
    uniq = f"cmhit{time.time_ns()}"
    key = f"resp-cmhit-{uniq}"
    text = (
        f"comparison hit pad {uniq}: " + ("alpha-bravo " * 24)
        + f"\nReply with exactly: CMPHIT_{uniq[-5:]}"
    )
    c1, d1 = client.post_json(
        "/v1/responses", _cache_body(model, extra, text, key=key, max_output=16)
    )
    rid = d1.get("id") if isinstance(d1, dict) else None
    if c1 != 200 or not isinstance(rid, str) or not rid:
        report.add(
            "cache",
            "prompt_cache_diagnostics_hit",
            "FAIL",
            f"base create http={c1} id={rid!r}",
        )
        return
    body2 = _cache_body(model, extra, text, key=key, max_output=16)
    body2["prompt_cache_options"] = {"comparison_response_id": rid}
    c2, d2 = client.post_json("/v1/responses", body2)
    diag = _diagnostics(d2)
    inp, cached, _ = _usage_cache(d2 if isinstance(d2, dict) else {})
    hit_ok = diag is not None and diag.get("type") == "cache_hit"
    # the warm prefix must also show in usage (>= input - 4 local slack)
    reuse_ok = inp > 0 and cached >= inp - 4
    ok = c2 == 200 and hit_ok and reuse_ok
    report.add(
        "cache",
        "prompt_cache_diagnostics_hit",
        "PASS" if ok else "FAIL",
        f"http=({c1},{c2}) input={inp} cached={cached} reuse_ok={reuse_ok} diagnostics={diag}",
    )


def _check_comparison_diagnostics_tools_changed(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    """Same input, renamed tool: the miss must be attributed to tools_changed."""
    uniq = f"cmtool{time.time_ns()}"
    key = f"resp-cmtool-{uniq}"
    text = (
        f"comparison tools pad {uniq}: " + ("charlie-delta " * 24)
        + f"\nReply with exactly: CMPTOOL_{uniq[-5:]}"
    )

    def one_tool(name: str) -> list[dict[str, Any]]:
        return [{
            "type": "function",
            "name": name,
            "description": "acceptance probe",
            "parameters": {"type": "object", "properties": {}},
        }]

    body = _cache_body(model, extra, text, key=key, max_output=16)
    body["tools"] = one_tool("probe_one")
    c1, d1 = client.post_json("/v1/responses", body)
    rid = d1.get("id") if isinstance(d1, dict) else None
    if c1 != 200 or not isinstance(rid, str) or not rid:
        report.add(
            "cache",
            "prompt_cache_diagnostics_tools_changed",
            "FAIL",
            f"base create http={c1} id={rid!r}",
        )
        return
    body2 = _cache_body(model, extra, text, key=key, max_output=16)
    body2["tools"] = one_tool("probe_two")
    body2["prompt_cache_options"] = {"comparison_response_id": rid}
    c2, d2 = client.post_json("/v1/responses", body2)
    diag = _diagnostics(d2)
    missed = diag.get("cache_missed_tokens") if diag is not None else None
    ok = (
        c2 == 200
        and diag is not None
        and diag.get("type") == "cache_miss"
        and diag.get("reason") == "tools_changed"
        and isinstance(missed, int)
        and missed > 0
    )
    report.add(
        "cache",
        "prompt_cache_diagnostics_tools_changed",
        "PASS" if ok else "FAIL",
        f"http=({c1},{c2}) expected_reason=tools_changed diagnostics={diag}",
    )


def _check_comparison_diagnostics_retrieve(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    """GET must return the same prompt_cache_diagnostics object as create."""
    uniq = f"cmret{time.time_ns()}"
    body = _cache_body(
        model,
        extra,
        f"comparison retrieve pad {uniq}\nReply with exactly: CMRET_{uniq[-5:]}",
        key=f"resp-cmret-{uniq}",
        max_output=16,
    )
    body["prompt_cache_options"] = {"comparison_response_id": f"resp_nf_{uniq}"}
    code, data = client.post_json("/v1/responses", body)
    rid = data.get("id") if isinstance(data, dict) else None
    diag = _diagnostics(data)
    if code != 200 or not isinstance(rid, str) or not rid or diag is None:
        report.add(
            "cache",
            "prompt_cache_diagnostics_retrieve",
            "FAIL",
            f"create http={code} id={rid!r} diagnostics={diag}",
        )
        return
    gcode, got = client.get_json(f"/v1/responses/{rid}")
    got_diag = _diagnostics(got)
    ok = gcode == 200 and got_diag == diag
    report.add(
        "cache",
        "prompt_cache_diagnostics_retrieve",
        "PASS" if ok else "FAIL",
        f"create={diag} retrieve_http={gcode} retrieve={got_diag}",
    )


def _check_comparison_response_id_not_string(
    client: ResponsesHttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
) -> None:
    """comparison_response_id must be a string; other shapes are rejected with 400."""
    uniq = f"cmshp{time.time_ns()}"
    codes: list[int] = []
    for bad in (123, {"id": "resp_x"}):
        body = _cache_body(
            model,
            extra,
            f"comparison shape pad {uniq}\nReply with exactly: CMSHAPE_{uniq[-5:]}",
            key=f"resp-cmshape-{uniq}",
            max_output=16,
        )
        body["prompt_cache_options"] = {"comparison_response_id": bad}
        code, _ = client.post_json("/v1/responses", body)
        codes.append(code)
    ok = all(c >= 400 for c in codes)
    report.add(
        "cache",
        "comparison_response_id_not_string",
        "PASS" if ok else "FAIL",
        f"http={codes}",
    )
