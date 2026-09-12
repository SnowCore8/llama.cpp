"""Restart durability: plant interrupted stores, restart server, assert finalize + reload."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .http_client import HttpClient, output_text
from .report import Report


def _files_root() -> Path:
    return Path(os.environ.get("LLAMA_OPENAI_FILES_PATH", "/tmp/llama-openai-files"))


def _port_from_base(base_url: str) -> int:
    u = urlparse(base_url)
    if u.port:
        return int(u.port)
    return 443 if u.scheme == "https" else 80


def _wait_health(base_url: str, api_key: str, timeout_s: float = 600.0) -> bool:
    deadline = time.time() + timeout_s
    url = base_url.rstrip("/") + "/health"
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2.0)
    return False


def _port_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def restart_server(base_url: str, api_key: str, report: Report) -> bool:
    """Kill llama-server and start via LLAMA_SERVER_RESTART_CMD (required for restart checks)."""
    skip = os.environ.get("LOCAL_DURABILITY_SKIP_RESTART", "").strip() in ("1", "true", "yes")
    if skip:
        report.add("restart", "server_restart", "SKIP", "LOCAL_DURABILITY_SKIP_RESTART set")
        return False

    cmd = os.environ.get("LLAMA_SERVER_RESTART_CMD", "").strip()
    if not cmd:
        report.add(
            "restart",
            "server_restart",
            "SKIP",
            "LLAMA_SERVER_RESTART_CMD not set (launch script that brings llama-server back up)",
        )
        return False
    port = _port_from_base(base_url)
    log_path = Path(os.environ.get("LLAMA_SERVER_RESTART_LOG", "/tmp/llama-server-restart.log"))

    # Prefer exact binary name; avoid matching the test runner's command line.
    pids: list[str] = []
    for pattern in ("llama-server",):
        try:
            out = subprocess.check_output(["pgrep", "-x", pattern], text=True)
            pids = [p.strip() for p in out.splitlines() if p.strip()]
            if pids:
                break
        except subprocess.CalledProcessError:
            continue
    if not pids:
        try:
            out = subprocess.check_output(
                ["pgrep", "-f", r"/bin/llama-server|--alias .*--openai-files-path"],
                text=True,
            )
            pids = [p.strip() for p in out.splitlines() if p.strip()]
        except subprocess.CalledProcessError:
            pids = []

    for pid in pids:
        try:
            os.kill(int(pid), 15)
        except Exception as e:  # noqa: BLE001
            report.add("restart", "server_restart", "FAIL", f"kill {pid}: {e}")
            return False

    deadline = time.time() + 60
    while time.time() < deadline and _port_listening(port):
        time.sleep(0.5)
    if _port_listening(port):
        for pid in pids:
            try:
                os.kill(int(pid), 9)
            except Exception:
                pass
        time.sleep(1.0)
    if _port_listening(port):
        report.add("restart", "server_restart", "FAIL", f"port {port} still listening after kill")
        return False

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as logf:
        logf.write(b"\n===== local_durability restart =====\n")
        proc = subprocess.Popen(
            ["bash", cmd],
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    if not _wait_health(base_url, api_key, timeout_s=float(os.environ.get("LLAMA_RESTART_HEALTH_TIMEOUT", "600"))):
        report.add(
            "restart",
            "server_restart",
            "FAIL",
            f"health timeout after restart pid={proc.pid} log={log_path}",
        )
        return False

    report.add("restart", "server_restart", "PASS", f"pid={proc.pid} health=200")
    return True


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def plant_interrupted_fixtures(model: str) -> dict[str, str]:
    root = _files_root()
    now = int(time.time())
    ids = {
        "resp": f"resp_dur_inprog_{now}",
    }

    resp_entry = {
        "id": ids["resp"],
        "created_at": now,
        "expires_at": 0,
        "model": model,
        "instructions": None,
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "dur"}]}],
        "output": [],
        "usage": {},
        "response": {
            "id": ids["resp"],
            "object": "response",
            "created_at": now,
            "status": "in_progress",
            "background": True,
            "model": model,
            "output": [],
            "error": None,
        },
    }
    _write_json(root / "responses" / f"{ids['resp']}.json", resp_entry)

    return ids


def seed_completed_via_api(
    client: HttpClient,
    model: str,
    extra: dict[str, Any],
) -> dict[str, str]:
    """Create durable completed objects that must survive restart."""
    out: dict[str, str] = {}
    code, resp = client.post_json(
        "/v1/responses",
        {
            "model": model,
            **extra,
            "input": "Reply with exactly: DUR_RESP_OK",
            "temperature": 0,
            "store": True,
            "metadata": {"src": "local_durability_restart"},
            "user": "dur-user",
        },
    )
    if code == 200 and isinstance(resp, dict) and resp.get("id"):
        out["resp_completed"] = resp["id"]

    ch, chat = client.post_json(
        "/v1/chat/completions",
        {
            "model": model,
            **extra,
            "store": True,
            "metadata": {"src": "local_durability_restart"},
            "messages": [{"role": "user", "content": "Reply with exactly: DUR_CHAT_OK"}],
            "temperature": 0,
        },
    )
    if ch == 200 and isinstance(chat, dict) and chat.get("id"):
        out["chat"] = chat["id"]

    return out


def _cached_tokens(data: Any) -> int:
    """cached_tokens of a Responses body (0 when the field is absent)."""
    usage = data.get("usage") if isinstance(data, dict) else None
    details = usage.get("input_tokens_details") if isinstance(usage, dict) else None
    if not isinstance(details, dict):
        return 0
    return int(details.get("cached_tokens") or 0)


def seed_prompt_cache_warm(
    client: HttpClient,
    model: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Warm slot KV under one key; the state must not survive a server restart."""
    uniq = f"dur{time.time_ns()}"
    pad = f"durability cache pad {uniq}: " + ("romeo-sierra " * 40)
    nonce = f"DUR_CACHE_{uniq[-5:]}"
    body = {
        "model": model,
        **extra,
        "input": f"{pad}\nReply with exactly: {nonce}",
        "max_output_tokens": 64,
        "temperature": 0,
        "prompt_cache_key": f"resp-dur-cache-{uniq}",
        "prompt_cache_retention": "in_memory",
    }
    warm = 0
    for _ in range(2):
        code, data = client.post_json("/v1/responses", body)
        if code != 200:
            return {"error": f"warm-up http={code}", "nonce": nonce}
        warm = _cached_tokens(data)
    return {"body": body, "nonce": nonce, "warm_cached": warm}


def assert_prompt_cache_after_restart(
    client: HttpClient,
    report: Report,
    warmed: dict[str, Any],
) -> None:
    if "error" in warmed:
        report.add("restart", "prompt_cache_dropped_after_restart", "FAIL", warmed["error"])
        return
    if int(warmed.get("warm_cached") or 0) <= 0:
        report.add(
            "restart",
            "prompt_cache_dropped_after_restart",
            "SKIP",
            "no KV reuse before the restart, nothing to drop",
        )
        return
    code, data = client.post_json("/v1/responses", warmed["body"])
    text = output_text(data) if isinstance(data, dict) else ""
    cached = _cached_tokens(data)
    ok = code == 200 and warmed["nonce"] in text and cached == 0
    report.add(
        "restart",
        "prompt_cache_dropped_after_restart",
        "PASS" if ok else "FAIL",
        f"http={code} cached={cached} warm_before={warmed['warm_cached']} "
        f"nonce={warmed['nonce'] in text}",
    )


def assert_after_restart(
    client: HttpClient,
    report: Report,
    planted: dict[str, str],
    seeded: dict[str, str],
) -> None:
    rid = planted["resp"]
    code, data = client.get_json(f"/v1/responses/{rid}")
    err = data.get("error") if isinstance(data, dict) else None
    ok = (
        code == 200
        and isinstance(data, dict)
        and data.get("status") == "failed"
        and isinstance(err, dict)
        and err.get("code") == "server_restart"
    )
    report.add(
        "restart",
        "responses_in_progress_to_failed",
        "PASS" if ok else "FAIL",
        f"HTTP {code} status={data.get('status') if isinstance(data, dict) else None} err={err}",
    )

    if "resp_completed" in seeded:
        code, data = client.get_json(f"/v1/responses/{seeded['resp_completed']}")
        ok = code == 200 and isinstance(data, dict) and data.get("status") == "completed"
        # previous_response_id continuation
        fcode, follow = client.post_json(
            "/v1/responses",
            {
                "model": data.get("model") if isinstance(data, dict) else "",
                "previous_response_id": seeded["resp_completed"],
                "input": "Say CONT_OK",
                "temperature": 0,
                "store": True,
            },
        )
        ok2 = fcode == 200
        report.add(
            "restart",
            "responses_completed_reload_prev_id",
            "PASS" if ok and ok2 else "FAIL",
            f"get={code} follow={fcode}",
        )
    else:
        report.add("restart", "responses_completed_reload_prev_id", "FAIL", "missing seed")

    if "chat" in seeded:
        code, chat = client.get_json(f"/v1/chat/completions/{seeded['chat']}")
        ok = code == 200 and isinstance(chat, dict) and chat.get("id") == seeded["chat"]
        report.add(
            "restart",
            "chat_completion_store_reload",
            "PASS" if ok else "FAIL",
            f"HTTP {code}",
        )
    else:
        report.add("restart", "chat_completion_store_reload", "FAIL", "missing seed")


def run_restart_checks(
    client: HttpClient,
    report: Report,
    model: str,
    extra: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
) -> None:
    if os.environ.get("LOCAL_DURABILITY_SKIP_RESTART", "").strip() in ("1", "true", "yes"):
        report.add("restart", "server_restart", "SKIP", "LOCAL_DURABILITY_SKIP_RESTART set")
        return

    seeded = seed_completed_via_api(client, model, extra)
    planted = plant_interrupted_fixtures(model)
    warmed = seed_prompt_cache_warm(client, model, extra)
    report.add(
        "restart",
        "fixtures_planted",
        "PASS",
        f"planted={planted} seeded_keys={sorted(seeded)} warm_cached={warmed.get('warm_cached')}",
    )

    if not restart_server(base_url, api_key, report):
        return

    # New client after restart (same URL)
    client2 = HttpClient(base_url, api_key)
    assert_after_restart(client2, report, planted, seeded)
    assert_prompt_cache_after_restart(client2, report, warmed)
