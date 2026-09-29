"""CLI runner: strict official OpenAI Responses API acceptance."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .checks_completions import run_completions_checks
from .checks_conversations import run_conversation_checks, run_conversation_response_checks
from .checks_create_params import run_create_param_checks
from .checks_embeddings import run_embeddings_checks
from .checks_endpoints import run_endpoint_checks
from .checks_models import run_models_checks
from .checks_prompt_cache import run_prompt_cache_checks
from .checks_scenarios import run_scenario_checks
from .checks_sdk import run_sdk_checks
from .checks_semantics import run_semantic_checks
from .checks_streaming import (
    run_background_stream_checks,
    run_output_logprobs_stream_checks,
    run_response_object_checks,
    run_streaming_checks,
    run_web_search_stream_checks,
)
from .checks_ws import run_ws_checks
from .http_client import ResponsesHttpClient
from .report import Report

BLOCKING = frozenset({"FAIL", "PARTIAL", "NOT_IMPLEMENTED"})

SUITE_GROUPS = (
    "endpoint",
    "models",
    "completions",
    "embeddings",
    "create_param",
    "sdk",
    "streaming",
    "scenario",
    "semantic",
    "prompt_cache",
    "conversation",
    "ws",
)


def _selected_groups(only: str) -> set[str]:
    if not only.strip():
        return set(SUITE_GROUPS)
    groups = {g.strip() for g in only.split(",") if g.strip()}
    unknown = groups - set(SUITE_GROUPS)
    if unknown:
        raise SystemExit(f"--only: unknown group(s) {sorted(unknown)}; valid: {list(SUITE_GROUPS)}")
    return groups


def _run_suite(report: Report, area: str, fn, *args, **kwargs):
    """Run one suite; a crash is recorded as a FAIL row so later suites still run."""
    try:
        return fn(*args, **kwargs)
    except Exception as err:  # noqa: BLE001 - keep the report alive, record the crash
        report.add(area, f"{fn.__name__}.crash", "FAIL", f"{type(err).__name__}: {err}"[:300])
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Strict acceptance of a server against the full public OpenAI "
            "Responses API (docs + openai Python SDK). Exit 0 only when every "
            "official surface PASSes (SKIP allowed for inapplicable probes)."
        )
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("RESPONSES_BASE_URL", "http://127.0.0.1:8080"),
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("RESPONSES_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
    )
    p.add_argument(
        "--model",
        default=os.environ.get("RESPONSES_MODEL", "gpt-4.1"),
    )
    p.add_argument(
        "--extra-json",
        default=os.environ.get("RESPONSES_EXTRA_JSON", ""),
        help="Vendor-only JSON merged into create bodies (not scored as OpenAI fields)",
    )
    p.add_argument(
        "--only",
        default=os.environ.get("RESPONSES_ONLY", ""),
        help="Comma-separated suite groups to run (default: all): " + ",".join(SUITE_GROUPS),
    )
    p.add_argument("--report-json", default="")
    return p.parse_args(argv)


def _seed_prompt_fixtures() -> None:
    """Install local prompt templates into the server's --openai-files-path/prompts/."""
    env_set = "LLAMA_OPENAI_FILES_PATH" in os.environ
    root = Path(os.environ.get("LLAMA_OPENAI_FILES_PATH", "/tmp/llama-openai-files"))
    if not env_set and not root.is_dir():
        return
    src = Path(__file__).resolve().parent.parent / "fixtures" / "prompts"
    if not src.is_dir():
        return
    dst = root / "prompts"
    try:
        dst.mkdir(parents=True, exist_ok=True)
        for path in sorted(src.glob("*.json")):
            shutil.copyfile(path, dst / path.name)
    except OSError as err:
        print(f"prompt fixtures not installed under {dst}: {err}")
        return
    print(f"prompt fixtures: {dst}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    extra: dict = {}
    if args.extra_json.strip():
        extra = json.loads(args.extra_json)

    report = Report()
    sel = _selected_groups(args.only)
    print("Official OpenAI Responses stable API acceptance (strict, full surface)")
    print(f"base_url={args.base_url} model={args.model}")
    print(f"only={','.join(g for g in SUITE_GROUPS if g in sel)}")
    print("normative=openai Responses+Completions+Models + openai Python SDK")
    print("pass_rule=every official check PASS (SKIP ok); FAIL/PARTIAL/NOT_IMPLEMENTED => fail")
    print("=" * 72)

    # Under --reasoning-preserve, probes that expect exact reply text / response.completed
    # must disable thinking; suites may override per-request (e.g. reasoning.effort=low).
    if "reasoning" not in extra:
        extra["reasoning"] = {"effort": "none"}

    client = ResponsesHttpClient(args.base_url, args.api_key)

    _seed_prompt_fixtures()

    code, _, _raw = client.request("GET", "/health")
    if code == 200:
        report.add("meta", "health", "PASS", "HTTP 200")
    else:
        report.add("meta", "health", "SKIP", f"HTTP {code} (vendor preflight)")

    rid = (
        _run_suite(report, "endpoint", run_endpoint_checks, client, report, args.model, extra)
        if "endpoint" in sel
        else None
    )
    if "models" in sel:
        _run_suite(report, "endpoint", run_models_checks, client, report, args.model)
    if "completions" in sel:
        _run_suite(report, "endpoint", run_completions_checks, client, report, args.model, extra)
    if "embeddings" in sel:
        _run_suite(report, "endpoint", run_embeddings_checks, client, report, args.model)
    if "create_param" in sel:
        _run_suite(report, "create_param", run_create_param_checks, client, report, args.model, extra, rid)
    if "sdk" in sel:
        _run_suite(
            report,
            "sdk",
            run_sdk_checks,
            report,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            extra=extra,
        )
    if "streaming" in sel:
        _run_suite(report, "stream_event", run_streaming_checks, client, report, args.model, extra)
        _run_suite(report, "response_object", run_response_object_checks, client, report, args.model, extra)
        _run_suite(report, "background_stream", run_background_stream_checks, client, report, args.model, extra)
        _run_suite(report, "output_logprobs", run_output_logprobs_stream_checks, client, report, args.model, extra)
        _run_suite(report, "web_search_stream", run_web_search_stream_checks, client, report, args.model, extra)
    if "ws" in sel:
        _run_suite(report, "ws", run_ws_checks, client, report, args.model, extra)
    if "conversation" in sel:
        _run_suite(report, "conversation", run_conversation_checks, client, report, args.model, extra)
        _run_suite(report, "conversation", run_conversation_response_checks, client, report, args.model, extra)
    if "scenario" in sel:
        _run_suite(report, "scenario", run_scenario_checks, client, report, args.model, extra)
    if "semantic" in sel:
        _run_suite(report, "semantic", run_semantic_checks, client, report, args.model, extra)
    cache_stats = (
        _run_suite(report, "cache", run_prompt_cache_checks, client, report, args.model, extra)
        if "prompt_cache" in sel
        else None
    )

    print(report.summary_text())

    gaps = [r for r in report.rows if r.status in BLOCKING]
    if gaps:
        print("gaps (blocking):")
        for r in gaps:
            print(f"  {r.status:16} [{r.area}] {r.name} — {r.detail}")

    if args.report_json:
        payload = report.to_dict()
        payload["gaps"] = [
            {"area": r.area, "name": r.name, "status": r.status, "detail": r.detail}
            for r in gaps
        ]
        if cache_stats is not None:
            payload["prompt_cache"] = cache_stats
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"report: {args.report_json}")

    return 1 if gaps else 0


if __name__ == "__main__":
    raise SystemExit(main())
