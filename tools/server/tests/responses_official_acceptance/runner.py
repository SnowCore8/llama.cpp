"""CLI runner: strict official OpenAI Responses API acceptance."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .checks_completions import run_completions_checks
from .checks_create_params import run_create_param_checks
from .checks_endpoints import run_endpoint_checks
from .checks_models import run_models_checks
from .checks_prompt_cache import run_prompt_cache_checks
from .checks_scenarios import run_scenario_checks
from .checks_sdk import run_sdk_checks
from .checks_semantics import run_semantic_checks
from .checks_streaming import run_background_stream_checks, run_response_object_checks, run_streaming_checks
from .http_client import ResponsesHttpClient
from .report import Report

BLOCKING = frozenset({"FAIL", "PARTIAL", "NOT_IMPLEMENTED"})


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
    print("Official OpenAI Responses stable API acceptance (strict, full surface)")
    print(f"base_url={args.base_url} model={args.model}")
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

    rid = run_endpoint_checks(client, report, args.model, extra)
    run_models_checks(client, report, args.model)
    run_completions_checks(client, report, args.model, extra)
    run_create_param_checks(client, report, args.model, extra, rid)
    run_sdk_checks(
        report,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        extra=extra,
    )
    run_streaming_checks(client, report, args.model, extra)
    run_response_object_checks(client, report, args.model, extra)
    run_background_stream_checks(client, report, args.model, extra)
    run_scenario_checks(client, report, args.model, extra)
    run_semantic_checks(client, report, args.model, extra)
    cache_stats = run_prompt_cache_checks(client, report, args.model, extra)

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
