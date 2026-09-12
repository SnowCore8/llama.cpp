"""CLI runner: strict official OpenAI Chat Completions API acceptance."""

from __future__ import annotations

import argparse
import json
import os

from .checks_create_params import run_create_param_checks
from .checks_endpoints import run_endpoint_checks
from .checks_models import run_models_checks
from .checks_scenarios import run_scenario_checks
from .checks_sdk import run_sdk_checks
from .checks_semantics import run_semantic_checks
from .checks_streaming import run_completion_object_checks, run_streaming_checks
from .http_client import ChatHttpClient
from .report import Report

BLOCKING = frozenset({"FAIL", "PARTIAL", "NOT_IMPLEMENTED"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Strict acceptance of a server against the public OpenAI "
            "Chat Completions API (docs + openai Python SDK). Exit 0 only when "
            "every official surface PASSes (SKIP allowed for inapplicable probes)."
        )
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("CHAT_BASE_URL", "http://127.0.0.1:8080"),
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get(
            "CHAT_API_KEY",
            os.environ.get("OPENAI_API_KEY", "sk-1234567890"),
        ),
    )
    p.add_argument(
        "--model",
        default=os.environ.get("CHAT_MODEL", "Qwen3.5-9B-Q4_K_M"),
    )
    p.add_argument(
        "--extra-json",
        default=os.environ.get("CHAT_EXTRA_JSON", ""),
        help="Vendor-only JSON merged into create bodies (via extra_body in SDK)",
    )
    p.add_argument("--report-json", default="")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    extra: dict = {}
    if args.extra_json.strip():
        extra = json.loads(args.extra_json)

    report = Report()
    print("Official OpenAI Chat Completions stable API acceptance (strict, full surface)")
    print(f"base_url={args.base_url} model={args.model}")
    print("normative=OpenAI Chat Completions + Models + openai Python SDK")
    print("pass_rule=every official check PASS (SKIP ok); FAIL/PARTIAL/NOT_IMPLEMENTED => fail")
    print("=" * 72)

    # Under --reasoning-preserve, exact-text probes need thinking off (vendor kwargs).
    ctk = dict(extra.get("chat_template_kwargs") or {})
    ctk.setdefault("enable_thinking", False)
    extra["chat_template_kwargs"] = ctk

    client = ChatHttpClient(args.base_url, args.api_key)

    code, _, _raw = client.request("GET", "/health")
    if code == 200:
        report.add("meta", "health", "PASS", "HTTP 200")
    else:
        report.add("meta", "health", "SKIP", f"HTTP {code} (vendor preflight)")

    run_endpoint_checks(client, report, args.model, extra)
    run_models_checks(client, report, args.model)
    run_create_param_checks(client, report, args.model, extra)
    run_sdk_checks(
        report,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        extra=extra,
    )
    run_streaming_checks(client, report, args.model, extra)
    run_completion_object_checks(client, report, args.model, extra)
    run_scenario_checks(client, report, args.model, extra)
    run_semantic_checks(client, report, args.model, extra)

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
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"report: {args.report_json}")

    return 1 if gaps else 0


if __name__ == "__main__":
    raise SystemExit(main())
