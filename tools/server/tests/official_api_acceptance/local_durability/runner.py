"""CLI runner: local durability / discovery acceptance."""

from __future__ import annotations

import argparse
import json
import os

from .checks_discovery import run_discovery_checks
from .checks_restart import run_restart_checks
from .checks_semantics import run_semantic_checks
from .checks_slot_tools import run_slot_and_tools_checks
from .http_client import HttpClient
from .report import Report

BLOCKING = frozenset({"FAIL", "PARTIAL", "NOT_IMPLEMENTED"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Local durability & discovery acceptance: /v1/tools, /props slot_save_path, "
            "compact expand, store restart finalize/reload."
        )
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get(
            "LOCAL_DURABILITY_BASE_URL",
            os.environ.get("OFFICIAL_API_BASE_URL", "http://127.0.0.1:8080"),
        ),
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get(
            "LOCAL_DURABILITY_API_KEY",
            os.environ.get("OFFICIAL_API_KEY", os.environ.get("OPENAI_API_KEY", "sk-1234567890")),
        ),
    )
    p.add_argument(
        "--model",
        default=os.environ.get(
            "LOCAL_DURABILITY_MODEL",
            os.environ.get("OFFICIAL_API_MODEL", "Qwen3.5-9B-Q4_K_M"),
        ),
    )
    p.add_argument(
        "--extra-json",
        default=os.environ.get("LOCAL_DURABILITY_EXTRA_JSON", os.environ.get("OFFICIAL_API_EXTRA_JSON", "")),
    )
    p.add_argument("--report-json", default="")
    p.add_argument(
        "--skip-restart",
        action="store_true",
        help="Skip kill/restart durability checks (also LOCAL_DURABILITY_SKIP_RESTART=1)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    extra: dict = {}
    if args.extra_json.strip():
        extra = json.loads(args.extra_json)
    if args.skip_restart:
        os.environ["LOCAL_DURABILITY_SKIP_RESTART"] = "1"

    report = Report()
    print("Local durability & discovery acceptance")
    print(f"base_url={args.base_url} model={args.model}")
    print("pass_rule=every check PASS (SKIP ok); FAIL/PARTIAL/NOT_IMPLEMENTED => fail")
    print("=" * 72)

    client = HttpClient(args.base_url, args.api_key)
    code, _, _ = client.request("GET", "/health")
    if code == 200:
        report.add("meta", "health", "PASS", "HTTP 200")
    else:
        report.add("meta", "health", "FAIL", f"HTTP {code}")
        print(report.summary_text())
        if args.report_json:
            report.write_json(args.report_json)
        return 1

    run_discovery_checks(client, report)
    run_slot_and_tools_checks(client, report, args.model)
    run_semantic_checks(client, report, args.model, extra)
    run_restart_checks(
        client,
        report,
        args.model,
        extra,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    print(report.summary_text())
    if args.report_json:
        report.write_json(args.report_json)

    bad = [r for r in report.rows if r.status in BLOCKING]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
