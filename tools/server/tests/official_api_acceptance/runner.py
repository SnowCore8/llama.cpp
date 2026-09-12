"""Sequential meta-runner for official API acceptance suites."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

TESTS_DIR = Path(__file__).resolve().parent.parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

SUITE_ORDER: list[tuple[str, str, str]] = [
    ("responses", "responses_official_acceptance", "responses.json"),
    ("chat_completions", "chat_completions_official_acceptance", "chat_completions.json"),
    # Last: may kill/restart llama-server to verify durable stores.
    ("local_durability", "local_durability_acceptance", "local_durability.json"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run official API acceptance suites sequentially "
            "(OpenAI Responses, OpenAI Chat Completions, "
            "local durability/discovery). "
            "Local durability may restart the server."
        )
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("OFFICIAL_API_BASE_URL", "http://127.0.0.1:8080"),
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get(
            "OFFICIAL_API_KEY",
            os.environ.get("OPENAI_API_KEY", "sk-1234567890"),
        ),
    )
    p.add_argument(
        "--model",
        default=os.environ.get("OFFICIAL_API_MODEL", "Qwen3.5-9B-Q4_K_M"),
    )
    p.add_argument(
        "--extra-json",
        default=os.environ.get("OFFICIAL_API_EXTRA_JSON", ""),
        help="Vendor-only JSON merged into create bodies",
    )
    p.add_argument(
        "--report-dir",
        default=os.environ.get("OFFICIAL_API_REPORT_DIR", ""),
        help="Directory for per-suite JSON reports and summary.json",
    )
    return p.parse_args(argv)


def _load_main(package: str) -> Callable[..., int] | None:
    try:
        mod = importlib.import_module(f"{package}.runner")
    except ImportError:
        return None
    main_fn = getattr(mod, "main", None)
    return main_fn if callable(main_fn) else None


def _run_suite(
    label: str,
    package: str,
    report_file: Path | None,
    *,
    base_url: str,
    api_key: str,
    model: str,
    extra_json: str,
) -> dict[str, Any]:
    print()
    print("=" * 72)
    print(f"SUITE: {label} ({package})")
    print("=" * 72)

    main_fn = _load_main(package)
    if main_fn is None:
        print(f"FAIL missing suite: {package} not importable")
        result = {
            "suite": label,
            "package": package,
            "exit_code": 1,
            "status": "FAIL",
            "detail": "missing suite package",
        }
        if report_file is not None:
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    argv = [
        "--base-url",
        base_url,
        "--api-key",
        api_key,
        "--model",
        model,
    ]
    argv.extend(["--extra-json", extra_json])
    if report_file is not None:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        argv.extend(["--report-json", str(report_file)])

    code = int(main_fn(argv))
    status = "PASS" if code == 0 else "FAIL"
    return {
        "suite": label,
        "package": package,
        "exit_code": code,
        "status": status,
        "report_json": str(report_file) if report_file else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_dir = Path(args.report_dir) if args.report_dir.strip() else None

    print("Official API acceptance meta-runner")
    print(f"base_url={args.base_url} model={args.model}")
    if report_dir:
        print(f"report_dir={report_dir}")

    results: list[dict[str, Any]] = []
    for label, package, filename in SUITE_ORDER:
        report_file = report_dir / filename if report_dir else None
        results.append(
            _run_suite(
                label,
                package,
                report_file,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                extra_json=args.extra_json,
            )
        )

    all_pass = all(r["exit_code"] == 0 for r in results)
    summary = {
        "base_url": args.base_url,
        "model": args.model,
        "overall_status": "PASS" if all_pass else "FAIL",
        "suites": results,
    }

    print()
    print("=" * 72)
    print("META SUMMARY")
    for r in results:
        print(f"  {r['status']:4}  {r['suite']} (exit={r['exit_code']})")
    print(f"overall: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 72)

    if report_dir:
        report_dir.mkdir(parents=True, exist_ok=True)
        summary_path = report_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"summary: {summary_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
