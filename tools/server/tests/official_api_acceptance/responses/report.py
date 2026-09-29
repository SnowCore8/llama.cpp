"""Report row helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any
import json


@dataclass
class Row:
    area: str
    name: str
    status: str
    detail: str = ""


class Report:
    def __init__(self) -> None:
        self.rows: list[Row] = []

    def add(self, area: str, name: str, status: str, detail: str = "") -> None:
        self.rows.append(Row(area, name, status, detail))
        print(f"{status:16} [{area}] {name}" + (f" — {detail}" if detail else ""))

    def counts(self) -> Counter:
        return Counter(r.status for r in self.rows)

    def to_dict(self) -> dict[str, Any]:
        c = self.counts()
        return {
            "counts": dict(c),
            "rows": [asdict(r) for r in self.rows],
        }

    def write_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def summary_text(self) -> str:
        c = self.counts()
        lines = [
            "=" * 72,
            (
                f"TOTAL {sum(c.values())}  PASS={c['PASS']}  FAIL={c['FAIL']}  "
                f"NOT_IMPLEMENTED={c['NOT_IMPLEMENTED']}  PARTIAL={c['PARTIAL']}  "
                f"SKIP={c['SKIP']}"
            ),
        ]
        for area in (
            "endpoint",
            "create_param",
            "sdk",
            "stream_event",
            "response_object",
            "conversation",
            "background_stream",
            "output_logprobs",
            "web_search_stream",
            "scenario",
            "semantic",
            "cache",
            "ws",
            "meta",
        ):
            subset = [r for r in self.rows if r.area == area]
            if not subset:
                continue
            lines.append(f"{area}: {len(subset)}")
            for r in subset:
                if r.status in ("FAIL", "NOT_IMPLEMENTED", "PARTIAL"):
                    lines.append(f"  {r.status:16} {r.name} — {r.detail}")
        return "\n".join(lines)
