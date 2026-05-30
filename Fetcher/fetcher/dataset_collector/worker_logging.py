from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable


def worker_log(component: str, message: str) -> None:
    """Human-readable lines for worker log files (stderr → merged into worker .log)."""
    print(f"[{component}] {message}", file=sys.stderr, flush=True)


def count_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def count_glob_files(directory: Path, pattern: str) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.glob(pattern) if p.is_file() and p.stat().st_size > 0)


def log_pass_header(component: str, title: str) -> None:
    worker_log(component, f"{'=' * 8} {title} {'=' * 8}")


def log_pass_footer(component: str, result: dict[str, Any]) -> None:
    parts = ", ".join(f"{k}={v}" for k, v in sorted(result.items()) if k != "error")
    worker_log(component, f"--- pass done: {parts} ---")
    if result.get("error"):
        worker_log(component, f"error: {result['error']}")


def log_kv_block(component: str, rows: Iterable[tuple[str, Any]]) -> None:
    for key, value in rows:
        worker_log(component, f"  {key}: {value}")
