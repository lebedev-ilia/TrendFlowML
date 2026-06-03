from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, TextIO

_STDERR_MAX_LINE = 500
_STDERR_DROP_MIN_LEN = 400


class _SanitizedStderr:
    """Prevent huge third-party stderr (e.g. Node botGuard.js) from filling worker logs."""

    def __init__(self, original: TextIO, *, max_line: int = _STDERR_MAX_LINE) -> None:
        self._original = original
        self._max_line = max_line

    def write(self, data: str) -> int:
        if not data:
            return 0
        written = 0
        for line in data.splitlines(keepends=True):
            if _should_suppress_stderr_line(line):
                line = "[worker] suppressed huge stderr chunk from third-party library\n"
            elif len(line) > self._max_line:
                line = line[: self._max_line] + "... [truncated]\n"
            n = self._original.write(line)
            written += n if n is not None else len(line)
        return written

    def flush(self) -> None:
        self._original.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


def _should_suppress_stderr_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < _STDERR_DROP_MIN_LEN:
        return False
    if stripped.startswith("["):
        return False
    if "botGuard.js" in stripped:
        return True
    if stripped.startswith("!function") or stripped.startswith("(function"):
        return True
    return False


def configure_worker_process_logging() -> None:
    """Quiet noisy libraries in worker subprocesses (stderr is merged into .log files)."""
    for name in ("pytubefix", "urllib3", "httpx"):
        logging.getLogger(name).setLevel(logging.ERROR)


def install_sanitized_worker_stderr() -> None:
    """Install once per download worker subprocess (stderr → .log file)."""
    configure_worker_process_logging()
    if not isinstance(sys.stderr, _SanitizedStderr):
        sys.stderr = _SanitizedStderr(sys.stderr)


@contextmanager
def sanitize_worker_stderr():
    """Wrap download (and similar) work so stderr noise does not bloat log files."""
    install_sanitized_worker_stderr()
    yield


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
