from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator

VERBOSE = os.getenv("DATASET_COLLECTOR_VERBOSE_TIMING", "").lower() in {"1", "true", "yes"}


class TimingStats:
    """Accumulates phase durations; printed only in verbose mode."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def record(self, phase: str, seconds: float) -> None:
        self.totals[phase] += seconds
        self.counts[phase] += 1

    def summary(self) -> str:
        if not self.totals:
            return ""
        parts = []
        for phase in sorted(self.totals, key=self.totals.get, reverse=True)[:5]:
            total = self.totals[phase]
            count = self.counts[phase]
            avg = total / count if count else 0.0
            parts.append(f"{phase} {avg:.1f}s avg ({count})")
        return ", ".join(parts)


_TIMING = TimingStats()


def get_timing_stats() -> TimingStats:
    return _TIMING


def reset_timing_stats() -> None:
    _TIMING.totals.clear()
    _TIMING.counts.clear()


def log_timing(phase: str, seconds: float, **context: Any) -> None:
    _TIMING.record(phase, seconds)
    if not VERBOSE:
        return
    ctx = " ".join(f"{key}={value}" for key, value in context.items() if value is not None)
    suffix = f" {ctx}" if ctx else ""
    print(f"[timing] {phase} {seconds:.2f}s{suffix}", file=sys.stderr, flush=True)


@contextmanager
def timed_phase(phase: str, **context: Any) -> Iterator[None]:
    if not VERBOSE:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        log_timing(phase, time.perf_counter() - start, **context)
