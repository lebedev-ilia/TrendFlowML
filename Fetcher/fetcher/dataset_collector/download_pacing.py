from __future__ import annotations

import time
from typing import Literal

from fetcher.dataset_collector.schemas import CampaignConfig
from fetcher.dataset_collector.worker_logging import worker_log
from fetcher.dataset_collector.worker_shutdown import should_stop

DownloadPacingOutcome = Literal["success", "fail", "bot", "unavailable", "cookie_bot"]

_consecutive_bot_streak: int = 0


def reset_download_pacing() -> None:
    global _consecutive_bot_streak
    _consecutive_bot_streak = 0


def consecutive_bot_streak() -> int:
    return _consecutive_bot_streak


def compute_download_pause_seconds(
    config: CampaignConfig,
    outcome: DownloadPacingOutcome,
) -> float:
    """Pause length before the next queue item (seconds)."""
    global _consecutive_bot_streak

    if outcome == "success":
        _consecutive_bot_streak = 0
        return float(config.download_pause_after_success_seconds)

    if outcome == "unavailable":
        return float(config.download_pause_after_unavailable_seconds)

    if outcome == "cookie_bot":
        return float(config.download_pause_after_cookie_bot_seconds)

    if outcome == "bot":
        _consecutive_bot_streak += 1
        base = float(config.download_pause_after_bot_seconds)
        mult = float(config.download_pause_bot_backoff_multiplier)
        cap = float(config.download_pause_after_bot_max_seconds)
        delay = base * (mult ** max(_consecutive_bot_streak - 1, 0))
        return min(delay, cap)

    # generic fail (yt-dlp, merge, exhausted clients without bot exception)
    return float(config.download_pause_after_fail_seconds)


def interruptible_sleep(seconds: float) -> None:
    if seconds <= 0:
        return
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if should_stop():
            return
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def apply_download_pause(config: CampaignConfig, outcome: DownloadPacingOutcome) -> float:
    seconds = compute_download_pause_seconds(config, outcome)
    if seconds <= 0:
        return 0.0
    extra = ""
    if outcome == "bot":
        extra = f", bot_streak={_consecutive_bot_streak}"
    worker_log("download", f"pace sleep {seconds:.1f}s after {outcome}{extra}")
    interruptible_sleep(seconds)
    return seconds
