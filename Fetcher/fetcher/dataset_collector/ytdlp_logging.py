"""yt-dlp logger: suppress known noisy YouTube metadata warnings."""

from __future__ import annotations

import re

_SUBTITLE_HTTP_NOISE = re.compile(
    r"(HTTP Error 403|Unable to download (?:auto )?subtitles|"
    r"subtitles for |timedtext|caption)",
    re.IGNORECASE,
)

_YOUTUBE_FORMAT_NOISE = re.compile(
    r"(nsig extraction failed|Signature extraction failed|"
    r"n challenge solving failed|challenge solving failed|"
    r"Only images are available for download|"
    r"Falling back to generic n function search|"
    r"Some .* client https formats have been skipped|"
    r"YouTube is forcing SABR streaming|"
    r"Unable to download format|Requested format is not available|"
    r"Please report this issue|player = https://www\\.youtube\\.com/s/player/)",
    re.IGNORECASE,
)


class YtdlpEnrichLogger:
    """Drop 403 / missing-caption messages; keep real extraction failures."""

    def debug(self, msg: str) -> None:
        return

    def info(self, msg: str) -> None:
        return

    def warning(self, msg: str) -> None:
        if _SUBTITLE_HTTP_NOISE.search(msg) or _YOUTUBE_FORMAT_NOISE.search(msg):
            return
        print(f"[yt-dlp] WARNING: {msg}", flush=True)

    def error(self, msg: str) -> None:
        if _SUBTITLE_HTTP_NOISE.search(msg) or _YOUTUBE_FORMAT_NOISE.search(msg):
            return
        print(f"[yt-dlp] ERROR: {msg}", flush=True)
