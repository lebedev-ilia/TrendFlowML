"""Download and parse YouTube subtitle / automatic-caption text (ru, en)."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from typing import Any, Callable

from fetcher.dataset_collector.training_format import CAPTION_LANG_PREFERENCE, normalize_caption_lang

# Avoid multi‑MiB shard entries on pathological videos.
MAX_CAPTION_TEXT_CHARS = 250_000

_FORMAT_PREFERENCE = ("vtt", "srt", "json3", "srv3", "srv2", "srv1", "ttml")


def _pick_caption_format(formats: list[dict]) -> dict | None:
    if not formats:
        return None
    for ext in _FORMAT_PREFERENCE:
        matches = [f for f in formats if (f.get("ext") or "").lower() == ext]
        if matches:
            return matches[-1]
    return formats[-1]


def _parse_vtt_like(content: str) -> dict:
    cues: list[dict[str, str]] = []
    start: str | None = None
    end: str | None = None
    text_lines: list[str] = []

    def flush() -> None:
        nonlocal start, end, text_lines
        text = " ".join(line.strip() for line in text_lines if line.strip()).strip()
        if text:
            cue: dict[str, str] = {"text": text}
            if start:
                cue["start"] = start
            if end:
                cue["end"] = end
            cues.append(cue)
        start = None
        end = None
        text_lines = []

    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if line == "WEBVTT":
            continue
        if "-->" in line:
            flush()
            raw_start, raw_end = line.split("-->", 1)
            start = raw_start.strip()
            end = raw_end.strip().split(" ", 1)[0]
            continue
        if line.isdigit() or line.startswith("NOTE") or line.startswith("STYLE"):
            continue
        if re.match(r"^[\d:.,\s\-]+$", line):
            continue
        text_lines.append(line)
    flush()
    return {"text": "\n".join(cue["text"] for cue in cues), "cues": cues}


def _parse_json3(content: str) -> dict:
    data = json.loads(content)
    cues: list[dict[str, Any]] = []
    for event in data.get("events") or []:
        if not isinstance(event, dict):
            continue
        parts: list[str] = []
        for seg in event.get("segs") or []:
            if not isinstance(seg, dict):
                continue
            piece = seg.get("utf8") or ""
            if piece and piece != "\n":
                parts.append(str(piece))
        text = "".join(parts).strip()
        if text:
            cue: dict[str, Any] = {"text": text}
            if event.get("tStartMs") is not None:
                cue["start_ms"] = event.get("tStartMs")
            if event.get("dDurationMs") is not None:
                cue["duration_ms"] = event.get("dDurationMs")
            cues.append(cue)
    return {"text": "\n".join(cue["text"] for cue in cues), "cues": cues}


def _parse_xml_caption(content: str) -> dict:
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return _parse_vtt_like(content)
    cues: list[dict[str, Any]] = []
    for elem in root.iter():
        text = (elem.text or "").strip()
        if not text:
            continue
        cue: dict[str, Any] = {"text": text}
        for attr in ("start", "dur", "t", "d"):
            if attr in elem.attrib:
                cue[attr] = elem.attrib[attr]
        cues.append(cue)
    return {"text": "\n".join(cue["text"] for cue in cues), "cues": cues}


def parse_subtitle_content(content: str, ext: str | None) -> str:
    return str(parse_subtitle_payload(content, ext).get("text") or "")


def parse_subtitle_payload(content: str, ext: str | None) -> dict:
    ext_norm = (ext or "vtt").lower()
    if ext_norm in ("json3", "json"):
        try:
            return _parse_json3(content)
        except (json.JSONDecodeError, TypeError, KeyError):
            return {"text": "", "cues": []}
    if ext_norm in ("srv3", "srv2", "srv1", "ttml", "xml"):
        return _parse_xml_caption(content)
    return _parse_vtt_like(content)


def _truncate(text: str) -> str:
    if len(text) <= MAX_CAPTION_TEXT_CHARS:
        return text
    return text[:MAX_CAPTION_TEXT_CHARS] + "\n…[truncated]"


def _download_format_text(
    urlopen: Callable[..., Any],
    fmt: dict,
) -> tuple[str, dict] | None:
    url = fmt.get("url")
    if not url:
        return None
    ext = str(fmt.get("ext") or "vtt")
    try:
        response = urlopen(url)
        raw = response.read()
        if isinstance(raw, bytes):
            content = raw.decode("utf-8", errors="replace")
        else:
            content = str(raw)
    except Exception:
        return None
    payload = parse_subtitle_payload(content, ext)
    text = _truncate(str(payload.get("text") or ""))
    if not text:
        return None
    payload["text"] = text
    return ext, payload


def _texts_for_track_dict(
    tracks: dict | None,
    *,
    urlopen: Callable[..., Any],
) -> dict[str, dict[str, Any]]:
    """Return {ru|en: {ext, text}} from yt-dlp subtitles / automatic_captions dict."""
    if not tracks or not isinstance(tracks, dict):
        return {}
    by_lang: dict[str, tuple[str, dict, int]] = {}
    for lang, formats in tracks.items():
        canon = normalize_caption_lang(str(lang))
        if not canon or not isinstance(formats, list):
            continue
        fmt = _pick_caption_format(formats)
        if not fmt:
            continue
        downloaded = _download_format_text(urlopen, fmt)
        if not downloaded:
            continue
        ext, payload = downloaded
        text = str(payload.get("text") or "")
        prev = by_lang.get(canon)
        if prev is None or len(text) > prev[2]:
            by_lang[canon] = (ext, payload, len(text))
    return {
        lang: {"language": lang, "ext": ext, **payload}
        for lang, (ext, payload, _) in by_lang.items()
    }


def fetch_caption_texts_from_info(
    info: dict,
    *,
    urlopen: Callable[..., Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    manual = _texts_for_track_dict(info.get("subtitles"), urlopen=urlopen)
    auto = _texts_for_track_dict(info.get("automatic_captions"), urlopen=urlopen)
    return manual, auto


def build_caption_metadata(
    tracks: dict | None,
    texts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Merge track list with downloaded text; prefer ru then en."""
    from fetcher.dataset_collector.training_format import slim_caption_tracks

    ext_only = slim_caption_tracks(tracks)
    result: dict[str, Any] = {}
    for lang in CAPTION_LANG_PREFERENCE:
        if lang in texts:
            result[lang] = dict(texts[lang])
        elif lang in ext_only:
            exts = ext_only[lang]
            if exts:
                result[lang] = {"ext": exts[0].get("ext", "vtt")}
    return result


def caption_block_has_text(block: dict | None) -> bool:
    if not block or not isinstance(block, dict):
        return False
    for val in block.values():
        if isinstance(val, dict) and (val.get("text") or "").strip():
            return True
    return False


def captions_need_text_download(metadata: dict) -> bool:
    """True when enrich must fetch caption bodies (legacy ext-only entries)."""
    subs = metadata.get("subtitles") or {}
    auto = metadata.get("automatic_captions") or {}
    if caption_block_has_text(subs) or caption_block_has_text(auto):
        return False
    for block in (subs, auto):
        for val in block.values():
            if isinstance(val, list) and val:
                return True
            if isinstance(val, dict) and val.get("ext") and not val.get("text"):
                return True
    # Empty {} after enrich means no captions on this video — do not re-queue.
    return False
