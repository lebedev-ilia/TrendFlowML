from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple


@dataclass(frozen=True)
class BigJsonStats:
    total_records_seen: int
    total_records_yielded: int


def iter_top_level_object(path: str) -> Iterator[Tuple[str, Any]]:
    """
    Stream-parse a JSON file whose top-level structure is an object:
      { "<key>": <value>, "<key2>": <value2>, ... }

    This avoids loading the full file into memory (important for large datasets).
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        decoder = json.JSONDecoder()
        buf = ""

        def _read_more() -> bool:
            nonlocal buf
            chunk = f.read(1024 * 1024)
            if not chunk:
                return False
            buf += chunk
            return True

        # prime buffer
        if not _read_more():
            return

        i = 0
        # skip whitespace
        while True:
            while i < len(buf) and buf[i].isspace():
                i += 1
            if i < len(buf):
                break
            if not _read_more():
                return

        if i >= len(buf) or buf[i] != "{":
            raise ValueError(f"Expected '{{' at start of JSON object: {path}")
        i += 1

        while True:
            # skip whitespace / commas
            while True:
                while i < len(buf) and buf[i].isspace():
                    i += 1
                if i < len(buf) and buf[i] == ",":
                    i += 1
                    continue
                break

            # ensure buffer
            while i >= len(buf):
                if not _read_more():
                    raise ValueError(f"Unexpected EOF while parsing {path}")

            # end object?
            if buf[i] == "}":
                return

            # parse key (JSON string)
            while True:
                try:
                    key, next_i = decoder.raw_decode(buf, i)
                    break
                except json.JSONDecodeError:
                    if not _read_more():
                        raise
            if not isinstance(key, str):
                raise ValueError(f"Expected string key at pos={i} in {path}")
            i = next_i

            # skip whitespace, expect colon
            while True:
                while i < len(buf) and buf[i].isspace():
                    i += 1
                if i < len(buf):
                    break
                if not _read_more():
                    raise ValueError(f"Unexpected EOF after key in {path}")
            if buf[i] != ":":
                raise ValueError(f"Expected ':' after key at pos={i} in {path}")
            i += 1

            # skip whitespace, parse value
            while True:
                while i < len(buf) and buf[i].isspace():
                    i += 1
                if i < len(buf):
                    break
                if not _read_more():
                    raise ValueError(f"Unexpected EOF before value in {path}")
            while True:
                try:
                    value, next_i = decoder.raw_decode(buf, i)
                    break
                except json.JSONDecodeError:
                    if not _read_more():
                        raise

            i = next_i
            yield key, value

            # periodically drop consumed buffer to keep memory bounded
            if i > 4 * 1024 * 1024:
                buf = buf[i:]
                i = 0


def load_video_records_subset(
    path: str,
    *,
    include_video_ids: Optional[set[str]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], BigJsonStats]:
    """
    Load only a subset of video records (by video_id) from a large top-level JSON object file.
    """
    out: Dict[str, Dict[str, Any]] = {}
    seen = 0
    yielded = 0
    for vid, payload in iter_top_level_object(path):
        seen += 1
        if include_video_ids is not None and vid not in include_video_ids:
            continue
        if isinstance(payload, dict):
            out[vid] = payload
            yielded += 1
    return out, BigJsonStats(total_records_seen=seen, total_records_yielded=yielded)


