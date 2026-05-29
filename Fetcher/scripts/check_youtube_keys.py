#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import httpx


YOUTUBE_VIDEOS_URL = "https://youtube.googleapis.com/youtube/v3/videos"
DEFAULT_TEST_VIDEO_ID = "dQw4w9WgXcQ"


@dataclass
class KeyCheckResult:
    index: int
    key: str
    status: str
    reason: str
    http_status: Optional[int] = None
    quota_project: Optional[str] = None
    elapsed_ms: Optional[int] = None


def parse_keys(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    keys: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for token in re.split(r"[\s,;]+", line):
            token = token.strip().strip("\"'")
            if not token or token.startswith("#"):
                break
            if token not in seen:
                keys.append(token)
                seen.add(token)
    return keys


def mask_key(key: str) -> str:
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]}"


def normalize_proxy_url(value: str) -> str:
    proxy = value.strip()
    if proxy and "://" not in proxy:
        proxy = f"http://{proxy}"
    return proxy


def classify_error(payload: dict, status_code: int) -> tuple[str, str]:
    error = payload.get("error") or {}
    message = str(error.get("message") or "")
    reasons = []
    for item in error.get("errors") or []:
        reason = item.get("reason")
        if reason:
            reasons.append(str(reason))
    reason_text = ",".join(reasons) or message or f"http_{status_code}"
    lowered = reason_text.lower()
    if "keyinvalid" in lowered or "badrequest" in lowered or "api key not valid" in message.lower():
        return "invalid", reason_text
    if "accessnotconfigured" in lowered or "apihasnotbeenused" in lowered:
        return "api_disabled", reason_text
    if "quotaexceeded" in lowered or "dailylimitexceeded" in lowered:
        return "quota_exceeded", reason_text
    if status_code == 403:
        return "forbidden", reason_text
    if status_code == 429:
        return "rate_limited", reason_text
    return "error", reason_text


def check_key(
    client: httpx.Client,
    api_key: str,
    *,
    index: int,
    video_id: str,
) -> KeyCheckResult:
    started = time.monotonic()
    try:
        response = client.get(
            YOUTUBE_VIDEOS_URL,
            params={
                "part": "id",
                "id": video_id,
                "key": api_key,
            },
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
    except httpx.RequestError as exc:
        return KeyCheckResult(
            index=index,
            key=api_key,
            status="network_error",
            reason=str(exc)[:500],
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )

    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if response.status_code == 200 and payload.get("items"):
        return KeyCheckResult(
            index=index,
            key=api_key,
            status="ok",
            reason="valid",
            http_status=response.status_code,
            elapsed_ms=elapsed_ms,
        )

    status, reason = classify_error(payload, response.status_code)
    return KeyCheckResult(
        index=index,
        key=api_key,
        status=status,
        reason=reason,
        http_status=response.status_code,
        elapsed_ms=elapsed_ms,
    )


def serialize_results(results: Iterable[KeyCheckResult], *, show_keys: bool) -> list[dict]:
    serialized = []
    for result in results:
        row = asdict(result)
        if not show_keys:
            row["key"] = mask_key(result.key)
        serialized.append(row)
    return serialized


def write_working_keys(results: Iterable[KeyCheckResult], path: Path) -> None:
    working = [result.key for result in results if result.status == "ok"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(working) + ("\n" if working else ""), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check YouTube Data API keys with a cheap videos.list request.")
    parser.add_argument("keys_file", type=Path, help="Path to keys.txt or another text file with API keys.")
    parser.add_argument("--output", type=Path, default=Path("youtube_key_check_results.json"))
    parser.add_argument("--working-output", type=Path, default=Path("youtube_working_keys.txt"))
    parser.add_argument("--video-id", default=DEFAULT_TEST_VIDEO_ID)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay between key checks in seconds.")
    parser.add_argument("--proxy", help="Optional HTTP/HTTPS proxy URL. Env HTTP_PROXY/HTTPS_PROXY is also honored.")
    parser.add_argument("--show-keys", action="store_true", help="Write full keys into the JSON output.")
    parser.add_argument("--no-working-output", action="store_true", help="Do not write working keys file.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    keys = parse_keys(args.keys_file)
    if not keys:
        print(f"No keys found in {args.keys_file}", file=sys.stderr)
        return 2

    client_kwargs = {"timeout": args.timeout}
    if args.proxy:
        client_kwargs["proxy"] = normalize_proxy_url(args.proxy)

    results: list[KeyCheckResult] = []
    with httpx.Client(**client_kwargs) as client:
        for index, api_key in enumerate(keys, start=1):
            result = check_key(client, api_key, index=index, video_id=args.video_id)
            results.append(result)
            print(f"[{index}/{len(keys)}] {mask_key(api_key)} -> {result.status} ({result.reason})")
            if args.sleep > 0:
                time.sleep(args.sleep)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(serialize_results(results, show_keys=args.show_keys), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not args.no_working_output:
        write_working_keys(results, args.working_output)

    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    print(json.dumps({"checked": len(results), "counts": counts}, ensure_ascii=False))
    return 0 if counts.get("ok", 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
