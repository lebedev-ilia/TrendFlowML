from __future__ import annotations

import argparse
import os
import time
import urllib.error
import urllib.request
from typing import Iterable, List, Optional, Tuple


def _env_first(*names: str) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


def _get(url: str, *, timeout_sec: float) -> Tuple[int, str]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            status = int(getattr(resp, "status", 200) or 200)
            body = resp.read() or b""
            return status, body.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raw = ""
        try:
            raw = e.read().decode("utf-8", errors="replace")
        except Exception:
            raw = ""
        return int(e.code), raw
    except Exception as e:
        return 0, str(e)


def _normalize_base_url(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if not u:
        return ""
    if not (u.startswith("http://") or u.startswith("https://")):
        u = "http://" + u
    return u.rstrip("/")


def _parse_models(x: str) -> List[str]:
    raw = (x or "").strip()
    if not raw:
        return []
    out: List[str] = []
    for part in raw.split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out


def _models_for_preset(preset: str) -> List[str]:
    p = (preset or "").strip().lower()
    if not p:
        return []
    if p in ("core_low", "core-low", "audit_v3_core_low", "audit-v3-core-low"):
        # Models present in DataProcessor/triton/models_core_low
        return [
            "preprocess_clip_image_224",
            "clip_image_224",
            "clip_text",
            "preprocess_raft_256",
            "raft_256",
            "preprocess_midas_256",
            "midas_256",
            "preprocess_places365_224",
            "places365_resnet50_224",
        ]
    if p in ("clip_224", "clip-image-text-224", "clip_image_text_224"):
        return [
            "preprocess_clip_image_224",
            "clip_image_224",
            "clip_text",
        ]
    return []


def _check_ready(base_url: str, *, timeout_sec: float) -> Tuple[bool, str]:
    status, body = _get(f"{base_url}/v2/health/ready", timeout_sec=timeout_sec)
    if status == 200:
        return True, "ready"
    if status == 0:
        return False, f"unreachable: {body}"
    return False, f"HTTP {status}: {body[:200]}".strip()


def _check_model_ready(base_url: str, *, model: str, timeout_sec: float) -> Tuple[bool, str]:
    status, body = _get(f"{base_url}/v2/models/{model}/ready", timeout_sec=timeout_sec)
    if status == 200:
        return True, "ready"
    if status == 0:
        return False, f"unreachable: {body}"
    # 404 can be valid if model is not loaded / not in repo
    return False, f"HTTP {status}: {body[:200]}".strip()


def _backoff_sleep_sec(*, attempt_idx: int, base_sec: float, multiplier: float, max_sec: float) -> float:
    # attempt_idx: 0..N-1, but sleep is typically used between attempts
    try:
        x = float(base_sec) * (float(multiplier) ** max(0, int(attempt_idx)))
        return min(float(max_sec), max(0.0, x))
    except Exception:
        return 0.0


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Triton preflight: readiness and optional model readiness checks.")
    p.add_argument(
        "--triton-http-url",
        default=None,
        help="Triton HTTP base URL (default: TRITON_HTTP_URL or TRITON_ENDPOINT or http://localhost:8000)",
    )
    p.add_argument("--timeout-sec", type=float, default=3.0, help="Timeout per request (seconds). Default: 3.")
    p.add_argument(
        "--attempts",
        type=int,
        default=5,
        help="How many readiness attempts to make (with backoff). Default: 5.",
    )
    p.add_argument(
        "--backoff-sec",
        type=float,
        default=0.5,
        help="Base backoff delay (seconds). Default: 0.5.",
    )
    p.add_argument(
        "--backoff-multiplier",
        type=float,
        default=2.0,
        help="Backoff multiplier. Default: 2.0.",
    )
    p.add_argument(
        "--backoff-max-sec",
        type=float,
        default=8.0,
        help="Maximum backoff delay (seconds). Default: 8.0.",
    )
    p.add_argument(
        "--models-preset",
        default="",
        help="Optional preset for model readiness checks. Example: core_low.",
    )
    p.add_argument(
        "--models",
        default="",
        help="Comma-separated model names to check via /v2/models/<name>/ready (optional).",
    )
    p.add_argument(
        "--require",
        action="store_true",
        help="If set, non-ready Triton or non-ready models -> exit 2. Otherwise prints status and exits 0.",
    )

    args = p.parse_args(list(argv) if argv is not None else None)

    base_url = _normalize_base_url(
        args.triton_http_url
        or _env_first("TRITON_HTTP_URL", "TRITON_ENDPOINT")
        or "http://localhost:8000"
    )
    timeout_sec = float(args.timeout_sec)
    models: List[str] = []
    models.extend(_models_for_preset(str(args.models_preset or "")))
    models.extend(_parse_models(args.models))
    # de-dupe while preserving order
    if models:
        seen = set()
        deduped: List[str] = []
        for m in models:
            if m not in seen:
                seen.add(m)
                deduped.append(m)
        models = deduped

    attempts = max(1, int(args.attempts))
    ok = False
    msg = "not_checked"
    for i in range(attempts):
        ok, msg = _check_ready(base_url, timeout_sec=timeout_sec)
        print(f"[triton] url={base_url} health.ready={ok} ({msg}) attempt={i+1}/{attempts}")
        if ok:
            break
        if i < attempts - 1:
            s = _backoff_sleep_sec(
                attempt_idx=i,
                base_sec=float(args.backoff_sec),
                multiplier=float(args.backoff_multiplier),
                max_sec=float(args.backoff_max_sec),
            )
            if s > 0:
                time.sleep(s)

    model_fail = False
    for m in models:
        mok, mmsg = _check_model_ready(base_url, model=m, timeout_sec=timeout_sec)
        print(f"[triton] model={m} ready={mok} ({mmsg})")
        if not mok:
            model_fail = True

    if args.require and (not ok or model_fail):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

