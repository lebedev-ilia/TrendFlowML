#!/usr/bin/env python3
"""
E2E (extended smoke): Fetcher artifact -> DataProcessor (segmenter + audio [+ optional visual]).

Use-case:
  - You already have a completed Fetcher run_id (artifacts are READY).
  - You want to verify DataProcessor can execute Segmenter + AudioProcessor
    (default extractors: clap, tempo, loudness), optionally VisualProcessor (without Text).

Example (audio only):
  cd backend
  source scripts/e2e_env.sh
  .venv/bin/python -u scripts/e2e_dataprocessor_audio_smoke.py \
    --fetcher-run-id 3bee7a40-e835-49bc-a135-e3a3092d8953

Example (segmenter + audio + visual, text off):
  .venv/bin/python -u scripts/e2e_dataprocessor_audio_smoke.py \
    --fetcher-run-id <uuid> --with-visual

Override visual YAML (audit profiles live under DataProcessor/configs/audit_v3/visual/):
  ... --with-visual --visual-cfg-path /abs/path/to/visual_*.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import httpx


def _out(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def _repo_root() -> Path:
    # backend/scripts/<this_file> -> backend -> repo_root
    return Path(__file__).resolve().parents[2]


def _default_paths() -> dict[str, Path]:
    repo_root = _repo_root()
    storage_root = repo_root / "storage"
    return {
        "repo_root": repo_root,
        "storage_root": storage_root,
        "rs_base": storage_root / "result_store",
        "output": storage_root / "frames_dir",
        "visual_cfg_path": repo_root / "DataProcessor" / "configs" / "audit_v3" / "visual" / "visual_core_5_only.yaml",
        "dag_path": repo_root / "DataProcessor" / "docs" / "reference" / "component_graph.yaml",
    }


def _fetcher_manifest(client: httpx.Client, fetcher_url: str, run_id: str) -> dict[str, Any]:
    url = f"{fetcher_url.rstrip('/')}/api/v1/runs/{run_id}/manifest"
    r = client.get(url, timeout=20.0)
    r.raise_for_status()
    return r.json()


def _fetcher_video_url(client: httpx.Client, fetcher_url: str, run_id: str) -> str:
    url = f"{fetcher_url.rstrip('/')}/api/v1/runs/{run_id}/artifacts"
    r = client.get(url, timeout=20.0)
    r.raise_for_status()
    data = r.json()
    for item in (data.get("artifacts") or []):
        if isinstance(item, dict) and item.get("artifact_type") == "video_file":
            u = item.get("download_url")
            if isinstance(u, str) and u.strip():
                return u
    raise RuntimeError("Fetcher artifacts did not contain video_file.download_url")


def _dp_process(
    client: httpx.Client,
    dataprocessor_url: str,
    api_key: Optional[str],
    *,
    run_id: str,
    video_id: str,
    platform_id: str,
    video_url: str,
    rs_base: Path,
    output: Path,
    visual_cfg_path: Path,
    dag_path: Path,
    with_visual: bool,
) -> dict[str, Any]:
    endpoint = f"{dataprocessor_url.rstrip('/')}/api/v1/process"
    headers = {"X-API-Key": api_key} if api_key else None

    config_hash = (
        "e2e-segmenter-audio-visual-no-text-smoke" if with_visual else "e2e-segmenter-audio-smoke"
    )
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "video_id": video_id,
        "platform_id": platform_id,
        "video_url": video_url,
        "config_hash": config_hash,
        "profile_config": {
            "config_hash": config_hash,
            "processors": {
                "segmenter": {"enabled": True, "required": True},
                "audio": {"enabled": True, "required": False},
                "text": {"enabled": False, "required": False},
                "visual": {"enabled": bool(with_visual), "required": False},
            },
        },
        "run_audio": True,
        "run_text": False,
        "rs_base": str(rs_base),
        "output": str(output),
        "visual_cfg_path": str(visual_cfg_path),
        "dag_path": str(dag_path),
        "dag_stage": "baseline",
        "sampling_policy_version": "v1",
        "dataprocessor_version": "dev",
        "chunk_size": 64,
    }

    r = client.post(endpoint, json=payload, headers=headers, timeout=30.0)
    if r.status_code >= 400:
        detail = None
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise RuntimeError(
            f"DataProcessor /process failed: status={r.status_code} detail={detail}"
        )
    return r.json()


def _dp_status(
    client: httpx.Client,
    dataprocessor_url: str,
    api_key: Optional[str],
    run_id: str,
) -> dict[str, Any]:
    url = f"{dataprocessor_url.rstrip('/')}/api/v1/runs/{run_id}/status"
    headers = {"X-API-Key": api_key} if api_key else None
    r = client.get(url, headers=headers, timeout=20.0)
    r.raise_for_status()
    return r.json()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Extended DataProcessor smoke: segmenter + audio; optional visual (no text)."
    )
    p.add_argument("--fetcher-run-id", required=True, help="Existing Fetcher run_id (UUID)")
    p.add_argument(
        "--with-visual",
        action="store_true",
        help="Enable VisualProcessor via profile (still run_text=false). Needs a working visual_cfg_path (Triton/models per YAML).",
    )
    p.add_argument(
        "--visual-cfg-path",
        type=str,
        default=None,
        help="Override VisualProcessor YAML (default: audit_v3 visual_core_5_only.yaml from repo)",
    )
    p.add_argument(
        "--fetcher-url",
        default=os.environ.get("TF_BACKEND_FETCHER_API_URL", "http://localhost:8000"),
        help="Fetcher API URL (default from TF_BACKEND_FETCHER_API_URL)",
    )
    p.add_argument(
        "--dataprocessor-url",
        default=os.environ.get("TF_BACKEND_DATAPROCESSOR_API_URL", "http://localhost:8002"),
        help="DataProcessor API URL (default from TF_BACKEND_DATAPROCESSOR_API_URL)",
    )
    p.add_argument(
        "--dataprocessor-api-key",
        default=os.environ.get("TF_BACKEND_DATAPROCESSOR_API_KEY", ""),
        help="DataProcessor API key (default from TF_BACKEND_DATAPROCESSOR_API_KEY)",
    )
    p.add_argument("--timeout", type=int, default=1200, help="Overall timeout seconds")
    p.add_argument("--poll-interval", type=float, default=5.0, help="Polling interval seconds")
    args = p.parse_args()

    api_key = args.dataprocessor_api_key.strip() or None
    paths = _default_paths()
    if args.visual_cfg_path:
        paths = {**paths, "visual_cfg_path": Path(args.visual_cfg_path).resolve()}

    try:
        with httpx.Client() as client:
            man = _fetcher_manifest(client, args.fetcher_url, args.fetcher_run_id)
            video_id = str(man.get("video_id") or "").strip()
            platform_id = str(man.get("platform") or "").strip()
            if not video_id:
                raise RuntimeError("Fetcher manifest missing video_id")
            if platform_id not in ("youtube", "upload"):
                raise RuntimeError(f"Fetcher manifest returned unsupported platform='{platform_id}'")
            video_url = _fetcher_video_url(client, args.fetcher_url, args.fetcher_run_id)

            run_id = str(uuid.uuid4())
            _out(f"Fetcher run_id: {args.fetcher_run_id}")
            _out(f"New DataProcessor run_id: {run_id}")
            _out(f"Video: platform_id={platform_id} video_id={video_id}")

            resp = _dp_process(
                client,
                args.dataprocessor_url,
                api_key,
                run_id=run_id,
                video_id=video_id,
                platform_id=platform_id,
                video_url=video_url,
                rs_base=paths["rs_base"],
                output=paths["output"],
                visual_cfg_path=paths["visual_cfg_path"],
                dag_path=paths["dag_path"],
                with_visual=bool(args.with_visual),
            )
            _out(f"Process accepted: {json.dumps(resp, ensure_ascii=False)}")

            t0 = time.time()
            while True:
                if time.time() - t0 > float(args.timeout):
                    _out(f"Timed out after {args.timeout}s waiting for completion (run_id={run_id})")
                    _out(f"Status URL: {args.dataprocessor_url.rstrip('/')}/api/v1/runs/{run_id}/status")
                    return 2

                st = _dp_status(client, args.dataprocessor_url, api_key, run_id)
                progress = st.get("progress") or {}
                comps = progress.get("components") or {}
                compact = {
                    "status": st.get("status"),
                    "stage": st.get("stage") or progress.get("current_processor"),
                    "overall": progress.get("overall"),
                    "segmenter": (comps.get("segmenter") or {}).get("status"),
                    "audio": (comps.get("audio") or {}).get("status"),
                    "visual": (comps.get("visual") or {}).get("status"),
                }
                elapsed = int(time.time() - t0)
                _out(f"[{elapsed:>4}s] {json.dumps(compact, ensure_ascii=False)}")

                if st.get("status") in ("success", "error", "cancelled"):
                    break
                time.sleep(float(args.poll_interval))
    except Exception as e:
        _out(str(e))
        return 2

    run_rs_path = paths["rs_base"] / platform_id / video_id / run_id
    _out(f"Result store: {run_rs_path}")
    _out("Expected audio render artifacts:")
    _out(f"- {run_rs_path}/loudness_extractor/_render/render.html")
    _out(f"- {run_rs_path}/tempo_extractor/_render/render.html")
    _out(f"- {run_rs_path}/clap_extractor/_render/render.html")
    if args.with_visual:
        _out("VisualProcessor: check manifests / component outputs under:")
        _out(f"- {run_rs_path}/ (visual_processor and module dirs per enabled DAG nodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

