from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..config import Settings


@dataclass
class RunPaths:
    run_rs_path: Path
    frames_dir: Optional[Path]
    manifest_path: Path
    state_events_path: Path


def build_profile_yaml(config_json: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_json, f, sort_keys=False, allow_unicode=True)


def resolve_run_paths(
    *,
    platform_id: str,
    video_id: str,
    run_id: str,
    result_store_base: Path,
) -> RunPaths:
    run_rs_path = result_store_base / platform_id / video_id / run_id
    manifest_path = run_rs_path / "manifest.json"
    runs_root = result_store_base.parent
    state_events_path = runs_root / "state" / platform_id / video_id / run_id / "state_events.jsonl"
    return RunPaths(
        run_rs_path=run_rs_path,
        frames_dir=None,
        manifest_path=manifest_path,
        state_events_path=state_events_path,
    )


def run_dataprocessor(
    *,
    video_path: Path,
    platform_id: str,
    video_id: str,
    run_id: str,
    profile_config: Dict[str, Any],
    result_store_base: Path,
    frames_dir_base: Path,
    visual_cfg_default: Path,
) -> RunPaths:
    settings = Settings()
    paths = settings.resolve_paths()

    profile_dir = result_store_base.parent / "profiles_cache" / run_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / "profile.yaml"
    build_profile_yaml(profile_config, profile_path)

    dp_main = paths.dataproc_root / "main.py"
    if not dp_main.exists():
        raise FileNotFoundError(f"DataProcessor main not found: {dp_main}")

    cmd = [
        os.environ.get("PYTHON", "python3"),
        str(dp_main),
        "--video-path",
        str(video_path),
        "--output",
        str(frames_dir_base),
        "--chunk-size",
        "64",
        "--visual-cfg-path",
        str(visual_cfg_default),
        "--profile-path",
        str(profile_path),
        "--dag-path",
        str(paths.dataproc_root / "docs" / "reference" / "component_graph.yaml"),
        "--dag-stage",
        "baseline",
        "--platform-id",
        platform_id,
        f"--video-id={video_id}",
        "--run-id",
        run_id,
        "--sampling-policy-version",
        "v1",
        "--dataprocessor-version",
        "dev",
        "--rs-base",
        str(result_store_base),
    ]

    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    run_paths = resolve_run_paths(
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        result_store_base=result_store_base,
    )
    frames_dir = frames_dir_base / video_id / "video"
    run_paths.frames_dir = frames_dir if frames_dir.exists() else None
    return run_paths

