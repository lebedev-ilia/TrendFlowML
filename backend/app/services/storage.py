from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from ..config import Settings


@dataclass
class VideoMeta:
    duration_sec: Optional[int]
    width: Optional[int]
    height: Optional[int]


def _require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable not found in PATH: {name}")


def probe_video(path: str) -> VideoMeta:
    _require_executable("ffprobe")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=width,height",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        os.path.abspath(path),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {p.stderr.strip()}")
    lines = [ln.strip() for ln in (p.stdout or "").splitlines() if ln.strip()]
    duration = None
    width = None
    height = None
    if lines:
        try:
            duration = int(float(lines[0]))
        except Exception:
            duration = None
    if len(lines) >= 3:
        try:
            width = int(lines[1])
            height = int(lines[2])
        except Exception:
            width = None
            height = None
    return VideoMeta(duration_sec=duration, width=width, height=height)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs() -> None:
    settings = Settings()
    paths = settings.resolve_paths()
    paths.storage_root.mkdir(parents=True, exist_ok=True)
    paths.raw_uploads_dir.mkdir(parents=True, exist_ok=True)
    paths.frames_dir_base.mkdir(parents=True, exist_ok=True)
    paths.result_store_base.mkdir(parents=True, exist_ok=True)
    paths.example_videos_dir.mkdir(parents=True, exist_ok=True)


def move_upload_to_storage(
    upload_temp_path: str,
    video_id: str,
    filename: Optional[str] = None,
) -> Tuple[str, str]:
    settings = Settings()
    paths = settings.resolve_paths()
    ensure_dirs()

    ext = Path(filename or upload_temp_path).suffix or ".mp4"
    out_dir = paths.raw_uploads_dir / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"video{ext}"
    shutil.move(upload_temp_path, out_path)

    example_copy = paths.example_videos_dir / f"{video_id}{ext}"
    try:
        shutil.copy2(out_path, example_copy)
    except Exception:
        pass

    return str(out_path), str(example_copy)

