#!/usr/bin/env python3
"""
Прогон полного E2E на реальных коротких видео из HF Ilialebedev/videos11.

Fetcher Celery worker читает FETCHER_YOUTUBE_MOCK_SAMPLE_VIDEO_DIR только при старте
(из e2e_env.sh → example/example_videos). Поэтому перед каждым прогоном копируем
HF mp4 как example/example_videos/{video_id}.mp4 и чистим кеш Segmenter.

Предусловия:
  ./backend/scripts/start_e2e_stack.sh --with-infra
  python example/scripts/download_hf_videos11_samples.py --count 5

Usage:
  cd backend && source scripts/e2e_env.sh && source .venv/bin/activate
  python scripts/e2e_run_hf_videos11.py --count 5 --with-triton-docker
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    videos = data.get("videos") or []
    if not isinstance(videos, list):
        raise ValueError("manifest.videos must be a list")
    return [v for v in videos if isinstance(v, dict) and v.get("video_id")]


def _ffprobe_duration(path: Path) -> Optional[float]:
    ffprobe = shutil.which("ffprobe") or str(_repo_root() / "tools" / "bin" / "ffprobe")
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode != 0:
            return None
        return float((r.stdout or "").strip())
    except (ValueError, subprocess.TimeoutExpired, OSError):
        return None


def _prepare_mock_video(root: Path, video_id: str, hf_path: Path) -> Path:
    """Fetcher worker видит example/example_videos — кладём туда {id}.mp4."""
    if not hf_path.is_file():
        raise FileNotFoundError(f"HF video missing: {hf_path}")
    dest = root / "example" / "example_videos" / f"{video_id}.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(hf_path, dest)
    dur = _ffprobe_duration(dest)
    print(f"  mock video: {dest} ({dur:.1f}s)" if dur else f"  mock video: {dest}", flush=True)
    return dest


def _clear_caches(root: Path, video_id: str) -> None:
    storage = root / "storage"
    url_cache = storage / "videos" / "_url_cache"
    if url_cache.is_dir():
        for p in url_cache.glob("*.mp4"):
            p.unlink(missing_ok=True)
    frames = storage / "frames_dir" / video_id
    if frames.is_dir():
        for rel in ("audio/audio.wav", "audio/metadata.json", "audio/segments.json"):
            (frames / rel).unlink(missing_ok=True)
    # Старые NPZ/manifest с mock-прогона и тот же run_id ломают §0.2 (stale audio_too_short).
    rs = storage / "result_store" / "youtube" / video_id
    if rs.is_dir():
        shutil.rmtree(rs)
    state = storage / "state" / "youtube" / video_id
    if state.is_dir():
        shutil.rmtree(state)


def _verify_run_audio(root: Path, video_id: str, run_id: str, expected_sec: float) -> Optional[str]:
    meta = root / "storage" / "frames_dir" / video_id / "audio" / "metadata.json"
    if not meta.is_file():
        return f"segmenter metadata missing: {meta}"
    data = json.loads(meta.read_text(encoding="utf-8"))
    dur = float(data.get("duration_sec") or 0)
    if dur < expected_sec * 0.85:
        return f"segmenter audio duration {dur:.2f}s << expected {expected_sec:.2f}s (likely mock sample_*.mp4 cache)"
    return None


def _seed_face_for_hf_video(root: Path, video_id: str, hf_mp4: Path) -> Optional[str]:
    """Добавить seed-лицо в Embedding Service до E2E (face_identity match)."""
    seed_py = root / "DataProcessor" / "embedding_service" / "scripts" / "seed_e2e_hf_face_from_video.py"
    dp_py = root / "DataProcessor" / ".data_venv" / "bin" / "python"
    if not seed_py.is_file() or not dp_py.is_file():
        print(f"WARN: skip face seed (missing {seed_py} or {dp_py})", flush=True)
        return "skip_missing_script"
    seed_name = f"hf_seed_{video_id}"
    proc = subprocess.run(
        [
            str(dp_py),
            str(seed_py),
            "--video",
            str(hf_mp4),
            "--name",
            seed_name,
        ],
        cwd=str(root / "DataProcessor"),
        capture_output=True,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout.strip(), flush=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        print(f"WARN: face seed failed for {video_id}: {err}", file=sys.stderr)
        return err[:200]
    return None


def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="E2E on HF videos11 samples (skip core_identity)")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=root / "example" / "hf_videos11" / "manifest.json",
    )
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--with-triton-docker", action="store_true")
    ap.add_argument("--e2e-low-vram", action="store_true", default=True)
    ap.add_argument("--no-e2e-low-vram", action="store_false", dest="e2e_low_vram")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument(
        "--with-core-identity",
        action="store_true",
        help="Включить semantic-хеды core_identity (по умолчанию отключены)",
    )
    ap.add_argument(
        "--fresh-results",
        action="store_true",
        help="Не мержить с существующим hf_videos11_results.json (перезаписать batch)",
    )
    ap.add_argument(
        "--no-seed-face",
        action="store_true",
        help="Не добавлять seed-лицо в Embedding Service (с --with-core-identity)",
    )
    ap.add_argument(
        "--sync-fixture",
        action="store_true",
        help="После batch (worst=0) скопировать results → tests/fixtures/hf_videos11_results.json для CI",
    )
    args = ap.parse_args()

    if not args.manifest.is_file():
        print(f"FAIL: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    videos = _load_manifest(args.manifest)
    if args.start_index:
        videos = videos[args.start_index :]
    videos = videos[: max(1, args.count)]
    if not videos:
        print("FAIL: no videos in manifest slice", file=sys.stderr)
        return 2

    hf_dir = args.manifest.parent.resolve()
    backend = root / "backend"
    e2e_py = backend / "scripts" / "e2e_full_max_run.py"
    quality_py = backend / "scripts" / "e2e_validate_output_quality.py"
    green_py = backend / "scripts" / "e2e_validate_full_green.py"
    dp_py = root / "DataProcessor" / ".data_venv" / "bin" / "python"

    out = backend / ".e2e" / "state" / "hf_videos11_results.json"
    results: List[Dict[str, Any]] = []
    batch_ids = {str(v["video_id"]) for v in videos}
    if out.is_file() and not getattr(args, "fresh_results", False):
        try:
            prev = json.loads(out.read_text(encoding="utf-8")).get("results") or []
            results = [r for r in prev if isinstance(r, dict) and r.get("video_id") not in batch_ids]
        except (json.JSONDecodeError, OSError):
            pass
    worst = max((int(r.get("e2e_exit") or 0) for r in results), default=0)

    for i, vid in enumerate(videos, start=1):
        video_id = str(vid["video_id"])
        expected_dur = float(vid.get("duration_sec") or 0)
        hf_mp4 = hf_dir / f"{video_id}.mp4"
        source_url = vid.get("source_url") or f"https://www.youtube.com/watch?v={video_id}"
        print(f"\n=== HF video {i}/{len(videos)}: {video_id} ({expected_dur}s) ===", flush=True)

        try:
            _prepare_mock_video(root, video_id, hf_mp4)
            _clear_caches(root, video_id)
        except FileNotFoundError as e:
            print(f"FAIL: {e}", file=sys.stderr)
            results.append({"video_id": video_id, "e2e_exit": 2, "error": str(e)})
            worst = max(worst, 2)
            continue

        if args.with_core_identity and not args.no_seed_face:
            seed_err = _seed_face_for_hf_video(root, video_id, hf_mp4)
            if seed_err and seed_err != "skip_missing_script":
                print(f"  face_seed: {seed_err}", flush=True)

        cmd = [
            sys.executable,
            str(e2e_py),
            "--offline-example",
            "--example-youtube-id",
            video_id,
            "--source-url",
            source_url,
            "--platform-video-id",
            video_id,
            "--real-video",
            "--cold-ingestion",
            "--timeout",
            str(args.timeout),
        ]
        if not args.with_core_identity:
            cmd.append("--skip-core-identity")
        if args.e2e_low_vram:
            cmd.append("--e2e-low-vram")
        if args.with_triton_docker:
            cmd.append("--with-triton-docker")

        proc = subprocess.run(cmd, cwd=str(backend))
        code = int(proc.returncode)
        entry: Dict[str, Any] = {"video_id": video_id, "e2e_exit": code}
        if args.with_core_identity:
            entry["core_identity"] = True

        if code == 0:
            rs_base = root / "storage" / "result_store" / "youtube" / video_id
            run_dirs = sorted(rs_base.iterdir()) if rs_base.is_dir() else []
            run_id = run_dirs[-1].name if run_dirs else None
            entry["run_id"] = run_id
            if run_id and expected_dur > 0:
                audio_err = _verify_run_audio(root, video_id, run_id, expected_dur)
                if audio_err:
                    print(f"FAIL audio check: {audio_err}", file=sys.stderr)
                    entry["audio_check"] = audio_err
                    code = 1
            if run_id and dp_py.is_file() and code == 0:
                q_args = [
                    str(dp_py),
                    str(quality_py),
                    "--run-id",
                    run_id,
                    "--video-id",
                    video_id,
                    "--real-video",
                ]
                if not args.with_core_identity:
                    q_args.append("--skip-core-identity")
                q = subprocess.run(q_args, cwd=str(backend))
                entry["quality_exit"] = q.returncode
                if q.returncode != 0:
                    code = q.returncode
                g = subprocess.run(
                    [sys.executable, str(green_py), "--run-id", run_id, "--video-id", video_id],
                    cwd=str(backend),
                )
                entry["green_exit"] = g.returncode
                if g.returncode != 0:
                    code = g.returncode

        entry["e2e_exit"] = code
        results.append(entry)
        worst = max(worst, code)
        print(f"=== done {video_id} exit={code} ===", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")
    print(f"\nSummary: {out}")
    for r in results:
        print(
            f"  {r.get('video_id')}: exit={r.get('e2e_exit')} "
            f"run={r.get('run_id', '-')} audio_check={r.get('audio_check', 'ok')}"
        )

    if args.sync_fixture and worst == 0:
        fixture = backend / "tests" / "fixtures" / "hf_videos11_results.json"
        sync_py = backend / "scripts" / "ci_sync_hf_results_fixture.sh"
        if sync_py.is_file():
            subprocess.run(["bash", str(sync_py)], cwd=str(backend), check=False)
            print(f"CI fixture synced: {fixture}", flush=True)
        else:
            fixture.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out, fixture)
            print(f"CI fixture copied: {out} -> {fixture}", flush=True)

    return worst


if __name__ == "__main__":
    raise SystemExit(main())
