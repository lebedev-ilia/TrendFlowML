#!/usr/bin/env python3
"""
Human-friendly quality demo for scene_classification (Places365 + core_clip semantics):

This script generates an HTML report for manual inspection:
- scenes timeline (scene_id, label, duration)
- key per-scene aggregates (Places365 confidence/entropy + CLIP semantics aggregates)
- thumbnails for a few representative frames per scene
- basic sanity checks (alignment, meta fields via validate_npz)

Important:
- scene_classification depends on core_clip (no-fallback). This demo expects that
  `rs_path/core_clip/embeddings.npz` already exists and covers the same frame_indices.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np

# Add VisualProcessor to path
_visual_processor_path = Path(__file__).parent.parent.parent / "VisualProcessor"
sys.path.insert(0, str(_visual_processor_path))

from utils.artifact_validator import validate_npz
from utils.frame_manager import FrameManager
from utils.logger import get_logger

logger = get_logger("demo_scene_classification_quality")


def _npz_to_dict(npz_path: str) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    out: Dict[str, Any] = {}
    for k in data.files:
        v = data[k]
        if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
            try:
                out[k] = v.item()
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


def _img_to_data_uri_bgr(img_bgr: np.ndarray, quality: int = 85) -> Optional[str]:
    try:
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            return None
        b64 = base64.b64encode(buf).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def _resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    s = float(max_side) / float(m)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def run_segmenter(
    *,
    video_path: str,
    out_dir: str,
    visual_cfg_path: Optional[str],
) -> str:
    segmenter_script = Path(__file__).parent.parent.parent / "Segmenter" / "segmenter.py"
    if not segmenter_script.exists():
        raise FileNotFoundError(f"Segmenter script not found: {segmenter_script}")

    video_id = Path(video_path).stem
    frames_dir = os.path.join(out_dir, video_id, "video")
    meta_path = os.path.join(frames_dir, "metadata.json")
    if os.path.exists(meta_path):
        logger.info("Reusing existing frames_dir: %s", frames_dir)
        return frames_dir

    cmd = [
        sys.executable,
        str(segmenter_script),
        "--video-path",
        str(video_path),
        "--out-dir",
        str(out_dir),
    ]
    if visual_cfg_path:
        cmd.extend(["--visual-cfg-path", str(visual_cfg_path)])

    logger.info("Running Segmenter: %s", " ".join(cmd))
    subprocess.check_call(cmd)
    if not os.path.exists(meta_path):
        raise RuntimeError(f"Segmenter finished but metadata.json not found at: {meta_path}")
    return frames_dir


def run_scene_classification(
    *,
    frames_dir: str,
    rs_path: str,
    runtime: str,
    triton_model_spec: str,
    input_size: int,
    batch_size: int,
    triton_http_url: Optional[str],
    label_fusion: str,
) -> str:
    """
    Runs VisualProcessor/modules/scene_classification/main.py to produce:
      rs_path/scene_classification/scene_classification_features.npz
    """
    module_main = Path(__file__).parent.parent.parent / "VisualProcessor" / "modules" / "scene_classification" / "main.py"
    if not module_main.exists():
        raise FileNotFoundError(f"scene_classification main.py not found: {module_main}")

    out_npz = os.path.join(rs_path, "scene_classification", "scene_classification_features.npz")
    if os.path.exists(out_npz):
        logger.info("Reusing existing artifact: %s", out_npz)
        return out_npz

    env = os.environ.copy()
    if triton_http_url:
        env["TRITON_HTTP_URL"] = str(triton_http_url)
    # ModelManager needs DP_MODELS_ROOT; for baseline demos default to the repo bundle if missing.
    if not env.get("DP_MODELS_ROOT"):
        cand = Path(__file__).parent.parent.parent / "dp_models" / "bundled_models"
        if cand.exists():
            env["DP_MODELS_ROOT"] = str(cand)
    # Ensure repo-root packages (dp_models, dp_triton, etc.) are importable when running module entrypoints directly.
    repo_root = str(Path(__file__).parent.parent.parent)
    env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    # scene_classification depends on torch; prefer VisualProcessor venv if present.
    vp_python = (Path(__file__).parent.parent.parent / "VisualProcessor" / ".vp_venv" / "bin" / "python")
    py_exec = str(vp_python) if vp_python.exists() else sys.executable

    cmd = [
        py_exec,
        str(module_main),
        "--frames-dir",
        str(frames_dir),
        "--rs-path",
        str(rs_path),
        "--runtime",
        str(runtime),
        "--triton-model-spec",
        str(triton_model_spec),
        "--input-size",
        str(int(input_size)),
        "--batch-size",
        str(int(batch_size)),
        "--enable-advanced-features",
        "--use-clip-for-semantics",
        "--label-fusion",
        str(label_fusion),
    ]
    # Enable temporal smoothing by default for better stability on videos.
    cmd.extend(["--temporal-smoothing", "--smoothing-window", "7"])
    logger.info("Running scene_classification: %s", " ".join(cmd))
    subprocess.check_call(cmd, env=env)
    if not os.path.exists(out_npz):
        raise RuntimeError(f"scene_classification finished but artifact not found at: {out_npz}")
    return out_npz


def _pick_scene_frames(indices: List[int]) -> List[int]:
    if not indices:
        return []
    if len(indices) <= 3:
        return list(indices)
    return [indices[0], indices[len(indices) // 2], indices[-1]]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser("demo_scene_classification_quality")
    ap.add_argument("--video-path", default=None, help="Optional: run Segmenter to create frames_dir")
    ap.add_argument("--frames-dir", default=None, help="Optional: existing frames_dir (with metadata.json)")
    ap.add_argument("--rs-path", required=True, help="Results store path (must contain core_clip/embeddings.npz)")
    ap.add_argument("--out-dir", required=True, help="Output directory for HTML report")
    ap.add_argument("--visual-cfg-path", default=None, help="Optional Segmenter visual cfg to generate per-component frame_indices")

    ap.add_argument("--runtime", default="triton", choices=["triton", "inprocess"])
    ap.add_argument("--triton-http-url", default=None, help="e.g. http://localhost:18000")
    ap.add_argument("--triton-model-spec", default="places365_resnet50_224_triton")
    ap.add_argument("--input-size", type=int, default=224, choices=[224, 336, 448])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--label-fusion", default="clip", choices=["places", "clip"])
    ap.add_argument("--thumb-max-side", type=int, default=240)
    args = ap.parse_args(argv)

    rs_path = os.path.abspath(str(args.rs_path))
    if not os.path.isdir(rs_path):
        raise SystemExit(f"--rs-path does not exist: {rs_path}")
    core_clip_npz = os.path.join(rs_path, "core_clip", "embeddings.npz")
    if not os.path.isfile(core_clip_npz):
        raise SystemExit(
            "core_clip artifact is required (no-fallback) but not found:\n"
            f"  {core_clip_npz}\n"
            "Run core_clip first for the same rs_path and aligned frame_indices."
        )

    if args.frames_dir:
        frames_dir = os.path.abspath(str(args.frames_dir))
    else:
        if not args.video_path:
            raise SystemExit("Provide either --frames-dir or --video-path")
        frames_dir = run_segmenter(video_path=str(args.video_path), out_dir=str(args.out_dir), visual_cfg_path=args.visual_cfg_path)

    npz_path = run_scene_classification(
        frames_dir=frames_dir,
        rs_path=rs_path,
        runtime=str(args.runtime),
        triton_model_spec=str(args.triton_model_spec),
        input_size=int(args.input_size),
        batch_size=int(args.batch_size),
        triton_http_url=str(args.triton_http_url) if args.triton_http_url else None,
        label_fusion=str(args.label_fusion),
    )

    ok, issues, meta = validate_npz(npz_path)
    data = _npz_to_dict(npz_path)
    scenes_obj = data.get("scenes")
    if isinstance(scenes_obj, np.ndarray) and scenes_obj.dtype == object and scenes_obj.shape == ():
        scenes_obj = scenes_obj.item()
    scenes = scenes_obj if isinstance(scenes_obj, dict) else {}

    fm = FrameManager(frames_dir=frames_dir, chunk_size=32, cache_size=2)
    try:
        rows_html: List[str] = []
        for sid in sorted(scenes.keys()):
            s = scenes.get(sid) or {}
            label = str(s.get("scene_label") or "")
            idxs = [int(x) for x in (s.get("indices") or [])]
            pick = _pick_scene_frames(idxs)
            thumbs = []
            for fi in pick:
                img = fm.get(int(fi))
                if img is None:
                    continue
                # FrameManager.get() returns RGB
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                bgr = _resize_max_side(bgr, int(args.thumb_max_side))
                uri = _img_to_data_uri_bgr(bgr)
                if uri:
                    thumbs.append(f'<img class="thumb" src="{uri}" title="fi={fi}"/>')

            rows_html.append(
                "<tr>"
                f"<td>{sid}</td>"
                f"<td>{label}</td>"
                f"<td>{int(s.get('length_frames') or 0)}</td>"
                f"<td>{float(s.get('length_seconds') or 0.0):.2f}</td>"
                f"<td>{float(s.get('mean_score') or 0.0):.3f}</td>"
                f"<td>{float(s.get('class_entropy_mean') or 0.0):.3f}</td>"
                f"<td>{float(s.get('top1_prob_mean') or 0.0):.3f}</td>"
                f"<td>{float(s.get('mean_aesthetic_score') or 0.0):.3f}</td>"
                f"<td>{float(s.get('mean_luxury_score') or 0.0):.3f}</td>"
                f"<td>{''.join(thumbs)}</td>"
                "</tr>"
            )

    finally:
        fm.close()

    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    # Make filename stable + unique (video_id may start with '-').
    vid_for_name = str(meta.get("video_id") or Path(frames_dir).parent.name or "unknown_video")
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    out_html = os.path.join(out_dir, f"demo_scene_classification_quality_{vid_for_name}_{stamp}.html")

    issues_html = "".join([f"<li><b>{i.level}</b>: {i.message}</li>" for i in issues])
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>scene_classification quality demo</title>
  <style>
    body {{ font-family: Arial, sans-serif; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; vertical-align: top; }}
    th {{ background: #f4f4f4; }}
    .thumb {{ max-width: 240px; max-height: 240px; margin-right: 6px; border: 1px solid #ccc; }}
    .ok {{ color: #0a0; font-weight: bold; }}
    .bad {{ color: #a00; font-weight: bold; }}
    code {{ background: #f7f7f7; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h2>scene_classification quality demo</h2>
  <p><b>artifact</b>: <code>{npz_path}</code></p>
  <p><b>validate_npz</b>: <span class="{ 'ok' if ok else 'bad' }">{'OK' if ok else 'FAIL'}</span></p>
  <ul>{issues_html}</ul>

  <h3>meta (excerpt)</h3>
  <pre>{json.dumps({k: meta.get(k) for k in ['producer','schema_version','platform_id','video_id','run_id','config_hash','sampling_policy_version','dataprocessor_version','model_signature']}, ensure_ascii=False, indent=2)}</pre>

  <h3>Scenes</h3>
  <table>
    <thead>
      <tr>
        <th>scene_id</th>
        <th>label</th>
        <th>len_frames</th>
        <th>len_seconds</th>
        <th>places mean_score</th>
        <th>places entropy</th>
        <th>places top1_prob</th>
        <th>aesthetic</th>
        <th>luxury</th>
        <th>thumbnails (first/mid/last)</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Wrote HTML: %s", out_html)
    print(out_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


