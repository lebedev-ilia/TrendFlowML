#!/usr/bin/env python3
"""
Локальный CPU-раннер компонента action_recognition (без Cursor, без GPU, без сервисного стека).

Прогоняет цепочку на КОРОТКОМ клипе: ffmpeg-трим → Segmenter → core_object_detections(+tracker)
→ action_recognition(v3) → input+output валидаторы. Предназначен для smoke-валидации логики/схем
самим Claude в сэндбоксе. Реальные длинные/200k прогоны — на GPU (Cursor/GPU-бокс).

Пример:
  DataProcessor/VisualProcessor/.vp_venv/bin/python DataProcessor/scripts/run_ar_local.py \
    --video <path.mp4> --seconds 10 --fps 8 --workdir /tmp/ar_local
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time, json
from pathlib import Path

DP = Path(__file__).resolve().parents[1]              # DataProcessor
ROOT = DP.parent                                       # repo root

def _pick_py(*cands):
    """venv-питон если есть (сэндбокс), иначе системный (контейнер k8s-Job)."""
    for c in cands:
        if Path(c).exists():
            return Path(c)
    return Path(sys.executable or "python3")

DATA_PY = _pick_py(DP / ".data_venv/bin/python")
VPY = _pick_py(DP / "VisualProcessor/.vp_venv/bin/python", DP / ".data_venv/bin/python")
SEG = DP / "Segmenter/segmenter.py"
COD = DP / "VisualProcessor/core/model_process/core_object_detections/main.py"
AR = DP / "VisualProcessor/modules/action_recognition/main.py"
AR_UTILS = DP / "VisualProcessor/modules/action_recognition/utils"
import shutil
FFMPEG = ROOT / "tools/bin/ffmpeg"
if not FFMPEG.exists():
    FFMPEG = Path(shutil.which("ffmpeg") or "ffmpeg")  # системный ffmpeg в контейнере
VCFG = DP / "configs/audit_v3/visual/visual_minimal_object_detections_action_recognition.yaml"
_DP_MODELS = Path(os.environ.get("DP_MODELS_ROOT") or (DP / "dp_models"))
YOLO = _DP_MODELS / "visual/object_detection/yolo11l/yolo11l.pt"


def sh(cmd, log, env=None, timeout=3600):
    log.write(f"\n$ {' '.join(map(str,cmd))}\n"); log.flush()
    p = subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT,
                       env=env, cwd=str(DP), timeout=timeout)
    return p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--seconds", type=int, default=10, help="Обрезать до N секунд; 0 = полный клип (GPU)")
    ap.add_argument("--fps", type=float, default=8.0, help="Анализ-fps (на GPU можно 25)")
    ap.add_argument("--workdir", default="/tmp/ar_local")
    ap.add_argument("--video-id", default="ar_local")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Устройство инференса компонентов")
    ap.add_argument("--batch-size", type=int, default=None, help="Batch (по умолчанию 4/2 CPU, 16/8 GPU)")
    args = ap.parse_args()
    det_bs = str(args.batch_size or (16 if args.device == "cuda" else 4))
    ar_bs = str(args.batch_size or (8 if args.device == "cuda" else 2))

    wd = Path(args.workdir); wd.mkdir(parents=True, exist_ok=True)
    log = open(wd / "run.log", "w", encoding="utf-8")
    env = os.environ.copy()
    env["DP_MODELS_ROOT"] = str(DP / "dp_models")
    env["PYTHONPATH"] = f"{DP/'VisualProcessor'}:{DP}:" + env.get("PYTHONPATH", "")
    run_id = f"local_{int(time.time())}"
    rs = wd / "rs"; rs.mkdir(exist_ok=True)
    summary = {"video_id": args.video_id, "run_id": run_id, "stages": {}}

    # 1) trim (или полный клип при --seconds 0)
    if int(args.seconds) > 0:
        clip = wd / "clip.mp4"
        t = time.time()
        rc = sh([FFMPEG, "-y", "-i", args.video, "-t", str(args.seconds), "-an", str(clip)], log, env)
        summary["stages"]["trim"] = {"rc": rc, "s": round(time.time()-t, 1)}
        if rc != 0:
            print(json.dumps(summary)); return 2
    else:
        clip = Path(args.video)
        summary["stages"]["trim"] = {"rc": 0, "skipped": True}

    # 2) Segmenter
    t = time.time()
    rc = sh([DATA_PY, SEG, "--video-path", clip, "--output", str(wd/"seg"),
             "--visual-cfg-path", VCFG, "--platform-id", "youtube",
             "--video-id", args.video_id, "--run-id", run_id,
             "--sampling-policy-version", "ar_local_v1",
             "--config-hash", "local", "--dataprocessor-version", "ar_local",
             "--analysis-fps", str(args.fps), "--analysis-width", str(args.width)], log, env)
    summary["stages"]["segmenter"] = {"rc": rc, "s": round(time.time()-t, 1)}
    frames_dir = wd/"seg"/args.video_id/"video"
    if rc != 0 or not (frames_dir/"metadata.json").is_file():
        summary["error"] = f"segmenter failed / no metadata ({frames_dir})"
        print(json.dumps(summary, ensure_ascii=False)); return 3

    # 3) core_object_detections (+ appearance tracker), CPU
    t = time.time()
    rc = sh([VPY, COD, "--frames-dir", frames_dir, "--rs-path", str(rs),
             "--model", str(YOLO), "--runtime", "ultralytics", "--batch-size", det_bs,
             "--box-threshold", "0.5", "--device", args.device,
             "--track-enabled", "true", "--track-embedder", "histogram"], log, env)
    summary["stages"]["detections"] = {"rc": rc, "s": round(time.time()-t, 1)}

    # 4) action_recognition v3, CPU (penultimate + tubelet + track_anchored)
    t = time.time()
    rc = sh([VPY, AR, "--frames-dir", frames_dir, "--rs-path", str(rs),
             "--clip-len", "32", "--batch-size", ar_bs, "--device", args.device,
             "--embedding-mode", "penultimate", "--localization", "track_anchored",
             "--tubelet-crop", "true", "--min-clip-real-frames", "16"], log, env)
    summary["stages"]["action_recognition"] = {"rc": rc, "s": round(time.time()-t, 1)}

    # 5) validators + summary from npz
    det_npz = rs/"core_object_detections"/"detections.npz"
    ar_npz = rs/"action_recognition"/"action_recognition_features.npz"
    sh([DATA_PY, AR_UTILS/"validate_action_recognition_input.py", frames_dir, det_npz], log, env)
    sh([DATA_PY, AR_UTILS/"validate_action_recognition_npz.py", ar_npz], log, env)
    try:
        import numpy as np
        a = np.load(ar_npz, allow_pickle=True)
        meta = json.loads(str(a["meta_json"].reshape(-1)[0])) if "meta_json" in a.files else {}
        summary["result"] = {
            "clip_count": int(np.asarray(a["clip_count"]).reshape(-1)[0]),
            "num_tracks": int(np.asarray(a["num_tracks"]).reshape(-1)[0]),
            "mean_clips_per_track": float(np.asarray(a["mean_clips_per_track"]).reshape(-1)[0]),
            "num_action_segments": int(np.asarray(a.get("num_action_segments", 0)).reshape(-1)[0]) if "num_action_segments" in a.files else None,
            "embedding_shape": list(np.asarray(a["clip_embeddings"]).shape),
            "embedding_mode": meta.get("embedding_mode"), "embedding_dim": meta.get("embedding_dim"),
            "classes_available": bool(np.asarray(a["classes_available"]).reshape(-1)[0]) if "classes_available" in a.files else None,
            "status": meta.get("status"),
        }
    except Exception as e:
        summary["result_error"] = str(e)
    _p = wd/"summary.json"; _p.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
