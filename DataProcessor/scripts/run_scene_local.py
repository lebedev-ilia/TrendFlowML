#!/usr/bin/env python3
"""
Локальный раннер цепочки scene_classification: Segmenter → VisualProcessor-оркестратор
(core_clip inprocess + cut_detection + scene_classification) → валидаторы.

Профиль (какие компоненты включены + их runtime) — visual cfg (по умолчанию inprocess-цепочка
scene_chain_inprocess.yaml, БЕЗ Triton). Оркестратор сам гонит core-providers и модули по зависимостям.

Пример (на поде, GPU):
  DP_MODELS_ROOT=.../dp_models python3 scripts/run_scene_local.py \
    --video <mp4> --seconds 0 --fps 8 --device cuda \
    --cfg configs/audit_v3/visual/scene_chain_inprocess.yaml --workdir /workspace/scene_out
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys, time, json
from pathlib import Path

DP = Path(__file__).resolve().parents[1]
ROOT = DP.parent

def _pick_py(*c):
    for x in c:
        if Path(x).exists():
            return str(x)
    return sys.executable or "python3"

DATA_PY = _pick_py(DP / ".data_venv/bin/python")
VPY = _pick_py(DP / "VisualProcessor/.vp_venv/bin/python", DP / ".data_venv/bin/python")
SEG = DP / "Segmenter/segmenter.py"
VMAIN = DP / "VisualProcessor/main.py"
SC_UTILS = DP / "VisualProcessor/modules/scene_classification/utils"
FFMPEG = ROOT / "tools/bin/ffmpeg"
if not FFMPEG.exists():
    FFMPEG = Path(shutil.which("ffmpeg") or "ffmpeg")
DP_MODELS = Path(os.environ.get("DP_MODELS_ROOT") or (DP / "dp_models"))


def sh(cmd, log, env, timeout=5400):
    log.write(f"\n$ {' '.join(map(str,cmd))}\n"); log.flush()
    return subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(DP), timeout=timeout).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--video-id", default="scene_local")
    ap.add_argument("--seconds", type=int, default=0)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cfg", default=str(DP / "configs/audit_v3/visual/scene_chain_inprocess.yaml"))
    ap.add_argument("--label-fusion", default="places", choices=["places", "clip"])
    ap.add_argument("--workdir", default="/tmp/scene_out")
    args = ap.parse_args()

    import yaml
    wd = Path(args.workdir); wd.mkdir(parents=True, exist_ok=True)
    log = open(wd / "run.log", "w", encoding="utf-8")
    env = os.environ.copy()
    env["DP_MODELS_ROOT"] = str(DP_MODELS)
    env["PYTHONPATH"] = f"{DP/'VisualProcessor'}:{DP}:" + env.get("PYTHONPATH", "")
    run_id = f"scene_{int(time.time())}"
    rs = wd / "rs"; rs.mkdir(exist_ok=True)
    summary = {"video_id": args.video_id, "run_id": run_id, "label_fusion": args.label_fusion, "stages": {}}

    # 1) trim
    if int(args.seconds) > 0:
        clip = wd / "clip.mp4"; t = time.time()
        rc = sh([FFMPEG, "-y", "-i", args.video, "-t", str(args.seconds), "-an", str(clip)], log, env)
        summary["stages"]["trim"] = {"rc": rc, "s": round(time.time()-t, 1)}
        if rc: print(json.dumps(summary)); return 2
    else:
        clip = Path(args.video); summary["stages"]["trim"] = {"skipped": True}

    # 2) Segmenter (scene visual cfg → frames + metadata для всех включённых компонентов)
    t = time.time()
    rc = sh([DATA_PY, SEG, "--video-path", clip, "--output", str(wd/"seg"),
             "--visual-cfg-path", args.cfg, "--platform-id", "youtube",
             "--video-id", args.video_id, "--run-id", run_id,
             "--sampling-policy-version", "scene_local_v1", "--config-hash", "local",
             "--dataprocessor-version", "scene_local", "--analysis-fps", str(args.fps),
             "--analysis-width", str(args.width)], log, env)
    summary["stages"]["segmenter"] = {"rc": rc, "s": round(time.time()-t, 1)}
    frames_dir = wd/"seg"/args.video_id/"video"
    if rc or not (frames_dir/"metadata.json").is_file():
        summary["error"] = f"segmenter failed ({frames_dir})"; print(json.dumps(summary, ensure_ascii=False)); return 3

    # 3) Прямые вызовы компонентов (обход оркестратора и Triton): core_clip inprocess →
    #    cut_detection (внутренний farneback, --no-require-core-optical-flow) → scene inprocess.
    CC = DP / "VisualProcessor/core/model_process/core_clip/main.py"
    CD = DP / "VisualProcessor/modules/cut_detection/main.py"
    SM = DP / "VisualProcessor/modules/scene_classification/main.py"
    t = time.time()
    rc_cc = sh([VPY, CC, "--frames-dir", frames_dir, "--rs-path", str(rs),
                "--runtime", "inprocess", "--model-name", "ViT-B/32", "--batch-size", "16"], log, env)
    summary["stages"]["core_clip"] = {"rc": rc_cc, "s": round(time.time()-t, 1)}
    t = time.time()
    rc_cd = sh([VPY, CD, "--frames-dir", frames_dir, "--rs-path", str(rs),
                "--no-require-core-optical-flow", "--no-use-clip"], log, env)
    summary["stages"]["cut_detection"] = {"rc": rc_cd, "s": round(time.time()-t, 1)}
    t = time.time()
    rc_sc = sh([VPY, SM, "--frames-dir", frames_dir, "--rs-path", str(rs),
                "--runtime", "inprocess", "--model-arch", "resnet50", "--device", args.device,
                "--label-fusion", args.label_fusion, "--enable-advanced-features"], log, env)
    summary["stages"]["scene_classification"] = {"rc": rc_sc, "s": round(time.time()-t, 1)}

    # 4) валидаторы + сводка
    sc_npz = rs/"scene_classification"/"scene_classification_features.npz"
    sh([DATA_PY, SC_UTILS/"validate_scene_classification_input.py", frames_dir, str(rs), "--label-fusion", args.label_fusion], log, env)
    sh([DATA_PY, SC_UTILS/"validate_scene_classification_npz.py", str(sc_npz)], log, env)
    try:
        import numpy as np
        a = np.load(sc_npz, allow_pickle=True)
        summary["result"] = {
            "keys": [k for k in a.files][:40],
            "N": int(np.asarray(a["frame_indices"]).size) if "frame_indices" in a.files else 0,
            "scene_embedding": list(np.asarray(a["frame_scene_embedding"]).shape) if "frame_scene_embedding" in a.files else None,
            "n_scenes": int(np.asarray(a["scene_ids"]).size) if "scene_ids" in a.files else None,
        }
    except Exception as e:
        summary["result_error"] = str(e)
    (wd/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
