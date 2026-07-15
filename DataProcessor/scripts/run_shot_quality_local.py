#!/usr/bin/env python3
"""
Локальный раннер shot_quality (самая тяжёлая цепочка, БЕЗ Triton через обходы):
Segmenter(shot_quality профиль) → core_clip(inprocess) + core_object_detections(ultralytics) +
core_face_landmarks(mediapipe) + cut_detection(farneback,no-clip) + core_depth_midas(inprocess MiDaS bypass)
→ shot_quality(numpy) → валидаторы.
Все deps должны иметь ОДИНАКОВЫЕ frame_indices (Segmenter aligned sampling group).
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys, time, json
from pathlib import Path

DP = Path(__file__).resolve().parents[1]; ROOT = DP.parent
def _pick(*c):
    for x in c:
        if Path(x).exists(): return str(x)
    return sys.executable or "python3"
PY = _pick("/workspace/venv/bin/python", DP / "VisualProcessor/.vp_venv/bin/python", DP / ".data_venv/bin/python")
SEG = DP / "Segmenter/segmenter.py"
VP = DP / "VisualProcessor"
CC = VP / "core/model_process/core_clip/main.py"
OD = VP / "core/model_process/core_object_detections/main.py"
FL = VP / "core/model_process/core_face_landmarks/main.py"
CD = VP / "modules/cut_detection/main.py"
SQ = VP / "modules/shot_quality/main.py"
SQU = VP / "modules/shot_quality/utils"
MIDAS = DP / "scripts/midas_depth_inprocess.py"
CFG = DP / "configs/audit_v3/visual/visual_shot_quality_only.yaml"
FFMPEG = ROOT / "tools/bin/ffmpeg"
if not FFMPEG.exists(): FFMPEG = Path(shutil.which("ffmpeg") or "ffmpeg")
YOLO = Path(os.environ.get("DP_MODELS_ROOT") or (DP/"dp_models")) / "visual/object_detection/yolo11l/yolo11l.pt"


def sh(cmd, log, env, timeout=5400):
    log.write(f"\n$ {' '.join(map(str,cmd))}\n"); log.flush()
    return subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(DP), timeout=timeout).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--video-id", default="sq_local")
    ap.add_argument("--fps", type=float, default=6.0); ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--workdir", default="/tmp/sq_out")
    a = ap.parse_args()
    wd = Path(a.workdir); wd.mkdir(parents=True, exist_ok=True)
    log = open(wd/"run.log", "w", encoding="utf-8")
    env = os.environ.copy(); env["DP_MODELS_ROOT"] = str(Path(os.environ.get("DP_MODELS_ROOT") or (DP/"dp_models")))
    env["PYTHONPATH"] = f"{VP}:{DP}:" + env.get("PYTHONPATH", "")
    run_id = f"sq_{int(time.time())}"; rs = wd/"rs"; rs.mkdir(exist_ok=True)
    S = {"video_id": a.video_id, "run_id": run_id, "stages": {}}

    def stage(name, cmd):
        t = time.time(); rc = sh(cmd, log, env); S["stages"][name] = {"rc": rc, "s": round(time.time()-t, 1)}; return rc

    # 1) Segmenter (shot_quality профиль → aligned frames для всех deps)
    if stage("segmenter", [PY, SEG, "--video-path", a.video, "--output", str(wd/"seg"),
             "--visual-cfg-path", CFG, "--platform-id", "youtube", "--video-id", a.video_id,
             "--run-id", run_id, "--sampling-policy-version", "sq_local_v1", "--config-hash", "local",
             "--dataprocessor-version", "sq_local", "--analysis-fps", str(a.fps), "--analysis-width", str(a.width)]):
        S["error"] = "segmenter"; print(json.dumps(S, ensure_ascii=False)); return 3
    fd = wd/"seg"/a.video_id/"video"
    if not (fd/"metadata.json").is_file():
        S["error"] = "no metadata"; print(json.dumps(S, ensure_ascii=False)); return 3

    # 2) deps (прямые вызовы, inprocess/farneback/bypass)
    stage("core_clip", [PY, CC, "--frames-dir", fd, "--rs-path", str(rs), "--runtime", "inprocess", "--model-name", "ViT-B/32", "--batch-size", "16"])
    stage("object_det", [PY, OD, "--frames-dir", fd, "--rs-path", str(rs), "--model", str(YOLO), "--runtime", "ultralytics", "--batch-size", "16", "--device", a.device])
    stage("face_landmarks", [PY, FL, "--frames-dir", fd, "--rs-path", str(rs), "--use-face-mesh", "--use-person-mask"])
    stage("cut_detection", [PY, CD, "--frames-dir", fd, "--rs-path", str(rs), "--no-require-core-optical-flow", "--no-use-clip"])
    stage("depth_midas_bypass", [PY, MIDAS, "--frames-dir", fd, "--rs-path", str(rs), "--device", a.device])
    # 3) shot_quality
    stage("shot_quality", [PY, SQ, "--frames-dir", fd, "--rs-path", str(rs), "--device", a.device])

    # 4) валидаторы + сводка
    sq_npz = rs/"shot_quality"/"shot_quality.npz"
    sh([PY, SQU/"validate_shot_quality_input.py", fd, str(rs)], log, env)
    v = SQU/"validate_shot_quality_npz.py"
    if v.exists(): sh([PY, v, str(sq_npz)], log, env)
    try:
        import numpy as np
        aa = np.load(sq_npz, allow_pickle=True)
        S["result"] = {"keys": [k for k in aa.files][:30],
                       "frame_features": list(np.asarray(aa["frame_features"]).shape) if "frame_features" in aa.files else None,
                       "feature_names": int(np.asarray(aa["feature_names"]).size) if "feature_names" in aa.files else None}
    except Exception as e:
        S["result_error"] = str(e)
    (wd/"summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=2))
    print(json.dumps(S, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
