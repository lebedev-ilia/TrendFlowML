#!/usr/bin/env python3
"""
Локальный раннер behavioral:
Segmenter(shot_quality профиль — даёт aligned OD+face_landmarks) → core_object_detections(ultralytics) →
core_face_landmarks(mediapipe) → behavioral(numpy) → validate_behavioral.
behavioral требует только landmarks.npz от core_face_landmarks (который требует detections.npz от OD).
Профиль shot_quality используется т.к. он выделяет aligned frame_indices для OD+landmarks (visual_behavioral_only
не включает OD в sampling group). Лишние deps (clip/cut/depth) НЕ запускаем — behavioral их не читает.
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
OD = VP / "core/model_process/core_object_detections/main.py"
FL = VP / "core/model_process/core_face_landmarks/main.py"
BEH = VP / "modules/behavioral/main.py"
BVAL = VP / "modules/behavioral/utils/validate_behavioral.py"
CFG = DP / "configs/audit_v3/visual/visual_shot_quality_only.yaml"
YOLO = Path(os.environ.get("DP_MODELS_ROOT") or (DP/"dp_models")) / "visual/object_detection/yolo11l/yolo11l.pt"


def sh(cmd, log, env, timeout=5400):
    log.write(f"\n$ {' '.join(map(str,cmd))}\n"); log.flush()
    return subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(DP), timeout=timeout).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--video-id", default="beh_local")
    ap.add_argument("--fps", type=float, default=6.0); ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--workdir", default="/tmp/beh_out")
    a = ap.parse_args()
    wd = Path(a.workdir); wd.mkdir(parents=True, exist_ok=True)
    log = open(wd/"run.log", "w", encoding="utf-8")
    env = os.environ.copy(); env["DP_MODELS_ROOT"] = str(Path(os.environ.get("DP_MODELS_ROOT") or (DP/"dp_models")))
    env["PYTHONPATH"] = f"{VP}:{DP}:" + env.get("PYTHONPATH", "")
    run_id = f"beh_{int(time.time())}"; rs = wd/"rs"; rs.mkdir(exist_ok=True)
    S = {"video_id": a.video_id, "run_id": run_id, "stages": {}}

    def stage(name, cmd):
        t = time.time(); rc = sh(cmd, log, env); S["stages"][name] = {"rc": rc, "s": round(time.time()-t, 1)}; return rc

    if stage("segmenter", [PY, SEG, "--video-path", a.video, "--output", str(wd/"seg"),
             "--visual-cfg-path", CFG, "--platform-id", "youtube", "--video-id", a.video_id,
             "--run-id", run_id, "--sampling-policy-version", "beh_local_v1", "--config-hash", "local",
             "--dataprocessor-version", "beh_local", "--analysis-fps", str(a.fps), "--analysis-width", str(a.width)]):
        S["error"] = "segmenter"; print(json.dumps(S, ensure_ascii=False)); return 3
    fd = wd/"seg"/a.video_id/"video"
    if not (fd/"metadata.json").is_file():
        S["error"] = "no metadata"; print(json.dumps(S, ensure_ascii=False)); return 3

    stage("object_det", [PY, OD, "--frames-dir", fd, "--rs-path", str(rs), "--model", str(YOLO), "--runtime", "ultralytics", "--batch-size", "16", "--device", a.device])
    stage("face_landmarks", [PY, FL, "--frames-dir", fd, "--rs-path", str(rs), "--use-pose", "--use-hands", "--use-face-mesh", "--use-person-mask"])
    # Профиль shot_quality не выставляет секцию `behavioral` в metadata.json (behavioral не в его sampling
    # group). behavioral работает на тех же кадрах, что core_face_landmarks → копируем секцию (aligned).
    try:
        mpath = fd/"metadata.json"; m = json.loads(mpath.read_text())
        if "behavioral" not in m and "core_face_landmarks" in m:
            m["behavioral"] = m["core_face_landmarks"]; m["behavioral"]["modality"] = "behavioral"
            mpath.write_text(json.dumps(m, ensure_ascii=False)); S["patched_behavioral_section"] = True
    except Exception as e:
        S["metadata_patch_error"] = str(e)
    stage("behavioral", [PY, BEH, "--frames-dir", fd, "--rs-path", str(rs)])

    beh_npz = rs/"behavioral"/"behavioral_features.npz"
    if BVAL.exists():
        S["validate_struct_rc"] = sh([PY, BVAL, str(beh_npz), "--struct"], log, env)
    try:
        import numpy as np
        aa = np.load(beh_npz, allow_pickle=True)
        lp = np.asarray(aa["landmarks_present"]) if "landmarks_present" in aa.files else None
        meta = aa["meta"].reshape(-1)[0] if "meta" in aa.files else {}
        S["result"] = {"keys_n": len(aa.files), "N": int(lp.size) if lp is not None else None,
                       "landmarks_present_true": int(lp.sum()) if lp is not None else None,
                       "status": meta.get("status"), "empty_reason": meta.get("empty_reason"),
                       "producer_version": meta.get("producer_version"), "schema_version": meta.get("schema_version")}
    except Exception as e:
        S["result_error"] = str(e)
    (wd/"summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=2))
    print(json.dumps(S, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
