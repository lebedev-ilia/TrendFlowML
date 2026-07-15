#!/usr/bin/env python3
"""
Локальный раннер emotion_face:
Segmenter(detalize_face профиль — aligned OD+face_landmarks) → core_object_detections(ultralytics) →
core_face_landmarks(mediapipe) → emotion_face(EmoNet) → validate_emotion_face.

emotion_face требует landmarks.npz от core_face_landmarks (no-fallback), берёт face_present+landmarks
для выбора кадров с лицом и построения crop-ов под EmoNet.

Профиль detalize_face_only выделяет aligned frame_indices для OD+landmarks, но не проставляет секцию
`emotion_face` в metadata.json → копируем её из core_face_landmarks (aligned), чтобы axis_source="emotion_face"
(штатный путь). Иначе emotion_face сам сделает fallback на ось core_face_landmarks (тоже валидно).

Golden/детерминизм: запускать с env EMOTION_FACE_USE_AMP=0 (fp32, детерминированный EmoNet).
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time, json
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
EMO = VP / "modules/emotion_face/main.py"
EVAL = VP / "modules/emotion_face/utils/validate_emotion_face.py"
CFG = DP / "configs/audit_v3/visual/visual_detalize_face_only.yaml"
YOLO = Path(os.environ.get("DP_MODELS_ROOT") or (DP/"dp_models")) / "visual/object_detection/yolo11l/yolo11l.pt"


def sh(cmd, log, env, timeout=5400):
    log.write(f"\n$ {' '.join(map(str,cmd))}\n"); log.flush()
    return subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(DP), timeout=timeout).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--video-id", default="emo_local")
    ap.add_argument("--fps", type=float, default=6.0); ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--workdir", default="/tmp/emo_out")
    ap.add_argument("--stride", type=int, default=None); ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-faces", type=int, default=None); ap.add_argument("--emo-path", default=None)
    a = ap.parse_args()
    wd = Path(a.workdir); wd.mkdir(parents=True, exist_ok=True)
    log = open(wd/"run.log", "w", encoding="utf-8")
    env = os.environ.copy(); env["DP_MODELS_ROOT"] = str(Path(os.environ.get("DP_MODELS_ROOT") or (DP/"dp_models")))
    env["PYTHONPATH"] = f"{VP}:{DP}:" + env.get("PYTHONPATH", "")
    run_id = f"emo_{int(time.time())}"; rs = wd/"rs"; rs.mkdir(exist_ok=True)
    S = {"video_id": a.video_id, "run_id": run_id, "use_amp_env": env.get("EMOTION_FACE_USE_AMP"), "stages": {}}

    def stage(name, cmd):
        t = time.time(); rc = sh(cmd, log, env); S["stages"][name] = {"rc": rc, "s": round(time.time()-t, 1)}; return rc

    if stage("segmenter", [PY, SEG, "--video-path", a.video, "--output", str(wd/"seg"),
             "--visual-cfg-path", CFG, "--platform-id", "youtube", "--video-id", a.video_id,
             "--run-id", run_id, "--sampling-policy-version", "emo_local_v1", "--config-hash", "local",
             "--dataprocessor-version", "emo_local", "--analysis-fps", str(a.fps), "--analysis-width", str(a.width)]):
        S["error"] = "segmenter"; print(json.dumps(S, ensure_ascii=False)); return 3
    fd = wd/"seg"/a.video_id/"video"
    if not (fd/"metadata.json").is_file():
        S["error"] = "no metadata"; print(json.dumps(S, ensure_ascii=False)); return 3

    stage("object_det", [PY, OD, "--frames-dir", fd, "--rs-path", str(rs), "--model", str(YOLO), "--runtime", "ultralytics", "--batch-size", "16", "--device", a.device])
    stage("face_landmarks", [PY, FL, "--frames-dir", fd, "--rs-path", str(rs), "--use-face-mesh", "--use-person-mask"])

    # Профиль detalize_face не выставляет секцию `emotion_face` в metadata.json → копируем aligned-секцию
    # core_face_landmarks (тот же sampling), чтобы axis_source="emotion_face" (штатная ось).
    try:
        mpath = fd/"metadata.json"; m = json.loads(mpath.read_text())
        if "emotion_face" not in m and "core_face_landmarks" in m:
            m["emotion_face"] = dict(m["core_face_landmarks"]); m["emotion_face"]["modality"] = "emotion_face"
            mpath.write_text(json.dumps(m, ensure_ascii=False)); S["patched_emotion_face_section"] = True
    except Exception as e:
        S["metadata_patch_error"] = str(e)

    emo_cmd = [PY, EMO, "--frames-dir", fd, "--rs-path", str(rs), "--device", a.device]
    if a.stride is not None: emo_cmd += ["--face-frame-stride", str(a.stride)]
    if a.max_frames is not None: emo_cmd += ["--max-frames", str(a.max_frames)]
    if a.max_faces is not None: emo_cmd += ["--max-faces-per-frame", str(a.max_faces)]
    if a.emo_path: emo_cmd += ["--emo-path", a.emo_path]
    stage("emotion_face", emo_cmd)

    emo_npz = rs/"emotion_face"/"emotion_face.npz"
    if EVAL.exists() and emo_npz.exists():
        S["validate_struct_rc"] = sh([PY, EVAL, str(emo_npz), "--struct"], log, env)
        S["validate_qa_rc"] = sh([PY, EVAL, str(emo_npz), "--qa"], log, env)
    try:
        import numpy as np
        z = np.load(emo_npz, allow_pickle=True)
        meta = z["meta"].reshape(-1)[0] if "meta" in z.files else {}
        fi = z["frame_indices"]; va = z["valence"]; ar = z["arousal"]
        pm = z["processed_mask"]; fp = z["face_present"]; ep = z["emotion_probs"]
        kf = z["keyframes"]
        proc = np.asarray(pm, dtype=bool)
        vproc = np.asarray(va, dtype=np.float32)[proc]
        aproc = np.asarray(ar, dtype=np.float32)[proc]
        epproc = np.asarray(ep, dtype=np.float32)[proc]
        finite_v = np.isfinite(vproc)
        S["result"] = {
            "status": meta.get("status"), "empty_reason": meta.get("empty_reason"),
            "schema_version": meta.get("schema_version"), "producer_version": meta.get("producer_version"),
            "N": int(fi.size), "face_present_true": int(np.asarray(fp).sum()),
            "processed": int(proc.sum()), "keyframes_n": int(getattr(kf, "size", 0)),
            "valence_range": [float(np.nanmin(vproc)), float(np.nanmax(vproc))] if finite_v.any() else None,
            "arousal_range": [float(np.nanmin(aproc)), float(np.nanmax(aproc))] if np.isfinite(aproc).any() else None,
            "valence_mean": float(np.nanmean(vproc)) if finite_v.any() else None,
            "probs_rowsum_range": [float(np.nanmin(epproc.sum(1))), float(np.nanmax(epproc.sum(1)))] if epproc.size and np.isfinite(epproc).any() else None,
            "proc_subset_of_facepresent": bool(np.all(np.asarray(fp, dtype=bool)[proc])) if proc.any() else True,
            "timing_ms": meta.get("stage_timings_ms"),
        }
    except Exception as e:
        S["result_error"] = str(e)
    (wd/"summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=2))
    print(json.dumps(S, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
