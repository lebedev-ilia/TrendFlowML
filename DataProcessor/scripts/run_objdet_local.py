#!/usr/bin/env python3
"""Локальный раннер core_object_detections (Segmenter → core_object_detections с appearance-трекером).
Прогоняет несколько видео, собирает метрики трекера (num_tracks, длины, intra/inter cosine)."""
from __future__ import annotations
import argparse, os, subprocess, sys, time, json, glob
from pathlib import Path
import numpy as np

DP = Path(__file__).resolve().parents[1]; ROOT = DP.parent
def _pick(*c):
    for x in c:
        if Path(x).exists(): return str(x)
    return sys.executable or "python3"
PY = _pick("/workspace/venv/bin/python", DP/".data_venv/bin/python")
SEG = DP/"Segmenter/segmenter.py"
OD = DP/"VisualProcessor/core/model_process/core_object_detections/main.py"
CFG = DP/"configs/audit_v3/visual/visual_core_object_detections_only.yaml"
YOLO = Path(os.environ.get("DP_MODELS_ROOT") or (DP/"dp_models"))/"visual/object_detection/yolo11l/yolo11l.pt"


def sh(cmd, log, env, timeout=3600):
    log.write(f"\n$ {' '.join(map(str,cmd))}\n"); log.flush()
    return subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(DP), timeout=timeout).returncode


def tracker_metrics(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    tid = np.asarray(d["track_ids"]); vm = np.asarray(d["valid_mask"]); cls = np.asarray(d["class_ids"])
    mj = json.loads(str(d["meta_json"].reshape(-1)[0])) if "meta_json" in d.files else {}
    trk = mj.get("tracking", {})
    ids, counts = np.unique(tid[tid >= 0], return_counts=True)
    return {"num_tracks": int(len(ids)), "mean_track_len": round(float(counts.mean()), 1) if len(ids) else 0,
            "max_track_len": int(counts.max()) if len(ids) else 0,
            "person_dets": int((vm & (cls == 0)).sum()), "embedder": trk.get("embedder"),
            "frac_single": trk.get("frac_single_len")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--fps", type=float, default=6.0); ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--workdir", default="/tmp/od_out")
    ap.add_argument("--embedder", default="histogram")
    a = ap.parse_args()
    wd = Path(a.workdir); wd.mkdir(parents=True, exist_ok=True)
    log = open(wd/"run.log", "w", encoding="utf-8")
    env = os.environ.copy(); env["DP_MODELS_ROOT"] = str(Path(os.environ.get("DP_MODELS_ROOT") or (DP/"dp_models")))
    env["PYTHONPATH"] = f"{DP/'VisualProcessor'}:{DP}:" + env.get("PYTHONPATH", "")
    results = []
    for i, v in enumerate(a.videos):
        vid = f"od_{i}_{Path(v).stem[:12]}"; run_id = f"od_{int(time.time())}_{i}"
        rs = wd/vid/"rs"; rs.mkdir(parents=True, exist_ok=True)
        t = time.time()
        rc_s = sh([PY, SEG, "--video-path", v, "--output", str(wd/vid/"seg"), "--visual-cfg-path", CFG,
                   "--platform-id", "youtube", "--video-id", vid, "--run-id", run_id,
                   "--sampling-policy-version", "od_v1", "--config-hash", "local",
                   "--dataprocessor-version", "od_local", "--analysis-fps", str(a.fps), "--analysis-width", str(a.width)], log, env)
        seg_s = round(time.time()-t, 1)
        fd = wd/vid/"seg"/vid/"video"
        rec = {"video": Path(v).name, "segmenter_s": seg_s, "seg_rc": rc_s}
        if rc_s == 0 and (fd/"metadata.json").is_file():
            t = time.time()
            rc_o = sh([PY, OD, "--frames-dir", fd, "--rs-path", str(rs), "--model", str(YOLO),
                       "--runtime", "ultralytics", "--batch-size", "16", "--box-threshold", "0.5",
                       "--device", a.device, "--track-enabled", "true", "--track-embedder", a.embedder], log, env)
            rec["objdet_s"] = round(time.time()-t, 1); rec["objdet_rc"] = rc_o
            g = glob.glob(str(rs/"core_object_detections"/"*.npz"))
            if rc_o == 0 and g:
                rec.update(tracker_metrics(g[0]))
        results.append(rec); log.flush()
    out = {"results": results}
    (wd/"summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
