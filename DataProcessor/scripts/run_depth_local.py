#!/usr/bin/env python3
"""
Локальный раннер core_depth_midas (Triton-free, inprocess MiDaS, полная v3-схема):
Segmenter(depth-профиль) → core_depth_midas(--runtime inprocess) → валидаторы (--struct --ranges) →
golden-детерминизм (2 прогона, побайтовое сравнение ключевых массивов).

Компонент выдаёт ПОЛНЫЙ v3-артефакт (depth_maps + depth_maps_norm + прокси + preview + meta с
models_used/model_signature), идентичный Triton-пути. Всё inprocess, без Triton.
Использование:
  python run_depth_local.py --video <mp4> [--model MiDaS_small] [--out-hw 256 256] [--device cuda]
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

DP = Path(__file__).resolve().parents[1]; ROOT = DP.parent


def _pick(*c):
    for x in c:
        if Path(x).exists():
            return str(x)
    return sys.executable or "python3"


PY = _pick("/workspace/venv/bin/python", DP / "VisualProcessor/.vp_venv/bin/python", DP / ".data_venv/bin/python")
SEG = DP / "Segmenter/segmenter.py"
VP = DP / "VisualProcessor"
CD = VP / "core/model_process/core_depth_midas/main.py"
CDU = VP / "core/model_process/core_depth_midas/utils/validate_core_depth_midas_npz.py"
CFG = DP / "configs/audit_v3/visual/visual_core_depth_midas_only.yaml"


def sh(cmd, log, env, timeout=5400):
    log.write(f"\n$ {' '.join(map(str, cmd))}\n"); log.flush()
    return subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(DP), timeout=timeout).returncode


def _depth_signature(npz_path: Path) -> dict:
    """Хэши ключевых массивов для golden-детерминизма."""
    import numpy as np
    z = np.load(npz_path, allow_pickle=True)
    out = {}
    for k in ("depth_maps", "depth_maps_norm", "depth_mean", "depth_std",
              "depth_p05", "depth_p95", "depth_complexity_score",
              "foreground_background_separation_proxy", "frame_indices", "times_s"):
        if k in z.files:
            a = np.asarray(z[k])
            out[k] = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
    z.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--video-id", default="depth_local")
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="MiDaS_small")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-h", type=int, default=256)
    ap.add_argument("--out-w", type=int, default=256)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--golden", action="store_true", help="Второй прогон + сравнение сигнатур массивов")
    ap.add_argument("--workdir", default="/tmp/depth_out")
    a = ap.parse_args()

    wd = Path(a.workdir); wd.mkdir(parents=True, exist_ok=True)
    log = open(wd / "run.log", "w", encoding="utf-8")
    env = os.environ.copy()
    env["DP_MODELS_ROOT"] = str(Path(os.environ.get("DP_MODELS_ROOT") or (DP / "dp_models")))
    env["PYTHONPATH"] = f"{VP}:{DP}:" + env.get("PYTHONPATH", "")
    run_id = f"depth_{int(time.time())}"
    rs = wd / "rs"; rs.mkdir(exist_ok=True)
    S = {"video_id": a.video_id, "run_id": run_id, "model": a.model, "device": a.device, "stages": {}}

    def stage(name, cmd):
        t = time.time(); rc = sh(cmd, log, env)
        S["stages"][name] = {"rc": rc, "s": round(time.time() - t, 1)}
        return rc

    # 1) Segmenter (depth-профиль)
    if stage("segmenter", [PY, SEG, "--video-path", a.video, "--output", str(wd / "seg"),
             "--visual-cfg-path", CFG, "--platform-id", "youtube", "--video-id", a.video_id,
             "--run-id", run_id, "--sampling-policy-version", "depth_local_v1", "--config-hash", "local",
             "--dataprocessor-version", "depth_local", "--analysis-fps", str(a.fps), "--analysis-width", str(a.width)]):
        S["error"] = "segmenter"; print(json.dumps(S, ensure_ascii=False)); return 3
    fd = wd / "seg" / a.video_id / "video"
    if not (fd / "metadata.json").is_file():
        S["error"] = "no metadata"; print(json.dumps(S, ensure_ascii=False)); return 3

    depth_cmd = [PY, CD, "--frames-dir", str(fd), "--rs-path", str(rs),
                 "--runtime", "inprocess", "--inprocess-model", a.model, "--device", a.device,
                 "--batch-size", str(a.batch_size), "--triton-preprocess-preset", "midas_256",
                 "--out-width", str(a.out_w), "--out-height", str(a.out_h)]
    if a.fp16:
        depth_cmd.append("--inprocess-fp16")

    # 2) core_depth_midas (inprocess, полная v3-схема)
    stage("core_depth_midas", depth_cmd)
    npz = rs / "core_depth_midas" / "depth.npz"

    # 3) валидаторы
    if npz.is_file():
        stage("validate", [PY, CDU, str(npz), "--struct", "--ranges"])
        try:
            import numpy as np
            z = np.load(npz, allow_pickle=True)
            meta = z["meta"].item() if "meta" in z.files else {}
            dm = np.asarray(z["depth_maps"]) if "depth_maps" in z.files else None
            S["result"] = {
                "keys": list(z.files),
                "depth_maps": list(dm.shape) if dm is not None else None,
                "finite_frac": round(float(np.isfinite(dm).mean()), 4) if dm is not None else None,
                "runtime": meta.get("runtime"), "model": meta.get("model_name"),
                "model_signature": str(meta.get("model_signature", ""))[:16],
                "stage_timings_ms": meta.get("stage_timings_ms"),
            }
            z.close()
            S["result"]["sig"] = _depth_signature(npz)
        except Exception as e:
            S["result_error"] = str(e)

    # 4) golden-детерминизм (второй прогон в отдельный rs)
    if a.golden and npz.is_file():
        rs2 = wd / "rs2"; rs2.mkdir(exist_ok=True)
        depth_cmd2 = list(depth_cmd)
        depth_cmd2[depth_cmd2.index(str(rs))] = str(rs2)
        stage("core_depth_midas_golden", depth_cmd2)
        npz2 = rs2 / "core_depth_midas" / "depth.npz"
        if npz2.is_file():
            sig1 = _depth_signature(npz)
            sig2 = _depth_signature(npz2)
            same = sig1 == sig2
            S["golden"] = {"identical": same,
                           "diff_keys": [k for k in sig1 if sig1.get(k) != sig2.get(k)]}

    (wd / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=2))
    print(json.dumps(S, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
