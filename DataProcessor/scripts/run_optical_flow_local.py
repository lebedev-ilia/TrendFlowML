#!/usr/bin/env python3
"""
Локальный раннер core_optical_flow (Triton-free, inprocess torchvision RAFT, полная v3-схема):
Segmenter(optical_flow-профиль) → core_optical_flow(--runtime inprocess) → валидаторы (--struct --ranges) →
golden-детерминизм (2 прогона, побайтовое сравнение ключевых массивов).

Компонент выдаёт ПОЛНЫЙ v3-артефакт (motion + audit-статистики + cam_* + preview + meta с
models_used/model_signature), идентичный Triton-пути. Всё inprocess, без Triton.

Использование:
  python run_optical_flow_local.py --video <mp4> [--model raft_small|raft_large]
      [--preset raft_256|raft_384|raft_512] [--device cuda] [--fps 4] [--width 480] [--golden]
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
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
CF = VP / "core/model_process/core_optical_flow/main.py"
CFU = VP / "core/model_process/core_optical_flow/utils/validate_core_optical_flow_npz.py"
CFG = DP / "configs/audit_v3/visual/visual_core_optical_flow_only.yaml"


def sh(cmd, log, env, timeout=5400):
    log.write(f"\n$ {' '.join(map(str, cmd))}\n"); log.flush()
    return subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(DP), timeout=timeout).returncode


_SIG_KEYS = (
    "frame_indices", "times_s", "motion_norm_per_sec_mean", "dt_seconds",
    "flow_mag_std_per_sec_norm", "flow_mag_p95_per_sec_norm",
    "flow_dx_mean_per_sec_norm", "flow_dy_mean_per_sec_norm",
    "flow_dir_sin_mean", "flow_dir_cos_mean", "flow_dir_dispersion",
    "flow_div_abs_mean", "flow_consistency",
    "cam_affine_scale", "cam_affine_rotation", "cam_tx_per_sec_norm",
    "cam_ty_per_sec_norm", "cam_shake_std_norm", "bg_ratio",
    "preview_flow_mag_map_norm",
)


def _flow_signature(npz_path: Path) -> dict:
    """Хэши ключевых массивов для golden-детерминизма (NaN → 0 для стабильного tobytes)."""
    import numpy as np
    z = np.load(npz_path, allow_pickle=True)
    out = {}
    for k in _SIG_KEYS:
        if k in z.files:
            a = np.asarray(z[k])
            a = np.nan_to_num(a.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            out[k] = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
    z.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--video-id", default="flow_local")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="raft_small", choices=["raft_small", "raft_large"])
    ap.add_argument("--preset", default="raft_256", choices=["raft_256", "raft_384", "raft_512"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--golden", action="store_true", help="Второй прогон + сравнение сигнатур массивов")
    ap.add_argument("--workdir", default="/tmp/flow_out")
    a = ap.parse_args()

    wd = Path(a.workdir); wd.mkdir(parents=True, exist_ok=True)
    log = open(wd / "run.log", "w", encoding="utf-8")
    env = os.environ.copy()
    env["DP_MODELS_ROOT"] = str(Path(os.environ.get("DP_MODELS_ROOT") or (DP / "dp_models")))
    env["PYTHONPATH"] = f"{VP}:{DP}:" + env.get("PYTHONPATH", "")
    run_id = f"flow_{int(time.time())}"
    rs = wd / "rs"; rs.mkdir(exist_ok=True)
    S = {"video_id": a.video_id, "run_id": run_id, "model": a.model, "preset": a.preset,
         "device": a.device, "fps": a.fps, "width": a.width, "stages": {}}

    def stage(name, cmd):
        t = time.time(); rc = sh(cmd, log, env)
        S["stages"][name] = {"rc": rc, "s": round(time.time() - t, 1)}
        return rc

    # 1) Segmenter (optical_flow-профиль)
    if stage("segmenter", [PY, SEG, "--video-path", a.video, "--output", str(wd / "seg"),
             "--visual-cfg-path", CFG, "--platform-id", "youtube", "--video-id", a.video_id,
             "--run-id", run_id, "--sampling-policy-version", "flow_local_v1", "--config-hash", "local",
             "--dataprocessor-version", "flow_local", "--analysis-fps", str(a.fps), "--analysis-width", str(a.width)]):
        S["error"] = "segmenter"; print(json.dumps(S, ensure_ascii=False)); return 3
    fd = wd / "seg" / a.video_id / "video"
    if not (fd / "metadata.json").is_file():
        S["error"] = "no metadata"; print(json.dumps(S, ensure_ascii=False)); return 3

    flow_cmd = [PY, CF, "--frames-dir", str(fd), "--rs-path", str(rs),
                "--runtime", "inprocess", "--inprocess-model", a.model, "--device", a.device,
                "--batch-size", str(a.batch_size), "--triton-preprocess-preset", a.preset]
    if a.fp16:
        flow_cmd.append("--inprocess-fp16")

    # 2) core_optical_flow (inprocess, полная v3-схема)
    stage("core_optical_flow", flow_cmd)
    npz = rs / "core_optical_flow" / "flow.npz"

    # 3) валидаторы
    if npz.is_file():
        stage("validate", [PY, CFU, str(npz), "--struct", "--ranges"])
        try:
            import numpy as np
            z = np.load(npz, allow_pickle=True)
            meta = z["meta"].item() if "meta" in z.files else {}
            mn = np.asarray(z["motion_norm_per_sec_mean"]) if "motion_norm_per_sec_mean" in z.files else None
            fi = np.asarray(z["frame_indices"]) if "frame_indices" in z.files else None
            res = {
                "keys": list(z.files),
                "N": int(fi.size) if fi is not None else None,
                "runtime": meta.get("runtime"), "model": meta.get("inprocess_model") or meta.get("model_name"),
                "model_signature": str(meta.get("model_signature", ""))[:16],
                "stage_timings_ms": meta.get("stage_timings_ms"),
            }
            if mn is not None and mn.size > 1:
                mv = mn[1:]  # exclude first (0 by design)
                mvf = mv[np.isfinite(mv)]
                if mvf.size:
                    res["motion"] = {
                        "mean": round(float(np.mean(mvf)), 6),
                        "median": round(float(np.median(mvf)), 6),
                        "p95": round(float(np.percentile(mvf, 95)), 6),
                        "max": round(float(np.max(mvf)), 6),
                        "std": round(float(np.std(mvf)), 6),
                        "finite_frac": round(float(np.isfinite(mv).mean()), 4),
                    }
            z.close()
            S["result"] = res
            S["result"]["sig"] = _flow_signature(npz)
        except Exception as e:
            S["result_error"] = str(e)

    # 4) golden-детерминизм (второй прогон в отдельный rs)
    if a.golden and npz.is_file():
        rs2 = wd / "rs2"; rs2.mkdir(exist_ok=True)
        flow_cmd2 = list(flow_cmd)
        flow_cmd2[flow_cmd2.index(str(rs))] = str(rs2)
        stage("core_optical_flow_golden", flow_cmd2)
        npz2 = rs2 / "core_optical_flow" / "flow.npz"
        if npz2.is_file():
            sig1 = _flow_signature(npz)
            sig2 = _flow_signature(npz2)
            same = sig1 == sig2
            S["golden"] = {"identical": same,
                           "diff_keys": [k for k in sig1 if sig1.get(k) != sig2.get(k)]}

    (wd / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=2))
    print(json.dumps(S, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
