#!/usr/bin/env python3
"""
Локальный раннер optical_flow (module, потребитель core_optical_flow):
  Segmenter → core_optical_flow(inprocess RAFT) → optical_flow(module) → валидаторы → golden

Использование:
  python run_optical_flow_module_local.py --video <mp4> [--golden] [--expected-empty]
      [--fps 4] [--width 480] [--device cuda] [--workdir /tmp/of_module_out]
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

DP = Path(__file__).resolve().parents[1]
ROOT = DP.parent


def _pick(*candidates):
    for x in candidates:
        if Path(x).exists():
            return str(x)
    return sys.executable or "python3"


PY = _pick(
    "/workspace/venv/bin/python",
    DP / "VisualProcessor/.vp_venv/bin/python",
    DP / ".data_venv/bin/python",
)
SEG = DP / "Segmenter/segmenter.py"
VP = DP / "VisualProcessor"
CF_MAIN = VP / "core/model_process/core_optical_flow/main.py"
CF_VAL = VP / "core/model_process/core_optical_flow/utils/validate_core_optical_flow_npz.py"
OF_MAIN = VP / "modules/optical_flow/main.py"
OF_VAL = VP / "modules/optical_flow/utils/validate_optical_flow_npz.py"
CFG = DP / "configs/audit_v3/visual/visual_optical_flow_only.yaml"


def sh(cmd, log, env, timeout=5400):
    log.write(f"\n$ {' '.join(map(str, cmd))}\n")
    log.flush()
    return subprocess.run(
        [str(c) for c in cmd],
        stdout=log, stderr=subprocess.STDOUT,
        env=env, cwd=str(DP), timeout=timeout,
    ).returncode


def _sig_keys():
    return (
        "frame_indices", "times_s", "motion_norm_per_sec_mean",
        "frame_feature_values", "feature_values",
    )


def _signature(npz_path: Path) -> dict:
    """SHA256 хэши ключевых массивов (NaN→0 для стабильности)."""
    import numpy as np
    z = np.load(npz_path, allow_pickle=True)
    out = {}
    try:
        for k in _sig_keys():
            if k in z.files:
                a = np.asarray(z[k])
                if a.dtype.kind in ("f", "c"):
                    a = np.nan_to_num(a.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                out[k] = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
    finally:
        z.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--video-id", default="of_module_local")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="raft_small", choices=["raft_small", "raft_large"])
    ap.add_argument("--preset", default="raft_256", choices=["raft_256", "raft_384", "raft_512"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--golden", action="store_true", help="Второй прогон optical_flow → golden сравнение")
    ap.add_argument("--workdir", default="/tmp/of_module_out")
    a = ap.parse_args()

    wd = Path(a.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    log = open(wd / "run.log", "w", encoding="utf-8")
    env = os.environ.copy()
    env["DP_MODELS_ROOT"] = str(Path(os.environ.get("DP_MODELS_ROOT") or (DP / "dp_models")))
    env["PYTHONPATH"] = f"{VP}:{DP}:" + env.get("PYTHONPATH", "")

    run_id = f"of_{int(time.time())}"
    rs = wd / "rs"
    rs.mkdir(exist_ok=True)
    S: dict = {
        "video_id": a.video_id, "run_id": run_id,
        "model": a.model, "preset": a.preset,
        "stages": {},
    }

    def stage(name, cmd):
        t = time.time()
        rc = sh(cmd, log, env)
        S["stages"][name] = {"rc": rc, "s": round(time.time() - t, 1)}
        return rc

    # 1) Segmenter
    if stage("segmenter", [
        PY, SEG,
        "--video-path", a.video,
        "--output", str(wd / "seg"),
        "--visual-cfg-path", CFG,
        "--platform-id", "youtube",
        "--video-id", a.video_id,
        "--run-id", run_id,
        "--sampling-policy-version", "of_module_v1",
        "--config-hash", "local",
        "--dataprocessor-version", "of_module_local",
        "--analysis-fps", str(a.fps),
        "--analysis-width", str(a.width),
    ]):
        S["error"] = "segmenter"
        print(json.dumps(S, ensure_ascii=False, indent=2))
        return 3

    fd = wd / "seg" / a.video_id / "video"
    if not (fd / "metadata.json").is_file():
        S["error"] = "no metadata.json"
        print(json.dumps(S, ensure_ascii=False, indent=2))
        return 3

    # 2) core_optical_flow (inprocess RAFT)
    core_cmd = [
        PY, CF_MAIN,
        "--frames-dir", str(fd),
        "--rs-path", str(rs),
        "--runtime", "inprocess",
        "--inprocess-model", a.model,
        "--device", a.device,
        "--batch-size", str(a.batch_size),
        "--triton-preprocess-preset", a.preset,
    ]
    stage("core_optical_flow", core_cmd)

    core_npz = rs / "core_optical_flow" / "flow.npz"
    if not core_npz.is_file():
        S["error"] = "no core_optical_flow/flow.npz"
        print(json.dumps(S, ensure_ascii=False, indent=2))
        return 4

    # 3) validate core
    stage("validate_core", [PY, CF_VAL, str(core_npz), "--struct", "--ranges"])

    # 4) optical_flow module (consumer, CPU-only)
    stage("optical_flow_module", [
        PY, OF_MAIN,
        "--frames-dir", str(fd),
        "--rs-path", str(rs),
    ])

    of_npz = rs / "optical_flow" / "optical_flow.npz"
    if not of_npz.is_file():
        S["error"] = "no optical_flow/optical_flow.npz"
        print(json.dumps(S, ensure_ascii=False, indent=2))
        return 5

    # 5) validate optical_flow module
    stage("validate_struct", [PY, OF_VAL, str(of_npz), "--struct"])
    stage("validate_ranges", [PY, OF_VAL, str(of_npz), "--ranges"])

    # Собрать числа
    try:
        import numpy as np
        z = np.load(of_npz, allow_pickle=True)
        meta = z["meta"].item() if "meta" in z.files else {}
        fi = np.asarray(z["frame_indices"], dtype=np.int32).ravel()
        mot = np.asarray(z["motion_norm_per_sec_mean"], dtype=np.float32).ravel()
        ffv = np.asarray(z["frame_feature_values"], dtype=np.float32)
        fn = [str(x) for x in np.asarray(z["feature_names"], dtype=object).ravel().tolist()]
        fv = np.asarray(z["feature_values"], dtype=np.float32).ravel()
        mot_valid = mot[1:][np.isfinite(mot[1:])]
        # NaN rate
        nan_total = int(np.isnan(ffv).sum())
        total_el = int(ffv.size)
        res = {
            "status": str(meta.get("status")),
            "N": int(fi.size),
            "D": int(ffv.shape[1]) if ffv.ndim == 2 else 0,
            "nan_rate": round(nan_total / max(1, total_el), 4),
            "motion": {
                "mean": round(float(np.mean(mot_valid)), 6) if mot_valid.size else None,
                "median": round(float(np.median(mot_valid)), 6) if mot_valid.size else None,
                "std": round(float(np.std(mot_valid)), 6) if mot_valid.size else None,
                "p95": round(float(np.percentile(mot_valid, 95)), 6) if mot_valid.size else None,
            },
            "agg": {fn[i]: round(float(fv[i]), 6) for i in range(len(fn))},
            "per_col_std": {
                str(np.asarray(z["frame_feature_names"], dtype=object).ravel().tolist()[i]):
                round(float(np.nanstd(ffv[:, i])), 6)
                for i in range(ffv.shape[1])
            } if ffv.ndim == 2 else {},
        }
        S["result"] = res
        S["sig"] = _signature(of_npz)
        z.close()
    except Exception as e:
        S["result_error"] = str(e)

    # 6) Golden детерминизм (второй прогон optical_flow module — без перезапуска core)
    if a.golden and of_npz.is_file():
        rs2 = wd / "rs2"
        rs2.mkdir(exist_ok=True)
        # Копируем core_optical_flow в rs2
        import shutil
        shutil.copytree(str(rs / "core_optical_flow"), str(rs2 / "core_optical_flow"))
        stage("optical_flow_module_golden", [
            PY, OF_MAIN,
            "--frames-dir", str(fd),
            "--rs-path", str(rs2),
        ])
        of_npz2 = rs2 / "optical_flow" / "optical_flow.npz"
        if of_npz2.is_file():
            sig1 = _signature(of_npz)
            sig2 = _signature(of_npz2)
            same = sig1 == sig2
            S["golden"] = {
                "identical": same,
                "diff_keys": [k for k in sig1 if sig1.get(k) != sig2.get(k)],
            }

    (wd / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=2))
    print(json.dumps(S, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
