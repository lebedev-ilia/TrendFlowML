#!/usr/bin/env python3
"""
Локальный раннер video_pacing (Triton-free):
Segmenter(video_pacing профиль) → core_clip(inprocess) + core_optical_flow(inprocess raft_small) +
cut_detection(farneback, no-clip) → video_pacing(numpy/opencv) → валидаторы (--struct --ranges --qa) →
golden-детерминизм (2 прогона video_pacing, побайтовое сравнение model-facing массивов).

Все deps должны иметь ОДИНАКОВЫЕ frame_indices (Segmenter aligned sampling group).
Дефолт feature-гейтинга = вариант B: entropy+histograms ON, pace_peaks/periodicity/bursts OFF.

Использование:
  python run_video_pacing_local.py --video <mp4> [--video-id ID] [--fps 4] [--width 480]
      [--device cuda] [--golden] [--workdir /tmp/vp_out]
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
CC = VP / "core/model_process/core_clip/main.py"
CF = VP / "core/model_process/core_optical_flow/main.py"
CD = VP / "modules/cut_detection/main.py"
VPM = VP / "modules/video_pacing/main.py"
VPU = VP / "modules/video_pacing/utils/validate_video_pacing.py"
CFG = DP / "configs/audit_v3/visual/visual_video_pacing_only.yaml"

_SIG_KEYS = (
    "frame_indices", "times_s", "shot_boundary_frame_indices",
    "motion_norm_per_sec_mean", "semantic_change_rate_per_sec", "color_change_rate_per_sec",
    "feature_values",
)


def sh(cmd, log, env, timeout=5400):
    log.write(f"\n$ {' '.join(map(str, cmd))}\n"); log.flush()
    return subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(DP), timeout=timeout).returncode


def _vp_signature(npz_path: Path) -> dict:
    import numpy as np
    z = np.load(npz_path, allow_pickle=True)
    out = {}
    for k in _SIG_KEYS:
        if k in z.files:
            a = np.asarray(z[k])
            if a.dtype == object:
                continue
            a = np.nan_to_num(a.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            out[k] = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
    z.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--video-id", default="vp_local")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--golden", action="store_true", help="Второй прогон video_pacing + сравнение сигнатур массивов")
    ap.add_argument("--expected-empty", action="store_true", help="Сдвиг индексов деп → нет пересечения (тест no-fallback пути)")
    ap.add_argument("--workdir", default="/tmp/vp_out")
    a = ap.parse_args()

    wd = Path(a.workdir); wd.mkdir(parents=True, exist_ok=True)
    log = open(wd / "run.log", "w", encoding="utf-8")
    env = os.environ.copy()
    env["DP_MODELS_ROOT"] = str(Path(os.environ.get("DP_MODELS_ROOT") or (DP / "dp_models")))
    env["PYTHONPATH"] = f"{VP}:{DP}:" + env.get("PYTHONPATH", "")
    # Пиннинг потоков для строгого golden (numpy/opencv редукции).
    env["OMP_NUM_THREADS"] = "1"; env["OPENBLAS_NUM_THREADS"] = "1"; env["MKL_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    run_id = f"vp_{int(time.time())}"
    rs = wd / "rs"; rs.mkdir(exist_ok=True)
    S = {"video_id": a.video_id, "run_id": run_id, "device": a.device, "fps": a.fps, "width": a.width, "stages": {}}

    def stage(name, cmd):
        t = time.time(); rc = sh(cmd, log, env); S["stages"][name] = {"rc": rc, "s": round(time.time() - t, 1)}; return rc

    # 1) Segmenter (video_pacing профиль → aligned frames для всех deps)
    if stage("segmenter", [PY, SEG, "--video-path", a.video, "--output", str(wd / "seg"),
             "--visual-cfg-path", CFG, "--platform-id", "youtube", "--video-id", a.video_id,
             "--run-id", run_id, "--sampling-policy-version", "vp_local_v1", "--config-hash", "local",
             "--dataprocessor-version", "vp_local", "--analysis-fps", str(a.fps), "--analysis-width", str(a.width)]):
        S["error"] = "segmenter"; print(json.dumps(S, ensure_ascii=False)); return 3
    fd = wd / "seg" / a.video_id / "video"
    if not (fd / "metadata.json").is_file():
        S["error"] = "no metadata"; print(json.dumps(S, ensure_ascii=False)); return 3

    # 2) deps
    stage("core_clip", [PY, CC, "--frames-dir", fd, "--rs-path", str(rs), "--runtime", "inprocess",
                        "--model-name", "ViT-B/32", "--batch-size", "16"])
    stage("core_optical_flow", [PY, CF, "--frames-dir", fd, "--rs-path", str(rs), "--runtime", "inprocess",
                                "--inprocess-model", "raft_small", "--device", a.device, "--batch-size", "8",
                                "--triton-preprocess-preset", "raft_256"])
    stage("cut_detection", [PY, CD, "--frames-dir", fd, "--rs-path", str(rs),
                            "--no-require-core-optical-flow", "--no-use-clip"])

    # 2b) expected-empty: сдвинуть frame_indices всех деп → нет пересечения с модулем
    if a.expected_empty:
        _shift_dep_indices(rs, log)

    # 3) video_pacing (вариант B: entropy+histograms ON)
    stage("video_pacing", [PY, VPM, "--frames-dir", fd, "--rs-path", str(rs),
                           "--enable-entropy-features", "--enable-histograms"])
    npz = rs / "video_pacing" / "video_pacing_features.npz"

    # 4) валидаторы
    if npz.is_file():
        stage("validate", [PY, VPU, str(npz), "--struct", "--ranges", "--qa"])
        try:
            import numpy as np
            z = np.load(npz, allow_pickle=True)
            meta = z["meta"].item() if "meta" in z.files else {}
            fi = np.asarray(z["frame_indices"]) if "frame_indices" in z.files else None
            fn = np.asarray(z["feature_names"]) if "feature_names" in z.files else None
            fv = np.asarray(z["feature_values"], dtype=np.float64) if "feature_values" in z.files else None
            res = {
                "keys": list(z.files),
                "N": int(fi.size) if fi is not None else None,
                "S": int(np.asarray(z["shot_boundary_frame_indices"]).size) if "shot_boundary_frame_indices" in z.files else None,
                "nfeat": int(fn.size) if fn is not None else None,
                "status": meta.get("status"), "producer_version": meta.get("producer_version"),
                "stage_timings_ms": meta.get("stage_timings_ms"),
            }
            if fv is not None and fn is not None:
                res["nan_features"] = [str(fn[i]) for i in range(fn.size) if not np.isfinite(fv[i])]
                # ключевые скаляры
                tab = {str(fn[i]): float(fv[i]) for i in range(fn.size)}
                res["key_scalars"] = {k: round(tab[k], 5) for k in (
                    "shots_count", "cuts_per_10s", "shot_duration_mean", "mean_motion_speed_per_shot",
                    "frame_embedding_diff_mean", "color_change_rate_mean") if k in tab}
            z.close()
            S["result"] = res
            S["result"]["sig"] = _vp_signature(npz)
        except Exception as e:
            S["result_error"] = str(e)

    # 5) golden (2-й прогон video_pacing в rs2 = копия rs, тот же вход)
    if a.golden and npz.is_file():
        rs2 = wd / "rs2"
        if rs2.exists():
            shutil.rmtree(rs2)
        shutil.copytree(rs, rs2)
        # copytree выравнивает mtime всех файлов → max(key=mtime) нестабилен между features/model_facing.
        # Обновляем mtime у features-NPZ (не model_facing) чтобы они всегда были НОВЕЕ.
        import time as _time
        for _npz in rs2.rglob("*.npz"):
            if "features" in _npz.name and "model_facing" not in _npz.name:
                _t = _time.time(); _npz.touch()
        # удалить старый артефакт vp в rs2, чтобы пересчитать
        vp2_dir = rs2 / "video_pacing"
        if vp2_dir.exists():
            shutil.rmtree(vp2_dir)
        stage("video_pacing_golden", [PY, VPM, "--frames-dir", fd, "--rs-path", str(rs2),
                                      "--enable-entropy-features", "--enable-histograms"])
        npz2 = rs2 / "video_pacing" / "video_pacing_features.npz"
        if npz2.is_file():
            sig1 = _vp_signature(npz); sig2 = _vp_signature(npz2)
            S["golden"] = {"identical": sig1 == sig2,
                           "diff_keys": [k for k in sig1 if sig1.get(k) != sig2.get(k)]}

    (wd / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=2))
    print(json.dumps(S, ensure_ascii=False, indent=2))
    return 0


def _shift_dep_indices(rs: Path, log) -> None:
    """Сдвиг frame_indices у core_clip/core_optical_flow/cut_detection на +10^6 → пересечение с модулем пусто."""
    import numpy as np
    targets = [
        rs / "core_clip" / "embeddings.npz",
        rs / "core_optical_flow" / "flow.npz",
    ]
    for p in targets:
        if not p.is_file():
            continue
        z = np.load(p, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        z.close()
        if "frame_indices" in d:
            d["frame_indices"] = np.asarray(d["frame_indices"], dtype=np.int32) + 1000000
        np.savez(p, **d)
        log.write(f"\n[expected-empty] shifted {p}\n"); log.flush()


if __name__ == "__main__":
    raise SystemExit(main())
