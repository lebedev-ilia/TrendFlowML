#!/usr/bin/env python3
"""
Локальный раннер frames_composition (GPU для deps, CPU для самого модуля).

Цепочка: Segmenter → core_object_detections(YOLO) + core_face_landmarks(mediapipe) +
         core_depth_midas(inprocess MiDaS) → frames_composition(pure CV) →
         валидаторы (--struct --ranges) → golden-детерминизм.

Под-модуль frames_composition детерминирован (чистый numpy/opencv) → max|Δ|=0.0.

Использование:
  python run_frames_composition_local.py --video <mp4> [--video-id ID]
      [--fps 4] [--width 480] [--device cuda] [--golden] [--expected-empty]
      [--workdir /tmp/fc_out]
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

DP = Path(__file__).resolve().parents[1]
ROOT = DP.parent


def _pick(*c):
    for x in c:
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
OBJ = VP / "core/model_process/core_object_detections/main.py"
FACE = VP / "core/model_process/core_face_landmarks/main.py"
DEPTH = VP / "core/model_process/core_depth_midas/main.py"
FC = VP / "modules/frames_composition/main.py"
VLD = VP / "modules/frames_composition/utils/validate_frames_composition.py"
CFG = DP / "configs/audit_v3/visual/visual_frames_composition_only.yaml"

# Подпись для golden-сравнения: model-facing массивы
_SIG_KEYS = (
    "frame_indices",
    "times_s",
    "frame_feature_values",
    "frame_feature_present_ratio",
    "feature_values",
)


def sh(cmd, log, env, timeout=5400):
    log.write(f"\n$ {' '.join(map(str, cmd))}\n")
    log.flush()
    return subprocess.run(
        [str(c) for c in cmd],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(DP),
        timeout=timeout,
    ).returncode


def _fc_signature(npz_path: Path) -> dict:
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
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--video", required=True, help="Путь к .mp4")
    ap.add_argument("--video-id", default="fc_local", help="video_id для директорий")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--golden", action="store_true", help="2-й прогон fc + сравнение сигнатур")
    ap.add_argument(
        "--expected-empty",
        action="store_true",
        help="Сдвинуть frame_indices core_face_landmarks на +1e6 → no_faces_in_video path",
    )
    ap.add_argument("--workdir", default="/tmp/fc_out")
    a = ap.parse_args()

    wd = Path(a.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    log = open(wd / "run.log", "w", encoding="utf-8")
    env = os.environ.copy()
    env["DP_MODELS_ROOT"] = str(
        Path(os.environ.get("DP_MODELS_ROOT") or (DP / "dp_models"))
    )
    env["PYTHONPATH"] = f"{VP}:{DP}:" + env.get("PYTHONPATH", "")
    # Пиннинг потоков для строгого golden (numpy/opencv редукции)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    run_id = f"fc_{int(time.time())}"
    rs = wd / "rs"
    rs.mkdir(exist_ok=True)
    S: dict = {
        "video_id": a.video_id,
        "run_id": run_id,
        "device": a.device,
        "fps": a.fps,
        "width": a.width,
        "stages": {},
    }

    def stage(name, cmd):
        t = time.time()
        rc = sh(cmd, log, env)
        S["stages"][name] = {"rc": rc, "s": round(time.time() - t, 1)}
        return rc

    # 1) Segmenter — генерирует frames_dir с метаданными
    cfg_path = CFG
    if not cfg_path.is_file():
        # Фолбэк: взять любой существующий visual cfg
        for alt in (
            DP / "configs/audit_v3/visual/visual_depth_only.yaml",
            DP / "configs/audit_v3/visual/visual_video_pacing_only.yaml",
        ):
            if alt.is_file():
                cfg_path = alt
                break
    seg_cmd = [
        PY, SEG,
        "--video-path", a.video,
        "--output", str(wd / "seg"),
        "--platform-id", "youtube",
        "--video-id", a.video_id,
        "--run-id", run_id,
        "--sampling-policy-version", "fc_local_v1",
        "--config-hash", "local",
        "--dataprocessor-version", "fc_local",
        "--analysis-fps", str(a.fps),
        "--analysis-width", str(a.width),
    ]
    if cfg_path.is_file():
        seg_cmd += ["--visual-cfg-path", str(cfg_path)]
    if stage("segmenter", seg_cmd):
        S["error"] = "segmenter"
        print(json.dumps(S, ensure_ascii=False))
        log.write(f"\nERROR: segmenter failed\n")
        return 3

    fd = wd / "seg" / a.video_id / "video"
    if not (fd / "metadata.json").is_file():
        S["error"] = "no metadata.json"
        print(json.dumps(S, ensure_ascii=False))
        return 3

    # 2) Core deps (GPU/CPU)
    # core_object_detections (на поде используем yolo11l.pt, если нет yolo11x_41_best.pt)
    if OBJ.is_file():
        # Пробуем kанонический путь, фолбэк на доступный
        yolo_path = str(DP / "dp_models" / "visual" / "yolo" / "yolo11x_41_best.pt")
        if not Path(yolo_path).exists():
            yolo_path = str(DP / "dp_models" / "visual" / "object_detection" / "yolo11l" / "yolo11l.pt")
        stage(
            "core_object_detections",
            [PY, OBJ, "--frames-dir", fd, "--rs-path", str(rs),
             "--model", yolo_path, "--device", a.device],
        )
    else:
        log.write(f"\nSkip core_object_detections (main.py not found at {OBJ})\n")
        S["stages"]["core_object_detections"] = {"rc": -1, "s": 0, "skip": True}

    # core_face_landmarks (baseline требует --use-face-mesh --use-person-mask)
    if FACE.is_file():
        stage(
            "core_face_landmarks",
            [PY, FACE, "--frames-dir", fd, "--rs-path", str(rs),
             "--use-face-mesh", "--use-person-mask"],
        )
    else:
        log.write(f"\nSkip core_face_landmarks (main.py not found at {FACE})\n")
        S["stages"]["core_face_landmarks"] = {"rc": -1, "s": 0, "skip": True}

    # core_depth_midas (inprocess)
    if DEPTH.is_file():
        stage(
            "core_depth_midas",
            [
                PY, DEPTH,
                "--frames-dir", fd,
                "--rs-path", str(rs),
                "--runtime", "inprocess",
                "--device", a.device,
            ],
        )
    else:
        log.write(f"\nSkip core_depth_midas (main.py not found at {DEPTH})\n")
        S["stages"]["core_depth_midas"] = {"rc": -1, "s": 0, "skip": True}

    # 3) expected-empty: сдвинуть face_landmarks frame_indices → нет пересечения с fc
    if a.expected_empty:
        _shift_face_indices(rs, log)
        log.write("\n[expected-empty] face_indices shifted → frames_composition будет no_faces_in_video\n")

    # 4) frames_composition (pure CV, CPU)
    fc_rc = stage(
        "frames_composition",
        [PY, FC, "--frames-dir", fd, "--rs-path", str(rs)],
    )
    npz = rs / "frames_composition" / "frames_composition.npz"

    # 5) Валидатор (--struct --ranges)
    if VLD.is_file() and npz.is_file():
        stage("validate", [PY, VLD, str(npz), "--struct", "--ranges"])
    else:
        S["stages"]["validate"] = {"skip": True}

    # 6) Анализ результатов
    if npz.is_file():
        try:
            import numpy as np

            z = np.load(npz, allow_pickle=True)
            meta = z["meta"].item() if "meta" in z.files else {}
            fi = np.asarray(z["frame_indices"]) if "frame_indices" in z.files else None
            fn = np.asarray(z["feature_names"]) if "feature_names" in z.files else None
            fv = (
                np.asarray(z["feature_values"], dtype=np.float64)
                if "feature_values" in z.files
                else None
            )
            ffv = (
                np.asarray(z["frame_feature_values"])
                if "frame_feature_values" in z.files
                else None
            )
            res = {
                "keys": list(z.files),
                "N": int(fi.size) if fi is not None else None,
                "D": int(ffv.shape[1]) if ffv is not None and ffv.ndim == 2 else None,
                "F": int(fn.size) if fn is not None else None,
                "status": meta.get("status"),
                "producer_version": meta.get("producer_version"),
                "stage_timings_ms": meta.get("stage_timings_ms"),
            }
            if ffv is not None:
                res["nan_frame_features_frac"] = float(np.isnan(ffv).mean())
            if fv is not None:
                res["nan_video_features_frac"] = float(np.isnan(fv).mean())
                res["nan_video_features"] = [
                    str(fn[i]) for i in range(fn.size) if not np.isfinite(fv[i])
                ]
            # Ключевые скаляры для отчёта
            if fv is not None and fn is not None:
                tab = {str(fn[i]): float(fv[i]) for i in range(fn.size)}
                key_scalars = {}
                for k in (
                    "has_faces",
                    "frames_n",
                    "face_present__mean",
                    "edge_density__mean",
                    "line_strength__mean",
                    "symmetry_score__mean",
                    "negative_space_ratio__mean",
                    "depth_std__mean",
                    "style_prob__minimalist__mean",
                    "style_prob__cinematic__mean",
                    "style_prob__vlog__mean",
                    "style_prob__product_centered__mean",
                    "style_dominant_id",
                ):
                    if k in tab:
                        key_scalars[k] = round(tab[k], 5)
                res["key_scalars"] = key_scalars
                res["sig"] = _fc_signature(npz)
            z.close()
            S["result"] = res
        except Exception as e:
            S["result_error"] = str(e)

    # 7) Golden (2-й прогон frames_composition в rs2)
    if a.golden and npz.is_file():
        rs2 = wd / "rs2"
        if rs2.exists():
            shutil.rmtree(rs2)
        shutil.copytree(rs, rs2)
        # Удалить старый артефакт fc в rs2 для пересчёта
        fc2_dir = rs2 / "frames_composition"
        if fc2_dir.exists():
            shutil.rmtree(fc2_dir)
        stage(
            "frames_composition_golden",
            [PY, FC, "--frames-dir", fd, "--rs-path", str(rs2)],
        )
        npz2 = rs2 / "frames_composition" / "frames_composition.npz"
        if npz2.is_file():
            sig1 = _fc_signature(npz)
            sig2 = _fc_signature(npz2)
            S["golden"] = {
                "identical": sig1 == sig2,
                "diff_keys": [k for k in sig1 if sig1.get(k) != sig2.get(k)],
            }

    (wd / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=2))
    print(json.dumps(S, ensure_ascii=False, indent=2))
    return 0


def _shift_face_indices(rs: Path, log) -> None:
    """Сдвиг frame_indices в core_face_landmarks на +10^6 → нет пересечения с fc."""
    import numpy as np

    targets = [rs / "core_face_landmarks" / "face_landmarks.npz"]
    for p in targets:
        if not p.is_file():
            log.write(f"\n[expected-empty] skip {p} (not found)\n")
            continue
        z = np.load(p, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        z.close()
        if "frame_indices" in d:
            d["frame_indices"] = np.asarray(d["frame_indices"], dtype=np.int32) + 1_000_000
        np.savez(p, **d)
        log.write(f"\n[expected-empty] shifted frame_indices in {p}\n")
        log.flush()


if __name__ == "__main__":
    raise SystemExit(main())
