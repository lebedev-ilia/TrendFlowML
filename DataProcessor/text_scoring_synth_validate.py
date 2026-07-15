#!/usr/bin/env python3
"""
Синтетический тест text_scoring — запускать локально (CPU-only, без пода).
Проверяет: ok-путь, 3 empty-пути, golden, U2 (ось времени), U6 (разные N).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import shutil
import traceback
import numpy as np

# Добавляем пути
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VP_DIR = os.path.join(THIS_DIR, "VisualProcessor")
if VP_DIR not in sys.path:
    sys.path.insert(0, VP_DIR)

# ============================================================
# Вспомогательные функции
# ============================================================

def _write_metadata(frames_dir: str, *, N: int, fps: float = 6.0,
                    run_id: str = "test-run-001",
                    platform_id: str = "youtube",
                    video_id: str = "synth_test") -> list:
    """Создать metadata.json с union_timestamps_sec и batches (FrameManager-совместимый)."""
    # frame_indices: последовательные 0..N-1 (union_timestamps_sec[fi] должен быть in-bounds)
    frame_indices = list(range(N))
    # union_timestamps_sec = [i/fps for i in range(total_frames)]
    total_frames = N
    union_timestamps = [round(i / fps, 6) for i in range(total_frames)]

    # Создаём реальные .npy батчи (нулевые кадры — text_scoring кадры не читает)
    os.makedirs(frames_dir, exist_ok=True)
    H, W = 240, 320
    chunk_size = 32
    batches = []
    for b in range((total_frames + chunk_size - 1) // chunk_size):
        s = b * chunk_size
        e = min(s + chunk_size, total_frames)
        chunk = np.zeros((e - s, H, W, 3), dtype=np.uint8)
        chunk_path = f"batch_{b:05d}.npy"
        np.save(os.path.join(frames_dir, chunk_path), chunk)
        batches.append({
            "batch_index": b,
            "path": chunk_path,
            "start_frame": s,
            "end_frame": e - 1,
            "num_frames": e - s,
        })

    meta = {
        "total_frames": total_frames,
        "fps": fps,
        "analysis_fps": fps,
        "analysis_width": W,
        "analysis_height": H,
        "height": H,
        "width": W,
        "channels": 3,
        "chunk_size": chunk_size,
        "batch_size": chunk_size,
        "cache_size": 2,
        "color_space": "RGB",
        "platform_id": platform_id,
        "video_id": video_id,
        "run_id": run_id,
        "sampling_policy_version": "synth_v1",
        "config_hash": "synth0000",
        "dataprocessor_version": "test",
        "batches": batches,
        "union_timestamps_sec": union_timestamps,
        "text_scoring": {
            "frame_indices": frame_indices,
            "num_indices": len(frame_indices),
        },
    }
    with open(os.path.join(frames_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return frame_indices


def _write_ocr_npz(path: str, *, frame_indices: list, mode: str = "ok") -> None:
    """
    mode:
      ok           — несколько OCR-детекций
      empty_nodata — ocr_raw = []
      out_of_fi    — OCR-детекции за пределами frame_indices
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if mode == "ok":
        detections = []
        # несколько детекций на первых кадрах
        for i, fi in enumerate(frame_indices[:6]):
            detections.append({
                "frame": int(fi),
                "bbox": [50, 30, 400, 80],
                "text_raw": f"Subscribe Now {i}",
                "text_norm": f"subscribe now {i}",
                "det_confidence": 0.85 + i * 0.01,
                "is_cta_candidate": (i == 0),
                "language": "en",
            })
        # Ещё несколько CTA-похожих
        for i, fi in enumerate(frame_indices[6:10]):
            detections.append({
                "frame": int(fi),
                "bbox": [100, 500, 600, 560],
                "text_raw": "SUBSCRIBE",
                "text_norm": "subscribe",
                "det_confidence": 0.92,
                "is_cta_candidate": True,
                "language": "en",
            })
    elif mode == "empty_nodata":
        detections = []
    elif mode == "out_of_fi":
        # frame_indices в metadata — [0,5,10,...], а здесь детекции на кадрах 1000+
        detections = [
            {
                "frame": 9999,
                "bbox": [10, 10, 200, 50],
                "text_raw": "out of range",
                "text_norm": "out of range",
                "det_confidence": 0.9,
                "is_cta_candidate": False,
            }
        ]
    else:
        detections = []

    meta_info = {
        "producer": "ocr_extractor",
        "schema_version": "ocr_npz_v1",
        "status": "ok" if detections else "empty",
        "empty_reason": None if detections else "no_text_detected",
    }
    np.savez_compressed(
        path,
        frame_indices=np.asarray([d["frame"] for d in detections], dtype=np.int32),
        times_s=np.zeros(len(detections), dtype=np.float32),
        ocr_raw=np.asarray(detections, dtype=object),
        meta=np.asarray(meta_info, dtype=object),
    )


def _run_text_scoring(frames_dir: str, rs_path: str, config: dict) -> dict:
    """Запустить TextScoringModule.process() и вернуть результат."""
    from modules.text_scoring.utils.text_scoring import TextScoringModule

    module = TextScoringModule(rs_path=rs_path)
    metadata = module.load_metadata(frames_dir)
    fi = module.get_frame_indices(metadata, fallback_to_all=False)
    fm = module.create_frame_manager(frames_dir, metadata)
    try:
        result = module.process(frame_manager=fm, frame_indices=fi, config=config)
    finally:
        fm.close()
    return result


def _validate_result(result: dict, *, expect_ok: bool, N: int, F: int = 35) -> list[str]:
    """Структурная проверка результата. Возвращает список ошибок."""
    errors = []
    for key in ("frame_indices", "times_s", "text_present", "text_presence",
                "text_count_per_frame", "feature_names", "feature_values",
                "ocr_raw", "ocr_unique_elements"):
        if key not in result:
            errors.append(f"отсутствует ключ: {key}")

    fi = np.asarray(result.get("frame_indices", []), dtype=np.int64)
    ts = np.asarray(result.get("times_s", []), dtype=np.float64)
    if fi.size != N:
        errors.append(f"frame_indices.size={fi.size} != N={N}")
    if ts.size != N:
        errors.append(f"times_s.size={ts.size} != N={N}")
    if fi.size > 1 and not np.all(np.diff(fi) > 0):
        errors.append("frame_indices не строго возрастает")
    if ts.size > 1 and np.any(np.diff(ts) < -1e-6):
        errors.append("times_s убывает")

    fv = np.asarray(result.get("feature_values", []), dtype=np.float64)
    fn = np.asarray(result.get("feature_names", []), dtype=object)
    if fv.size != F:
        errors.append(f"feature_values.size={fv.size} != F={F}")
    if fn.size != F:
        errors.append(f"feature_names.size={fn.size} != F={F}")

    text_present = bool(result.get("text_present", np.array(False)).item()
                        if isinstance(result.get("text_present"), np.ndarray)
                        else result.get("text_present", False))

    nan_count = int(np.sum(np.isnan(fv))) if fv.size else 0

    if expect_ok:
        if not text_present:
            errors.append("expect_ok: text_present должен быть True")
        if nan_count > 10:  # допускаем NaN для отключённых фичефлагов (peaks/entropy/speed/CTA)
            errors.append(f"expect_ok: слишком много NaN={nan_count}/35 (ожидаем ≤10)")
        # проверяем core-блок (должен быть без NaN при ok)
        fn_list = [str(x) for x in fn.tolist()]
        core = ["text_frames_ratio", "text_count_mean", "text_count_p95",
                "text_on_screen_continuity", "text_readability_score",
                "text_action_sync_score"]
        for name in core:
            if name in fn_list:
                idx = fn_list.index(name)
                if idx < fv.size and np.isnan(fv[idx]):
                    errors.append(f"expect_ok: core-фича {name!r} = NaN")
        # доли в [0,1]
        for name in ("text_frames_ratio", "text_area_fraction", "text_on_screen_continuity_normalized"):
            if name in fn_list:
                idx = fn_list.index(name)
                if idx < fv.size and np.isfinite(fv[idx]):
                    v = float(fv[idx])
                    if v < -1e-3 or v > 1.0 + 1e-3:
                        errors.append(f"{name}={v:.4f} вне [0,1]")
    else:
        if text_present:
            errors.append("expect_empty: text_present должен быть False")
        if nan_count != 34:
            errors.append(f"expect_empty: ожидаем 34 NaN, получили {nan_count}")

    return errors


# ============================================================
# Тесты
# ============================================================

PASS = "PASS"
FAIL = "FAIL"

results = {}


def test_ok_path(tmpdir: str) -> tuple[str, str]:
    """U1+U2+U3+C1+C4: ok-путь с реальными OCR-детекциями, N=30."""
    frames_dir = os.path.join(tmpdir, "frames_ok")
    rs_path = os.path.join(tmpdir, "rs_ok")
    N = 30
    fi = _write_metadata(frames_dir, N=N)
    ocr_path = os.path.join(rs_path, "ocr_extractor", "ocr.npz")
    _write_ocr_npz(ocr_path, frame_indices=fi, mode="ok")
    result = _run_text_scoring(frames_dir, rs_path, config={})
    errors = _validate_result(result, expect_ok=True, N=N)
    if errors:
        return FAIL, "; ".join(errors)
    fv = np.asarray(result["feature_values"], dtype=np.float64)
    fn = [str(x) for x in np.asarray(result["feature_names"], dtype=object).tolist()]
    text_frac = fv[fn.index("text_frames_ratio")] if "text_frames_ratio" in fn else float("nan")
    cta_pres = fv[fn.index("cta_presence")] if "cta_presence" in fn else float("nan")
    return PASS, f"N={N} text_frames_ratio={text_frac:.3f} cta_presence={cta_pres:.3f} NaN={int(np.sum(np.isnan(fv)))}/35"


def test_empty_no_ocr(tmpdir: str) -> tuple[str, str]:
    """U4a: нет ocr.npz вовсе → status=empty."""
    frames_dir = os.path.join(tmpdir, "frames_noocr")
    rs_path = os.path.join(tmpdir, "rs_noocr")
    N = 30
    _write_metadata(frames_dir, N=N)
    # НЕ создаём ocr.npz
    result = _run_text_scoring(frames_dir, rs_path, config={})
    errors = _validate_result(result, expect_ok=False, N=N)
    if errors:
        return FAIL, "; ".join(errors)
    return PASS, f"N={N} status=empty (no ocr.npz)"


def test_empty_nodata(tmpdir: str) -> tuple[str, str]:
    """U4b: ocr.npz есть, но ocr_raw=[] → status=empty."""
    frames_dir = os.path.join(tmpdir, "frames_emptyocr")
    rs_path = os.path.join(tmpdir, "rs_emptyocr")
    N = 30
    fi = _write_metadata(frames_dir, N=N)
    ocr_path = os.path.join(rs_path, "ocr_extractor", "ocr.npz")
    _write_ocr_npz(ocr_path, frame_indices=fi, mode="empty_nodata")
    result = _run_text_scoring(frames_dir, rs_path, config={})
    errors = _validate_result(result, expect_ok=False, N=N)
    if errors:
        return FAIL, "; ".join(errors)
    return PASS, f"N={N} status=empty (empty ocr_raw)"


def test_empty_out_of_fi(tmpdir: str) -> tuple[str, str]:
    """U4c: ocr.npz есть, все детекции за пределами frame_indices → status=empty."""
    frames_dir = os.path.join(tmpdir, "frames_outfi")
    rs_path = os.path.join(tmpdir, "rs_outfi")
    N = 30
    fi = _write_metadata(frames_dir, N=N)
    ocr_path = os.path.join(rs_path, "ocr_extractor", "ocr.npz")
    _write_ocr_npz(ocr_path, frame_indices=fi, mode="out_of_fi")
    result = _run_text_scoring(frames_dir, rs_path, config={})
    errors = _validate_result(result, expect_ok=False, N=N)
    if errors:
        return FAIL, "; ".join(errors)
    return PASS, f"N={N} status=empty (OCR out of frame_indices)"


def test_golden(tmpdir: str) -> tuple[str, str]:
    """U5: golden детерминизм — два прогона с одними данными → feature_values идентичны."""
    frames_dir = os.path.join(tmpdir, "frames_golden")
    rs_path1 = os.path.join(tmpdir, "rs_g1")
    rs_path2 = os.path.join(tmpdir, "rs_g2")
    N = 30
    fi = _write_metadata(frames_dir, N=N)
    for rs in (rs_path1, rs_path2):
        ocr_path = os.path.join(rs, "ocr_extractor", "ocr.npz")
        _write_ocr_npz(ocr_path, frame_indices=fi, mode="ok")
    r1 = _run_text_scoring(frames_dir, rs_path1, config={})
    r2 = _run_text_scoring(frames_dir, rs_path2, config={})
    fv1 = np.asarray(r1["feature_values"], dtype=np.float64)
    fv2 = np.asarray(r2["feature_values"], dtype=np.float64)
    # сравниваем только finite (NaN == NaN в nan-equal)
    diff_finite = np.nanmax(np.abs(fv1 - fv2)) if fv1.size else 0.0
    nan_eq = np.sum(np.isnan(fv1)) == np.sum(np.isnan(fv2))
    if not np.isclose(diff_finite, 0.0, atol=1e-8) or not nan_eq:
        return FAIL, f"golden diff={diff_finite:.2e} nan_eq={nan_eq}"
    return PASS, f"golden diff={diff_finite:.2e} (побайтово)"


def test_different_lengths(tmpdir: str) -> tuple[str, str]:
    """U6: N=5, N=30, N=200 — всё отрабатывает без падений."""
    errors = []
    for N in (5, 30, 200):
        frames_dir = os.path.join(tmpdir, f"frames_N{N}")
        rs_path = os.path.join(tmpdir, f"rs_N{N}")
        fi = _write_metadata(frames_dir, N=N)
        ocr_path = os.path.join(rs_path, "ocr_extractor", "ocr.npz")
        _write_ocr_npz(ocr_path, frame_indices=fi, mode="ok")
        result = _run_text_scoring(frames_dir, rs_path, config={})
        errs = _validate_result(result, expect_ok=True, N=N)
        if errs:
            errors.append(f"N={N}: {errs}")
    if errors:
        return FAIL, "; ".join(str(e) for e in errors)
    return PASS, "N=5/30/200 — все валидны"


def test_fallback_motion_signal(tmpdir: str) -> tuple[str, str]:
    """
    Баг-фикс: при повреждённом optical_flow.npz должен пробоваться core_optical_flow.
    Создаём damaged optical_flow.npz (не NPZ-формат) + валидный core_optical_flow.
    """
    frames_dir = os.path.join(tmpdir, "frames_mot")
    rs_path = os.path.join(tmpdir, "rs_mot")
    N = 30
    fi = _write_metadata(frames_dir, N=N)

    # ocr
    ocr_path = os.path.join(rs_path, "ocr_extractor", "ocr.npz")
    _write_ocr_npz(ocr_path, frame_indices=fi, mode="ok")

    # Повреждённый optical_flow.npz
    of_dir = os.path.join(rs_path, "optical_flow")
    os.makedirs(of_dir, exist_ok=True)
    with open(os.path.join(of_dir, "optical_flow.npz"), "w") as f:
        f.write("NOT A VALID NPZ FILE - BROKEN")

    # Валидный core_optical_flow
    cof_dir = os.path.join(rs_path, "core_optical_flow")
    os.makedirs(cof_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(cof_dir, "flow.npz"),
        frame_indices=np.asarray(fi, dtype=np.int32),
        motion_norm_per_sec_mean=np.random.RandomState(42).uniform(0, 1, len(fi)).astype(np.float32),
    )

    result = _run_text_scoring(frames_dir, rs_path, config={"use_motion_data": True, "motion_weight": 0.3})
    errors = _validate_result(result, expect_ok=True, N=N)
    # Проверяем что fallback сработал (нет exception)
    if errors:
        return FAIL, f"исключение при broken optical_flow.npz: {errors}"
    return PASS, "fallback на core_optical_flow при broken optical_flow.npz"


def test_validate_script(tmpdir: str) -> tuple[str, str]:
    """U1: validate_text_scoring.py --struct --ranges на синтетическом NPZ."""
    import subprocess
    frames_dir = os.path.join(tmpdir, "frames_val")
    rs_path = os.path.join(tmpdir, "rs_val")
    N = 30
    fi = _write_metadata(frames_dir, N=N)
    ocr_path = os.path.join(rs_path, "ocr_extractor", "ocr.npz")
    _write_ocr_npz(ocr_path, frame_indices=fi, mode="ok")

    # Сначала запускаем process через run() чтобы получить реальный NPZ
    from modules.text_scoring.utils.text_scoring import TextScoringModule
    module = TextScoringModule(rs_path=rs_path)
    saved_path = module.run(frames_dir=frames_dir, config={})

    validate_script = os.path.join(
        THIS_DIR, "VisualProcessor", "modules", "text_scoring", "utils", "validate_text_scoring.py"
    )
    cmd = [sys.executable, validate_script, saved_path, "--struct", "--ranges"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=THIS_DIR)
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        return FAIL, f"rc={r.returncode} output={out[:300]}"
    return PASS, f"rc=0 | {out[:200]}"


# ============================================================
# Запуск
# ============================================================

def main():
    tmpdir = tempfile.mkdtemp(prefix="text_scoring_synth_")
    print(f"tmpdir: {tmpdir}\n")
    try:
        tests = [
            ("ok_path (U1+U2+U3+C1+C4)", test_ok_path),
            ("empty_no_ocr (U4a)", test_empty_no_ocr),
            ("empty_nodata (U4b)", test_empty_nodata),
            ("empty_out_of_fi (U4c)", test_empty_out_of_fi),
            ("golden_determinism (U5)", test_golden),
            ("different_lengths (U6)", test_different_lengths),
            ("fallback_motion_signal (bug fix)", test_fallback_motion_signal),
            ("validate_script (U1 full)", test_validate_script),
        ]
        total_pass = 0
        total_fail = 0
        for name, fn in tests:
            try:
                status, detail = fn(tmpdir)
            except Exception as e:
                status, detail = FAIL, f"EXCEPTION: {traceback.format_exc()}"
            icon = "✅" if status == PASS else "❌"
            print(f"{icon} [{status}] {name}")
            print(f"   {detail}")
            if status == PASS:
                total_pass += 1
            else:
                total_fail += 1

        print(f"\n{'='*60}")
        print(f"Итог: {total_pass}/{total_pass+total_fail} PASS, {total_fail} FAIL")
        return 0 if total_fail == 0 else 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
