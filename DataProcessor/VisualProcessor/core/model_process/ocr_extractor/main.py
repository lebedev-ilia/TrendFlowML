#!/usr/bin/env python3
"""
ocr_extractor (v1)

Run OCR over `text_region` crops from core_object_detections.

MVP engine: tesseract CLI via subprocess (no-network).
If tesseract is not installed -> write a valid empty artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_vp_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _vp_root not in sys.path:
    sys.path.append(_vp_root)

from utils.frame_manager import FrameManager  # type: ignore  # noqa: E402
from utils.logger import get_logger  # type: ignore  # noqa: E402
from utils.utilites import load_metadata  # type: ignore  # noqa: E402
from utils.meta_builder import apply_models_meta  # type: ignore  # noqa: E402

NAME = "ocr_extractor"
VERSION = "0.1"
SCHEMA_VERSION = "ocr_extractor_npz_v1"
LOGGER = get_logger(NAME)


def _load_npz(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise RuntimeError(f"{NAME} | required artifact not found: {path}")
    z = np.load(path, allow_pickle=True)
    out: Dict[str, Any] = {}
    for k in z.files:
        v = z[k]
        if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
            try:
                out[k] = v.item()
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


def _norm_text(s: str) -> str:
    s = str(s or "")
    s = s.lower()
    s = s.replace("\u200b", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _require_frame_indices(meta: dict) -> List[int]:
    block = meta.get("core_object_detections")
    if not isinstance(block, dict) or "frame_indices" not in block:
        raise RuntimeError(f"{NAME} | frames metadata missing core_object_detections.frame_indices (no-fallback)")
    frame_indices = block.get("frame_indices")
    if not isinstance(frame_indices, list) or not frame_indices:
        raise RuntimeError(f"{NAME} | core_object_detections.frame_indices empty/invalid (no-fallback)")
    return [int(x) for x in frame_indices]


def _class_id_map(class_names_arr: Any) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if class_names_arr is None:
        return out
    for s in np.asarray(class_names_arr).tolist():
        try:
            ss = str(s)
            k, v = ss.split(":", 1)
            out[int(k)] = str(v)
        except Exception:
            continue
    return out


def _crop_rgb(frame_rgb: np.ndarray, xyxy: np.ndarray, margin_frac: float) -> Optional[np.ndarray]:
    if frame_rgb is None:
        return None
    h, w = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
    x1, y1, x2, y2 = [float(v) for v in xyxy[:4].tolist()]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    mx = float(margin_frac) * bw
    my = float(margin_frac) * bh
    x1 = max(0.0, x1 - mx)
    y1 = max(0.0, y1 - my)
    x2 = min(float(w - 1), x2 + mx)
    y2 = min(float(h - 1), y2 + my)
    ix1, iy1, ix2, iy2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    if ix2 <= ix1 + 1 or iy2 <= iy1 + 1:
        return None
    return frame_rgb[iy1:iy2, ix1:ix2, :].copy()


def _append_state_event_if_possible(*, rs_path: str, event: Dict[str, Any]) -> None:
    """
    Best-effort writer for `state_events.jsonl` (backend tails this file).
    """
    try:
        run_rs = Path(rs_path).resolve()
        rs_base = run_rs.parents[2]  # <rs_base>/<platform>/<video>/<run>
        runs_root = rs_base.parent
        platform_id = str(event.get("platform_id") or "")
        video_id = str(event.get("video_id") or "")
        run_id = str(event.get("run_id") or "")
        if not (platform_id and video_id and run_id):
            return
        p = runs_root / "state" / platform_id / video_id / run_id / "state_events.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        event["platform_id"] = platform_id
        event["video_id"] = video_id
        event["run_id"] = run_id
        line = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
        with open(p, "ab") as f:
            f.write(line)
    except Exception:
        return


def _emit_stage(*, rs_path: str, platform_id: str, video_id: str, run_id: str, stage: str) -> None:
    """Emit stage event to state_events.jsonl."""
    _append_state_event_if_possible(
        rs_path=rs_path,
        event={
            "ts": datetime.utcnow().isoformat() + "Z",
            "scope": "progress",
            "processor": "visual",
            "component": NAME,
            "status": "running",
            "stage": str(stage),
            "platform_id": platform_id,
            "video_id": video_id,
            "run_id": run_id,
        },
    )


def _emit_progress(
    *,
    rs_path: str,
    platform_id: str,
    video_id: str,
    run_id: str,
    done: int,
    total: int,
    stage: str,
) -> None:
    """Emit progress event to state_events.jsonl."""
    if total <= 0:
        return
    progress = float(done) / float(total)
    _append_state_event_if_possible(
        rs_path=rs_path,
        event={
            "ts": datetime.utcnow().isoformat() + "Z",
            "scope": "progress",
            "processor": "visual",
            "component": NAME,
            "status": "running",
            "progress": progress,
            "done": int(done),
            "total": int(total),
            "stage": str(stage),
            "platform_id": platform_id,
            "video_id": video_id,
            "run_id": run_id,
        },
    )


def atomic_save_npz(path: str, **kwargs) -> None:
    """
    Атомарно сохраняет np.savez_compressed через временный файл.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # IMPORTANT: tmp must have .npz suffix, otherwise numpy will write to tmp + ".npz"
    # leaving tmp empty and corrupting the final artifact on os.replace().
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".npz", dir=os.path.dirname(path))
    os.close(fd)
    try:
        np.savez_compressed(tmp, **kwargs)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise


def _run_tesseract(crop_rgb_uint8: np.ndarray, *, lang: str, psm: int) -> str:
    """
    Returns raw text (may be empty).
    """
    from PIL import Image  # type: ignore

    img = Image.fromarray(crop_rgb_uint8)
    img = img.convert("L")  # grayscale
    # upscale small crops for better OCR
    w, h = img.size
    if max(w, h) < 200:
        scale = 2
        img = img.resize((w * scale, h * scale), resample=Image.BICUBIC)

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "crop.png")
        out_base = os.path.join(td, "out")
        img.save(in_path, format="PNG")
        cmd = ["tesseract", in_path, out_base, "-l", str(lang), "--psm", str(int(psm))]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        txt_path = out_base + ".txt"
        if not os.path.isfile(txt_path):
            return ""
        try:
            return open(txt_path, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            return ""


def main() -> int:
    ap = argparse.ArgumentParser(NAME)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--rs-path", required=True)
    ap.add_argument("--proposal-class", default="text_region")
    ap.add_argument("--min-det-score", type=float, default=0.5)
    ap.add_argument("--max-boxes-per-frame", type=int, default=5)
    ap.add_argument("--max-total-boxes", type=int, default=5000)
    ap.add_argument("--crop-margin-frac", type=float, default=0.02)
    ap.add_argument("--tesseract-lang", default="eng+rus")
    ap.add_argument("--tesseract-psm", type=int, default=6)
    args = ap.parse_args()

    # Initialize timing dictionary
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    meta = load_metadata(os.path.join(args.frames_dir, "metadata.json"), NAME)
    
    # Extract run identity for state_events
    platform_id = str(meta.get("platform_id") or "")
    video_id = str(meta.get("video_id") or "")
    run_id = str(meta.get("run_id") or "")

    # Baseline contract: emit start stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="start",
    )

    t_init = time.perf_counter()
    timings["initialization"] = t_init - t0

    frame_indices = _require_frame_indices(meta)

    uts = meta.get("union_timestamps_sec")
    if uts is None:
        raise RuntimeError(f"{NAME} | metadata.json missing union_timestamps_sec (contract)")
    uts_arr = np.asarray(uts, dtype=np.float32).reshape(-1)
    fi_np = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    if fi_np.size == 0:
        raise RuntimeError(f"{NAME} | frame_indices is empty (no-fallback)")
    if np.any(fi_np < 0) or np.any(fi_np >= int(uts_arr.shape[0])):
        raise RuntimeError(f"{NAME} | frame_indices out of range for union_timestamps_sec")
    times_s = uts_arr[fi_np].astype(np.float32)

    out_dir = os.path.join(str(args.rs_path), NAME)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ocr.npz")

    # Baseline contract: emit load_deps stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="load_deps",
    )

    t_load_deps = time.perf_counter()
    timings["load_deps"] = t_load_deps - t_init

    # If OCR engine is missing -> valid empty artifact
    if shutil.which("tesseract") is None:
        timings["saving"] = 0.0
        timings["total"] = time.perf_counter() - t0
        stage_timings_ms: Dict[str, float] = {}
        for key, value in timings.items():
            stage_timings_ms[key] = float(value) * 1000.0

        meta_out = {
            "producer": NAME,
            "producer_version": VERSION,
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "status": "empty",
            "empty_reason": "dependency_missing",
            "platform_id": meta.get("platform_id") or "unknown",
            "video_id": meta.get("video_id") or "unknown",
            "run_id": meta.get("run_id") or "unknown",
            "config_hash": meta.get("config_hash") or "unknown",
            "sampling_policy_version": meta.get("sampling_policy_version") or "unknown",
            "dataprocessor_version": meta.get("dataprocessor_version") or "unknown",
            "engine": "tesseract",
            "tesseract_lang": str(args.tesseract_lang),
            "tesseract_psm": int(args.tesseract_psm),
            "models_used": [],
            "stage_timings_ms": stage_timings_ms,
        }
        meta_out = apply_models_meta(meta_out, models_used=meta_out.get("models_used"))
        
        # Baseline contract: emit save stage
        _emit_stage(
            rs_path=args.rs_path,
            platform_id=platform_id,
            video_id=video_id,
            run_id=run_id,
            stage="save",
        )

        t_save_start = time.perf_counter()
        atomic_save_npz(
            out_path,
            frame_indices=np.asarray(frame_indices, dtype=np.int32),
            times_s=times_s,
            ocr_raw=np.asarray([], dtype=object),
            meta=np.asarray(meta_out, dtype=object),
        )
        timings["saving"] = time.perf_counter() - t_save_start
        timings["total"] = time.perf_counter() - t0

        # Update stage_timings_ms with final timings
        stage_timings_ms = {}
        for key, value in timings.items():
            stage_timings_ms[key] = float(value) * 1000.0
        meta_out["stage_timings_ms"] = stage_timings_ms

        # Validate artifact
        from utils.artifact_validator import validate_npz  # type: ignore

        ok, issues, _ = validate_npz(out_path)
        if not ok:
            error_messages = [f"{i.level}: {i.message}" for i in issues if i.level == "error"]
            os.remove(out_path)
            raise RuntimeError(f"{NAME} | Artifact validation failed: {', '.join(error_messages)}")

        # Baseline contract: emit done stage
        _emit_stage(
            rs_path=args.rs_path,
            platform_id=platform_id,
            video_id=video_id,
            run_id=run_id,
            stage="done",
        )

        LOGGER.warning("%s | tesseract not found; wrote empty artifact: %s", NAME, out_path)
        return 0

    # Load detections and find text_region class id
    det_path = os.path.join(str(args.rs_path), "core_object_detections", "detections.npz")
    det = _load_npz(det_path)
    fi_det = np.asarray(det.get("frame_indices"), dtype=np.int32).reshape(-1)
    if fi_det.size == 0:
        raise RuntimeError(f"{NAME} | core_object_detections.detections.npz missing frame_indices (no-fallback)")
    if fi_det.shape[0] != len(frame_indices) or not np.all(fi_det == np.asarray(frame_indices, dtype=np.int32)):
        raise RuntimeError(f"{NAME} | frame_indices mismatch vs core_object_detections (no-fallback)")

    boxes = np.asarray(det.get("boxes"), dtype=np.float32)  # (N,MAX,4)
    scores = np.asarray(det.get("scores"), dtype=np.float32)  # (N,MAX)
    class_ids = np.asarray(det.get("class_ids"), dtype=np.int32)  # (N,MAX)
    valid_mask = np.asarray(det.get("valid_mask"))  # (N,MAX)
    class_id_to_name = _class_id_map(det.get("class_names"))
    proposal = str(args.proposal_class).strip()
    proposal_ids = {cid for cid, nm in class_id_to_name.items() if nm == proposal}
    if not proposal_ids and class_id_to_name:
        LOGGER.warning("%s | proposal class '%s' not found in detections taxonomy; OCR will be empty.", NAME, proposal)

    # Baseline contract: emit process_frames stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="process_frames",
    )

    t_process_start = time.perf_counter()

    fm = FrameManager(frames_dir=str(args.frames_dir), chunk_size=meta.get("chunk_size", 32), cache_size=2)
    ocr_rows: List[Dict[str, Any]] = []
    try:
        total = 0
        N, MAX = int(boxes.shape[0]), int(boxes.shape[1])
        
        # Progress callback
        def progress_cb(done: int, total_frames: int) -> None:
            _emit_progress(
                rs_path=args.rs_path,
                platform_id=platform_id,
                video_id=video_id,
                run_id=run_id,
                done=done,
                total=total_frames,
                stage="process_frames",
            )

        for n_i in range(N):
            if total >= int(args.max_total_boxes):
                break
            # select candidate boxes in this frame
            cand: List[Tuple[float, int]] = []  # (score_area, j)
            for j in range(MAX):
                if not bool(valid_mask[n_i, j]):
                    continue
                sc = float(scores[n_i, j])
                if sc < float(args.min_det_score):
                    continue
                if proposal_ids and int(class_ids[n_i, j]) not in proposal_ids:
                    continue
                xyxy = boxes[n_i, j]
                area = max(1.0, float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])))
                cand.append((sc * area, j))
            if not cand:
                continue
            cand.sort(reverse=True)
            cand = cand[: int(args.max_boxes_per_frame)]
            fr_idx = int(frame_indices[n_i])
            t = float(times_s[n_i])
            frame_rgb = fm.get(fr_idx)
            for _, j in cand:
                if total >= int(args.max_total_boxes):
                    break
                crop = _crop_rgb(frame_rgb, boxes[n_i, j], margin_frac=float(args.crop_margin_frac))
                if crop is None:
                    continue
                txt = _run_tesseract(crop, lang=str(args.tesseract_lang), psm=int(args.tesseract_psm))
                txt_raw = str(txt or "").strip()
                txt_norm = _norm_text(txt_raw)
                if not txt_norm:
                    continue
                ocr_rows.append(
                    {
                        "frame": fr_idx,
                        "time_s": t,
                        "bbox": [float(x) for x in boxes[n_i, j].tolist()],
                        "text_raw": txt_raw,
                        "text_norm": txt_norm,
                        "det_confidence": float(scores[n_i, j]),
                        "engine": "tesseract",
                        "lang": str(args.tesseract_lang),
                    }
                )
                total += 1
            
            # Emit progress every 10% or at least 10 times
            if (n_i + 1) % max(1, N // 10) == 0 or n_i == N - 1:
                progress_cb(n_i + 1, N)
    finally:
        try:
            fm.close()
        except Exception:
            pass

    t_process_end = time.perf_counter()
    timings["process_frames"] = t_process_end - t_process_start

    # Baseline contract: stage_timings_ms in meta
    timings["saving"] = 0.0  # Will be updated after save
    timings["total"] = time.perf_counter() - t0
    stage_timings_ms: Dict[str, float] = {}
    for key, value in timings.items():
        stage_timings_ms[key] = float(value) * 1000.0

    meta_out = {
        "producer": NAME,
        "producer_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "status": "ok" if ocr_rows else "empty",
        "empty_reason": None if ocr_rows else "no_text_available",
        "platform_id": meta.get("platform_id") or "unknown",
        "video_id": meta.get("video_id") or "unknown",
        "run_id": meta.get("run_id") or "unknown",
        "config_hash": meta.get("config_hash") or "unknown",
        "sampling_policy_version": meta.get("sampling_policy_version") or "unknown",
        "dataprocessor_version": meta.get("dataprocessor_version") or "unknown",
        "engine": "tesseract",
        "tesseract_lang": str(args.tesseract_lang),
        "tesseract_psm": int(args.tesseract_psm),
        "proposal_class": proposal,
        "models_used": [],
        "stage_timings_ms": stage_timings_ms,
    }
    meta_out = apply_models_meta(meta_out, models_used=meta_out.get("models_used"))

    # Baseline contract: emit save stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="save",
    )

    t_save_start = time.perf_counter()

    atomic_save_npz(
        out_path,
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        times_s=times_s,
        ocr_raw=np.asarray(ocr_rows, dtype=object),
        meta=np.asarray(meta_out, dtype=object),
    )

    timings["saving"] = time.perf_counter() - t_save_start
    timings["total"] = time.perf_counter() - t0

    # Update stage_timings_ms with final timings
    stage_timings_ms = {}
    for key, value in timings.items():
        stage_timings_ms[key] = float(value) * 1000.0
    meta_out["stage_timings_ms"] = stage_timings_ms

    # Validate artifact
    from utils.artifact_validator import validate_npz  # type: ignore

    ok, issues, _ = validate_npz(out_path)
    if not ok:
        error_messages = [f"{i.level}: {i.message}" for i in issues if i.level == "error"]
        os.remove(out_path)
        raise RuntimeError(f"{NAME} | Artifact validation failed: {', '.join(error_messages)}")

    # Baseline contract: emit done stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="done",
    )

    LOGGER.info("%s | wrote %s (rows=%d)", NAME, out_path, len(ocr_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


