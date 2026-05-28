#!/usr/bin/env python3
"""
ocr_extractor (v1)

Run OCR over `text_region` crops from core_object_detections.

MVP engine: tesseract CLI via subprocess (no-network).
If tesseract is not installed -> write a valid empty artifact (status=empty, not error).
"""

from __future__ import annotations

import argparse
import hashlib
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

# VisualProcessor root: …/core/model_process/ocr_extractor/main.py -> parents[3]
_vp_root = str(Path(__file__).resolve().parents[3])
# Must be first: ocr_extractor/utils/ would shadow VisualProcessor/utils.
if _vp_root not in sys.path:
    sys.path.insert(0, _vp_root)
elif sys.path[0] != _vp_root:
    try:
        sys.path.remove(_vp_root)
    except ValueError:
        pass
    sys.path.insert(0, _vp_root)

from utils.frame_manager import FrameManager  # type: ignore  # noqa: E402
from utils.logger import get_logger  # type: ignore  # noqa: E402
from utils.utilites import load_metadata  # type: ignore  # noqa: E402
from utils.meta_builder import apply_models_meta  # type: ignore  # noqa: E402

NAME = "ocr_extractor"
VERSION = "0.2"
SCHEMA_VERSION = "ocr_extractor_npz_v2"
LOGGER = get_logger(NAME)


def _resource_profile_snapshot() -> Dict[str, Any]:
    """
    Best-effort resource snapshot for audit/profiling.
    Enabled only when VP_RESOURCE_PROFILE=1|true|yes.
    """
    v = str(os.environ.get("VP_RESOURCE_PROFILE") or "").strip().lower()
    if v not in ("1", "true", "yes", "y", "on"):
        return {}

    out: Dict[str, Any] = {}
    try:
        import psutil  # type: ignore

        p = psutil.Process(os.getpid())
        rss = int(getattr(p.memory_info(), "rss", 0) or 0)
        out["rss_bytes"] = rss
        out["rss_mib"] = float(rss) / (1024.0 * 1024.0)
    except Exception:
        pass

    try:
        import torch  # type: ignore

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                out["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                out["cuda_max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            except Exception:
                pass
    except Exception:
        pass

    return out


def _load_npz(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise RuntimeError(f"{NAME} | required artifact not found: {path}")
    z = np.load(path, allow_pickle=True)
    try:
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
    finally:
        try:
            z.close()
        except Exception:
            pass


def _norm_text(s: str) -> str:
    s = str(s or "")
    s = s.lower()
    s = s.replace("\u200b", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def _require_meta_str(meta: dict, key: str) -> str:
    v = meta.get(key)
    if v is None:
        raise RuntimeError(f"{NAME} | metadata.json missing required key '{key}' (no-fallback)")
    s = str(v).strip()
    if not s:
        raise RuntimeError(f"{NAME} | metadata.json has empty required key '{key}' (no-fallback)")
    return s


def _load_ppocr_dict(dict_path: str) -> List[str]:
    """
    Loads PP-OCR recognition dictionary: one character per line (without blank).
    CTC blank is assumed to be class 0.
    """
    if not os.path.isfile(dict_path):
        raise RuntimeError(f"{NAME} | OCR dict file not found: {dict_path}")
    chars: List[str] = []
    with open(dict_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ch = line.rstrip("\n\r")
            if ch == "":
                continue
            chars.append(ch)
    if not chars:
        raise RuntimeError(f"{NAME} | OCR dict is empty: {dict_path}")
    return chars


def _ppocr_preprocess(crop_rgb_uint8: np.ndarray, *, img_h: int, img_w: int) -> np.ndarray:
    """
    Preprocess crop for PP-OCR recognizer ONNX models.
    Output: float32 (1,3,H,W) in range ~[-1,1].
    """
    from PIL import Image  # type: ignore

    img_h = int(img_h)
    img_w = int(img_w)
    if img_h <= 0 or img_w <= 0:
        raise RuntimeError(f"{NAME} | invalid ppocr image shape: {(img_h, img_w)}")

    im = Image.fromarray(crop_rgb_uint8).convert("RGB")
    w, h = im.size
    if w <= 1 or h <= 1:
        return np.zeros((1, 3, img_h, img_w), dtype=np.float32)

    # Keep aspect ratio by scaling height to img_h and padding width to img_w.
    new_w = int(round(float(img_h) * float(w) / float(h)))
    new_w = max(1, min(new_w, img_w))
    im = im.resize((new_w, img_h), resample=Image.BICUBIC)

    # Pad to (img_w, img_h)
    canvas = Image.new("RGB", (img_w, img_h), color=(0, 0, 0))
    canvas.paste(im, (0, 0))

    x = np.asarray(canvas, dtype=np.float32)  # (H,W,3), RGB
    x = x / 255.0
    x = (x - 0.5) / 0.5  # [-1,1]
    x = np.transpose(x, (2, 0, 1))  # (3,H,W)
    x = x.reshape(1, 3, img_h, img_w).astype(np.float32)
    return x


def _ppocr_ctc_greedy_decode(logits: np.ndarray, chars: List[str]) -> Tuple[str, float]:
    """
    logits: (1,T,C) float32.
    Returns (text, confidence_proxy).
    """
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise RuntimeError(f"{NAME} | unexpected ppocr logits shape: {getattr(logits, 'shape', None)}")
    x = np.asarray(logits[0], dtype=np.float32)  # (T,C)
    # softmax
    x = x - np.max(x, axis=1, keepdims=True)
    expx = np.exp(x)
    probs = expx / np.maximum(1e-9, np.sum(expx, axis=1, keepdims=True))
    idx = np.argmax(probs, axis=1).astype(np.int32)  # (T,)
    conf = np.max(probs, axis=1).astype(np.float32)  # (T,)

    out_chars: List[str] = []
    out_confs: List[float] = []
    prev = -1
    for t in range(int(idx.shape[0])):
        i = int(idx[t])
        if i == 0:
            prev = i
            continue
        if i == prev:
            continue
        j = i - 1
        if 0 <= j < len(chars):
            out_chars.append(chars[j])
            out_confs.append(float(conf[t]))
        prev = i

    text = "".join(out_chars).strip()
    if not out_confs:
        return text, 0.0
    return text, float(sum(out_confs) / max(1, len(out_confs)))


def _run_ppocr_rec_onnx(
    crop_rgb_uint8: np.ndarray,
    *,
    session: Any,
    chars: List[str],
    img_h: int,
    img_w: int,
) -> Tuple[str, float]:
    x = _ppocr_preprocess(crop_rgb_uint8, img_h=img_h, img_w=img_w)
    input_name = session.get_inputs()[0].name
    outs = session.run(None, {input_name: x})
    if not outs:
        return "", 0.0
    logits = np.asarray(outs[0])
    return _ppocr_ctc_greedy_decode(logits, chars)


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
    ap.add_argument(
        "--engine",
        default="tesseract",
        help="OCR engine: tesseract | ppocr_rec_onnx. Recommended: ppocr_rec_onnx (better quality, offline via dp_models).",
    )
    ap.add_argument(
        "--rec-model-spec",
        default="ppocr_rec_onnx_v1_inprocess",
        help="dp_models ModelManager spec name for OCR recognizer (used for engine=ppocr_rec_onnx).",
    )
    ap.add_argument("--ppocr-img-h", type=int, default=48)
    ap.add_argument("--ppocr-img-w", type=int, default=320)
    ap.add_argument("--min-rec-score", type=float, default=0.0)
    ap.add_argument("--proposal-class", default="text_region")
    ap.add_argument("--min-det-score", type=float, default=0.5)
    ap.add_argument("--max-boxes-per-frame", type=int, default=5)
    ap.add_argument("--max-total-boxes", type=int, default=5000)
    ap.add_argument("--crop-margin-frac", type=float, default=0.02)
    ap.add_argument("--tesseract-lang", default="eng+rus")
    ap.add_argument("--tesseract-psm", type=int, default=6)
    ap.add_argument(
        "--retain-raw-ocr-text",
        action="store_true",
        help="If set, store raw OCR text in the artifact (dev/debug). Default: do NOT retain raw OCR text.",
    )
    args = ap.parse_args()

    # Initialize timing dictionary
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    meta = load_metadata(os.path.join(args.frames_dir, "metadata.json"), NAME)
    
    # Extract run identity for state_events
    platform_id = _require_meta_str(meta, "platform_id")
    video_id = _require_meta_str(meta, "video_id")
    run_id = _require_meta_str(meta, "run_id")
    config_hash = _require_meta_str(meta, "config_hash")
    sampling_policy_version = _require_meta_str(meta, "sampling_policy_version")
    dataprocessor_version = _require_meta_str(meta, "dataprocessor_version")

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

    engine = str(args.engine or "tesseract").strip().lower()
    if engine not in ("tesseract", "ppocr_rec_onnx"):
        raise RuntimeError(f"{NAME} | unsupported engine={engine}. Expected: tesseract|ppocr_rec_onnx")

    # Engine init (may load local model via ModelManager; no-network)
    rec_session = None
    rec_chars: List[str] = []
    rec_models_used: List[Dict[str, Any]] = []
    skip_ocr_processing = False
    skip_ocr_reason: Optional[str] = None
    if engine == "tesseract":
        if shutil.which("tesseract") is None:
            LOGGER.warning(
                "%s | tesseract not in PATH — emitting empty OCR artifact (install tesseract or use engine=ppocr_rec_onnx).",
                NAME,
            )
            skip_ocr_processing = True
            skip_ocr_reason = "tesseract_not_in_path"
    elif engine == "ppocr_rec_onnx":
        # Quieter ONNX Runtime stderr (e.g. graph.CleanUnusedInitializers warnings). Env alone is
        # unreliable if ORT was imported early; force level and set API before ModelManager loads ONNX.
        os.environ["ORT_LOG_SEVERITY_LEVEL"] = "4"  # 4=FATAL in ORT Python bindings
        try:
            import onnxruntime as _ort  # type: ignore

            _ort.set_default_logger_severity(4)
        except Exception:
            pass
        try:
            # Ensure DataProcessor root is importable for dp_models
            dp_root = Path(__file__).resolve().parents[4]
            if str(dp_root) not in sys.path:
                sys.path.insert(0, str(dp_root))
            from dp_models import get_global_model_manager  # type: ignore
        except Exception as e:
            raise RuntimeError(f"{NAME} | dp_models is required for engine=ppocr_rec_onnx: {e}") from e

        # IMPORTANT: VisualProcessor sets DP_MODELS_ROOT to ".../dp_models/bundled_models" for assets/caches.
        # ModelManager specs in this repo use local_artifacts like "bundled_models/...".
        # Therefore ModelManager models_root must be the parent dir ".../dp_models".
        try:
            env_root = str(os.environ.get("DP_MODELS_ROOT") or "").strip()
            if env_root and os.path.basename(env_root) == "bundled_models":
                os.environ["DP_MODELS_ROOT"] = os.path.abspath(os.path.join(env_root, os.pardir))
        except Exception:
            pass

        mm = get_global_model_manager()
        rm = mm.get(model_name=str(args.rec_model_spec))
        handle = rm.handle or {}
        rec_session = handle.get("session")
        if rec_session is None:
            raise RuntimeError(f"{NAME} | ModelManager returned empty session handle for {args.rec_model_spec}")

        dict_path = None
        for rel, abs_path in (rm.resolved_artifacts or {}).items():
            if str(rel).lower().endswith("dict.txt"):
                dict_path = abs_path
                break
        if dict_path is None:
            for _rel, abs_path in (rm.resolved_artifacts or {}).items():
                if str(abs_path).lower().endswith(".txt"):
                    dict_path = abs_path
                    break
        if dict_path is None:
            raise RuntimeError(f"{NAME} | OCR dict artifact not found in model spec {args.rec_model_spec}")

        rec_chars = _load_ppocr_dict(str(dict_path))
        rec_models_used = [rm.models_used_entry]

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

    ocr_rows: List[Dict[str, Any]] = []
    fm: Optional[FrameManager] = None
    try:
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

        if skip_ocr_processing:
            N_skip = int(boxes.shape[0])
            if N_skip > 0:
                progress_cb(N_skip, N_skip)
        else:
            fm = FrameManager(frames_dir=str(args.frames_dir), chunk_size=meta.get("chunk_size", 32), cache_size=2)
            total = 0
            N, MAX = int(boxes.shape[0]), int(boxes.shape[1])
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
                assert fm is not None
                frame_rgb = fm.get(fr_idx)
                for _, j in cand:
                    if total >= int(args.max_total_boxes):
                        break
                    crop = _crop_rgb(frame_rgb, boxes[n_i, j], margin_frac=float(args.crop_margin_frac))
                    if crop is None:
                        continue
                    rec_conf = None
                    if engine == "tesseract":
                        txt = _run_tesseract(crop, lang=str(args.tesseract_lang), psm=int(args.tesseract_psm))
                        txt_raw = str(txt or "").strip()
                    else:
                        if rec_session is None:
                            raise RuntimeError(f"{NAME} | ppocr_rec_onnx session not initialized")
                        txt_raw, rc = _run_ppocr_rec_onnx(
                            crop,
                            session=rec_session,
                            chars=rec_chars,
                            img_h=int(args.ppocr_img_h),
                            img_w=int(args.ppocr_img_w),
                        )
                        rec_conf = float(rc)
                        if rec_conf < float(args.min_rec_score):
                            continue
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
                            "engine": engine,
                            "lang": (str(args.tesseract_lang) if engine == "tesseract" else None),
                            "rec_confidence": rec_conf,
                        }
                    )
                    total += 1

                # Emit progress every 10% or at least 10 times
                if (n_i + 1) % max(1, N // 10) == 0 or n_i == N - 1:
                    progress_cb(n_i + 1, N)
    finally:
        if fm is not None:
            try:
                fm.close()
            except Exception:
                pass

    t_process_end = time.perf_counter()
    timings["process_frames"] = t_process_end - t_process_start

    # Privacy: optionally redact OCR text before saving
    retain_raw_ocr_text = bool(args.retain_raw_ocr_text)
    if not retain_raw_ocr_text:
        redacted: List[Dict[str, Any]] = []
        for r in ocr_rows:
            if not isinstance(r, dict):
                continue
            txt_norm = str(r.get("text_norm") or _norm_text(str(r.get("text_raw") or "")))
            rr = {k: v for k, v in r.items() if k not in ("text_raw", "text_norm")}
            rr["text_sha256"] = _sha256_text(txt_norm)
            rr["text_len"] = int(len(txt_norm))
            redacted.append(rr)
        ocr_rows = redacted

    _empty_reason_meta = None if ocr_rows else (skip_ocr_reason or "no_text_available")
    meta_out = {
        "producer": NAME,
        "producer_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "status": "ok" if ocr_rows else "empty",
        "empty_reason": _empty_reason_meta,
        "platform_id": platform_id,
        "video_id": video_id,
        "run_id": run_id,
        "config_hash": config_hash,
        "sampling_policy_version": sampling_policy_version,
        "dataprocessor_version": dataprocessor_version,
        "engine": engine,
        "tesseract_lang": str(args.tesseract_lang),
        "tesseract_psm": int(args.tesseract_psm),
        "proposal_class": proposal,
        "retain_raw_ocr_text": retain_raw_ocr_text,
        "rec_model_spec": (str(args.rec_model_spec) if engine != "tesseract" else None),
        "ppocr_img_h": (int(args.ppocr_img_h) if engine != "tesseract" else None),
        "ppocr_img_w": (int(args.ppocr_img_w) if engine != "tesseract" else None),
        "models_used": rec_models_used,
    }

    # Baseline contract: stage timings in meta (two-pass save to include saving time)
    timings["saving"] = 0.0
    timings["total"] = time.perf_counter() - t0
    meta_out["stage_timings_ms"] = {k: float(v) * 1000.0 for k, v in timings.items()}
    rp_before = _resource_profile_snapshot()
    if isinstance(rp_before, dict) and rp_before:
        meta_out["resource_profile_before"] = dict(rp_before)
    meta_out = apply_models_meta(meta_out, models_used=meta_out.get("models_used"))
    meta_json = json.dumps(meta_out, ensure_ascii=False, sort_keys=True)

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
        meta_json=np.asarray(meta_json),
        meta=np.asarray(meta_out, dtype=object),
    )

    timings["saving"] = time.perf_counter() - t_save_start
    timings["total"] = time.perf_counter() - t0

    # Two-pass: rewrite artifact with final stage timings and meta_json
    meta_out["stage_timings_ms"] = {k: float(v) * 1000.0 for k, v in timings.items()}
    meta_json = json.dumps(meta_out, ensure_ascii=False, sort_keys=True)
    atomic_save_npz(
        out_path,
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        times_s=times_s,
        ocr_raw=np.asarray(ocr_rows, dtype=object),
        meta_json=np.asarray(meta_json),
        meta=np.asarray(meta_out, dtype=object),
    )

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


