#!/usr/bin/env python3
"""
Production-ready object detection + tracking extractor.

Поддерживает:
- YOLO (ultralytics)

Выход:
- detections.npz с фиксированными numpy-массивами:
    - boxes       (N_frames, MAX_DETECTIONS, 4)  float32 (x1,y1,x2,y2)
    - scores      (N_frames, MAX_DETECTIONS)     float32
    - class_ids   (N_frames, MAX_DETECTIONS)     int32
    - valid_mask  (N_frames, MAX_DETECTIONS)     bool
    - class_names (M,)                            unicode array "id:name"
    - metadata fields...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

import cv2
import numpy as np

_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _path not in sys.path:
    sys.path.append(_path)

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.resource_probe import pick_device
from utils.utilites import load_metadata
from utils.meta_builder import apply_models_meta, model_used

NAME = "core_object_detections"
VERSION = "2.1"
SCHEMA_VERSION = "core_object_detections_npz_v1"
LOGGER = get_logger(NAME)

MAX_DETECTIONS = 100
BBOX_DIMS = 4

# Final taxonomy v1.0 (41 classes) - source of truth for baseline and production.
# See: yolo_fine_tune/YOLO_CLASSES_V1_FINAL.md
_FINAL_TAXONOMY_V1_CLASSES = [
    "person",
    "crowd",
    "car",
    "motorcycle",
    "bicycle",
    "bus",
    "truck",
    "pet",
    "sports_ball",
    "phone",
    "laptop",
    "tablet",
    "smartwatch",
    "watch",
    "headphones",
    "camera",
    "microphone",
    "game_controller",
    "tv_device",
    "monitor_device",
    "clothing_top",
    "clothing_bottom",
    "outerwear",
    "suit",
    "dress",
    "shoes",
    "bag",
    "hat",
    "glasses",
    "ring",
    "bracelet",
    "earrings",
    "pendant",
    "logo_region",
    "text_region",
    "cosmetics_product",
    "screen_phone",
    "screen_laptop",
    "screen_monitor",
    "tv_screen",
    "food_item",
]

# COCO-80 class names (legacy fallback, deprecated in favor of _FINAL_TAXONOMY_V1_CLASSES).
# Kept for backward compatibility only.
_COCO80_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def _load_final_taxonomy_v1_classes() -> Dict[int, str]:
    """
    Загружает финальный набор классов (v1.0, 41 класс) из файла.
    
    Источник правды: yolo_fine_tune/labels_v1_40.txt
    Документация: yolo_fine_tune/YOLO_CLASSES_V1_FINAL.md
    
    Returns:
        Dict[int, str]: маппинг class_id -> class_name для ID 0..40
    """
    taxonomy_file = os.path.join(os.path.dirname(__file__), "DETECTOR_TAXONOMY_V1_40_NAMES.txt")
    
    # Fallback to hardcoded list if file not found
    if not os.path.exists(taxonomy_file):
        LOGGER.warning(
            "%s | taxonomy file not found: %s, using hardcoded final taxonomy v1.0",
            NAME,
            taxonomy_file,
        )
        return {i: name for i, name in enumerate(_FINAL_TAXONOMY_V1_CLASSES)}
    
    try:
        with open(taxonomy_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if len(lines) != 41:
            LOGGER.warning(
                "%s | taxonomy file has %d classes (expected 41), using hardcoded list",
                NAME,
                len(lines),
            )
            return {i: name for i, name in enumerate(_FINAL_TAXONOMY_V1_CLASSES)}
        
        return {i: name for i, name in enumerate(lines)}
    except Exception as e:
        LOGGER.warning(
            "%s | failed to load taxonomy file: %s, using hardcoded list. Error: %s",
            NAME,
            taxonomy_file,
            e,
        )
        return {i: name for i, name in enumerate(_FINAL_TAXONOMY_V1_CLASSES)}


def _load_triton_spec_via_model_manager(model_spec_name: str) -> dict:
    """
    Resolve Triton model spec via dp_models.ModelManager (no-network, reproducible).
    Returns dict with keys:
      - client: TritonHttpClient
      - rp: runtime_params
      - models_used_entry: dict (model_used)
    """
    from dp_models import get_global_model_manager  # type: ignore

    mm = get_global_model_manager()
    rm = mm.get(model_name=str(model_spec_name))
    rp = rm.spec.runtime_params or {}
    handle = rm.handle or {}
    client = None
    if isinstance(handle, dict):
        client = handle.get("client")
    if client is None:
        raise RuntimeError(f"{NAME} | ModelManager returned empty Triton client handle for: {model_spec_name}")
    if not isinstance(rp, dict) or not rp:
        raise RuntimeError(f"{NAME} | ModelManager returned empty runtime_params for: {model_spec_name}")
    return {"client": client, "rp": rp, "models_used_entry": rm.models_used_entry}


def _letterbox_bgr_no_upscale(
    img_bgr: np.ndarray,
    *,
    new_size: int,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Ultralytics-style letterbox to square (new_size x new_size), but with NO UPSCALE.

    Returns:
      - img_lb: (new_size, new_size, 3) uint8 BGR
      - r: resize ratio applied (<=1.0)
      - pad: (left, top) padding applied in pixels
    """
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"{NAME} | invalid image shape: {img_bgr.shape}")
    h0, w0 = int(img_bgr.shape[0]), int(img_bgr.shape[1])
    s = int(new_size)
    if s <= 0:
        raise ValueError(f"{NAME} | invalid new_size={new_size}")

    r = min(float(s) / float(h0), float(s) / float(w0))
    r = min(r, 1.0)  # no upscale

    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    if (new_w, new_h) != (w0, h0):
        img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        img = img_bgr

    dw = s - new_w
    dh = s - new_h
    left = int(round(dw / 2.0 - 0.1))
    right = int(round(dw / 2.0 + 0.1))
    top = int(round(dh / 2.0 - 0.1))
    bottom = int(round(dh / 2.0 + 0.1))

    img_lb = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    if img_lb.shape[0] != s or img_lb.shape[1] != s:
        # Defensive: ensure exact shape
        img_lb = cv2.resize(img_lb, (s, s), interpolation=cv2.INTER_LINEAR)
        left, top = 0, 0
        r = float(s) / float(max(h0, w0))
        r = min(r, 1.0)

    return img_lb, float(r), (int(left), int(top))


def _prep_yolo_tensor_from_rgb_uint8(
    frame_rgb_uint8: np.ndarray,
    *,
    input_size: int,
) -> Tuple[np.ndarray, float, Tuple[int, int], Tuple[int, int]]:
    """
    Preprocess one RGB uint8 frame for Ultralytics-exported YOLO ONNX:
      RGB -> BGR -> letterbox(no-upscale) -> RGB -> FP32 /255 -> NCHW

    Returns:
      - x: (1,3,S,S) float32
      - r: resize ratio
      - pad: (left, top)
      - orig_hw: (h0, w0)
    """
    bgr = cv2.cvtColor(frame_rgb_uint8, cv2.COLOR_RGB2BGR)
    img_lb, r, pad = _letterbox_bgr_no_upscale(bgr, new_size=int(input_size))
    rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]  # (1,3,S,S)
    return x.astype(np.float32), float(r), (int(pad[0]), int(pad[1])), (int(frame_rgb_uint8.shape[0]), int(frame_rgb_uint8.shape[1]))


def _scale_boxes_back(
    boxes_xyxy: np.ndarray,
    *,
    r: float,
    pad: Tuple[int, int],
    orig_hw: Tuple[int, int],
) -> np.ndarray:
    """
    Reverse letterbox: boxes in letterboxed image coords -> original image coords.
    """
    if boxes_xyxy.size == 0:
        return boxes_xyxy.astype(np.float32)
    left, top = int(pad[0]), int(pad[1])
    b = boxes_xyxy.astype(np.float32).copy()
    b[:, [0, 2]] -= float(left)
    b[:, [1, 3]] -= float(top)
    rr = float(max(r, 1e-9))
    b[:, :4] /= rr
    h0, w0 = int(orig_hw[0]), int(orig_hw[1])
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0.0, float(w0 - 1))
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0.0, float(h0 - 1))
    return b


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    IOU между bbox a и b в формате [x1,y1,x2,y2]
    a: (4,), b: (N,4)  -> returns (N,)
    """
    ax1, ay1, ax2, ay2 = a
    bx1 = b[:, 0]
    by1 = b[:, 1]
    bx2 = b[:, 2]
    by2 = b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = np.maximum(0.0, (bx2 - bx1)) * np.maximum(0.0, (by2 - by1))

    union = area_a + area_b - inter_area + 1e-12
    return inter_area / union


def run_yolo(
    frame_manager: FrameManager,
    frame_indices: List[int],
    model_path: str,
    box_threshold: float,
    batch_size: int,
    device: Optional[str] = None,
    progress_callback: Optional[callable] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str], List[np.ndarray]]:
    """
    Запускает ультраликтикс YOLO на батчах и возвращает фиксированные тензоры + raw detections per frame.

    Возвращает:
        boxes (N, MAX_DETECTIONS, 4),
        scores (N, MAX_DETECTIONS),
        class_ids (N, MAX_DETECTIONS),
        valid_mask (N, MAX_DETECTIONS),
        class_names dict,
        raw_per_frame: list length N, each element - ndarray (M,5) [x1,y1,x2,y2,score]
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        LOGGER.exception("YOLO import failed: %s", e)
        raise

    LOGGER.info("%s | YOLO | loading model: %s", NAME, model_path)
    model = YOLO(model_path)

    n = len(frame_indices)
    boxes = np.zeros((n, MAX_DETECTIONS, BBOX_DIMS), dtype=np.float32)
    scores = np.zeros((n, MAX_DETECTIONS), dtype=np.float32)
    class_ids = np.zeros((n, MAX_DETECTIONS), dtype=np.int32)
    valid_mask = np.zeros((n, MAX_DETECTIONS), dtype=bool)
    class_names: Dict[int, str] = {}

    raw_per_frame: List[np.ndarray] = [np.zeros((0, 5), dtype=np.float32) for _ in frame_indices]

    for start in range(0, n, batch_size):
        batch_idx = frame_indices[start : start + batch_size]
        # FrameManager.get() returns RGB; ultralytics accepts numpy images, but many CV pipelines assume BGR.
        # Convert RGB->BGR for stability with OpenCV-based preprocessing.
        batch_frames = [cv2.cvtColor(frame_manager.get(i), cv2.COLOR_RGB2BGR) for i in batch_idx]

        results = model(batch_frames, device=device, verbose=False)

        for i_local, res in enumerate(results):
            out_i = start + i_local
            # res.boxes may be empty
            if res.boxes is None or len(res.boxes) == 0:
                continue

            detections = []
            # iterate boxes
            for j in range(min(len(res.boxes), MAX_DETECTIONS)):
                try:
                    conf = float(res.boxes.conf[j].item())
                    if conf < box_threshold:
                        continue
                    xyxy = res.boxes.xyxy[j].cpu().numpy().astype(np.float32)
                    cls_id = int(res.boxes.cls[j].item())
                except Exception:
                    # fallback if API shape differs
                    try:
                        box = res.boxes.data[j].cpu().numpy()
                        xyxy = box[:4].astype(np.float32)
                        conf = float(box[4])
                        cls_id = int(box[5]) if box.shape[0] > 5 else 0
                        if conf < box_threshold:
                            continue
                    except Exception:
                        raise RuntimeError(f"{NAME} | YOLO | cannot parse detection output (ultralytics API drift?)")

                boxes[out_i, j] = xyxy
                scores[out_i, j] = conf
                class_ids[out_i, j] = cls_id
                valid_mask[out_i, j] = True
                detections.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf])

                if cls_id not in class_names:
                    try:
                        class_names[cls_id] = res.names.get(cls_id, f"class_{cls_id}")
                    except Exception:
                        class_names[cls_id] = f"class_{cls_id}"

            raw_per_frame[out_i] = np.array(detections, dtype=np.float32)

        processed = min(start + batch_size, n)
        LOGGER.info("%s | YOLO | processed %d/%d", NAME, processed, n)
        if progress_callback:
            progress_callback(done=processed, total=n)

    return boxes, scores, class_ids, valid_mask, class_names, raw_per_frame


def run_yolo_triton(
    *,
    frame_manager: FrameManager,
    frame_indices: List[int],
    triton_client,
    triton_model_name: str,
    triton_model_version: Optional[str],
    triton_input_name: str,
    triton_output_name: str,
    input_size: int,
    box_threshold: float,
    iou_threshold: float,
    class_names: Dict[int, str],
    progress_callback: Optional[callable] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str], List[np.ndarray]]:
    """
    Triton-backed YOLO inference.

    Notes:
    - Fixed-shape baseline models are batch=1, so we process frames one-by-one.
    - We implement NMS locally to avoid ultralytics API drift.
    """
    def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
        # xywh: (N,4) with center x,y,w,h
        x = xywh[:, 0]
        y = xywh[:, 1]
        w = xywh[:, 2]
        h = xywh[:, 3]
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    def _nms_single_class(boxes_xyxy: np.ndarray, scores_: np.ndarray, iou_th: float, max_det: int) -> List[int]:
        if boxes_xyxy.size == 0:
            return []
        order = scores_.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0 and len(keep) < int(max_det):
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            ious = iou_xyxy(boxes_xyxy[i], boxes_xyxy[rest])
            order = rest[ious <= float(iou_th)]
        return keep

    def _decode_and_nms_yolo(out_b84n: np.ndarray) -> np.ndarray:
        """
        out_b84n: (1,84,N)
        returns det: (M,6) [x1,y1,x2,y2,conf,cls_id]
        """
        if out_b84n.ndim != 3 or out_b84n.shape[0] != 1 or out_b84n.shape[1] < 6:
            raise RuntimeError(f"{NAME} | unexpected YOLO output shape: {out_b84n.shape}")
        pred = out_b84n[0].T  # (N,84)
        boxes_xywh = pred[:, :4].astype(np.float32)
        cls_scores = pred[:, 4:].astype(np.float32)  # (N,nc)
        cls_id = np.argmax(cls_scores, axis=1).astype(np.int32)
        conf = np.max(cls_scores, axis=1).astype(np.float32)
        m = conf >= float(box_threshold)
        if not np.any(m):
            return np.zeros((0, 6), dtype=np.float32)
        boxes_xyxy = _xywh_to_xyxy(boxes_xywh[m])
        conf_m = conf[m]
        cls_m = cls_id[m]

        dets: List[np.ndarray] = []
        for c in np.unique(cls_m):
            idx = np.where(cls_m == c)[0]
            if idx.size == 0:
                continue
            keep = _nms_single_class(boxes_xyxy[idx], conf_m[idx], float(iou_threshold), int(MAX_DETECTIONS))
            if not keep:
                continue
            kk = idx[np.asarray(keep, dtype=np.int64)]
            cc = np.full((kk.size, 1), float(c), dtype=np.float32)
            dets.append(np.concatenate([boxes_xyxy[kk], conf_m[kk, None], cc], axis=1))

        if not dets:
            return np.zeros((0, 6), dtype=np.float32)
        det = np.concatenate(dets, axis=0)
        # global top-k by conf
        order = det[:, 4].argsort()[::-1]
        det = det[order[: int(MAX_DETECTIONS)]]
        return det.astype(np.float32)

    n = len(frame_indices)
    boxes = np.zeros((n, MAX_DETECTIONS, BBOX_DIMS), dtype=np.float32)
    scores = np.zeros((n, MAX_DETECTIONS), dtype=np.float32)
    class_ids = np.zeros((n, MAX_DETECTIONS), dtype=np.int32)
    valid_mask = np.zeros((n, MAX_DETECTIONS), dtype=bool)
    raw_per_frame: List[np.ndarray] = [np.zeros((0, 5), dtype=np.float32) for _ in frame_indices]

    for i_out, fi in enumerate(frame_indices):
        fr_rgb = frame_manager.get(int(fi))  # RGB uint8
        x, r, pad, orig_hw = _prep_yolo_tensor_from_rgb_uint8(fr_rgb, input_size=int(input_size))
        try:
            res = triton_client.infer(
                model_name=str(triton_model_name),
                model_version=str(triton_model_version) if triton_model_version else None,
                input_name=str(triton_input_name),
                input_tensor=x,
                output_name=str(triton_output_name),
                datatype="FP32",
            )
        except Exception as e:
            raise RuntimeError(f"{NAME} | Triton infer failed: {e}") from e

        out = np.asarray(res.output, dtype=np.float32)  # expected (1,84,N)
        det_np = _decode_and_nms_yolo(out)
        if det_np.size == 0:
            continue
        # scale boxes back to original
        det_np[:, :4] = _scale_boxes_back(det_np[:, :4], r=float(r), pad=pad, orig_hw=orig_hw)

        detections: List[List[float]] = []
        for j in range(min(det_np.shape[0], MAX_DETECTIONS)):
            xyxy = det_np[j, :4].astype(np.float32)
            conf = float(det_np[j, 4])
            cls_id = int(det_np[j, 5])
            if conf < float(box_threshold):
                continue
            boxes[i_out, j] = xyxy
            scores[i_out, j] = conf
            class_ids[i_out, j] = cls_id
            valid_mask[i_out, j] = True
            detections.append([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), float(conf)])

            if cls_id not in class_names:
                class_names[cls_id] = f"class_{cls_id}"

        raw_per_frame[i_out] = np.asarray(detections, dtype=np.float32)

        processed = i_out + 1
        if processed % 25 == 0:
            LOGGER.info("%s | Triton YOLO | processed %d/%d", NAME, processed, n)
        if progress_callback:
            progress_callback(done=processed, total=n)

    return boxes, scores, class_ids, valid_mask, class_names, raw_per_frame


# Tracking removed - no longer using StrongSORT or any tracking algorithm


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


def main():
    parser = argparse.ArgumentParser(description="Production object detection + tracking provider")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--rs-path", required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--runtime", type=str, default="ultralytics", choices=["ultralytics", "triton"])
    # Triton (preferred via ModelManager specs)
    parser.add_argument("--triton-model-spec", type=str, default=None, help="dp_models spec name (e.g., yolo11x_640_triton)")
    parser.add_argument("--triton-http-url", type=str, default=None)
    parser.add_argument("--triton-model-name", type=str, default=None)
    parser.add_argument("--triton-model-version", type=str, default=None)
    parser.add_argument("--triton-input-name", type=str, default="images")
    parser.add_argument("--triton-output-name", type=str, default="output0")
    parser.add_argument("--triton-preprocess-preset", type=str, default="yolo11x_640", choices=["yolo11x_320", "yolo11x_640", "yolo11x_960"])
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size (must be provided by scheduler/orchestrator)")
    parser.add_argument("--box-threshold", type=float, default=0.6)
    parser.add_argument("--device", type=str, default="auto", help="'auto'|'cpu'|'cuda'")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    args = parser.parse_args()
    # Expand env vars in --model (so configs can use ${DP_MODELS_ROOT}/...).
    if isinstance(args.model, str):
        args.model = os.path.expandvars(str(args.model))

    # Initialize timing dictionary
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    meta = load_metadata(os.path.join(args.frames_dir, "metadata.json"), NAME)
    total_frames = int(meta["total_frames"])

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

    # Strict sampling contract: prefer metadata[NAME].frame_indices, no fallback allowed.
    block = meta.get(NAME)
    if not isinstance(block, dict) or "frame_indices" not in block:
        raise RuntimeError(
            f"{NAME} | metadata missing '{NAME}.frame_indices'. "
            "Segmenter must provide per-provider frame_indices. No fallback is allowed."
        )
    frame_indices_raw = block.get("frame_indices")
    if not isinstance(frame_indices_raw, list) or not frame_indices_raw:
        raise RuntimeError(f"{NAME} | metadata '{NAME}.frame_indices' is empty/invalid.")
    frame_indices = [int(x) for x in frame_indices_raw]
    LOGGER.info("%s | sampled frames: %d / total=%d", NAME, len(frame_indices), total_frames)
    if len(frame_indices) <= 0:
        raise RuntimeError(f"{NAME} | empty frame_indices is not allowed (no-fallback)")

    frame_manager = FrameManager(
        frames_dir=args.frames_dir,
        chunk_size=meta.get("chunk_size", 32),
        cache_size=meta.get("cache_size", 2),
    )

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

    try:
        device = pick_device(args.device)
        batch_size = int(args.batch_size)
        if batch_size <= 0:
            raise RuntimeError(f"{NAME} | --batch-size must be > 0 (scheduler-controlled); got {batch_size}")

        class_names: Dict[int, str] = {}
        
        # Resolve model path for both ultralytics and triton runtimes
        resolved_model_path = str(args.model) if args.model else ""
        if resolved_model_path and not os.path.exists(resolved_model_path):
            # If user passed a relative path and DP_MODELS_ROOT is set, try resolving from it.
            mr = os.environ.get("DP_MODELS_ROOT")
            if mr and not os.path.isabs(resolved_model_path):
                cand = os.path.join(str(mr), str(resolved_model_path))
                if os.path.exists(cand):
                    resolved_model_path = cand
                    LOGGER.info("%s | Resolved model path via DP_MODELS_ROOT: %s", NAME, resolved_model_path)
        
        # Create progress callback
        def progress_cb(done: int, total: int) -> None:
            _emit_progress(
                rs_path=args.rs_path,
                platform_id=platform_id,
                video_id=video_id,
                run_id=run_id,
                done=done,
                total=total,
                stage="process_frames",
            )

        # Baseline contract: emit process_frames stage
        _emit_stage(
            rs_path=args.rs_path,
            platform_id=platform_id,
            video_id=video_id,
            run_id=run_id,
            stage="process_frames",
        )

        t_process_start = time.perf_counter()

        if str(args.runtime).lower() == "triton":
            # Best-effort class names resolution (priority order):
            # 1. If local ultralytics weights exist, read names from them (may be fine-tuned model).
            # 2. Otherwise use final taxonomy v1.0 (41 classes) - baseline/production standard.
            #    This keeps runtime=triton offline-friendly (no-network) and ensures consistency.
            # Use already resolved model path from above
            if resolved_model_path and os.path.exists(resolved_model_path):
                try:
                    from ultralytics import YOLO  # type: ignore

                    y = YOLO(str(resolved_model_path))
                    # Use model's class names if available (may be fine-tuned on our taxonomy)
                    model_class_names = getattr(y, "names", {})
                    if model_class_names:
                        class_names = {int(k): str(v) for k, v in model_class_names.items()}
                        LOGGER.info(
                            "%s | runtime=triton: loaded class names from model weights (%d classes)",
                            NAME,
                            len(class_names),
                        )
                    else:
                        # Model exists but has no names -> use final taxonomy
                        class_names = _load_final_taxonomy_v1_classes()
                except Exception as e:
                    LOGGER.warning(
                        "%s | runtime=triton: failed to load class names from model: %s, using final taxonomy v1.0",
                        NAME,
                        e,
                    )
                    class_names = _load_final_taxonomy_v1_classes()
            else:
                # No model weights available -> use final taxonomy v1.0 (baseline standard)
                class_names = _load_final_taxonomy_v1_classes()
                LOGGER.info(
                    "%s | runtime=triton: using final taxonomy v1.0 (41 classes) as baseline standard. "
                    "Model path=%r resolved=%r",
                    NAME,
                    args.model,
                    resolved_model_path,
                )

            # Resolve Triton client/params
            from dp_triton import TritonHttpClient, TritonError  # type: ignore

            if args.triton_model_spec:
                mm_entry = _load_triton_spec_via_model_manager(str(args.triton_model_spec))
                client = mm_entry["client"]
                rp = mm_entry["rp"]
                args.triton_http_url = str(rp.get("triton_http_url") or args.triton_http_url or "")
                args.triton_model_name = str(rp.get("triton_model_name") or args.triton_model_name or "")
                args.triton_model_version = str(rp.get("triton_model_version") or "") or None
                args.triton_input_name = str(rp.get("triton_input_name") or args.triton_input_name)
                args.triton_output_name = str(rp.get("triton_output_name") or args.triton_output_name)
            else:
                if not args.triton_http_url or not str(args.triton_http_url).strip():
                    raise RuntimeError(f"{NAME} | runtime=triton requires --triton-http-url or --triton-model-spec (no-fallback)")
                if not args.triton_model_name or not str(args.triton_model_name).strip():
                    raise RuntimeError(f"{NAME} | runtime=triton requires --triton-model-name or --triton-model-spec (no-fallback)")
                client = TritonHttpClient(base_url=str(args.triton_http_url), timeout_sec=10.0)
                if not client.ready():
                    raise TritonError(f"{NAME} | Triton is not ready at {args.triton_http_url}", error_code="triton_unavailable")

            if batch_size != 1:
                LOGGER.info("%s | runtime=triton: forcing fixed batch=1 (was %d)", NAME, batch_size)

            preset = str(args.triton_preprocess_preset).strip().lower()
            if preset == "yolo11x_320":
                input_size = 320
            elif preset == "yolo11x_640":
                input_size = 640
            elif preset == "yolo11x_960":
                input_size = 960
            else:
                raise RuntimeError(f"{NAME} | unknown triton_preprocess_preset: {preset}")

            boxes, scores, class_ids, valid_mask, class_names, raw_per_frame = run_yolo_triton(
                frame_manager=frame_manager,
                frame_indices=frame_indices,
                triton_client=client,
                triton_model_name=str(args.triton_model_name),
                triton_model_version=str(args.triton_model_version) if args.triton_model_version else None,
                triton_input_name=str(args.triton_input_name),
                triton_output_name=str(args.triton_output_name),
                input_size=int(input_size),
                box_threshold=float(args.box_threshold),
                iou_threshold=float(args.iou_threshold),
                class_names=class_names,
                progress_callback=progress_cb,
            )
            impl = f"triton:{args.triton_model_name}"
        else:
            # For ultralytics runtime, use resolved model path
            if not resolved_model_path or not os.path.exists(resolved_model_path):
                raise RuntimeError(
                    f"{NAME} | Model file not found: {args.model} (resolved: {resolved_model_path}). "
                    f"Please ensure DP_MODELS_ROOT is set or provide absolute path."
                )
            boxes, scores, class_ids, valid_mask, class_names, raw_per_frame = run_yolo(
                frame_manager=frame_manager,
                frame_indices=frame_indices,
                model_path=resolved_model_path,
                box_threshold=float(args.box_threshold),
                batch_size=batch_size,
                device=device,
                progress_callback=progress_cb,
            )
            impl = "yolo"

        t_process_end = time.perf_counter()
        timings["process_frames"] = t_process_end - t_process_start

        # Tracking removed: no longer using StrongSORT or any tracking
        
        class_names_arr = np.array([f"{k}:{v}" for k, v in sorted(class_names.items())], dtype="U")

        # Get timestamps for times_s
        uts = meta.get("union_timestamps_sec")
        if uts is None:
            raise RuntimeError(f"{NAME} | metadata.json missing union_timestamps_sec (contract)")
        uts_arr = np.asarray(uts, dtype=np.float32).reshape(-1)
        fi_np = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
        if np.any(fi_np < 0) or np.any(fi_np >= int(uts_arr.shape[0])):
            raise RuntimeError(f"{NAME} | frame_indices out of range for union_timestamps_sec")
        times_s = uts_arr[fi_np].astype(np.float32)

        out_dir = os.path.join(args.rs_path, NAME)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "detections.npz")

        created_at = datetime.utcnow().isoformat() + "Z"
        
        # Check for valid empty: no detections above threshold across all frames
        total_valid_detections = int(np.sum(valid_mask))
        if total_valid_detections == 0:
            status = "empty"
            empty_reason = "no_detections_above_threshold"
            LOGGER.warning("%s | No detections found above threshold %.2f across %d frames", NAME, args.box_threshold, len(frame_indices))
        else:
            status = "ok"
            empty_reason = None
        
        meta_info = {
            "producer": NAME,
            "producer_version": VERSION,
            "schema_version": SCHEMA_VERSION,
            "created_at": created_at,
            "status": status,
            "empty_reason": empty_reason,
            "impl": impl,
            "model": args.model,
            "box_threshold": args.box_threshold,
            "batch_size": int(batch_size),
            "device": str(device),
            "total_frames": int(total_frames),
            "total_detections": total_valid_detections,
        }
        required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
        missing = [k for k in required_run_keys if not meta.get(k)]
        if missing:
            raise RuntimeError(f"{NAME} | frames metadata missing required run identity keys: {missing}")
        for k in required_run_keys:
            meta_info[k] = meta.get(k)
        # Required by contract (baseline may use "unknown")
        meta_info["dataprocessor_version"] = str(meta.get("dataprocessor_version") or "unknown")

        # PR-3: model system baseline
        if str(args.runtime).lower() == "triton":
            model_name = str(args.triton_model_name or "triton")
            meta_info = apply_models_meta(
                meta_info,
                models_used=[
                    model_used(
                        model_name=model_name,
                        model_version=str(args.triton_model_version or "1"),
                        weights_digest="provided_by_deploy",
                        runtime="triton",
                        engine="triton",
                        precision="fp32",
                        device="cuda",
                    )
                ],
            )
        else:
            model_name = str(args.model)
            engine = "ultralytics"
            meta_info = apply_models_meta(
                meta_info,
                models_used=[
                    model_used(
                        model_name=model_name,
                        model_version="unknown",
                        weights_digest="unknown",
                        runtime="inprocess",
                        engine=engine,
                        precision="fp32",
                        device=str(device),
                    )
                ],
            )

        # Baseline contract: stage_timings_ms in meta
        timings["saving"] = 0.0  # Will be updated after save
        timings["total"] = time.perf_counter() - t0
        stage_timings_ms: Dict[str, float] = {}
        for key, value in timings.items():
            stage_timings_ms[key] = float(value) * 1000.0
        meta_info["stage_timings_ms"] = stage_timings_ms

        # Baseline contract: emit save stage
        _emit_stage(
            rs_path=args.rs_path,
            platform_id=platform_id,
            video_id=video_id,
            run_id=run_id,
            stage="save",
        )

        t_save_start = time.perf_counter()

        # Сохраняем meta как JSON-строку для совместимости между виртуальными окружениями
        # Это предотвращает SIGSEGV при загрузке pickled объектов из разных версий numpy
        # Старый формат (meta как object-array) сохраняется для обратной совместимости
        meta_json_str = json.dumps(meta_info, ensure_ascii=False, default=str)
        
        atomic_save_npz(
            out_path,
            # metadata fields
            meta=np.asarray(meta_info, dtype=object),  # Старый формат для обратной совместимости
            meta_json=np.array(meta_json_str, dtype="U"),  # Новый формат: JSON-строка (безопасно для всех окружений)
            frame_indices=np.asarray(frame_indices, dtype=np.int32),
            times_s=times_s,
            # detection arrays
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            valid_mask=valid_mask,
            class_names=class_names_arr,
        )

        timings["saving"] = time.perf_counter() - t_save_start
        timings["total"] = time.perf_counter() - t0

        # Update stage_timings_ms with final timings
        stage_timings_ms = {}
        for key, value in timings.items():
            stage_timings_ms[key] = float(value) * 1000.0
        meta_info["stage_timings_ms"] = stage_timings_ms

        # Log stage timings for profiling
        LOGGER.info(f"{NAME} | stage timings (ms): {', '.join([f'{k}={v:.1f}' for k, v in sorted(stage_timings_ms.items())])}")

        # Validate artifact
        from utils.artifact_validator import validate_npz

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

        LOGGER.info("%s | saved NPZ artifact: %s", NAME, out_path)

    finally:
        try:
            frame_manager.close()
        except Exception:
            LOGGER.exception("Failed to close FrameManager")


if __name__ == "__main__":
    main()
