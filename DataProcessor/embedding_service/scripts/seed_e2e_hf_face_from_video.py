#!/usr/bin/env python3
"""
Добавить в Embedding Service (category=face) seed-лицо из HF/local видео + landmarks NPZ.

Нужно для E2E core_identity: placeholder из setup_e2e_infra.sql не в FAISS/ArcFace,
поэтому face_identity получает no_faces_processed при реальных лицах в кадре.

Usage (Embedding Service на :8005):
  cd DataProcessor
  .data_venv/bin/python embedding_service/scripts/seed_e2e_hf_face_from_video.py \\
    --video ../example/hf_videos11/-4WRepA-bss.mp4 \\
    --landmarks-npz ../storage/result_store/youtube/-4WRepA-bss/<run_id>/core_face_landmarks/landmarks.npz

Или offline (без HTTP, напишет в Postgres+FAISS; перезапустите embedding-service):
  EMBEDDING_SEED_OFFLINE=1 .data_venv/bin/python embedding_service/scripts/seed_e2e_hf_face_from_video.py ...
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DP = Path(__file__).resolve().parent.parent.parent
if str(_DP) not in sys.path:
    sys.path.insert(0, str(_DP))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402
from PIL import Image  # noqa: E402

# Reuse face_identity crop helpers (normalized landmarks → pixel bbox).
_FI_DIR = _DP / "VisualProcessor" / "core" / "model_process" / "core_identity" / "face_identity"
if str(_FI_DIR) not in sys.path:
    sys.path.insert(0, str(_FI_DIR))
from main import _crop_face, _extract_face_bbox_from_landmarks  # noqa: E402


def _load_landmarks_npz(path: Path) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _best_face_crop_from_video_insightface(video: Path, max_frames: int = 120) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // max_frames) if total > 0 else 1
    best: Optional[Tuple[float, np.ndarray, int]] = None
    idx = 0
    scanned = 0
    while scanned < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        faces = app.get(frame)
        if faces:
            face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
            x1, y1, x2, y2 = [int(v) for v in face.bbox[:4]]
            crop = frame[max(0, y1) : y2, max(0, x1) : x2]
            if crop.size > 0:
                area = float(crop.shape[0] * crop.shape[1])
                if best is None or area > best[0]:
                    best = (area, crop.copy(), idx)
        idx += step
        scanned += 1
        if total > 0 and idx >= total:
            break

    cap.release()
    if best is None:
        raise RuntimeError(f"No faces found in video (insightface scan): {video}")
    _, crop, frame_idx = best
    meta = {"source": "seed_e2e_hf_face_from_video", "video": video.name, "frame_idx": frame_idx, "mode": "video_only"}
    return crop, frame_idx, meta


def _best_face_crop(video: Path, landmarks_npz: Optional[Path]) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    if landmarks_npz and landmarks_npz.is_file():
        return _best_face_crop_from_landmarks(video, landmarks_npz)
    return _best_face_crop_from_video_insightface(video)


def _best_face_crop_from_landmarks(video: Path, landmarks_npz: Path) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    data = _load_landmarks_npz(landmarks_npz)
    fi = np.asarray(data["frame_indices"], dtype=np.int32).reshape(-1)
    fp = np.asarray(data["face_present"])
    has = np.any(fp, axis=1) if fp.ndim == 2 else fp.astype(bool)
    if not np.any(has):
        raise RuntimeError(f"No faces in landmarks NPZ: {landmarks_npz}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    best: Optional[Tuple[float, np.ndarray, int]] = None
    for pos in np.where(has)[0]:
        frame_idx = int(fi[int(pos)])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        lm = data["face_landmarks"][int(pos), 0]
        bbox = _extract_face_bbox_from_landmarks(np.asarray(lm, dtype=np.float32))
        if bbox is None:
            continue
        crop = _crop_face(frame, bbox)
        if crop is None or crop.size == 0:
            continue
        area = float(crop.shape[0] * crop.shape[1])
        if best is None or area > best[0]:
            best = (area, crop, frame_idx)

    cap.release()
    if best is None:
        raise RuntimeError("Could not extract any face crop from video+landmarks")
    _, crop, frame_idx = best
    meta = {"source": "seed_e2e_hf_face_from_video", "video": video.name, "frame_idx": frame_idx}
    return crop, frame_idx, meta


def _add_via_http(
    *,
    base_url: str,
    name: str,
    crop_bgr: np.ndarray,
    metadata: Dict[str, Any],
    timeout: float,
) -> str:
    url = f"{base_url.rstrip('/')}/objects/add"
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=92)
    buf.seek(0)
    r = requests.post(
        url,
        files={"image": ("face_seed.jpg", buf, "image/jpeg")},
        data={"category": "face", "name": name, "metadata": json.dumps(metadata)},
        timeout=timeout,
    )
    r.raise_for_status()
    payload = r.json()
    return str(payload.get("id") or "")


def _add_via_manager(*, name: str, crop_bgr: np.ndarray, metadata: Dict[str, Any]) -> str:
    from embedding_service.config.settings import EmbeddingServiceConfig  # noqa: E402
    from embedding_service.core.embedding_manager import EmbeddingManager  # noqa: E402

    config = EmbeddingServiceConfig()
    manager = EmbeddingManager(config)
    try:
        oid = manager.add(category="face", image=crop_bgr, name=name, metadata=metadata)
        return str(oid)
    finally:
        manager.close()


def _verify_search(base_url: str, crop_bgr: np.ndarray, name: str, timeout: float) -> List[Dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/search"
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=92)
    buf.seek(0)
    r = requests.post(
        url,
        files={"image": ("probe.jpg", buf, "image/jpeg")},
        data={"category": "face", "top_k": "5", "similarity_threshold": "0.0"},
        timeout=timeout,
    )
    r.raise_for_status()
    results = (r.json() or {}).get("results") or []
    hit = [x for x in results if str(x.get("name") or "") == name]
    print(f"  search: {len(results)} result(s), seed name match={len(hit)}")
    for row in results[:3]:
        print(f"    - {row.get('name')}: sim={row.get('similarity')}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed Embedding Service face from HF video landmarks")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument(
        "--landmarks-npz",
        type=Path,
        default=None,
        help="core_face_landmarks/landmarks.npz (optional; без него — InsightFace scan видео)",
    )
    ap.add_argument(
        "--name",
        default="hf_videos11_-4WRepA-bss_seed",
        help="Object name in Embedding Service",
    )
    ap.add_argument(
        "--embedding-service-url",
        default=os.environ.get("EMBEDDING_SERVICE_URL", "http://127.0.0.1:8005"),
    )
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"FAIL: video not found: {args.video}", file=sys.stderr)
        return 2
    if args.landmarks_npz is not None and not args.landmarks_npz.is_file():
        print(f"FAIL: landmarks NPZ not found: {args.landmarks_npz}", file=sys.stderr)
        return 2

    default_name = f"hf_seed_{args.video.stem}"
    seed_name = args.name if args.name != "hf_videos11_-4WRepA-bss_seed" else default_name
    if args.video.stem == "-4WRepA-bss" and args.name == "hf_videos11_-4WRepA-bss_seed":
        seed_name = "hf_videos11_-4WRepA-bss_seed"

    crop, frame_idx, meta = _best_face_crop(
        args.video.resolve(),
        args.landmarks_npz.resolve() if args.landmarks_npz else None,
    )
    print(f"Best face crop: frame={frame_idx} shape={crop.shape}")

    offline = os.environ.get("EMBEDDING_SEED_OFFLINE", "").strip().lower() in ("1", "true", "yes")
    if offline:
        oid = _add_via_manager(name=seed_name, crop_bgr=crop, metadata=meta)
        print(f"OK offline add face id={oid} name={seed_name} (restart embedding-service if it was running)")
    else:
        oid = _add_via_http(
            base_url=args.embedding_service_url,
            name=seed_name,
            crop_bgr=crop,
            metadata=meta,
            timeout=args.timeout,
        )
        print(f"OK HTTP add face id={oid} name={seed_name}")
        if not args.no_verify:
            _verify_search(args.embedding_service_url, crop, seed_name, args.timeout)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
