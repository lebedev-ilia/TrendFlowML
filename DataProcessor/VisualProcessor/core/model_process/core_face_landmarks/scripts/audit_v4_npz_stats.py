#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _load_npz(path: str) -> Dict[str, Any]:
    npz = np.load(path, allow_pickle=True)
    try:
        return {k: npz[k] for k in npz.files}
    finally:
        try:
            npz.close()
        except Exception:
            pass


def _unbox_meta(meta_arr: Any) -> Dict[str, Any]:
    if isinstance(meta_arr, np.ndarray) and meta_arr.dtype == object and meta_arr.shape == ():
        try:
            meta_arr = meta_arr.item()
        except Exception:
            pass
    if isinstance(meta_arr, dict):
        return dict(meta_arr)
    return {}


def _as_bool_array(x: Any) -> np.ndarray:
    try:
        return np.asarray(x, dtype=bool)
    except Exception:
        return np.asarray([], dtype=bool)


def _as_float_array(x: Any) -> np.ndarray:
    try:
        return np.asarray(x, dtype=np.float32)
    except Exception:
        return np.asarray([], dtype=np.float32)


def _count_all_nan(arr: np.ndarray, axis: Tuple[int, ...]) -> np.ndarray:
    a = np.asarray(arr)
    if a.size == 0:
        return np.asarray([], dtype=bool)
    return np.all(np.isnan(a), axis=axis)


@dataclass
class RunStats:
    npz_path: str
    platform_id: Optional[str]
    video_id: Optional[str]
    run_id: Optional[str]
    config_hash: Optional[str]
    sampling_policy_version: Optional[str]
    schema_version: Optional[str]
    producer_version: Optional[str]

    N: int
    FACES: int
    HANDS: int

    face_present_true: int
    face_present_total: int
    face_present_ratio: Optional[float]

    face_mesh_ran_true: int
    face_mesh_ran_total: int
    face_mesh_ran_ratio: Optional[float]

    face_present_implies_mesh_violations: int
    face_mesh_and_not_present: int

    face_landmarks_all_nan_when_absent_violations: int
    face_landmarks_any_nan_when_present_count: int

    hands_present_true: int
    hands_present_total: int
    hands_present_ratio: Optional[float]
    hands_landmarks_all_nan_when_absent_violations: int
    hands_landmarks_any_nan_when_present_count: int

    pose_present_true: Optional[int]
    pose_present_total: Optional[int]
    pose_present_ratio: Optional[float]
    pose_landmarks_nan_total: Optional[int]


def compute_run_stats(npz_path: str) -> RunStats:
    d = _load_npz(npz_path)
    meta = _unbox_meta(d.get("meta"))

    face_present = _as_bool_array(d.get("face_present"))
    face_mesh_ran = _as_bool_array(d.get("face_mesh_ran")).reshape(-1)
    face_landmarks = _as_float_array(d.get("face_landmarks"))

    hands_present = _as_bool_array(d.get("hands_present"))
    hands_landmarks = _as_float_array(d.get("hands_landmarks"))

    pose_present = d.get("pose_present", None)
    pose_landmarks = d.get("pose_landmarks", None)

    # shapes
    N = int(face_present.shape[0]) if face_present.ndim >= 1 else 0
    FACES = int(face_present.shape[1]) if face_present.ndim == 2 else 0
    HANDS = int(hands_present.shape[1]) if hands_present.ndim == 2 else 0

    # face ratios
    face_present_true = int(face_present.sum()) if face_present.size else 0
    face_present_total = int(face_present.size)
    face_present_ratio = (float(face_present_true) / float(face_present_total)) if face_present_total else None

    face_mesh_ran_true = int(face_mesh_ran.sum()) if face_mesh_ran.size else 0
    face_mesh_ran_total = int(face_mesh_ran.size)
    face_mesh_ran_ratio = (float(face_mesh_ran_true) / float(face_mesh_ran_total)) if face_mesh_ran_total else None

    # invariants: face_present => face_mesh_ran (per-frame)
    if face_present.ndim == 2 and face_mesh_ran.size == N:
        face_present_any = np.any(face_present, axis=1)
        face_present_implies_mesh_violations = int(np.sum(face_present_any & (~face_mesh_ran)))
        face_mesh_and_not_present = int(np.sum(face_mesh_ran & (~face_present_any)))
    else:
        face_present_implies_mesh_violations = 0
        face_mesh_and_not_present = 0

    # NaN policy: when absent => all NaN in landmark slot; when present => ideally no NaN
    face_landmarks_all_nan_when_absent_violations = 0
    face_landmarks_any_nan_when_present_count = 0
    if face_landmarks.ndim == 4 and face_present.ndim == 2:
        # face_landmarks: (N, FACES, 468, 3)
        all_nan_per_slot = _count_all_nan(face_landmarks, axis=(2, 3))  # (N,FACES)
        absent = ~face_present
        present = face_present
        face_landmarks_all_nan_when_absent_violations = int(np.sum(absent & (~all_nan_per_slot)))
        # if present: count slots where any NaN exists
        any_nan_per_slot = np.any(np.isnan(face_landmarks), axis=(2, 3))
        face_landmarks_any_nan_when_present_count = int(np.sum(present & any_nan_per_slot))

    hands_present_true = int(hands_present.sum()) if hands_present.size else 0
    hands_present_total = int(hands_present.size)
    hands_present_ratio = (float(hands_present_true) / float(hands_present_total)) if hands_present_total else None

    hands_landmarks_all_nan_when_absent_violations = 0
    hands_landmarks_any_nan_when_present_count = 0
    if hands_landmarks.ndim == 4 and hands_present.ndim == 2:
        # hands_landmarks: (N, HANDS, 21, 3)
        all_nan_per_slot_h = _count_all_nan(hands_landmarks, axis=(2, 3))  # (N,HANDS)
        absent_h = ~hands_present
        present_h = hands_present
        hands_landmarks_all_nan_when_absent_violations = int(np.sum(absent_h & (~all_nan_per_slot_h)))
        any_nan_per_slot_h = np.any(np.isnan(hands_landmarks), axis=(2, 3))
        hands_landmarks_any_nan_when_present_count = int(np.sum(present_h & any_nan_per_slot_h))

    # pose
    pose_present_true_v = None
    pose_present_total_v = None
    pose_present_ratio_v = None
    pose_landmarks_nan_total_v = None
    if pose_present is not None:
        pp = _as_bool_array(pose_present).reshape(-1)
        pose_present_true_v = int(pp.sum())
        pose_present_total_v = int(pp.size)
        pose_present_ratio_v = (float(pose_present_true_v) / float(pose_present_total_v)) if pose_present_total_v else None
    if pose_landmarks is not None:
        pl = _as_float_array(pose_landmarks)
        pose_landmarks_nan_total_v = int(np.isnan(pl).sum()) if pl.size else 0

    return RunStats(
        npz_path=npz_path,
        platform_id=str(meta.get("platform_id")) if meta.get("platform_id") is not None else None,
        video_id=str(meta.get("video_id")) if meta.get("video_id") is not None else None,
        run_id=str(meta.get("run_id")) if meta.get("run_id") is not None else None,
        config_hash=str(meta.get("config_hash")) if meta.get("config_hash") is not None else None,
        sampling_policy_version=str(meta.get("sampling_policy_version")) if meta.get("sampling_policy_version") is not None else None,
        schema_version=str(meta.get("schema_version")) if meta.get("schema_version") is not None else None,
        producer_version=str(meta.get("producer_version")) if meta.get("producer_version") is not None else None,
        N=N,
        FACES=FACES,
        HANDS=HANDS,
        face_present_true=face_present_true,
        face_present_total=face_present_total,
        face_present_ratio=face_present_ratio,
        face_mesh_ran_true=face_mesh_ran_true,
        face_mesh_ran_total=face_mesh_ran_total,
        face_mesh_ran_ratio=face_mesh_ran_ratio,
        face_present_implies_mesh_violations=face_present_implies_mesh_violations,
        face_mesh_and_not_present=face_mesh_and_not_present,
        face_landmarks_all_nan_when_absent_violations=face_landmarks_all_nan_when_absent_violations,
        face_landmarks_any_nan_when_present_count=face_landmarks_any_nan_when_present_count,
        hands_present_true=hands_present_true,
        hands_present_total=hands_present_total,
        hands_present_ratio=hands_present_ratio,
        hands_landmarks_all_nan_when_absent_violations=hands_landmarks_all_nan_when_absent_violations,
        hands_landmarks_any_nan_when_present_count=hands_landmarks_any_nan_when_present_count,
        pose_present_true=pose_present_true_v,
        pose_present_total=pose_present_total_v,
        pose_present_ratio=pose_present_ratio_v,
        pose_landmarks_nan_total=pose_landmarks_nan_total_v,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v4 stats for core_face_landmarks (VisualProcessor core)")
    ap.add_argument("--npz", action="append", required=True, help="Path to core_face_landmarks/landmarks.npz (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    out_path = os.fspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    runs = [compute_run_stats(os.fspath(p)) for p in (args.npz or [])]

    doc: Dict[str, Any] = {
        "component": "core_face_landmarks",
        "level": "L2 (A+B, 5 runs)",
        "n_runs": int(len(runs)),
        "aggregates": {
            "N_total": int(sum(r.N for r in runs)),
            "N_set": sorted({int(r.N) for r in runs}),
            "FACES_set": sorted({int(r.FACES) for r in runs}),
            "HANDS_set": sorted({int(r.HANDS) for r in runs}),
            "face_present_ratio_min": min([r.face_present_ratio for r in runs if r.face_present_ratio is not None], default=None),
            "face_present_ratio_max": max([r.face_present_ratio for r in runs if r.face_present_ratio is not None], default=None),
            "hands_present_ratio_min": min([r.hands_present_ratio for r in runs if r.hands_present_ratio is not None], default=None),
            "hands_present_ratio_max": max([r.hands_present_ratio for r in runs if r.hands_present_ratio is not None], default=None),
            "pose_present_ratio_min": min([r.pose_present_ratio for r in runs if r.pose_present_ratio is not None], default=None),
            "pose_present_ratio_max": max([r.pose_present_ratio for r in runs if r.pose_present_ratio is not None], default=None),
            "face_present_implies_mesh_violations_total": int(sum(r.face_present_implies_mesh_violations for r in runs)),
            "face_mesh_and_not_present_total": int(sum(r.face_mesh_and_not_present for r in runs)),
            "face_landmarks_all_nan_when_absent_violations_total": int(sum(r.face_landmarks_all_nan_when_absent_violations for r in runs)),
            "face_landmarks_any_nan_when_present_total": int(sum(r.face_landmarks_any_nan_when_present_count for r in runs)),
            "hands_landmarks_all_nan_when_absent_violations_total": int(sum(r.hands_landmarks_all_nan_when_absent_violations for r in runs)),
            "hands_landmarks_any_nan_when_present_total": int(sum(r.hands_landmarks_any_nan_when_present_count for r in runs)),
            "pose_landmarks_nan_total": int(sum(int(r.pose_landmarks_nan_total or 0) for r in runs)),
        },
        "per_run": [asdict(r) for r in runs],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

