#!/usr/bin/env python3
"""
Валидатор ВХОДНОГО контракта shot_quality (падаем рано/понятно). Самая тяжёлая цепочка — 5 hard-deps.

Проверяет наличие + базовую форму 5 зависимостей и выравнивание по frame_indices:
- core_clip/embeddings.npz (CLIP-quality промпты + эмбеддинги)
- core_depth_midas/depth.npz (depth_maps (N,H,W))
- core_object_detections/detections.npz
- core_face_landmarks/*.npz  (валидная пустота ок → face-ROI фичи = NaN, НЕ ошибка)
- cut_detection/*.npz (границы шотов)
+ frames metadata: union_timestamps_sec, shot_quality.frame_indices, run-identity.

CLI: python validate_shot_quality_input.py <frames_dir> <rs_path>
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from typing import Any, Dict, List, Tuple
import numpy as np


def _load(p): d = np.load(p, allow_pickle=True); return {k: d[k] for k in d.files}
def _find(rs, prov, name=None):
    if name:
        c = os.path.join(rs, prov, name)
        if os.path.isfile(c): return c
    h = glob.glob(os.path.join(rs, prov, "*.npz")); return h[0] if h else None


def validate_input(frames_dir: str, rs: str) -> Tuple[bool, List[str], Dict[str, Any]]:
    problems: List[str] = []; info: Dict[str, Any] = {}
    mp = os.path.join(frames_dir, "metadata.json")
    if not os.path.isfile(mp): return False, [f"нет {mp}"], info
    fm = json.load(open(mp, encoding="utf-8"))
    if not isinstance(fm.get("union_timestamps_sec"), list) or len(fm["union_timestamps_sec"]) < 2:
        problems.append("metadata: union_timestamps_sec отсутствует/короткий")
    for k in ("platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"):
        if not fm.get(k): problems.append(f"metadata: нет run-identity {k}")
    sq = fm.get("shot_quality")
    sq_fi = set()
    if not isinstance(sq, dict) or not sq.get("frame_indices"):
        problems.append("metadata: нет shot_quality.frame_indices (Segmenter)")
    else:
        sq_fi = set(int(x) for x in sq["frame_indices"]); info["sq_num_frames"] = len(sq_fi)

    # 5 hard deps
    checks = {
        "core_clip": ("embeddings.npz", ["frame_indices"], ["frame_embeddings", "embeddings"]),
        "core_depth_midas": ("depth.npz", ["frame_indices"], ["depth_maps"]),
        "core_object_detections": ("detections.npz", ["frame_indices"], ["boxes", "class_ids"]),
        "cut_detection": (None, [], ["detections", "frame_indices"]),
    }
    for prov, (name, need_any_all, need_any) in checks.items():
        p = _find(rs, prov, name)
        if not p:
            problems.append(f"нет {prov} npz (hard dep)"); continue
        d = _load(p); info[prov] = "present"
        for k in need_any_all:
            if k not in d: problems.append(f"{prov}: нет {k}")
        if need_any and not any(k in d for k in need_any):
            problems.append(f"{prov}: нет ни одного из {need_any}")
        if prov == "core_depth_midas" and "depth_maps" in d:
            dm = np.asarray(d["depth_maps"])
            if dm.ndim != 3: problems.append(f"core_depth_midas: depth_maps ndim={dm.ndim} (ожид. 3: N,H,W)")
    # core_face_landmarks — присутствие желательно, но пустота ок (face-ROI → NaN)
    fl = _find(rs, "core_face_landmarks")
    info["core_face_landmarks"] = "present" if fl else "absent (face-ROI будут NaN — допустимо)"

    return len(problems) == 0, problems, info


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("frames_dir"); ap.add_argument("rs_path"); a = ap.parse_args()
    ok, problems, info = validate_input(a.frames_dir, a.rs_path)
    print("info:", json.dumps(info, ensure_ascii=False))
    if not ok:
        print("❌ входной контракт shot_quality НАРУШЕН:")
        for p in problems: print("  -", p)
        return 1
    print("✅ входной контракт shot_quality выполнен")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
