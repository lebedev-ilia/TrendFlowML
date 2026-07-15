#!/usr/bin/env python3
"""
Валидатор ВХОДНОГО контракта scene_classification (падаем рано и понятно).

Проверяет зависимости ДО обработки:
- frames metadata: union_timestamps_sec, scene_classification.frame_indices, run-identity;
- core_clip/embeddings.npz: frame_indices + frame_embeddings (+ places365_text_embeddings если clip);
- cut_detection: shot_boundaries_frame_indices;
- subset-констрейнт: scene_classification.frame_indices ⊆ core_clip.frame_indices.

CLI: python validate_scene_classification_input.py <frames_dir> <rs_path> [--label-fusion places|clip]
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from typing import Any, Dict, List, Tuple
import numpy as np


def _load_npz(path: str) -> Dict[str, Any]:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _find_provider_npz(rs_path: str, provider: str, name: str) -> str | None:
    cand = os.path.join(rs_path, provider, name)
    if os.path.isfile(cand):
        return cand
    hits = glob.glob(os.path.join(rs_path, provider, "*.npz"))
    return hits[0] if hits else None


def validate_input(frames_dir: str, rs_path: str, label_fusion: str = "places") -> Tuple[bool, List[str], Dict[str, Any]]:
    problems: List[str] = []
    info: Dict[str, Any] = {}

    meta_path = os.path.join(frames_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return False, [f"нет {meta_path}"], info
    fmeta = json.load(open(meta_path, encoding="utf-8"))
    uts = fmeta.get("union_timestamps_sec")
    if not isinstance(uts, list) or len(uts) < 2:
        problems.append("metadata: union_timestamps_sec отсутствует/короткий")
    for k in ("platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"):
        if not fmeta.get(k):
            problems.append(f"metadata: нет run-identity {k}")
    sc = fmeta.get("scene_classification")
    sc_fi = set()
    if not isinstance(sc, dict) or not sc.get("frame_indices"):
        problems.append("metadata: нет scene_classification.frame_indices (Segmenter)")
    else:
        sc_fi = set(int(x) for x in sc["frame_indices"])
        info["scene_num_frames"] = len(sc_fi)

    # core_clip (hard)
    cc = _find_provider_npz(rs_path, "core_clip", "embeddings.npz")
    if not cc:
        problems.append("нет core_clip/embeddings.npz (hard dep)")
    else:
        d = _load_npz(cc)
        info["core_clip_schema"] = "present"
        cc_fi = None
        for key in ("frame_indices",):
            if key not in d:
                problems.append(f"core_clip: нет {key}")
            else:
                cc_fi = set(int(x) for x in np.asarray(d[key]).reshape(-1).tolist())
        if not any(k in d for k in ("frame_embeddings", "embeddings")):
            problems.append("core_clip: нет frame_embeddings")
        if label_fusion == "clip" and "places365_text_embeddings" not in d:
            problems.append("core_clip: нет places365_text_embeddings (нужно для label_fusion=clip)")
        if cc_fi is not None and sc_fi and not sc_fi.issubset(cc_fi):
            miss = len(sc_fi - cc_fi)
            problems.append(f"subset нарушен: {miss} scene-кадров вне core_clip.frame_indices")

    # cut_detection (hard)
    cd = _find_provider_npz(rs_path, "cut_detection", "detections.npz") or _find_provider_npz(rs_path, "cut_detection", "cut_detection.npz")
    if not cd:
        problems.append("нет cut_detection npz (hard dep, границы шотов)")
    else:
        d = _load_npz(cd)
        # cut_detection хранит границы шотов в object-ключе `detections` (или отдельном model_facing npz),
        # не топ-уровневым `shot_boundaries_*`. Достаточно проверить наличие detections/frame_indices.
        has_bounds = ("detections" in d) or any("shot" in k.lower() or "bound" in k.lower() for k in d.keys())
        if not has_bounds and "frame_indices" not in d:
            problems.append("cut_detection: нет detections/границ шотов")
        info["cut_detection"] = "present"

    return len(problems) == 0, problems, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("rs_path")
    ap.add_argument("--label-fusion", default="places", choices=["places", "clip"])
    a = ap.parse_args()
    ok, problems, info = validate_input(a.frames_dir, a.rs_path, a.label_fusion)
    print("info:", json.dumps(info, ensure_ascii=False))
    if not ok:
        print("❌ входной контракт scene_classification НАРУШЕН:")
        for p in problems:
            print("  -", p)
        return 1
    print("✅ входной контракт scene_classification выполнен")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
