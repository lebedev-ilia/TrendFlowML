#!/usr/bin/env python3
"""
Валидатор ВХОДНОГО контракта action_recognition (ASSESSMENT §3).

Проверяет, что зависимости готовы ДО обработки, чтобы падать рано и понятно:
- `frames_dir/metadata.json`: есть `union_timestamps_sec`, run-identity, окна action_recognition;
- `core_object_detections/detections.npz`: schema v3, `track_ids (N,M)`, согласованные формы,
  person-детекции присутствуют (иначе — валидный empty-путь, не ошибка).

Использование:
  validate_input(frames_dir, detections_npz_path) -> (ok: bool, problems: List[str], info: dict)
CLI:
  python validate_action_recognition_input.py <frames_dir> <detections.npz>
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

PERSON_CLASS_ID = 0


def _load_meta_json(npz: Dict[str, Any]) -> Dict[str, Any]:
    mj = npz.get("meta_json")
    if mj is not None:
        s = mj.item() if getattr(mj, "shape", None) == () else np.asarray(mj).reshape(-1)[0]
        try:
            return json.loads(str(s))
        except Exception:
            return {}
    return {}


def validate_input(frames_dir: str, detections_npz_path: str) -> Tuple[bool, List[str], Dict[str, Any]]:
    problems: List[str] = []
    info: Dict[str, Any] = {}

    # 1) frames metadata
    meta_path = os.path.join(frames_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return False, [f"нет {meta_path}"], info
    with open(meta_path, "r", encoding="utf-8") as f:
        fmeta = json.load(f)
    uts = fmeta.get("union_timestamps_sec")
    if not isinstance(uts, list) or len(uts) < 1:
        problems.append("metadata.json: union_timestamps_sec отсутствует/пуст")
    for k in ("platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"):
        if not fmeta.get(k):
            problems.append(f"metadata.json: нет run-identity ключа {k}")
    ar = fmeta.get("action_recognition")
    if not isinstance(ar, dict) or not ar.get("frame_indices"):
        problems.append("metadata.json: нет action_recognition.frame_indices (Segmenter не дал выборку)")
    else:
        info["ar_num_frames"] = len(ar.get("frame_indices") or [])
        info["ar_num_windows"] = int(ar.get("num_windows") or len(ar.get("windows") or []))
        if info["ar_num_windows"] == 0:
            problems.append("metadata.json: action_recognition.windows пуст (нет плотных окон R1/R3)")

    # 2) detections.npz (v3 + track_ids)
    if not os.path.isfile(detections_npz_path):
        return False, problems + [f"нет {detections_npz_path}"], info
    d = np.load(detections_npz_path, allow_pickle=True)
    dd = {k: d[k] for k in d.files}
    dmeta = _load_meta_json(dd)
    sv = str(dmeta.get("schema_version", ""))
    info["det_schema"] = sv
    if "core_object_detections" not in sv:
        problems.append(f"detections.npz: неожиданная schema_version={sv!r}")
    if "track_ids" not in dd:
        problems.append("detections.npz: нет track_ids (нужен appearance-tracker / schema v3)")
    else:
        tids = np.asarray(dd["track_ids"])
        vm = np.asarray(dd.get("valid_mask")) if "valid_mask" in dd else None
        cls = np.asarray(dd.get("class_ids")) if "class_ids" in dd else None
        if vm is not None and tids.shape != vm.shape:
            problems.append(f"detections.npz: track_ids{tids.shape} != valid_mask{vm.shape}")
        # person-присутствие (иначе валидный empty downstream — не ошибка входа)
        if vm is not None and cls is not None:
            person = vm & (cls == PERSON_CLASS_ID)
            n_person = int(person.sum())
            n_person_tracks = int(np.unique(tids[person & (tids >= 0)]).size) if n_person else 0
            info["n_person_detections"] = n_person
            info["n_person_tracks"] = n_person_tracks
            info["valid_empty_expected"] = bool(n_person == 0)

    ok = len(problems) == 0
    return ok, problems, info


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: validate_action_recognition_input.py <frames_dir> <detections.npz>", file=sys.stderr)
        return 2
    ok, problems, info = validate_input(sys.argv[1], sys.argv[2])
    print("info:", json.dumps(info, ensure_ascii=False))
    if not ok:
        print("❌ входной контракт action_recognition НАРУШЕН:")
        for p in problems:
            print("  -", p)
        return 1
    if info.get("valid_empty_expected"):
        print("✅ вход валиден; людей нет → ожидается валидный empty (no_person_detections)")
    else:
        print("✅ входной контракт action_recognition выполнен")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
