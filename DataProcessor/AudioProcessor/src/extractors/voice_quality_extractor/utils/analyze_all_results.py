#!/usr/bin/env python3
"""
Анализ всех результатов voice_quality_extractor для проверки качества.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# Добавляем путь к utils для импорта validate
_utils_dir = Path(__file__).resolve().parent
if str(_utils_dir) not in sys.path:
    sys.path.insert(0, str(_utils_dir))

from validate_voice_quality import load_npz, extract_meta, validate_voice_quality


def analyze_all_results(
    rs_base: str = "dp_results/youtube",
    run_id_prefix: str = "test_voice_quality_",
    component_name: str = "voice_quality_extractor",
    npz_name: str = "voice_quality_extractor_features.npz",
) -> Dict[str, Any]:
    """Анализирует все результаты voice_quality_extractor."""
    rs_path = Path(rs_base)
    platform = "youtube"

    all_stats: List[Dict[str, Any]] = []
    all_feature_values: List[np.ndarray] = []
    video_ids: List[str] = []

    # Ищем run_id по префиксу
    platform_dir = rs_path / platform
    if not platform_dir.exists():
        return {"total_videos": 0, "per_video": [], "summary": {}}

    for run_dir in sorted(platform_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith(run_id_prefix):
            continue
        video_id = run_dir.name
        # Структура: rs_base/youtube/video_id/run_id/component/npz (video_id=run_id в тестах)
        npz_path = run_dir / video_id / component_name / npz_name

        if not npz_path.exists():
            continue

        try:
            r = validate_voice_quality(str(npz_path))
            npz_data = load_npz(str(npz_path))
            meta = extract_meta(npz_data)

            fv = npz_data.get("feature_values")
            if fv is not None:
                arr = np.asarray(fv, dtype=np.float32).reshape(-1)
                valid = ~(np.isnan(arr) | np.isinf(arr))
                if np.any(valid):
                    all_feature_values.append(arr[valid])

            all_stats.append({
                "video_id": video_id,
                "valid": r["valid"],
                "segments_count": r["stats"].get("segments_count", 0),
                "status": meta.get("status", "unknown"),
            })
            video_ids.append(video_id)

        except Exception as e:
            all_stats.append({
                "video_id": video_id,
                "valid": False,
                "error": str(e),
            })

    # Сводная статистика по feature_values
    summary = {}
    if all_feature_values:
        concat = np.concatenate(all_feature_values)
        summary["feature_values"] = {
            "count": int(len(concat)),
            "min": float(np.min(concat)),
            "max": float(np.max(concat)),
            "mean": float(np.mean(concat)),
            "median": float(np.median(concat)),
            "std": float(np.std(concat)),
        }

    valid_count = sum(1 for s in all_stats if s.get("valid", False))
    summary["valid_count"] = valid_count
    summary["total_count"] = len(all_stats)

    return {
        "total_videos": len(all_stats),
        "per_video": all_stats,
        "summary": summary,
    }


def print_analysis(result: Dict[str, Any]) -> None:
    """Выводит анализ в читаемом формате."""
    print("=" * 80)
    print("АНАЛИЗ ВСЕХ РЕЗУЛЬТАТОВ voice_quality_extractor")
    print("=" * 80)
    print(f"\nВсего видео: {result['total_videos']}")
    print(f"Валидных: {result['summary'].get('valid_count', 0)}/{result['summary'].get('total_count', 0)}")

    if result["summary"].get("feature_values"):
        fv = result["summary"]["feature_values"]
        print("\n" + "=" * 80)
        print("СТАТИСТИКА feature_values")
        print("=" * 80)
        print(f"  Количество значений: {fv['count']}")
        print(f"  Среднее: {fv['mean']:.4f} ± {fv['std']:.4f}")
        print(f"  Диапазон: [{fv['min']:.4f}, {fv['max']:.4f}]")

    print("\n" + "=" * 80)
    print("ПО ВИДЕО")
    print("=" * 80)
    for s in result["per_video"]:
        status = "✅" if s.get("valid") else "❌"
        print(f"  {status} {s['video_id']}: segments={s.get('segments_count', '?')}, status={s.get('status', '?')}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Анализ результатов voice_quality_extractor")
    parser.add_argument("--rs-base", type=str, default="dp_results/youtube")
    parser.add_argument("--json", action="store_true", help="Вывод в формате JSON")
    args = parser.parse_args()

    result = analyze_all_results(args.rs_base)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_analysis(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
