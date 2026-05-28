#!/usr/bin/env python3
"""
Анализ всех результатов action_recognition для проверки качества.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any
import json

import numpy as np

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.action_recognition.utils.validate_action_recognition import load_npz, extract_meta


def analyze_all_results(rs_base: str = "dp_results/youtube", include_tests: bool = True, include_audit: bool = True) -> Dict[str, Any]:
    """Анализирует все результаты action_recognition."""
    rs_path = Path(rs_base)
    
    all_stats = []
    all_metrics = {
        "stability": [],
        "stability_centroid_dist": [],
        "max_temporal_jump": [],
        "mean_temporal_jump": [],
        "num_clips": [],
        "num_switches": [],
        "tracks_count": [],
        "total_clips": [],
    }
    
    video_ids = []
    
    # Собираем все результаты
    # Поддерживаем как старую структуру (плоскую), так и новую (организованную)
    search_paths = []
    if include_tests:
        search_paths.extend(rs_path.glob("tests/action_recognition/test_action_recognition_*"))
        search_paths.extend(rs_path.glob("test_action_recognition_*"))  # Старая структура для обратной совместимости
    if include_audit:
        search_paths.extend(rs_path.glob("audit/v3/smoke/audit3_action_recognition_*"))
        search_paths.extend(rs_path.glob("audit3_action_recognition_*"))  # Старая структура
    
    for video_dir in sorted(search_paths):
        video_id = video_dir.name
        ar_dir = video_dir / video_id / "action_recognition"
        npz_path = ar_dir / "action_recognition_features.npz"
        if not npz_path.exists():
            npz_path = ar_dir / "action_recognition_emb.npz"
        
        if not npz_path.exists():
            continue
        
        try:
            npz_data = load_npz(str(npz_path))
            meta = extract_meta(npz_data)
            results_json = npz_data.get("results_json", [])
            
            tracks = npz_data.get("tracks", [])
            tracks_count = len(tracks) if isinstance(tracks, (list, np.ndarray)) else 0
            
            total_clips = 0
            for rj in results_json:
                if hasattr(rj, 'item'):
                    rj = rj.item()
                if isinstance(rj, dict):
                    total_clips += int(rj.get("num_clips", 0))
                    
                    # Собираем метрики
                    for key in ["stability", "stability_centroid_dist", "max_temporal_jump", 
                               "mean_temporal_jump", "num_clips", "num_switches"]:
                        val = rj.get(key)
                        if val is not None and not (isinstance(val, float) and np.isnan(val)):
                            all_metrics[key].append(float(val))
            
            all_metrics["tracks_count"].append(tracks_count)
            all_metrics["total_clips"].append(total_clips)
            
            all_stats.append({
                "video_id": video_id,
                "tracks_count": tracks_count,
                "total_clips": total_clips,
                "status": meta.get("status", "unknown"),
            })
            video_ids.append(video_id)
            
        except Exception as e:
            print(f"Error processing {video_id}: {e}", file=sys.stderr)
    
    # Вычисляем статистику
    summary = {}
    for key, values in all_metrics.items():
        if values:
            arr = np.asarray(values, dtype=np.float32)
            summary[key] = {
                "count": len(values),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
            }
    
    return {
        "total_videos": len(all_stats),
        "summary": summary,
        "per_video": all_stats,
    }


def print_analysis(result: Dict[str, Any]):
    """Выводит анализ в читаемом формате."""
    print("=" * 80)
    print("АНАЛИЗ ВСЕХ РЕЗУЛЬТАТОВ action_recognition")
    print("=" * 80)
    print(f"\nВсего видео: {result['total_videos']}")
    
    print("\n" + "=" * 80)
    print("ОБЩАЯ СТАТИСТИКА ПО МЕТРИКАМ")
    print("=" * 80)
    
    for metric, stats in result["summary"].items():
        print(f"\n{metric}:")
        print(f"  Количество значений: {stats['count']}")
        print(f"  Среднее: {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"  Медиана: {stats['median']:.3f}")
        print(f"  Диапазон: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"  Перцентили: 25%={stats['p25']:.3f}, 75%={stats['p75']:.3f}")
    
    print("\n" + "=" * 80)
    print("СТАТИСТИКА ПО ВИДЕО")
    print("=" * 80)
    
    tracks_counts = [s["tracks_count"] for s in result["per_video"]]
    clips_counts = [s["total_clips"] for s in result["per_video"]]
    
    print(f"\nТреки:")
    print(f"  Всего: {sum(tracks_counts)}")
    print(f"  Среднее на видео: {np.mean(tracks_counts):.1f}")
    print(f"  Медиана: {np.median(tracks_counts):.1f}")
    print(f"  Диапазон: [{min(tracks_counts)}, {max(tracks_counts)}]")
    
    print(f"\nКлипы:")
    print(f"  Всего: {sum(clips_counts)}")
    print(f"  Среднее на видео: {np.mean(clips_counts):.1f}")
    print(f"  Медиана: {np.median(clips_counts):.1f}")
    print(f"  Диапазон: [{min(clips_counts)}, {max(clips_counts)}]")
    
    print("\n" + "=" * 80)
    print("НАБЛЮДЕНИЯ")
    print("=" * 80)
    
    # Анализ наблюдений
    stability_stats = result["summary"].get("stability", {})
    if stability_stats:
        mean_stab = stability_stats.get("mean", 0)
        if mean_stab > 0.95:
            print("\n⚠️  ВНИМАНИЕ: Средняя stability очень высокая (>0.95)")
            print("   Это может означать, что большинство треков имеют только 1 клип.")
            print("   Для более информативного анализа нужны треки с несколькими клипами.")
    
    num_clips_stats = result["summary"].get("num_clips", {})
    if num_clips_stats:
        mean_clips = num_clips_stats.get("mean", 0)
        if mean_clips <= 1.1:
            print("\n⚠️  ВНИМАНИЕ: Среднее количество клипов на трек ≈ 1")
            print("   Это означает, что треки слишком короткие для анализа temporal patterns.")
            print("   Рекомендуется проверить параметры сегментации (segment_gap_sec).")
    
    temporal_jump_stats = result["summary"].get("max_temporal_jump", {})
    if temporal_jump_stats:
        mean_jump = temporal_jump_stats.get("mean", 0)
        if mean_jump < 0.01:
            print("\n⚠️  ВНИМАНИЕ: Temporal jumps очень низкие (<0.01)")
            print("   Это может означать, что действия очень стабильны,")
            print("   или что треки слишком короткие для анализа изменений.")
    
    print("\n" + "=" * 80)
    print("РЕКОМЕНДАЦИИ")
    print("=" * 80)
    
    print("\n1. Проверьте HTML рендеры для визуальной оценки качества")
    print("2. Сравните результаты между похожими видео")
    print("3. Проверьте корректность сегментации (группировка person детекций)")
    print("4. Для более информативных метрик нужны треки с num_clips > 1")
    print("5. Рассмотрите настройку параметров segment_gap_sec и min_person_confidence")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Анализ всех результатов action_recognition")
    parser.add_argument(
        "--rs-base",
        type=str,
        default="dp_results/youtube",
        help="Базовая директория с результатами"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Вывод в формате JSON"
    )
    
    args = parser.parse_args()
    
    result = analyze_all_results(args.rs_base)
    
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_analysis(result)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

