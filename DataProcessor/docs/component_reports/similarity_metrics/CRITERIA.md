# Критерии приёмки: similarity_metrics

Версия: 2.0.2 · Схема: similarity_metrics_npz_v3 · Дата: 2026-07-16

## Универсальные хард-гейты

| Гейт | Критерий |
|------|----------|
| U1 | validate_similarity_metrics_npz.py → rc=0 на всех NPZ |
| U2 | times_s монотонно неубывает (ranges check) |
| U3 | centroid_sims, temporal_sim_next ∈ [-1, 1] (cosine similarity) |
| U4 | N=1 → temporal_sim_next len=0, temporal_sim_mean=NaN; нет core_clip → FileNotFoundError (no-fallback) |
| U5 | Golden-детерминизм: max\|Δ\| = 0.0 (чистый numpy, детерминирован) |
| U6 | F=39 стабильно при N=1/10/43/100 |

## Критерии под компонент

| # | Критерий | Порог / описание |
|---|----------|------------------|
| C1 | NaN by design | 24/39 feature_values = NaN когда reference_present=False — это норма (нет reference pack). uniqueness_score, uniqueness_clip, uniqueness_overall, все reference_similarity_* = NaN by design. |
| C2 | Вариативность centroid_sim_mean | CV ≥ 1% на корпусе видео (измерено: 4.8% на 23 видео) |
| C3 | feature_values shape | F=39 (стабильно при любом N) |
| C4 | Когерентность всегда finite | centroid_sims, temporal_sim_next, n_frames — always finite при status=ok |

## Примечание о мёртвом коде
Методы `compute_style_similarity`, `compute_text_similarity`, `compute_audio_similarity`, `compute_emotion_behavior_similarity`, `compute_temporal_similarity`, `compute_high_level_scores`, `compute_batch_metrics`, `extract_all` и `if __name__=='__main__'` блок в similarity_metrics.py — мёртвый код от другой версии (`SimilarityMetrics` класса которого нет). Используют незадекларированные импорты (cosine, wasserstein_distance, pearsonr). Не вызываются в production. Отмечены как WARNING (не FAIL).
