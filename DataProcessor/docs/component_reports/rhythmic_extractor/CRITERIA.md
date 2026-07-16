# Критерии приёмки — rhythmic_extractor

Согласованы: 2026-07-16

## Универсальные хард-гейты (U1–U6)

| Гейт | Описание |
|------|----------|
| U1 | `validate_rhythmic.py` возвращает rc=0 на всех NPZ из storage |
| U2 | Ось времени: `segment_start_sec` монотонна; `segment_end_sec > segment_start_sec`; длины segment_* массивов одинаковы |
| U3 | finite/health: `feature_values` finite там где не NaN by design; `rhythm_tempo_bpm` ∈ [40,300] когда finite; `rhythm_regularity` ∈ [0,1] когда finite |
| U4 | Expected-empty: 4 empty NPZ в storage имеют `status=empty`, валидную схему, `feature_values` all-NaN; rc=0 |
| U5 | Golden (синтетик/повтор): librosa CPU детерминирован → `max|Δ feature_values|=0.0` при повторном вычислении |
| U6 | Разные длины: в storage присутствуют NPZ с `segments_count` ∈ {5, 11, 12, 16, 17, 31} — отрабатывают без падений |

## Критерии компонента (C1–C4)

| Критерий | Описание |
|----------|----------|
| C1 | `rhythm_tempo_bpm` ∈ [40,300] когда finite — контракт компонента (на storage: 117–144 BPM) |
| C2 | `rhythm_regularity` ∈ [0,1] когда finite (на storage: 0.017–0.557) |
| C3 | NaN by design при `beats_count=0`: `rhythm_tempo_bpm`, `rhythm_regularity`, `rhythm_tempo_variation`, `rhythm_beat_consistency` = NaN — метрики неопределены без битов |
| C4 | `feature_names` фиксированы: ровно 9 имён в постоянном порядке: `rhythm_tempo_bpm, rhythm_beats_count, rhythm_beat_density, rhythm_regularity, rhythm_tempo_variation, rhythm_beat_consistency, duration_sec, sample_rate, segments_count` |

## Примечания
- Empty-path reason: `audio_missing_or_extract_failed` → feature_values all-NaN by design (нет аудио)
- Empty-path reason: `rhythmic_all_segments_failed` → `beats_count=0`, `beat_density=0.0`, `duration_sec/sample_rate/segments_count` конкретные значения by design (аудио есть, beat tracking упал)
- U2 исключение: нулевые сегменты (`end_sec == start_sec`) с `segment_mask=False` — by design (Segmenter передал нулевой сегмент, компонент его пометил как failed)
- beat_times сохраняются в NPZ при enable_beat_times=True; при > 10000 битов — в .npy (баг исправлен в v2.0.1)
- GPU не нужен (CPU-only librosa)
