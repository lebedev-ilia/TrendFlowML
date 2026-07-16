# CRITERIA.md — voice_quality_extractor

**Версия компонента:** 3.0.1  
**Схема:** voice_quality_extractor_npz_v1  
**Дата согласования:** 2026-07-16

---

## Универсальные хард-гейты (U1–U6)

| Гейт | Описание | Порог | Статус |
|------|----------|-------|--------|
| U1 | `validate_voice_quality.py --struct` → rc=0 | rc=0 | PASS ×17 |
| U2 | Ось времени: длины segment_mask, jitter_by_segment, shimmer_by_segment, hnr_by_segment = N | равны N | PASS ×17 |
| U3 | ok-путь: все 29 feature_values finite, не константа; empty-путь: 24/29 NaN by design | ok=finite, empty=NaN(24) | PASS |
| U4 | Expected-empty: audio_missing → `audio_missing_or_extract_failed`; all_segs_fail → `voice_quality_all_segments_failed` | valid empty NPZ | PASS ×17 |
| U5 | Golden-детерминизм (YIN CPU): max\|Δ\| = 0.0 по 5 прогонам | = 0.0 | PASS |
| U6 | Разные длины видео (3–58 сек, N=5–30) отрабатывают без падений | rc=0 | PASS ×5 |

---

## Компонентные критерии (C1–C4)

| Критерий | Описание | Порог |
|----------|----------|-------|
| C1 | vq_jitter ∈ [0,1]; vq_shimmer ∈ [0,1]; vq_hnr_like_db ∈ [-100,100] dB | в диапазоне |
| C2 | vq_voice_quality_score ∈ [0,1]; vq_breathiness_score ∈ [0,1] | в диапазоне |
| C3 | vq_f0_mean ∈ [f0_fmin, f0_fmax] = [50, 500] Hz | в диапазоне |
| C4 | NaN by design: при status=empty → все vq_* NaN, metadata finite (sample_rate/duration/f0_fmin/f0_fmax/segments_count) | ожидаемо |

---

## Задекларированные исключения

- **NaN by design (empty-путь):** при `status=empty` (`voice_quality_all_segments_failed` или `audio_missing_or_extract_failed`) 24/29 feature_values = NaN — это норма, не дефект.
- **vq_voice_presence_ratio в run_segments() режиме:** = n_ok/total_segments (доля успешных сегментов), не f0_frames/audio_samples как в run() legacy mode.
- **vq_f0_std в run_segments() режиме:** std межсегментных средних F0 (inter-segment variation), не intra-segment variation.
- **f0_method=torchcrepe → all_segments_failed:** если torchcrepe не установлен, все сегменты падают с ImportError — это корректное поведение. Рекоменд. метод для production: YIN.

---

## Рекомендуемая конфигурация (production)

```yaml
voice_quality:
  sample_rate: 22050  # или 16000 (оба работают корректно)
  f0_method: yin      # pyin/torchcrepe — только при установленных зависимостях
  enable_jitter: true
  enable_shimmer: true
  enable_hnr: true
  enable_f0_stats: false  # включать опционально
  enable_time_series: false  # включать опционально
```
