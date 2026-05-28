# `emotion_diarization_extractor` — описание фич и трассировка артефактов

Реализация: [`main.py`](main.py) (`EmotionDiarizationExtractor`), NPZ: [`npz_savers/emotion_diarization.py`](../../core/npz_savers/emotion_diarization.py). Схема: `docs/SCHEMA.md`, обзор: `docs/README.md`. Артефакт: `emotion_diarization_extractor/emotion_diarization_extractor_features.npz`.

---

## 1. Код: режимы и `payload` (Audit v3)

- **Только `run_segments()`** с family **`emotion`** в `segments.json`. `run()` отключён.
- Модель SpeechBrain / веса **только через `dp_models`** (без сети).
- **Короткое аудио:** если `max(ends) - …` < **5 с** — валидный **empty** (`empty_reason=audio_too_short`), маски/`-1`/`NaN` на длине N.
- **Тишина** (при `enable_silence_detection`): `audio_silent`, маска нулевая.
- **Строгая длина N** у всех сегментных массивов; невалидные окна — `segment_mask[i]=false`, `emotion_id=-1`, `confidence=NaN`.

| Группа | Поля (см. `main.py` + SCHEMA) |
|--------|------------------------------|
| Ось времени | `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask` |
| Per-segment | `emotion_id`, `emotion_confidence` |
| Агрегаты (tabular `add`) | `segments_count`, `emotion_entropy`, `dominant_emotion_id` / `prob`, `emotion_transitions_count`, `emotion_stability_score`, `emotion_diversity_score` |
| Опц. (флаги) | `enable_probs` → `emotion_probs` [N,C]; `enable_mean_probs`; `enable_dominant` → distribution dicts; `enable_quality_metrics` → `emotion_quality_metrics` |
| Мета-поля | `sample_rate`, `device_used`, `model_name`, `weights_digest`, `segments_total`, `silence_*_threshold` (в meta через савер), `stage_timings_ms`, `emotion_contract_version` |

**Тайминги в meta:** ключи `*_ms` (из `*_sec` внутри экстрактора × 1000): `load_segments_ms`, `silence_detection_ms`, `padding_ms`, `inference_ms`, `aggregates_ms`, `postprocess_ms`, `total_ms`.

---

## 2. NPZ

- **Всегда:** tabular, `segment_*`, `emotion_id`, `emotion_confidence`, `emotion_labels`, `meta`.
- **Опц.:** `emotion_probs`, `emotion_mean_probs`, object-scalars `emotion_distribution` / `emotion_segments_per_emotion` / `emotion_duration_per_emotion`, `emotion_quality_metrics`.
- `schema_version`: **`emotion_diarization_extractor_npz_v1`**.

Пример с включёнными probs/dominant/quality: **16** файлов в архиве (см. последний прогон).

---

## 3. Batch CSV

Плоский **`meta`**; tabular `feature_values` **не** дублируется в wide CSV (как у других audio-модулей).

---

## 4. Melt / пояснения

`view_csv_melt_interesting.json` → `emotion_diarization_extractor`; `view_csv_feature_qa.json`; `view_csv_feature_descriptions_ru.json`.

---

## 5. CLI

`utils/validate_emotion_diarization.py <npz> [--struct] [--qa]`

---

## 6. Чеклист

1. Family `emotion`, непустой список сегментов (иначе error).  
2. Длина N согласована по `segment_*`, `emotion_id`, `emotion_confidence`.  
3. При `enable_probs`: `emotion_probs.shape[0] == N`.  
4. Скалярные агрегаты — из NPZ `feature_values` / tabular, не из CSV-only.
