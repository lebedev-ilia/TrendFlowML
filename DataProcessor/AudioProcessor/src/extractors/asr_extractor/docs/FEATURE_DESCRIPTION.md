# `asr_extractor` — выходы и фичи

**Реализация:** [`../main.py`](../main.py) · **NPZ-схема:** [`../../../../schemas/asr_extractor_npz_v2.json`](../../../../schemas/asr_extractor_npz_v2.json) · **Валидатор:** [`../utils/validate_asr.py`](../utils/validate_asr.py).

## Табличный срез

Плоский слой: **`feature_names` / `feature_values`** (скаляры для отчёта/CSV). Полный набор и семантика — в SCHEMA/саёере NPZ.

## Массивы и сегментация (смысл)

| Ключ / группа | Смысл |
|---------------|--------|
| `segment_start_sec` / `end` / `center` | Временная ось сегментов, сек |
| `lang_code_by_segment`, `lang_conf_by_segment` | Язык и уверенность |
| `audio_duration_sec` | Длительность исходного аудио |
| `asr_window_sec`, `asr_stride_sec`, `asr_max_windows` | Параметры окон/stride |
| `token_ids_by_segment` | Токен-ID (Whisper), без сырого текста в артефактах |
| `meta` | `status`, тайминги, `stage_timings_ms`, device |

**Ориентиры (нормальные диапазоны):** длительности ≥ 0; `lang_conf` ∈ [0,1]; токен-ID в пределах словаря Whisper (см. код констант). Детали проверок — `--qa` / `--struct` в валидаторе.

## HTML / melt

Подписи к колонкам в melt: `view_csv_feature_descriptions_ru.json` + `view_csv_melt_captions_ru.py`. Пороги подсветки: `view_csv_feature_qa.json` → секция `asr_extractor` (при `--melt-qa`).
