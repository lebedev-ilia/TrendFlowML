# `asr_extractor` — описание фич и трассировка артефактов

Документ связывает **логику кода** (`main.py` + `npz_savers/asr.py`), **NPZ**, **сводный batch CSV** и **melt-HTML** (`view_csv.py` + `view_csv_melt_interesting.json`).

Детальная схема ключей NPZ: `docs/SCHEMA.md`, `docs/README.md` (пользовательский README).

---

## 1. Исход кода: что считается и что попадает в `payload`

Класс `ASRExtractor` в `main.py` формирует `payload` при `run_segments` / `extract_batch_segments`. Основные группы:

| Группа | Содержимое |
|--------|------------|
| Идентичность / декод | `whisper_model_name`, `tokenizer_model_name`, `tokenizer_weights_digest`, `device_used`, `asr_text_contract_version`, параметры `decode_*` |
| Сегментация / v2 | `audio_duration_sec`, `asr_sampling_profile`, `asr_window_sec`, `asr_stride_sec`, `asr_max_windows` |
| Сегменты (всегда при успехе) | `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `lang_id_by_segment`, `lang_code_by_segment`, `lang_conf_by_segment` |
| Токены | `token_ids_by_segment` (флаг `enable_token_sequences`) |
| Счёты/агрегаты | `token_counts`, `token_total`, `token_density_per_sec`, `speech_rate_wpm`, `lang_distribution`, `segments_with_speech`, `avg_segment_duration_sec`, `token_variance` — **по флагам** `enable_*` |
| Качество | `segment_quality_by_segment` (числа: `avg_logprob`, `compression_ratio`, `no_speech_prob`, `temperature`) |
| Текст (отладка) | `segment_texts_by_segment` только при `save_segment_text` |
| Профилирование | `asr_stage_timings_ms`, `asr_resource_profile` (`_merge_asr_profiler_meta`) |
| Служебное | `_features_enabled` — список имён включённых фич (для meta) |

Контракт версии текста: `ASR_TEXT_CONTRACT_VERSION` (`asr_text_contract_v1`).

---

## 2. NPZ: ключи и что где лежит

Имя файла: `asr_extractor_features.npz` (см. `npz_saver.save_component_npz`).

### 2.1. Табличный вектор

- `feature_names` (object) / `feature_values` (float32) — пары, собранные чрез `add()` в `npz_savers/asr.py`: числовые и скаляры агрегатов, в т.ч. `segments_count`, `sample_rate`, `token_total`, `token_*`, `asr_quality__*` (mean / p50 / p90 / present_rate), и т.д. Все значения приводятся к float через `as_float`.

### 2.2. Массивы

См. `docs/SCHEMA.md`: `token_ids_by_segment`, сегментные массивы, `lang_*`, `token_counts`, `lang_distribution`, `segment_quality_by_segment`, скаляры v2 `audio_duration_sec`, `asr_*`.

### 2.3. `meta` (словарь в `meta` object scalar)

- База `build_meta`: `producer`, `producer_version`, `schema_version`, `status`, `created_at` (+ `apply_models_meta`: `models_used`, `model_signature` и т.д.).
- Из `run_cli` (`extra_meta`): `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`, `device_used`, `models_used`, `scheduler_knobs`, **`stage_timings_ms`**, `resource_metrics`.
- Из савера `asr`: `error`, `empty_reason`, `asr_text_contract_version`, `features_enabled`, **`asr_stage_timings_ms`**, `asr_resource_profile`.

`stage_timings_ms` — **оркестратор** (стена компонента + этапы pipeline). `asr_stage_timings_ms` — **внутри ASR** (load / infer / mel / decode / aggregates / total / …).

---

## 3. Batch CSV (`batch_runs_feature_report.py`)

Используется `extract_meta` + `_flatten_meta(meta, prefix="meta_")`.

- Попадают: **скаляры** bool → int, int/float, строка длины **меньше 200 символов**.
- `stage_timings_ms` (dict) → **`meta_timing_<ключ>`** (только числовые подполя) — **специальный случай** в `_flatten_meta`.
- `asr_stage_timings_ms` (dict) → **`meta_asr_timing_<ключ>`** — специальный случай (те же правила, что для pipeline timings).
- **Не** попадают в плоский CSV: вложенные dict/list, кроме обработанных (например `features_enabled`, `asr_resource_profile`, `resource_metrics`, `scheduler_knobs`, `models_used` — списки/объекты без плоского представления).
- **Табличные** `feature_names` / `feature_values` **не** мержатся в строку CSV: сводка видит только `meta` + поля `manifest_*`, `device_used` merge и т.д. Чтобы сравнивать агрегаты `token_total` и т.д. в таблице, нужно грузить NPZ напрямую или расширить отчёт (см. раздел 5).

Колонки на уровне run: `platform_id`, `video_id`, `run_id`, `component`, `duration_ms` (из манифеста), `manifest_*`, `device_used`, `npz`, `npz_error`, `render_error`.

---

## 4. Melt-HTML / `view_csv_melt_interesting.json`

- `defaults.merge_into_each`: общие для всех колонок (`duration_ms`, `manifest_status`, `manifest_empty_reason`, `device_used`).
- Для `asr_extractor`: явный `include` (мета, контракт, версии) + `add_all_meta_timing` (все `meta_timing_*` из фактического CSV) + **`add_all_meta_asr_timing`** (все `meta_asr_timing_*` после выгрузки ASR-стадий).
- Секунды в ячейках: эвристика `*_ms` (в т.ч. `meta_asr_timing_*_ms`) — см. `view_csv._melt_column_is_milliseconds`.

---

## 5. Известные пробелы / улучшения

1. **Табличные агрегаты из NPZ** (`feature_names`/`feature_values`) не в batch CSV — сводка meta-only; аналитика по `token_total` / `asr_quality__*` — из NPZ или доработка `batch_runs_feature_report`.
2. **`resource_metrics` / `asr_resource_profile`**: в CSV не разворачиваются; смотреть `meta` в NPZ или render `_render/`.

---

## 6. Быстрый чеклист аудита

1. Код: флаги `enable_*` в `ASRExtractor.__init__` и заполнение `payload` / `_features_enabled`.
2. NPZ: `unzip -l …/asr_extractor_features.npz` или `numpy.load` — список ключей + `meta.item()`.
3. CSV: строка `component=asr_extractor` — `meta_*`, `meta_timing_*`, `meta_asr_timing_*` (после обновления `_flatten_meta`).
4. Melt: пересобрать HTML с `--melt-interesting` и убедиться, что важные поля не пустые по пилотным run’ам.

---

## 7. QA: нормальные диапазоны и валидатор

- **Единый JSON** (все компоненты, фрагмент для ASR): `storage/result_store/view_csv_feature_qa.json` — блоки `any_component` и `components.asr_extractor` (min/max в тех же единицах, что и в wide CSV: мс для `*_ms`).
- **Логика проверки** (плоский meta как в batch): `DataProcessor/qa/component_feature_qa.py` (`flatten_meta`, `QaConfig`).
- **Melt-HTML**: `view_csv.py --melt --melt-interesting --melt-qa` — ячейки вне диапазона подсвечиваются, текст причины в `title`.
- **CLI по NPZ**: `utils/validate_asr.py <path.npz> --qa` — схема + предупреждения по правилам для `asr_extractor` (колонки из плоского meta; `duration_ms` из манифеста в NPZ нет — смотрите wide CSV при полном аудите).

Нормы **пилотные**; при сравнении прогонов сужайте `max` для таймингов и фиксируйте изменения в `view_csv_feature_qa.json` и в этом разделе.
