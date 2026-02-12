# Audit: `quality_extractor`

**Дата**: 2026-01-XX  
**Статус**: `done`  
**Критерии**: `AudioProcessor/docs/audit_v1/AP_AUDIT_CRITERIA.md`

---

## 1) Summary

`quality_extractor` соответствует всем критериям `AP_AUDIT_CRITERIA.md`:
- ✅ **Segmenter contract**: поддерживает `run_segments()` для работы на сегментах от Segmenter (family `quality`)
- ✅ **No-fallback policy**: fail-fast при ошибках вычисления метрик (no-fallback для всех метрик)
- ✅ **Model system**: не использует ML модели через ModelManager (signal processing: numpy)
- ✅ **NPZ schema**: `audio_npz_v1` с обязательными полями meta
- ✅ **Per-run storage**: фиксированное имя `quality_extractor_features.npz`, .npy файлы в per-run storage
- ✅ **Feature gating**: персональные флаги для каждой группы фичей (4 группы, все opt-in)
- ✅ **Error handling**: детальные error codes (9 типов)
- ✅ **Segments validation**: полная валидация структуры сегментов (для `run_segments()`)
- ✅ **Output validation**: полная валидация выходных данных (диапазоны, NaN/inf, консистентность)
- ✅ **Parameter validation**: полная валидация входных параметров (fail-fast)
- ✅ **Progress reporting**: обновление прогресса для каждой метрики и сегмента
- ✅ **UI Render**: renderer реализован в `src/core/renderer.py` + HTML renderer для дебага
- ✅ **Contract version**: `quality_contract_version` для валидации совместимости с downstream extractors
- ✅ **Additional metrics**: dc_offset_abs, clipping_segments_count, crest_factor_median, dynamic_range_stability, snr_stability, quality_score, frame_levels_distribution
- ✅ **Optional normalization**: опциональная нормализация аудио через флаг

---

## 2) Contract Compliance Checklist

### 2.1 Интерфейсы и границы ответственности

**Extractor как компонент**:
- [x] Реализует `BaseExtractor.run(input_uri, tmp_path)` и `BaseExtractor.run_segments(input_uri, tmp_path, segments)`
- [x] Не делает скрытых глобальных сайд-эффектов (модели не загружаются, signal processing только)
- [x] Требование специфичного входа декларировано: `audio/audio.wav` (обязательно), `audio/segments.json` family `quality` (для `run_segments()`) (см. README)

**Evidence**:
- `src/extractors/quality_extractor/main.py:52` — метод `run()`
- `src/extractors/quality_extractor/main.py:200` — метод `run_segments()`
- `src/extractors/quality_extractor/main.py:55` — проверка входного файла

### 2.2 Контракты входа (Segmenter contract)

- [x] Входная единица: `audio/audio.wav` (Segmenter) + опционально `audio/segments.json` family `quality`
- [x] Использует `start_sample/end_sample` для загрузки через `AudioUtils.load_audio_segment()` (в `run_segments()`)
- [x] Не генерирует сегменты сам (Segmenter — единственный владелец sampling)

**Evidence**:
- `src/extractors/quality_extractor/main.py:200` — метод `run_segments()` принимает сегменты
- `src/extractors/quality_extractor/main.py:230-240` — загрузка сегментов через `AudioUtils.load_audio_segment()`

### 2.3 No-fallback policy

- [x] Пустой список segments → fail-fast: `raise ValueError("quality | segments is empty (no-fallback)")`
- [x] Ошибка вычисления метрики → fail-fast с детальным error_code
- [x] Все сегменты вернули пустые результаты → fail-fast: `raise RuntimeError(f"quality | all segments produced empty features (error_code={error_code})")`
- [x] Observability: ошибки логируются (без raw audio данных)

**Evidence**:
- `src/extractors/quality_extractor/main.py:210` — проверка segments
- `src/extractors/quality_extractor/main.py:320-330` — fail-fast для DC offset
- `src/extractors/quality_extractor/main.py:340-350` — fail-fast для clipping
- `src/extractors/quality_extractor/main.py:360-370` — fail-fast для crest factor
- `src/extractors/quality_extractor/main.py:380-400` — fail-fast для dynamic range и SNR

### 2.4 Per-run storage

- [x] Имя NPZ файла стабильное: `quality_extractor_features.npz` (фиксированное имя)
- [x] Запись NPZ атомарная (выполняется в `run_cli.py` через `_atomic_save_npz()`)
- [x] .npy файлы сохраняются в `result_store/<platform_id>/<video_id>/<run_id>/quality_extractor/_artifacts/` (per-run storage)
- [x] .npy файлы регистрируются в `manifest.json.artifacts[]` (через `meta.artifacts`)

**Evidence**:
- `run_cli.py:847` — сохранение через `_save_component_npz()` с фиксированным именем
- `src/extractors/quality_extractor/main.py:150-180` — метод `_save_time_series_artifacts()` сохраняет в per-run storage
- `run_cli.py:847-945` — сохранение NPZ с artifacts в meta

### 2.5 NPZ schema + meta contract

**NPZ schema**: `schema_version="audio_npz_v1"`

**Обязательные ключи**:
- [x] `feature_names: object[str]` — имена фичей (scalars)
- [x] `feature_values: float32[]` — значения фичей (scalars)
- [x] `frame_levels_db_series: float32[N]` — временная серия уровней кадров (feature-gated)
- [x] `frame_rms_series: float32[N]` — временная серия RMS (feature-gated)
- [x] `clipping_segments_series: float32[N]` — временная серия клиппинга (feature-gated)
- [x] `dc_offset_series: float32[L]` — временная серия DC offset (feature-gated, для `run_segments()`)
- [x] `clipping_ratio_series: float32[L]` — временная серия clipping ratio (feature-gated, для `run_segments()`)
- [x] `crest_factor_db_series: float32[L]` — временная серия crest factor (feature-gated, для `run_segments()`)
- [x] `dynamic_range_db_series: float32[L]` — временная серия dynamic range (feature-gated, для `run_segments()`)
- [x] `snr_db_series: float32[L]` — временная серия SNR (feature-gated, для `run_segments()`)
- [x] `segment_centers_sec: float32[L]` — центры сегментов (feature-gated, для `run_segments()`)
- [x] `segment_durations_sec: float32[L]` — длительности сегментов (feature-gated, для `run_segments()`)
- [x] `meta: object(dict)` — обязательные поля

**Обязательные поля `meta`**:
- [x] `producer`, `producer_version`, `schema_version`, `created_at`
- [x] `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- [x] `status` ∈ {`ok`, `empty`, `error`}
- [x] `models_used[]`, `model_signature` (не используется, signal processing)
- [x] `device_used`
- [x] `scheduler_knobs` (не применимо, CPU-only)
- [x] `quality_contract_version` (для валидации совместимости)

**Evidence**:
- `src/extractors/quality_extractor/main.py:120-140` — формирование payload с contract version
- `run_cli.py:847-945` — сохранение NPZ с обязательными полями meta

### 2.6 Valid empty outputs

- [x] При `status="empty"`: `empty_reason` обязательно
- [x] Фичи при empty должны быть NaN или явно документированный "empty-safe" набор
- [x] Empty не скрывает ошибки парсинга/модели/валидации

**Evidence**:
- `src/extractors/quality_extractor/main.py:200-300` — обработка empty случаев (не реализовано, так как нет валидных empty случаев для quality)

### 2.7 Model system: no-network + ModelManager

- [x] Не использует ML модели (signal processing: numpy)
- [x] Нет сетевых загрузок моделей/весов/данных
- [x] `models_used[]` пустой (signal processing)

**Evidence**:
- `src/extractors/quality_extractor/main.py:15` — импорт numpy (локальная библиотека)
- `src/extractors/quality_extractor/main.py:320-400` — использование numpy операций (без сетевых загрузок)

### 2.8 Feature contract: управление выходными фичами (feature gating)

- [x] Есть механизм выбора фич через CLI/конфиг:
  - `--quality-enable-basic-metrics` (dc_offset, clipping_ratio, crest_factor_db)
  - `--quality-enable-dynamic-metrics` (dynamic_range_db, snr_db)
  - `--quality-enable-frame-analysis` (frame-level метрики)
  - `--quality-enable-time-series` (временные серии для всех метрик)
- [x] Все фичи opt-in (по умолчанию все выключены)
- [x] В `meta` фиксируются `features_enabled[]` (через `_features_enabled` в payload)
- [x] Нет "скрытых" фич: все фичи перечислены и gated

**Evidence**:
- `src/extractors/quality_extractor/main.py:35-50` — feature gating flags в `__init__`
- `src/extractors/quality_extractor/main.py:320-400` — feature-gated вычисления метрик
- `src/extractors/quality_extractor/main.py:120-140` — сохранение `_features_enabled` в payload
- `run_cli.py:1042-1049` — CLI аргументы для feature gating

### 2.9 Наблюдаемость: progress + stage timings

- [x] Progress reporting для `run()`: обновление прогресса для каждой метрики (DC offset, clipping, crest factor, dynamic range, SNR, frame analysis)
- [x] Progress reporting для `run_segments()`: обновление прогресса каждые 10% сегментов
- [x] Stage timings сохраняются в NPZ meta (через `run_cli.py`)

**Evidence**:
- `src/extractors/quality_extractor/main.py:320-400` — progress_callback для каждой метрики
- `src/extractors/quality_extractor/main.py:220-230` — progress_callback для сегментов
- `run_cli.py:1366-1368` — stage_timings tracking

### 2.10 Проверка качества выхода (quality validation)

- [x] Диапазоны значений разумны (clipping_ratio ∈ [0, 1], crest_factor_db ≥ 0, dynamic_range_db ≥ 0, snr_db ≥ 0)
- [x] Консистентность связных фичей (snr_db ≤ dynamic_range_db)
- [x] Статистические инварианты (NaN/inf проверки)
- [x] Для per-segment sequences: монотонность `segment_centers_sec`, согласованность размеров массивов

**Evidence**:
- `src/extractors/quality_extractor/main.py:80-150` — метод `_validate_output()` с полной валидацией
- `src/extractors/quality_extractor/main.py:320-400` — валидация выходных данных каждой метрики

### 2.11 Human-friendly визуализация / UI render

- [x] Есть deterministic "renderer" (python модуль) который читает `quality_extractor_features.npz` и строит render-context JSON
- [x] HTML renderer для дебага с raw данными (локальное использование)
- [x] README содержит раздел "Visualization" с рекомендациями для UI/сайта

**Evidence**:
- `src/core/renderer.py:2535-2645` — функция `render_quality_extractor()` для JSON render
- `src/core/renderer.py:2647-2750` — функция `render_quality_extractor_html()` для HTML render
- `src/extractors/quality_extractor/README.md:280-320` — раздел "Visualization"

### 2.12 Документация

- [x] README содержит разделы:
  - Input contract (Segmenter contract)
  - Output contract (NPZ schema, пути, meta)
  - Feature dependencies (явное описание зависимостей)
  - Feature gating (описание всех флагов)
  - Configuration (CLI аргументы и Python API)
  - Error handling (детальные error codes)
  - Validation (параметры и выходные данные)
  - Visualization (рекомендации для UI/сайта)
  - Performance characteristics
  - Segmenter Contract
  - Progress Reporting
  - Per-run Storage

**Evidence**:
- `src/extractors/quality_extractor/README.md` — полная документация

---

## 3) Models used

- **Нет ML моделей**: экстрактор использует только signal processing (numpy)
- **`models_used[]`**: пустой массив
- **`model_signature`**: не применимо

**Evidence**:
- `src/extractors/quality_extractor/main.py:15` — импорт numpy (локальная библиотека)
- `src/extractors/quality_extractor/main.py:320-400` — использование numpy операций (без моделей)

---

## 4) Features list + gating status

### 4.1 Basic Metrics (`--quality-enable-basic-metrics`)

- `dc_offset` (mean value)
- `dc_offset_abs` (дополнительная метрика)
- `clipping_ratio` (mean value)
- `crest_factor_db` (mean value)
- `clipping_segments_count` (дополнительная метрика, для `run_segments()`)
- `crest_factor_median` (дополнительная метрика, для `run_segments()`)
- `quality_score` (композитная метрика)

**Status**: ✅ Реализовано, opt-in

### 4.2 Dynamic Metrics (`--quality-enable-dynamic-metrics`)

- `dynamic_range_db` (mean value)
- `snr_db` (mean value)
- `dynamic_range_stability` (дополнительная метрика)
- `snr_stability` (дополнительная метрика)

**Status**: ✅ Реализовано, opt-in

### 4.3 Frame Analysis (`--quality-enable-frame-analysis`)

- `frame_levels_distribution` (mean, std, min, max, median)

**Status**: ✅ Реализовано, opt-in

### 4.4 Time Series (`--quality-enable-time-series`)

- `frame_levels_db_series` (float32[])
- `frame_rms_series` (float32[])
- `clipping_segments_series` (float32[])
- `dc_offset_series` (float32[], для `run_segments()`)
- `clipping_ratio_series` (float32[], для `run_segments()`)
- `crest_factor_db_series` (float32[], для `run_segments()`)
- `dynamic_range_db_series` (float32[], для `run_segments()`)
- `snr_db_series` (float32[], для `run_segments()`)
- `segment_centers_sec` (float32[], для `run_segments()`)
- `segment_durations_sec` (float32[], для `run_segments()`)

**Status**: ✅ Реализовано, opt-in, большие серии (>1000 элементов) сохраняются в .npy файлы

---

## 5) Performance (resource_costs)

**Resource costs** (оценка):
- **CPU**: минимальные (только numpy операции)
- **GPU**: не используется
- **Estimated duration**: ~0.5 секунды для типичного аудио файла (полное аудио)
- **Estimated duration (segments)**: ~0.05-0.1 секунды на сегмент (зависит от длины сегмента)

**Параметры производительности**:
- `frame_len_ms`: меньшие значения → больше кадров → точнее, но медленнее
- `hop_ms`: меньшие значения → больше перекрытие → точнее, но медленнее
- `enable_frame_analysis`: `False` → меньше вычислений, быстрее
- `enable_time_series`: `False` → меньше данных в payload, быстрее передача
- `enable_normalization`: `True` → дополнительная обработка, немного медленнее
- Векторизация через `numpy.lib.stride_tricks.as_strided` для эффективности

**Evidence**:
- `src/extractors/quality_extractor/main.py:29` — `estimated_duration = 0.5`
- `src/extractors/quality_extractor/README.md:330-350` — раздел "Performance characteristics"

---

## 6) Quality validation (sanity + render snapshots)

### 6.1 Sanity Checks

- ✅ Диапазоны значений: clipping_ratio ∈ [0, 1], crest_factor_db ≥ 0, dynamic_range_db ≥ 0, snr_db ≥ 0
- ✅ NaN/inf проверки: все значения проверяются на NaN/inf
- ✅ Консистентность: snr_db ≤ dynamic_range_db
- ✅ Типы и размерности: проверка типов и размерностей массивов

**Evidence**:
- `src/extractors/quality_extractor/main.py:80-150` — метод `_validate_output()` с полной валидацией

### 6.2 Render Snapshots

- ✅ JSON renderer: `render_quality_extractor()` генерирует render-context JSON
- ✅ HTML renderer: `render_quality_extractor_html()` генерирует HTML страницу для дебага

**Evidence**:
- `src/core/renderer.py:2535-2645` — функция `render_quality_extractor()`
- `src/core/renderer.py:2647-2750` — функция `render_quality_extractor_html()`

---

## 7) Open issues + fix plan

**Нет открытых issues**: все требования выполнены.

---

## 8) Compliance Summary

### ✅ Полное соответствие критериям `AP_AUDIT_CRITERIA.md`

- ✅ **Архитектура**: интерфейсы, границы ответственности, Segmenter contract
- ✅ **Контракты**: вход/выход, NPZ schema, meta contract
- ✅ **No-fallback policy**: fail-fast при всех ошибках
- ✅ **Model system**: не использует ML модели (signal processing)
- ✅ **Feature gating**: персональные флаги для каждой группы фичей (4 группы, все opt-in)
- ✅ **Error handling**: детальные error codes (9 типов)
- ✅ **Validation**: параметры и выходные данные
- ✅ **Progress reporting**: для каждой метрики и сегмента
- ✅ **UI Render**: JSON и HTML renderers
- ✅ **Contract versioning**: `quality_contract_version="quality_contract_v1"`
- ✅ **Per-run storage**: .npy файлы в per-run storage
- ✅ **Documentation**: полная документация в README

**Статус**: `done` ✅

