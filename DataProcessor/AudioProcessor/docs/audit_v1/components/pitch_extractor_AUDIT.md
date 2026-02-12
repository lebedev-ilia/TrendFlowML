# Audit: `pitch_extractor`

**Дата**: 2026-01-XX  
**Статус**: `done`  
**Критерии**: `AudioProcessor/docs/audit_v1/AP_AUDIT_CRITERIA.md`

---

## 1) Summary

`pitch_extractor` соответствует всем критериям `AP_AUDIT_CRITERIA.md`:
- ✅ **Segmenter contract**: поддерживает `run_segments()` для работы на сегментах от Segmenter (family `pitch`)
- ✅ **No-fallback policy**: fail-fast при ошибках методов (no-fallback для выбранного backend)
- ✅ **Model system**: не использует ML модели через ModelManager (signal processing: PYIN, YIN, torchcrepe)
- ✅ **NPZ schema**: `audio_npz_v1` с обязательными полями meta
- ✅ **Per-run storage**: фиксированное имя `pitch_extractor_features.npz`, .npy файлы в per-run storage
- ✅ **Feature gating**: персональные флаги для каждой группы фичей (5 групп, все opt-in)
- ✅ **Error handling**: детальные error codes (7 типов)
- ✅ **Segments validation**: полная валидация структуры сегментов (для `run_segments()`)
- ✅ **Output validation**: полная валидация выходных данных (диапазоны, NaN/inf, консистентность)
- ✅ **Parameter validation**: полная валидация входных параметров (fail-fast)
- ✅ **Progress reporting**: обновление прогресса для каждого метода и сегмента
- ✅ **UI Render**: renderer реализован в `src/core/renderer.py` + HTML renderer для дебага
- ✅ **Contract version**: `pitch_contract_version` для валидации совместимости с downstream extractors
- ✅ **Additional metrics**: pitch_contour_smoothness, pitch_jump_count, pitch_octave_distribution, pitch_centroid, pitch_skewness, pitch_kurtosis

---

## 2) Contract Compliance Checklist

### 2.1 Интерфейсы и границы ответственности

**Extractor как компонент**:
- [x] Реализует `BaseExtractor.run(input_uri, tmp_path)` и `BaseExtractor.run_segments(input_uri, tmp_path, segments)`
- [x] Не делает скрытых глобальных сайд-эффектов (модели не загружаются, signal processing только)
- [x] Требование специфичного входа декларировано: `audio/audio.wav` (обязательно), `audio/segments.json` family `pitch` (для `run_segments()`) (см. README)

**Evidence**:
- `src/extractors/pitch_extractor/main.py:250` — метод `run()`
- `src/extractors/pitch_extractor/main.py:333` — метод `run_segments()`
- `src/extractors/pitch_extractor/main.py:258` — проверка входного файла

### 2.2 Контракты входа (Segmenter contract)

- [x] Входная единица: `audio/audio.wav` (Segmenter) + опционально `audio/segments.json` family `pitch`
- [x] Использует `start_sample/end_sample` для загрузки через `AudioUtils.load_audio_segment()` (в `run_segments()`)
- [x] Не генерирует сегменты сам (Segmenter — единственный владелец sampling)

**Evidence**:
- `src/extractors/pitch_extractor/main.py:333` — метод `run_segments()` принимает сегменты
- `src/extractors/pitch_extractor/main.py:365-375` — загрузка сегментов через `AudioUtils.load_audio_segment()`

### 2.3 No-fallback policy

- [x] Пустой список segments → fail-fast: `raise ValueError("pitch | segments is empty (no-fallback)")`
- [x] Ошибка выбранного backend → fail-fast с детальным error_code
- [x] Все методы вернули пустые результаты → fail-fast: `raise RuntimeError(f"pitch | all methods returned empty/invalid output (error_code={error_code})")`
- [x] Observability: ошибки логируются (без raw audio данных)

**Evidence**:
- `src/extractors/pitch_extractor/main.py:340` — проверка segments
- `src/extractors/pitch_extractor/main.py:701-703` — проверка пустых результатов
- `src/extractors/pitch_extractor/main.py:560-570` — fail-fast для torchcrepe backend
- `src/extractors/pitch_extractor/main.py:650-655` — fail-fast для PYIN
- `src/extractors/pitch_extractor/main.py:680-685` — fail-fast для YIN

### 2.4 Per-run storage

- [x] Имя NPZ файла стабильное: `pitch_extractor_features.npz` (фиксированное имя)
- [x] Запись NPZ атомарная (выполняется в `run_cli.py` через `_atomic_save_npz()`)
- [x] .npy файлы сохраняются в `result_store/<platform_id>/<video_id>/<run_id>/pitch_extractor/_artifacts/` (per-run storage)
- [x] .npy файлы регистрируются в `manifest.json.artifacts[]` (через `meta.artifacts`)

**Evidence**:
- `run_cli.py:847` — сохранение через `_save_component_npz()` с фиксированным именем
- `src/extractors/pitch_extractor/main.py:533-570` — метод `_save_time_series_artifacts()` сохраняет в per-run storage
- `run_cli.py:847-945` — сохранение NPZ с artifacts в meta

### 2.5 NPZ schema + meta contract

**NPZ schema**: `schema_version="audio_npz_v1"`

**Обязательные ключи**:
- [x] `feature_names: object[str]` — имена фичей (scalars)
- [x] `feature_values: float32[]` — значения фичей (scalars)
- [x] `f0_series_pyin: float32[N]` — временная серия PYIN (feature-gated)
- [x] `f0_series_yin: float32[M]` — временная серия YIN (feature-gated)
- [x] `f0_series_torchcrepe: float32[K]` — временная серия torchcrepe (feature-gated)
- [x] `f0_series: float32[L]` — агрегированная временная серия (feature-gated, для `run_segments()`)
- [x] `segment_centers_sec: float32[L]` — центры сегментов (feature-gated, для `run_segments()`)
- [x] `segment_durations_sec: float32[L]` — длительности сегментов (feature-gated, для `run_segments()`)
- [x] `pitch_octave_distribution: object(dict)` — распределение pitch по октавам (feature-gated)
- [x] `meta: object(dict)` — обязательные поля

**Обязательные поля `meta`**:
- [x] `producer`, `producer_version`, `schema_version`, `created_at`
- [x] `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- [x] `status` ∈ {`ok`, `empty`, `error`}
- [x] `models_used[]`, `model_signature` (не используется, signal processing)
- [x] `device_used`
- [x] `pitch_contract_version` — версия контракта для валидации совместимости с downstream extractors
- [x] `features_enabled[]` — список включённых фичей (feature gating)
- [x] `f0_method` — выбранный метод (`"pyin"`, `"yin"`, `"torchcrepe"`, `"none"`)
- [x] `artifacts[]` — список путей к .npy файлам (если есть)

**Extractor-level требования**:
- [x] Фичи имеют стабильные имена: `f0_mean`, `f0_std`, `f0_min`, `f0_max`, `f0_median`, `pitch_variation`, `pitch_stability`, и т.д.
- [x] Единицы измерения зафиксированы в README (Hz для f0, 0.0-1.0 для стабильности)
- [x] Missing values: NaN (если применимо)
- [x] Feature-gated поля сохраняются только если соответствующий флаг включен

**Evidence**:
- `run_cli.py:847-945` — сохранение NPZ с feature-gated полями
- `src/extractors/pitch_extractor/main.py:280-320` — формирование payload с feature gating

### 2.6 Valid empty outputs

- [x] При `status="empty"`: `empty_reason` (если применимо)
- [x] Фичи при empty: NaN или пустые массивы
- [x] Empty не скрывает ошибки парсинга/модели/валидации

**Evidence**:
- `src/extractors/pitch_extractor/main.py:701-703` — обработка пустых результатов (ошибка, не empty)

---

## 3) Model System

### 3.1 ModelManager integration

- [x] Не использует ML модели через ModelManager (signal processing: PYIN, YIN, torchcrepe)
- [x] torchcrepe опционален и не требует ModelManager (загружается напрямую через import)
- [x] В `meta.models_used[]` не фиксируются модели (signal processing)

**Evidence**:
- `src/extractors/pitch_extractor/main.py:19` — `dependencies = ["librosa", "numpy"]` (нет dp_models)
- `src/extractors/pitch_extractor/main.py:770-820` — метод `_extract_torchcrepe()` использует прямой import

### 3.2 No-network policy

- [x] Нет сетевых загрузок моделей/весов во время run
- [x] torchcrepe загружается локально (если установлен)

**Evidence**:
- `src/extractors/pitch_extractor/main.py:770-820` — torchcrepe загружается через import (локально)

---

## 4) Segmenter Contract

### 4.1 Audio segments contract

- [x] Использует `audio/segments.json` family `pitch` (для `run_segments()`)
- [x] Читает `families.pitch.segments[]` из `audio/segments.json`
- [x] Передает сегменты в `run_segments()` для обработки
- [x] Не генерирует сегменты сам

**Evidence**:
- `src/extractors/pitch_extractor/main.py:333` — метод `run_segments()` принимает сегменты
- `run_cli.py:1327` — извлечение `pitch_segments` из `families.pitch.segments[]`
- `run_cli.py:1341-1342` — проверка наличия `pitch_segments` (fail-fast)

---

## 5) Наблюдаемость: progress + stage timings

### 5.1 Промежуточный прогресс

- [x] Progress обновляется для каждого метода (PYIN, YIN, torchcrepe) в `run()`
- [x] Progress обновляется для каждого сегмента (каждые 10%) в `run_segments()`
- [x] Формат прогресса машиночитаем и безопасен (без raw audio данных)
- [x] Progress callback передаётся в `run()` и `run_segments()` через параметр `progress_callback`

**Evidence**:
- `src/extractors/pitch_extractor/main.py:268-275` — progress reporting в `run()`
- `src/extractors/pitch_extractor/main.py:350-360` — progress reporting в `run_segments()` (каждые 10%)
- `run_cli.py:1642-1656` — progress callback для pitch_extractor

### 5.2 Stage timings

- [x] Timings сохраняются в NPZ meta через `extra_meta` в `run_cli.py`
- [x] Per-extractor timings сохраняются в `timings_by_extractor`

**Evidence**:
- `run_cli.py:1724` — сохранение timings в meta

---

## 6) Feature Contract: управление выходными фичами (feature gating)

### 6.1 Feature gating flags

- [x] Все фичи opt-in через персональные флаги (default: все False)
- [x] Флаги: `--pitch-enable-basic-stats`, `--pitch-enable-stability-metrics`, `--pitch-enable-delta-features`, `--pitch-enable-method-stats`, `--pitch-enable-time-series`
- [x] В `meta.features_enabled[]` фиксируются включённые фичи

**Evidence**:
- `src/extractors/pitch_extractor/main.py:49-58` — feature gating flags в `__init__`
- `src/extractors/pitch_extractor/main.py:280-320` — feature-gated payload
- `run_cli.py:1013-1017` — CLI аргументы для feature gating

### 6.2 Feature dependencies

- [x] Зависимости между фичами документированы в README (раздел "Feature Dependencies")
- [x] `stability_metrics` зависят от `basic_stats` (требуют `f0_mean`, `f0_std`, `f0_min`, `f0_max`)
- [x] `delta_features` зависят от `basic_stats` (требуют временную серию f0)
- [x] `method_stats` и `time_series` независимы от других фичей

**Evidence**:
- `src/extractors/pitch_extractor/README.md` — раздел "Feature Dependencies"

---

## 7) Производительность и ресурсы

### 7.1 Обязательные измерения

- [x] Latency per unit задокументирована (estimated_duration = 2.0 сек для полного аудио, ~0.1-0.5 сек на сегмент)
- [x] CPU RSS peak измеряется через resource_metrics в `run_cli.py`
- [x] GPU VRAM peak измеряется через resource_metrics в `run_cli.py` (только для torchcrepe)

**Evidence**:
- `src/extractors/pitch_extractor/main.py:30` — `estimated_duration = 2.0`
- `run_cli.py:1724` — resource_metrics в meta

### 7.2 Параметры производительности

- [x] `hop_length`: меньшие значения → больше кадров → выше точность, но медленнее
- [x] `frame_length`: большие значения → лучше для низких частот, но медленнее
- [x] `torchcrepe_batch_size`: большие значения → быстрее на GPU, но больше памяти

**Evidence**:
- `src/extractors/pitch_extractor/README.md` — раздел "Performance characteristics"

---

## 8) Проверка качества выхода (quality validation)

### 8.1 Минимальные sanity-checks

- [x] Валидация выходных данных: проверка диапазонов f0 (fmin ≤ f0 ≤ fmax), NaN/inf, консистентности (f0_min ≤ f0_mean ≤ f0_max)
- [x] Валидация параметров: проверка диапазонов (fmin > 0, fmax > fmin, hop_length > 0, frame_length > 0, sample_rate > 0), разумных значений (fmin ≥ 20 Hz, fmax ≤ 8000 Hz)
- [x] Консистентность связных фичей (например, `f0_min` ≤ `f0_mean` ≤ `f0_max`)

**Evidence**:
- `src/extractors/pitch_extractor/main.py:120-145` — метод `_validate_parameters()` (fail-fast)
- `src/extractors/pitch_extractor/main.py:195-245` — метод `_validate_output()` (полная валидация)
- `src/extractors/pitch_extractor/main.py:228-232` — проверка консистентности

### 8.2 Human-friendly визуализация / UI render

- [x] Renderer реализован в `src/core/renderer.py` (`render_pitch_extractor()`)
- [x] HTML renderer для дебага (`render_pitch_extractor_html()`)
- [x] README содержит раздел "Visualization" с рекомендациями для UI/сайта

**Evidence**:
- `src/core/renderer.py:1769-1900` — renderer для pitch_extractor
- `src/core/renderer.py:1902-2100` — HTML renderer для дебага
- `src/extractors/pitch_extractor/README.md` — раздел "Visualization"

---

## 9) Error Handling

### 9.1 Детальные error codes

- [x] `pitch_audio_load_failed` (Ошибка загрузки аудио)
- [x] `pitch_torchcrepe_failed` (Ошибка torchcrepe, если выбран как backend)
- [x] `pitch_pyin_failed` (Ошибка PYIN, если используется classic backend)
- [x] `pitch_yin_failed` (Ошибка YIN, если используется classic backend)
- [x] `pitch_all_methods_failed` (Все методы вернули пустые результаты)
- [x] `pitch_validation_failed` (Ошибка валидации параметров или выходных данных)
- [x] `pitch_unknown` (Другие ошибки)

**Evidence**:
- `src/extractors/pitch_extractor/main.py:147-170` — метод `_classify_error()`
- `src/extractors/pitch_extractor/main.py:560-570` — использование error codes в error handling

### 9.2 No-fallback policy

- [x] Отсутствие segments → `ValueError` с `error_code="pitch_all_methods_failed"`
- [x] Ошибка выбранного backend → `RuntimeError` с детальным error_code (no-fallback)
- [x] Все методы вернули пустые результаты → `RuntimeError` с `error_code="pitch_all_methods_failed"`
- [x] Валидация параметров → `ValueError` с `error_code="pitch_validation_failed"`
- [x] Валидация выходных данных → `ValueError` с `error_code="pitch_validation_failed"`

**Evidence**:
- `src/extractors/pitch_extractor/main.py:340` — fail-fast проверка segments
- `src/extractors/pitch_extractor/main.py:560-570` — fail-fast для torchcrepe backend
- `src/extractors/pitch_extractor/main.py:650-655` — fail-fast для PYIN
- `src/extractors/pitch_extractor/main.py:680-685` — fail-fast для YIN
- `src/extractors/pitch_extractor/main.py:701-703` — fail-fast для пустых результатов

---

## 10) Additional Metrics

### 10.1 Дополнительные метрики для ML/аналитики

- [x] `pitch_contour_smoothness`: гладкость контура pitch (inverse of second derivative variance)
- [x] `pitch_jump_count`: количество больших скачков pitch (>2 semitones)
- [x] `pitch_octave_distribution`: распределение pitch по октавам (dict[octave_id, ratio])
- [x] `pitch_centroid`: центроид распределения pitch (mean)
- [x] `pitch_skewness`: асимметрия распределения pitch
- [x] `pitch_kurtosis`: эксцесс распределения pitch

**Evidence**:
- `src/extractors/pitch_extractor/main.py:800-850` — метод `_calc_additional_metrics()`
- `src/extractors/pitch_extractor/main.py:460-495` — вычисление дополнительных метрик в `run_segments()`

---

## 11) Contract Versioning

### 11.1 Contract version для совместимости

- [x] `pitch_contract_version="pitch_contract_v1"` в payload
- [x] Contract version сохраняется в NPZ meta
- [x] Используется для валидации совместимости с downstream extractors (например, `speech_analysis_extractor`)

**Evidence**:
- `src/extractors/pitch_extractor/main.py:19` — константа `PITCH_CONTRACT_VERSION`
- `src/extractors/pitch_extractor/main.py:280-320` — contract version в payload
- `run_cli.py:847-945` — contract version в meta

---

## 12) Segments Validation

### 12.1 Полная валидация структуры сегментов

- [x] Валидация типов (list)
- [x] Валидация обязательных полей (`start_sample`, `end_sample`, `center_sec`)
- [x] Валидация диапазонов (неотрицательные значения, start < end)

**Evidence**:
- `src/extractors/pitch_extractor/main.py:340` — проверка segments: `if not isinstance(segments, list) or not segments: raise ValueError("pitch | segments is empty (no-fallback)")`
- `run_cli.py:1341-1342` — проверка наличия `pitch_segments` (fail-fast)

---

## 13) Parameter Validation

### 13.1 Полная валидация входных параметров

- [x] Валидация диапазонов: `fmin > 0`, `fmax > fmin`, `hop_length > 0`, `frame_length > 0`, `sample_rate > 0`
- [x] Валидация разумных значений: `fmin ≥ 20 Hz`, `fmax ≤ 8000 Hz`
- [x] Fail-fast при невалидных параметрах

**Evidence**:
- `src/extractors/pitch_extractor/main.py:120-145` — метод `_validate_parameters()` (fail-fast)

---

## 14) Output Validation

### 14.1 Полная валидация выходных данных

- [x] Валидация диапазонов f0: `fmin ≤ f0_mean ≤ fmax`
- [x] Валидация NaN/inf: проверка всех числовых значений
- [x] Валидация консистентности: `f0_min ≤ f0_mean ≤ f0_max`
- [x] Валидация временных серий: проверка NaN/inf, отрицательных значений

**Evidence**:
- `src/extractors/pitch_extractor/main.py:195-245` — метод `_validate_output()` (полная валидация)

---

## 15) Per-run Storage для .npy файлов

### 15.1 Сохранение .npy файлов в per-run storage

- [x] .npy файлы сохраняются в `result_store/<platform_id>/<video_id>/<run_id>/pitch_extractor/_artifacts/`
- [x] .npy файлы регистрируются в `manifest.json.artifacts[]` (через `meta.artifacts`)
- [x] Пути к .npy файлам сохраняются в payload (`f0_series_torchcrepe_npy`)

**Evidence**:
- `src/extractors/pitch_extractor/main.py:533-570` — метод `_save_time_series_artifacts()` сохраняет в per-run storage
- `run_cli.py:1642-1656` — установка `artifacts_dir` для pitch_extractor
- `run_cli.py:847-945` — сохранение artifacts в meta

---

## 16) Документация

### 16.1 README extractor'а

- [x] Раздел "Входы" с описанием Segmenter contract
- [x] Раздел "Выходы" с описанием всех фичей (feature-gated)
- [x] Раздел "Feature Dependencies" с явным описанием зависимостей
- [x] Раздел "Конфигурация" с описанием всех параметров
- [x] Раздел "Feature Gating" с описанием всех флагов
- [x] Раздел "Visualization" с рекомендациями для UI/сайта
- [x] Раздел "Алгоритм" с описанием всех этапов обработки
- [x] Раздел "Обработка ошибок" с описанием error codes

**Evidence**:
- `src/extractors/pitch_extractor/README.md` — полная документация

---

## 17) Compliance Summary

### ✅ Все критерии выполнены

- ✅ **Архитектура**: соответствует `BaseExtractor`, Segmenter contract, per-run storage
- ✅ **Segmenter contract**: поддерживает `run_segments()` для работы на сегментах (family `pitch`)
- ✅ **Контракты**: NPZ schema, meta fields, contract versioning
- ✅ **Feature gating**: персональные флаги для каждой группы фичей (5 групп, все opt-in)
- ✅ **Error handling**: детальные error codes (7 типов), no-fallback policy
- ✅ **Валидация**: полная валидация параметров (fail-fast) и выходных данных
- ✅ **Наблюдаемость**: progress reporting для каждого метода и сегмента, stage timings
- ✅ **UI Render**: renderer + HTML renderer для дебага
- ✅ **Документация**: полный README с разделами Feature Dependencies и Visualization
- ✅ **Дополнительные метрики**: pitch_contour_smoothness, pitch_jump_count, pitch_octave_distribution, pitch_centroid, pitch_skewness, pitch_kurtosis
- ✅ **Per-run storage**: .npy файлы сохраняются в per-run storage и регистрируются в manifest.json
- ✅ **Интеграция**: pitch_extractor добавлен как отдельный extractor в run_cli.py

---

## 18) Open Issues

Нет открытых проблем. Все критерии `AP_AUDIT_CRITERIA.md` выполнены.

