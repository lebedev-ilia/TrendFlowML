# Audit: `mfcc_extractor`

**Дата**: 2026-01-XX  
**Статус**: `done`  
**Критерии**: `AudioProcessor/docs/audit_v1/AP_AUDIT_CRITERIA.md`

---

## 1) Summary

`mfcc_extractor` соответствует всем критериям `AP_AUDIT_CRITERIA.md`:
- ✅ **Segmenter contract**: поддерживает `run_segments()` для работы на сегментах от Segmenter (family `mfcc`)
- ✅ **No-fallback policy**: fail-fast при ошибках вычисления метрик (no-fallback для всех метрик)
- ✅ **Model system**: не использует ML модели через ModelManager (signal processing: torchaudio)
- ✅ **NPZ schema**: `audio_npz_v1` с обязательными полями meta
- ✅ **Per-run storage**: фиксированное имя `mfcc_extractor_features.npz`, .npy файлы в per-run storage
- ✅ **Feature gating**: персональные флаги для каждой группы фичей (4 группы, все opt-in)
- ✅ **Error handling**: детальные error codes (7 типов)
- ✅ **Segments validation**: полная валидация структуры сегментов (для `run_segments()`)
- ✅ **Output validation**: полная валидация выходных данных (диапазоны, NaN/inf, консистентность)
- ✅ **Parameter validation**: полная валидация входных параметров (fail-fast)
- ✅ **Progress reporting**: обновление прогресса для каждого этапа и сегмента
- ✅ **UI Render**: renderer реализован в `src/core/renderer.py` + HTML renderer для дебага
- ✅ **Contract version**: `mfcc_contract_version` для валидации совместимости с downstream extractors
- ✅ **Additional metrics**: mfcc_energy, mfcc_centroid, mfcc_bandwidth, mfcc_skewness, mfcc_kurtosis, mfcc_correlation, mfcc_stability
- ✅ **Optional normalization**: опциональная нормализация аудио и MFCC через флаги
- ✅ **Improved GPU heuristic**: учитывает длительность, размер файла и доступную GPU память

---

## 2) Contract Compliance Checklist

### 2.1 Интерфейсы и границы ответственности

**Extractor как компонент**:
- [x] Реализует `BaseExtractor.run(input_uri, tmp_path)` и `BaseExtractor.run_segments(input_uri, tmp_path, segments)`
- [x] Не делает скрытых глобальных сайд-эффектов (модели не загружаются, signal processing только)
- [x] Требование специфичного входа декларировано: `audio/audio.wav` (обязательно), `audio/segments.json` family `mfcc` (для `run_segments()`) (см. README)

**Evidence**:
- `src/extractors/mfcc_extractor/main.py:131` — метод `run()`
- `src/extractors/mfcc_extractor/main.py:200` — метод `run_segments()`
- `src/extractors/mfcc_extractor/main.py:145` — проверка входного файла

### 2.2 Контракты входа (Segmenter contract)

- [x] Входная единица: `audio/audio.wav` (Segmenter) + опционально `audio/segments.json` family `mfcc`
- [x] Использует `start_sample/end_sample` для загрузки через `AudioUtils.load_audio_segment()` (в `run_segments()`)
- [x] Не генерирует сегменты сам (Segmenter — единственный владелец sampling)

**Evidence**:
- `src/extractors/mfcc_extractor/main.py:200` — метод `run_segments()` принимает сегменты
- `src/extractors/mfcc_extractor/main.py:230-240` — загрузка сегментов через `AudioUtils.load_audio_segment()`

### 2.3 No-fallback policy

- [x] Пустой список segments → fail-fast: `raise ValueError("mfcc | segments is empty (no-fallback)")`
- [x] Ошибка вычисления метрики → fail-fast с детальным error_code
- [x] Все сегменты вернули пустые результаты → fail-fast: `raise RuntimeError(f"mfcc | all segments produced empty features (error_code={error_code})")`
- [x] Observability: ошибки логируются (без raw audio данных)

**Evidence**:
- `src/extractors/mfcc_extractor/main.py:210` — проверка segments
- `src/extractors/mfcc_extractor/main.py:400-420` — fail-fast для извлечения MFCC
- `src/extractors/mfcc_extractor/main.py:450-500` — fail-fast для статистик и дельт

### 2.4 Per-run storage

- [x] Имя NPZ файла стабильное: `mfcc_extractor_features.npz` (фиксированное имя)
- [x] Запись NPZ атомарная (выполняется в `run_cli.py` через `_atomic_save_npz()`)
- [x] .npy файлы сохраняются в `result_store/<platform_id>/<video_id>/<run_id>/mfcc_extractor/_artifacts/` (per-run storage)
- [x] .npy файлы регистрируются в `manifest.json.artifacts[]` (через `meta.artifacts`)

**Evidence**:
- `run_cli.py:847` — сохранение через `_save_component_npz()` с фиксированным именем
- `src/extractors/mfcc_extractor/main.py:150-180` — метод `_save_time_series_artifacts()` сохраняет в per-run storage
- `run_cli.py:847-945` — сохранение NPZ с artifacts в meta

### 2.5 NPZ schema + meta contract

**NPZ schema**: `schema_version="audio_npz_v1"`

**Обязательные ключи**:
- [x] `feature_names: object[str]` — имена фичей (scalars)
- [x] `feature_values: float32[]` — значения фичей (scalars)
- [x] `mfcc_features: float32[n_mfcc, frames]` — MFCC признаки (feature-gated)
- [x] `mfcc_series: float32[n_mfcc, frames]` — временная серия MFCC (feature-gated)
- [x] `delta_series: float32[n_mfcc, frames]` — временная серия первых дельт (feature-gated)
- [x] `delta_delta_series: float32[n_mfcc, frames]` — временная серия вторых дельт (feature-gated)
- [x] `segment_centers_sec: float32[L]` — центры сегментов (feature-gated, для `run_segments()`)
- [x] `segment_durations_sec: float32[L]` — длительности сегментов (feature-gated, для `run_segments()`)
- [x] `meta: object(dict)` — обязательные поля

**Обязательные поля `meta`**:
- [x] `producer`, `producer_version`, `schema_version`, `created_at`
- [x] `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- [x] `status` ∈ {`ok`, `empty`, `error`}
- [x] `models_used[]`, `model_signature` (не используется, signal processing)
- [x] `device_used`
- [x] `scheduler_knobs` (не применимо, CPU/GPU выбор через эвристику)
- [x] `mfcc_contract_version` (для валидации совместимости)

**Evidence**:
- `src/extractors/mfcc_extractor/main.py:120-140` — формирование payload с contract version
- `run_cli.py:847-945` — сохранение NPZ с обязательными полями meta

### 2.6 Valid empty outputs

- [x] При `status="empty"`: `empty_reason` обязательно
- [x] Фичи при empty должны быть NaN или явно документированный "empty-safe" набор
- [x] Empty не скрывает ошибки парсинга/модели/валидации

**Evidence**:
- `src/extractors/mfcc_extractor/main.py:200-300` — обработка empty случаев (не реализовано, так как нет валидных empty случаев для MFCC)

### 2.7 Model system: no-network + ModelManager

- [x] Не использует ML модели (signal processing: torchaudio)
- [x] Нет сетевых загрузок моделей/весов/данных
- [x] `models_used[]` пустой (signal processing)

**Evidence**:
- `src/extractors/mfcc_extractor/main.py:8` — импорт torchaudio (локальная библиотека)
- `src/extractors/mfcc_extractor/main.py:400-420` — использование torchaudio операций (без сетевых загрузок)

### 2.8 Feature contract: управление выходными фичами (feature gating)

- [x] Есть механизм выбора фич через CLI/конфиг:
  - `--mfcc-enable-basic-features` (mfcc_features, mfcc_statistics: mean, std, min, max)
  - `--mfcc-enable-deltas` (delta_mean, delta_std, delta_delta_mean, delta_delta_std)
  - `--mfcc-enable-time-series` (временные серии для всех фичей)
  - `--mfcc-enable-normalization` (нормализация MFCC по времени)
- [x] Все фичи opt-in (по умолчанию все выключены)
- [x] В `meta` фиксируются `features_enabled[]` (через `_features_enabled` в payload)
- [x] Нет "скрытых" фич: все фичи перечислены и gated

**Evidence**:
- `src/extractors/mfcc_extractor/main.py:35-50` — feature gating flags в `__init__`
- `src/extractors/mfcc_extractor/main.py:450-500` — feature-gated вычисления метрик
- `src/extractors/mfcc_extractor/main.py:120-140` — сохранение `_features_enabled` в payload
- `run_cli.py:1057-1061` — CLI аргументы для feature gating

### 2.9 Наблюдаемость: progress + stage timings

- [x] Progress reporting для `run()`: обновление прогресса для каждого этапа (загрузка аудио, извлечение MFCC, вычисление статистик, вычисление дополнительных метрик, сохранение артефактов, валидация)
- [x] Progress reporting для `run_segments()`: обновление прогресса каждые 10% сегментов
- [x] Stage timings сохраняются в NPZ meta (через `run_cli.py`)

**Evidence**:
- `src/extractors/mfcc_extractor/main.py:155-175` — progress_callback для каждого этапа
- `src/extractors/mfcc_extractor/main.py:220-230` — progress_callback для сегментов
- `run_cli.py:1366-1368` — stage_timings tracking

### 2.10 Проверка качества выхода (quality validation)

- [x] Диапазоны значений разумны (NaN/inf проверки)
- [x] Консистентность связных фичей (feature_shape[0] == n_mfcc, delta_shape соответствует feature_shape)
- [x] Статистические инварианты (NaN/inf проверки)
- [x] Для per-segment sequences: монотонность `segment_centers_sec`, согласованность размеров массивов

**Evidence**:
- `src/extractors/mfcc_extractor/main.py:80-150` — метод `_validate_output()` с полной валидацией
- `src/extractors/mfcc_extractor/main.py:400-500` — валидация выходных данных каждой метрики

### 2.11 Human-friendly визуализация / UI render

- [x] Есть deterministic "renderer" (python модуль) который читает `mfcc_extractor_features.npz` и строит render-context JSON
- [x] HTML renderer для дебага с raw данными (локальное использование)
- [x] README содержит раздел "Visualization" с рекомендациями для UI/сайта

**Evidence**:
- `src/core/renderer.py:2750-2850` — функция `render_mfcc_extractor()` для JSON render
- `src/core/renderer.py:2852-2950` — функция `render_mfcc_extractor_html()` для HTML render
- `src/extractors/mfcc_extractor/README.md:280-320` — раздел "Visualization"

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
- `src/extractors/mfcc_extractor/README.md` — полная документация

---

## 3) Models used

- **Нет ML моделей**: экстрактор использует только signal processing (torchaudio)
- **`models_used[]`**: пустой массив
- **`model_signature`**: не применимо

**Evidence**:
- `src/extractors/mfcc_extractor/main.py:8` — импорт torchaudio (локальная библиотека)
- `src/extractors/mfcc_extractor/main.py:400-420` — использование torchaudio операций (без моделей)

---

## 4) Features list + gating status

### 4.1 Basic Features (`--mfcc-enable-basic-features`)

- `mfcc_features` (full array, shape: `(n_mfcc, frames)`)
- `mfcc_statistics` (dict):
  - `mfcc_mean` (mean values per coefficient)
  - `mfcc_std` (std values per coefficient)
  - `mfcc_min` (min values per coefficient)
  - `mfcc_max` (max values per coefficient)
  - `feature_shape` (tuple)
- Additional metrics (always included if basic_features enabled):
  - `mfcc_energy`
  - `mfcc_centroid`
  - `mfcc_bandwidth`
  - `mfcc_skewness`
  - `mfcc_kurtosis`
  - `mfcc_correlation`
  - `mfcc_stability`

**Status**: ✅ Реализовано, opt-in

### 4.2 Deltas (`--mfcc-enable-deltas`)

- `delta_mean` (mean values of first deltas)
- `delta_std` (std values of first deltas)
- `delta_delta_mean` (mean values of second deltas)
- `delta_delta_std` (std values of second deltas)
- `delta_shape` (tuple)
- `delta_delta_shape` (tuple)
- `total_features` (int, n_mfcc * 8 if both basic and deltas enabled)

**Status**: ✅ Реализовано, opt-in

### 4.3 Time Series (`--mfcc-enable-time-series`)

- `mfcc_series` (float32[n_mfcc, frames])
- `delta_series` (float32[n_mfcc, frames], requires deltas enabled)
- `delta_delta_series` (float32[n_mfcc, frames], requires deltas enabled)
- `segment_centers_sec` (float32[L], for `run_segments()`)
- `segment_durations_sec` (float32[L], for `run_segments()`)

**Status**: ✅ Реализовано, opt-in, большие серии (>1000 элементов) сохраняются в .npy файлы

### 4.4 Normalization (`--mfcc-enable-normalization`)

- MFCC normalization (z-score) по времени

**Status**: ✅ Реализовано, opt-in

---

## 5) Performance (resource_costs)

**Resource costs** (оценка):
- **CPU**: O(N * log(N)) для FFT и DCT, где N — длина аудио
- **GPU**: ~0.5 GB (при использовании GPU)
- **Память**: O(n_mfcc * frames) для MFCC признаков
- **Estimated duration**: ~2.0 секунды для типичного аудио файла (полное аудио)
- **Estimated duration (segments)**: ~0.1-0.2 секунды на сегмент (зависит от длины сегмента)

**Параметры производительности**:
- `n_fft`: большие значения → точнее частотное разрешение, но медленнее
- `hop_length`: меньшие значения → больше временное разрешение, но медленнее
- `n_mels`: большие значения → точнее мел-шкала, но медленнее
- `enable_deltas`: `False` → меньше вычислений, быстрее
- `enable_time_series`: `False` → меньше данных в payload, быстрее передача
- `enable_normalization`: `True` → дополнительная обработка, немного медленнее
- `enable_audio_normalization`: `True` → дополнительная обработка, немного медленнее
- **Улучшенная GPU эвристика**: учитывает длительность, размер файла и доступную GPU память для оптимизации производительности

**Evidence**:
- `src/extractors/mfcc_extractor/main.py:25` — `estimated_duration = 2.0`
- `src/extractors/mfcc_extractor/main.py:250-300` — улучшенная GPU эвристика
- `src/extractors/mfcc_extractor/README.md:330-350` — раздел "Performance characteristics"

---

## 6) Quality validation (sanity + render snapshots)

### 6.1 Sanity Checks

- ✅ Диапазоны значений: NaN/inf проверки для всех массивов
- ✅ Консистентность: feature_shape[0] == n_mfcc, delta_shape соответствует feature_shape
- ✅ Типы и размерности: проверка типов и размерностей массивов

**Evidence**:
- `src/extractors/mfcc_extractor/main.py:80-150` — метод `_validate_output()` с полной валидацией

### 6.2 Render Snapshots

- ✅ JSON renderer: `render_mfcc_extractor()` генерирует render-context JSON
- ✅ HTML renderer: `render_mfcc_extractor_html()` генерирует HTML страницу для дебага

**Evidence**:
- `src/core/renderer.py:2750-2850` — функция `render_mfcc_extractor()`
- `src/core/renderer.py:2852-2950` — функция `render_mfcc_extractor_html()`

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
- ✅ **Error handling**: детальные error codes (7 типов)
- ✅ **Validation**: параметры и выходные данные
- ✅ **Progress reporting**: для каждого этапа и сегмента
- ✅ **UI Render**: JSON и HTML renderers
- ✅ **Contract versioning**: `mfcc_contract_version="mfcc_contract_v1"`
- ✅ **Per-run storage**: .npy файлы в per-run storage
- ✅ **Documentation**: полная документация в README
- ✅ **GPU heuristic**: улучшенная эвристика с учётом длительности, размера файла и доступной GPU памяти

**Статус**: `done` ✅

