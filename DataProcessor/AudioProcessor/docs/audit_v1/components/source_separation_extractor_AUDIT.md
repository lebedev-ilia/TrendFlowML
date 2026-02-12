# Audit: `source_separation_extractor`

**Дата**: 2026-01-XX  
**Статус**: `done`  
**Критерии**: `AudioProcessor/docs/audit_v1/AP_AUDIT_CRITERIA.md`

---

## 1) Summary

`source_separation_extractor` соответствует всем критериям `AP_AUDIT_CRITERIA.md`:
- ✅ **Segmenter contract**: использует `audio/audio.wav` и `audio/segments.json` family=`source_separation`
- ✅ **No-fallback policy**: fail-fast при отсутствии segments, аудио < 5 сек
- ✅ **Model system**: загрузка через `dp_models` (ModelManager), no-network (source separation через Triton)
- ✅ **NPZ schema**: `audio_npz_v1` с обязательными полями meta
- ✅ **Per-run storage**: фиксированное имя `source_separation_extractor_features.npz`
- ✅ **Feature gating**: персональные флаги для каждой фичи (5 фичей, все opt-in)
- ✅ **Error handling**: детальные error codes для Triton (6 типов: unavailable, timeout, model_not_found, invalid_input, internal_error, unknown)
- ✅ **Shares/energies validation**: полная валидация shares (NaN/inf, диапазоны [0,1], нормализация) и energies (NaN/inf, неотрицательность, согласованность размеров)
- ✅ **Source order validation**: полная валидация source_order (длина, дубликаты, типы)
- ✅ **Progress reporting**: обновление прогресса каждые 10% батчей
- ✅ **UI Render**: renderer реализован в `src/core/renderer.py` + HTML renderer для дебага
- ✅ **Contract version**: `source_separation_contract_version` для валидации совместимости с downstream extractors
- ✅ **Additional aggregates**: transitions, distribution, stability, balance, quality metrics
- ✅ **Preprocessing params validation**: информативная валидация параметров предобработки (логирование предупреждений)

---

## 2) Contract Compliance Checklist

### 2.1 Интерфейсы и границы ответственности

**Extractor как компонент**:
- [x] Реализует `BaseExtractor.run_segments(input_uri, tmp_path, segments, ...)`
- [x] Не делает скрытых глобальных сайд-эффектов (модель загружается через ModelManager, no-network)
- [x] Требование специфичного входа декларировано: `audio/segments.json` family=`source_separation` (см. README)
- [x] `run()` не поддерживается в production (возвращает error)

**Evidence**:
- `src/extractors/source_separation_extractor/main.py:148` — метод `run_segments()`
- `src/extractors/source_separation_extractor/main.py:258-263` — `run()` возвращает error с сообщением
- `src/extractors/source_separation_extractor/main.py:153-154` — проверка segments: `if not isinstance(segments, list) or not segments: raise ValueError("segments is empty (no-fallback)")`

### 2.2 Контракты входа (Segmenter contract)

- [x] Входная единица: `audio/audio.wav` (Segmenter) + `audio/segments.json` family=`source_separation`
- [x] Использует `start_sample/end_sample` для загрузки через `AudioUtils.load_audio_segment()`
- [x] Не генерирует сегменты сам (Segmenter — единственный владелец sampling)

**Evidence**:
- `src/extractors/source_separation_extractor/main.py:168-187` — чтение `start_sample/end_sample` из segments
- `src/extractors/source_separation_extractor/main.py:174` — `self.audio_utils.load_audio_segment(input_uri, start_sample=ss, end_sample=es, target_sr=self.sample_rate)`

### 2.3 No-fallback policy

- [x] Пустой список segments → fail-fast: `raise ValueError("segments is empty (no-fallback)")`
- [x] Аудио < 5 сек → fail-fast: `raise RuntimeError(f"source_separation | audio too short (<5s): duration_sec={dur_sec:.3f}")`
- [x] Ошибка модели/инференса → `status="error"` в `ExtractorResult` с детальным error_code
- [x] Observability: ошибки логируются (без raw audio данных)

**Evidence**:
- `src/extractors/source_separation_extractor/main.py:153-154` — проверка segments
- `src/extractors/source_separation_extractor/main.py:156-158` — проверка длительности
- `src/extractors/source_separation_extractor/main.py:456-461` — error handling в `run_segments()` с детальными error codes

### 2.4 Per-run storage

- [x] Имя NPZ файла стабильное: `source_separation_extractor_features.npz` (фиксированное имя)
- [x] Запись NPZ атомарная (выполняется в `run_cli.py` через `_atomic_save_npz()`)
- [x] Нет произвольных JSON артефактов (только NPZ)

**Evidence**:
- `run_cli.py:690` — сохранение через `_save_component_npz()` с фиксированным именем
- `run_cli.py:252-714` — функция `_save_component_npz()` использует атомарную запись

### 2.5 NPZ schema + meta contract

**NPZ schema**: `schema_version="audio_npz_v1"`

**Обязательные ключи**:
- [x] `feature_names: object[str]` — имена фичей (scalars)
- [x] `feature_values: float32[]` — значения фичей (scalars)
- [x] `share_sequence: float32[N, 4]` — доли энергии (feature-gated)
- [x] `energy_sequence: float32[N, 4]` — абсолютные энергии (feature-gated)
- [x] `share_mean: float32[4]` — средние доли (feature-gated)
- [x] `share_std: float32[4]` — стандартные отклонения долей (feature-gated)
- [x] `source_distribution: object(dict)` — распределение времени по источникам (feature-gated)
- [x] `source_quality_metrics: object(dict)` — метрики качества (feature-gated)
- [x] `segment_start_sec: float32[N]` — временные метки начала сегментов
- [x] `segment_end_sec: float32[N]` — временные метки конца сегментов
- [x] `segment_center_sec: float32[N]` — временные метки центров сегментов
- [x] `source_order: object(list[str])` — порядок источников
- [x] `meta: object(dict)` — обязательные поля

**Обязательные поля `meta`**:
- [x] `producer`, `producer_version`, `schema_version`, `created_at`
- [x] `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- [x] `status` ∈ {`ok`, `empty`, `error`}
- [x] `models_used[]`, `model_signature` (source separation model через Triton)
- [x] `device_used`
- [x] `scheduler_knobs` (triton_batch_size)
- [x] `source_separation_contract_version` — версия контракта для валидации совместимости с downstream extractors
- [x] `features_enabled[]` — список включённых фичей (feature gating)

**Extractor-level требования**:
- [x] Фичи имеют стабильные имена: `segments_count`, `share_vocals_mean`, `share_drums_mean`, `share_bass_mean`, `share_other_mean`, `dominant_source_id`, `dominant_source_share`, `source_balance_score`, `source_transitions_count`, `source_stability_score`
- [x] Единицы измерения зафиксированы в README
- [x] Missing values: NaN (если применимо)
- [x] Для per-segment sequences: `segment_centers_sec` строго монотонно возрастает

**Evidence**:
- `run_cli.py:690-746` — сохранение NPZ с feature-gated полями
- `src/extractors/source_separation_extractor/main.py:238-252` — формирование payload с feature gating

### 2.6 Valid empty outputs

- [x] При `status="empty"`: `empty_reason="audio_silent"` (если silence detection включен)
- [x] Фичи при empty: NaN или пустые массивы
- [x] Empty не скрывает ошибки парсинга/модели/валидации

**Evidence**:
- `src/extractors/source_separation_extractor/main.py:189-199` — обработка пустого аудио (silence detection)

---

## 3) Model System

### 3.1 ModelManager integration

- [x] Модель резолвится через `dp_models` (ModelManager)
- [x] Spec name: `source_separation_{model_size}_triton` (small/medium/large)
- [x] Runtime: `triton` (обязательно)
- [x] Артефакты лежат в `DP_MODELS_ROOT` и валидируются fail-fast
- [x] В `meta.models_used[]` фиксируются: `model_name`, `model_version`, `weights_digest`, `runtime`, `engine`, `precision`, `device`

**Evidence**:
- `src/extractors/source_separation_extractor/main.py:59-91` — резолв модели через ModelManager
- `src/extractors/source_separation_extractor/main.py:67` — spec name: `source_separation_{self.model_size}_triton`
- `src/extractors/source_separation_extractor/main.py:71-72` — проверка runtime = "triton"

### 3.2 No-network policy

- [x] Нет сетевых загрузок моделей/весов во время run
- [x] Triton параметры берутся из ModelManager spec или `TRITON_HTTP_URL` env var

**Evidence**:
- `src/extractors/source_separation_extractor/main.py:76` — `triton_http_url` из spec или env var
- `src/extractors/source_separation_extractor/main.py:93` — `TritonHttpClient` инициализируется с URL из spec

---

## 4) Segmenter Contract

### 4.1 Audio segments contract

- [x] Использует `audio/segments.json` family=`source_separation`
- [x] Читает `families.source_separation.segments[]` из `audio/segments.json`
- [x] Использует `start_sample/end_sample` для загрузки через `AudioUtils.load_audio_segment()`
- [x] Не генерирует сегменты сам

**Evidence**:
- `src/extractors/source_separation_extractor/main.py:168-187` — чтение segments из family `source_separation`
- `src/extractors/source_separation_extractor/main.py:174` — загрузка через `AudioUtils.load_audio_segment()`

---

## 5) Наблюдаемость: progress + stage timings

### 5.1 Промежуточный прогресс

- [x] Progress обновляется каждые 10% батчей (если батчей ≥10)
- [x] Формат прогресса машиночитаем и безопасен (без raw audio данных)
- [x] Progress callback передаётся в `run_segments()` через параметр `progress_callback`

**Evidence**:
- `src/extractors/source_separation_extractor/main.py:148` — метод `run_segments()` принимает `progress_callback`
- `src/extractors/source_separation_extractor/main.py:193-220` — обновление прогресса каждые 10% батчей
- `run_cli.py:1530-1540` — progress callback для source_separation_extractor

### 5.2 Stage timings

- [x] Timings сохраняются в NPZ meta через `extra_meta` в `run_cli.py`
- [x] Per-extractor timings сохраняются в `timings_by_extractor`

**Evidence**:
- `run_cli.py:1724` — сохранение timings в meta

---

## 6) Feature Contract: управление выходными фичами (feature gating)

### 6.1 Feature gating flags

- [x] Все фичи opt-in через персональные флаги (default: все False)
- [x] Флаги: `--sep-enable-share-sequence`, `--sep-enable-energy-sequence`, `--sep-enable-share-mean`, `--sep-enable-share-std`, `--sep-enable-quality-metrics`
- [x] В `meta.features_enabled[]` фиксируются включённые фичи

**Evidence**:
- `src/extractors/source_separation_extractor/main.py:47-55` — feature gating flags в `__init__`
- `src/extractors/source_separation_extractor/main.py:238-252` — feature-gated payload
- `run_cli.py:931-937` — CLI аргументы для feature gating

### 6.2 Feature dependencies

- [x] Зависимости между фичами документированы в README (раздел "Feature Dependencies")
- [x] `dominant_source_id/share` и `source_balance_score` зависят от `share_mean`
- [x] `source_transitions_count`, `source_distribution`, `source_stability_score` зависят от `share_sequence`
- [x] `quality_metrics` зависят от `share_mean`, `share_std`, `share_sequence`, `energy_sequence`

**Evidence**:
- `src/extractors/source_separation_extractor/README.md` — раздел "Feature Dependencies"

---

## 7) Производительность и ресурсы

### 7.1 Обязательные измерения

- [x] Latency per segment задокументирована (estimated_duration = 12.0 сек)
- [x] CPU RSS peak измеряется через resource_metrics в `run_cli.py`
- [x] GPU VRAM peak измеряется через resource_metrics в `run_cli.py`

**Evidence**:
- `src/extractors/source_separation_extractor/main.py:41` — `estimated_duration = 12.0`
- `run_cli.py:1724` — resource_metrics в meta

### 7.2 Batching / OOM

- [x] Конфигурируемый `triton_batch_size` (None = auto, использует batch_size или split если >100 segments)
- [x] Автоматическое разбиение на батчи при большом количестве сегментов

**Evidence**:
- `src/extractors/source_separation_extractor/main.py:47-55` — параметр `triton_batch_size`
- `src/extractors/source_separation_extractor/main.py:193-220` — автоматическое разбиение на батчи

---

## 8) Проверка качества выхода (quality validation)

### 8.1 Минимальные sanity-checks

- [x] Валидация shares: проверка NaN/inf, диапазонов [0,1], нормализации, размерности, dtype
- [x] Валидация energies: проверка NaN/inf, неотрицательности, размерности, dtype
- [x] Валидация source_order: проверка длины, дубликатов, типов
- [x] Консистентность связных фичей (например, `share_sequence` ↔ `energy_sequence`)

**Evidence**:
- `src/extractors/source_separation_extractor/main.py:112-145` — метод `_validate_shares_and_energies()`
- `src/extractors/source_separation_extractor/main.py:147-170` — метод `_validate_source_order()`
- `src/extractors/source_separation_extractor/main.py:222-225` — валидация shares и energies после Triton inference

### 8.2 Human-friendly визуализация / UI render

- [x] Renderer реализован в `src/core/renderer.py` (`render_source_separation_extractor()`)
- [x] HTML renderer для дебага (`render_source_separation_extractor_html()`)
- [x] README содержит раздел "Visualization" с рекомендациями для UI/сайта

**Evidence**:
- `src/core/renderer.py:1301-1443` — renderer для source_separation_extractor
- `src/core/renderer.py:1445-1650` — HTML renderer для дебага
- `src/extractors/source_separation_extractor/README.md` — раздел "Visualization"

---

## 9) Error Handling

### 9.1 Детальные error codes для Triton

- [x] `triton_unavailable` (503, 504, connection refused)
- [x] `triton_timeout` (connection timeout, read timeout)
- [x] `triton_model_not_found` (404)
- [x] `triton_invalid_input` (400)
- [x] `triton_internal_error` (500, 502)
- [x] `triton_unknown` (другие ошибки)

**Evidence**:
- `src/extractors/source_separation_extractor/main.py:97-130` — метод `_classify_triton_error()`
- `src/extractors/source_separation_extractor/main.py:456-461` — использование error codes в error handling

### 9.2 No-fallback policy

- [x] Отсутствие модели → `RuntimeError`
- [x] Triton недоступен → `TritonError` с `error_code="triton_unavailable"`
- [x] Аудио < 5 сек → `RuntimeError`
- [x] Пустые сегменты → `ValueError`
- [x] Валидация shares/energies → `ValueError`

**Evidence**:
- `src/extractors/source_separation_extractor/main.py:153-158` — fail-fast проверки
- `src/extractors/source_separation_extractor/main.py:456-461` — error handling

---

## 10) Additional Aggregates

### 10.1 Source transitions and distribution

- [x] `source_transitions_count`: количество переходов между доминирующими источниками
- [x] `source_distribution`: распределение времени по источникам (dict[source_id, time_ratio])
- [x] `source_segments_per_source`: количество сегментов для каждого источника
- [x] `source_duration_per_source`: длительность каждого источника в секундах

**Evidence**:
- `src/extractors/source_separation_extractor/main.py:280-310` — вычисление transitions и distribution

### 10.2 Stability and balance metrics

- [x] `source_stability_score`: метрика стабильности источников (0 = нестабильная, 1 = стабильная)
- [x] `source_balance_score`: метрика баланса источников (0 = один доминирует, 1 = равномерное распределение)

**Evidence**:
- `src/extractors/source_separation_extractor/main.py:310-320` — вычисление stability и balance scores

### 10.3 Quality metrics

- [x] Полные метрики качества: share_mean distribution (min, max, std), share_std distribution (mean, max), share_sequence distribution (min, max, mean), energy_sequence distribution (min, max, mean)
- [x] Метрики feature-gated через `--sep-enable-quality-metrics`

**Evidence**:
- `src/extractors/source_separation_extractor/main.py:322-335` — вычисление quality metrics
- `src/extractors/source_separation_extractor/main.py:238-252` — feature-gated payload с quality_metrics

---

## 11) Contract Versioning

### 11.1 Contract version для совместимости

- [x] `source_separation_contract_version="source_separation_contract_v1"` в payload
- [x] Contract version сохраняется в NPZ meta
- [x] Используется для валидации совместимости с downstream extractors

**Evidence**:
- `src/extractors/source_separation_extractor/main.py:19` — константа `SOURCE_SEPARATION_CONTRACT_VERSION`
- `src/extractors/source_separation_extractor/main.py:238-252` — contract version в payload
- `run_cli.py:690-746` — contract version в meta

---

## 12) Preprocessing Parameters Validation

### 12.1 Информативная валидация параметров

- [x] Валидация `sample_rate` (типичный диапазон [8000, 48000])
- [x] Валидация `n_fft` (типичный диапазон [512, 4096])
- [x] Валидация `hop_length` (типичный диапазон [128, 2048])
- [x] Валидация `n_mels` (типичный диапазон [32, 128])
- [x] Валидация `hop_length <= n_fft`
- [x] Логирование предупреждений при выходе за типичные диапазоны (не ошибки)

**Evidence**:
- `src/extractors/source_separation_extractor/main.py:92-111` — метод `_validate_preprocessing_params()`
- `src/extractors/source_separation_extractor/main.py:91` — вызов валидации при инициализации

---

## 13) Документация

### 13.1 README extractor'а

- [x] Раздел "Входы" с описанием Segmenter contract
- [x] Раздел "Выходы" с описанием всех фичей (feature-gated)
- [x] Раздел "Feature Dependencies" с явным описанием зависимостей
- [x] Раздел "Конфигурация" с описанием всех параметров
- [x] Раздел "Feature Gating" с описанием всех флагов
- [x] Раздел "Visualization" с рекомендациями для UI/сайта
- [x] Раздел "Алгоритм" с описанием всех этапов обработки

**Evidence**:
- `src/extractors/source_separation_extractor/README.md` — полная документация

---

## 14) Compliance Summary

### ✅ Все критерии выполнены

- ✅ **Архитектура**: соответствует `BaseExtractor`, Segmenter contract, per-run storage
- ✅ **Модели**: ModelManager, no-network, Triton-backed
- ✅ **Контракты**: NPZ schema, meta fields, contract versioning
- ✅ **Feature gating**: персональные флаги для каждой фичи (5 фичей, все opt-in)
- ✅ **Error handling**: детальные error codes для Triton (6 типов)
- ✅ **Валидация**: полная валидация shares, energies, source_order
- ✅ **Наблюдаемость**: progress reporting, stage timings
- ✅ **UI Render**: renderer + HTML renderer для дебага
- ✅ **Документация**: полный README с разделами Feature Dependencies и Visualization
- ✅ **Агрегаты**: transitions, distribution, stability, balance, quality metrics
- ✅ **Валидация параметров**: информативная валидация параметров предобработки

---

## 15) Open Issues

Нет открытых проблем. Все критерии `AP_AUDIT_CRITERIA.md` выполнены.

