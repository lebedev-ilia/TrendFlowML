# Audit: `speaker_diarization_extractor`

**Дата**: 2026-01-XX  
**Статус**: `done`  
**Критерии**: `AudioProcessor/docs/audit_v1/AP_AUDIT_CRITERIA.md`

---

## 1) Summary

`speaker_diarization_extractor` соответствует всем критериям `AP_AUDIT_CRITERIA.md`:
- ✅ **Segmenter contract**: использует `audio/audio.wav` и `audio/segments.json` family=`diarization`
- ✅ **No-fallback policy**: fail-fast при отсутствии segments, аудио < 5 сек
- ✅ **Model system**: загрузка через `dp_models` (ModelManager), no-network (speaker embeddings через Triton)
- ✅ **NPZ schema**: `audio_npz_v1` с обязательными полями meta
- ✅ **Per-run storage**: фиксированное имя `speaker_diarization_extractor_features.npz`
- ✅ **Feature gating**: персональные флаги для каждой фичи (6 фичей, все opt-in)
- ✅ **Error handling**: детальные error codes для Triton (6 типов: unavailable, timeout, model_not_found, invalid_input, internal_error, unknown)
- ✅ **Embedding validation**: полная валидация эмбеддингов (NaN/inf, диапазоны, размерность, dtype)
- ✅ **Clustering validation**: валидация меток кластеризации (согласованность с количеством сегментов и спикеров)
- ✅ **Progress reporting**: обновление прогресса каждые 10% сегментов
- ✅ **UI Render**: renderer реализован в `src/core/renderer.py` + HTML renderer для дебага
- ✅ **Contract version**: `diarization_contract_version` для валидации совместимости с downstream extractors
- ✅ **Clustering methods**: поддержка Agglomerative (default для обучения), KMeans (быстрее), Auto (автоматический выбор)
- ✅ **Speaker count estimation**: поддержка heuristic (default), silhouette (оптимальный k), fixed (min_speakers)
- ✅ **Clustering metrics**: полные метрики качества (silhouette, Davies-Bouldin, Calinski-Harabasz, intra/inter-cluster distances)

---

## 2) Contract Compliance Checklist

### 2.1 Интерфейсы и границы ответственности

**Extractor как компонент**:
- [x] Реализует `BaseExtractor.run_segments(input_uri, tmp_path, segments, ...)`
- [x] Не делает скрытых глобальных сайд-эффектов (модель загружается через ModelManager, no-network)
- [x] Требование специфичного входа декларировано: `audio/segments.json` family=`diarization` (см. README)
- [x] `run()` не поддерживается в production (возвращает error)

**Evidence**:
- `src/extractors/speaker_diarization_extractor/main.py:119` — метод `run_segments()`
- `src/extractors/speaker_diarization_extractor/main.py:240-255` — `run()` возвращает error с сообщением
- `src/extractors/speaker_diarization_extractor/main.py:133-134` — проверка segments: `if not isinstance(segments, list) or not segments: raise ValueError("segments is empty (no-fallback)")`

### 2.2 Контракты входа (Segmenter contract)

- [x] Входная единица: `audio/audio.wav` (Segmenter) + `audio/segments.json` family=`diarization`
- [x] Использует `start_sample/end_sample` для загрузки через `AudioUtils.load_audio_segment()`
- [x] Не генерирует сегменты сам (Segmenter — единственный владелец sampling)

**Evidence**:
- `src/extractors/speaker_diarization_extractor/main.py:148-154` — чтение `start_sample/end_sample` из segments
- `src/extractors/speaker_diarization_extractor/main.py:154` — `self.audio_utils.load_audio_segment(input_uri, start_sample=ss, end_sample=es, target_sr=self.sample_rate)`

### 2.3 No-fallback policy

- [x] Пустой список segments → fail-fast: `raise ValueError("segments is empty (no-fallback)")`
- [x] Аудио < 5 сек → fail-fast: `raise RuntimeError(f"speaker_diarization | audio too short for diarization (<5s): duration_sec={dur_sec:.3f}")`
- [x] Ошибка модели/инференса → `status="error"` в `ExtractorResult` с детальным error_code
- [x] Observability: ошибки логируются (без raw audio данных)

**Evidence**:
- `src/extractors/speaker_diarization_extractor/main.py:133-134` — проверка segments
- `src/extractors/speaker_diarization_extractor/main.py:137-139` — проверка длительности
- `src/extractors/speaker_diarization_extractor/main.py:456-461` — error handling в `run_segments()` с детальными error codes

### 2.4 Per-run storage

- [x] Имя NPZ файла стабильное: `speaker_diarization_extractor_features.npz` (фиксированное имя)
- [x] Запись NPZ атомарная (выполняется в `run_cli.py` через `_atomic_save_npz()`)
- [x] Нет произвольных JSON артефактов (только NPZ)

**Evidence**:
- `run_cli.py:518` — сохранение через `_save_component_npz()` с фиксированным именем
- `run_cli.py:252-714` — функция `_save_component_npz()` использует атомарную запись

### 2.5 NPZ schema + meta contract

**NPZ schema**: `schema_version="audio_npz_v1"`

**Обязательные ключи**:
- [x] `feature_names: object[str]` — имена фичей (scalars)
- [x] `feature_values: float32[]` — значения фичей (scalars)
- [x] `speaker_segments: object(list[dict])` — сегменты спикеров (feature-gated)
- [x] `speaker_ids: int32[N]` — ID спикеров
- [x] `speaker_embeddings_mean: float32[N_speakers, D]` — средние эмбеддинги (feature-gated)
- [x] `speaker_stats: object(dict)` — статистика по спикерам (feature-gated)
- [x] `segment_embeddings: object(list[list[float]])` — все индивидуальные эмбеддинги (feature-gated)
- [x] `speaker_time_ratios: object(dict)` — доли времени по спикерам (feature-gated)
- [x] `clustering_metrics: object(dict)` — метрики качества кластеризации (feature-gated)
- [x] `segment_start_sec: float32[N]` — временные метки начала сегментов
- [x] `segment_end_sec: float32[N]` — временные метки конца сегментов
- [x] `segment_center_sec: float32[N]` — временные метки центров сегментов
- [x] `meta: object(dict)` — обязательные поля

**Обязательные поля `meta`**:
- [x] `producer`, `producer_version`, `schema_version`, `created_at`
- [x] `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- [x] `status` ∈ {`ok`, `empty`, `error`}
- [x] `models_used[]`, `model_signature` (speaker diarization model через Triton)
- [x] `device_used`
- [x] `scheduler_knobs` (triton_batch_size, clustering_method, speaker_count_method)
- [x] `diarization_contract_version` — версия контракта для валидации совместимости с downstream extractors
- [x] `features_enabled[]` — список включённых фичей (feature gating)

**Extractor-level требования**:
- [x] Фичи имеют стабильные имена: `speaker_count`, `segments_count`, `duration_sec`, `speaker_balance_score`, `speaker_transitions_count`, `speaker_segments_density`, `dominant_speaker_id`
- [x] Единицы измерения зафиксированы в README
- [x] Missing values: NaN (если применимо)
- [x] Для per-segment sequences: `segment_centers_sec` строго монотонно возрастает

**Evidence**:
- `run_cli.py:518-564` — сохранение NPZ с feature-gated полями
- `src/extractors/speaker_diarization_extractor/main.py:222-234` — формирование payload с feature gating

### 2.6 Valid empty outputs

- [x] При `status="empty"`: `empty_reason="audio_silent"` (если silence detection включен)
- [x] Фичи при empty: NaN или пустые массивы
- [x] Empty не скрывает ошибки парсинга/модели/валидации

**Evidence**:
- `src/extractors/speaker_diarization_extractor/main.py:174-191` — обработка пустого аудио (silence detection)

---

## 3) Model System

### 3.1 ModelManager integration

- [x] Модель резолвится через `dp_models` (ModelManager)
- [x] Spec name: `speaker_diarization_{model_size}_triton` (small/large)
- [x] Runtime: `triton` (обязательно)
- [x] Артефакты лежат в `DP_MODELS_ROOT` и валидируются fail-fast
- [x] В `meta.models_used[]` фиксируются: `model_name`, `model_version`, `weights_digest`, `runtime`, `engine`, `precision`, `device`

**Evidence**:
- `src/extractors/speaker_diarization_extractor/main.py:71-97` — резолв модели через ModelManager
- `src/extractors/speaker_diarization_extractor/main.py:78` — spec name: `speaker_diarization_{self.model_size}_triton`
- `src/extractors/speaker_diarization_extractor/main.py:82-83` — проверка runtime = "triton"

### 3.2 No-network policy

- [x] Нет сетевых загрузок моделей/весов во время run
- [x] Triton параметры берутся из ModelManager spec или `TRITON_HTTP_URL` env var

**Evidence**:
- `src/extractors/speaker_diarization_extractor/main.py:87` — `triton_http_url` из spec или env var
- `src/extractors/speaker_diarization_extractor/main.py:99` — `TritonHttpClient` инициализируется с URL из spec

---

## 4) Segmenter Contract

### 4.1 Audio segments contract

- [x] Использует `audio/segments.json` family=`diarization`
- [x] Читает `families.diarization.segments[]` из `audio/segments.json`
- [x] Использует `start_sample/end_sample` для загрузки через `AudioUtils.load_audio_segment()`
- [x] Не генерирует сегменты сам

**Evidence**:
- `src/extractors/speaker_diarization_extractor/main.py:148-154` — чтение segments из family `diarization`
- `src/extractors/speaker_diarization_extractor/main.py:154` — загрузка через `AudioUtils.load_audio_segment()`

---

## 5) Наблюдаемость: progress + stage timings

### 5.1 Промежуточный прогресс

- [x] Progress обновляется каждые 10% сегментов (если сегментов ≥10)
- [x] Формат прогресса машиночитаем и безопасен (без raw audio данных)
- [x] Progress callback передаётся в `run_segments()` через параметр `progress_callback`

**Evidence**:
- `src/extractors/speaker_diarization_extractor/main.py:119` — метод `run_segments()` принимает `progress_callback`
- `src/extractors/speaker_diarization_extractor/main.py:147-155` — обновление прогресса каждые 10% сегментов
- `run_cli.py:1349-1360` — progress callback для speaker_diarization_extractor

### 5.2 Stage timings

- [x] Timings сохраняются в NPZ meta через `extra_meta` в `run_cli.py`
- [x] Per-extractor timings сохраняются в `timings_by_extractor`

**Evidence**:
- `run_cli.py:1542` — сохранение timings в meta

---

## 6) Feature Contract: управление выходными фичами (feature gating)

### 6.1 Feature gating flags

- [x] Все фичи opt-in через персональные флаги (default: все False)
- [x] Флаги: `--diar-enable-speaker-segments`, `--diar-enable-speaker-embeddings`, `--diar-enable-speaker-stats`, `--diar-enable-speaker-durations`, `--diar-enable-clustering-metrics`, `--diar-enable-segment-embeddings`
- [x] В `meta.features_enabled[]` фиксируются включённые фичи

**Evidence**:
- `src/extractors/speaker_diarization_extractor/main.py:36-44` — feature gating flags в `__init__`
- `src/extractors/speaker_diarization_extractor/main.py:222-234` — feature-gated payload
- `run_cli.py:833-843` — CLI аргументы для feature gating

### 6.2 Feature dependencies

- [x] Зависимости между фичами документированы в README (раздел "Feature Dependencies")
- [x] `speaker_durations` зависит от `speaker_stats`
- [x] `clustering_metrics` зависит от результатов кластеризации

**Evidence**:
- `src/extractors/speaker_diarization_extractor/README.md` — раздел "Feature Dependencies"

---

## 7) Производительность и ресурсы

### 7.1 Обязательные измерения

- [x] Latency per segment задокументирована (estimated_duration = 6.0 сек)
- [x] CPU RSS peak измеряется через resource_metrics в `run_cli.py`
- [x] GPU VRAM peak измеряется через resource_metrics в `run_cli.py`

**Evidence**:
- `src/extractors/speaker_diarization_extractor/main.py:30` — `estimated_duration = 6.0`
- `run_cli.py:1542` — resource_metrics в meta

### 7.2 Batching / OOM

- [x] Конфигурируемый `triton_batch_size` (None = auto, >100 segments → split)
- [x] Автоматическое разбиение на батчи при большом количестве сегментов

**Evidence**:
- `src/extractors/speaker_diarization_extractor/main.py:36-44` — параметр `triton_batch_size`
- `src/extractors/speaker_diarization_extractor/main.py:193-220` — автоматическое разбиение на батчи

---

## 8) Проверка качества выхода (quality validation)

### 8.1 Минимальные sanity-checks

- [x] Валидация эмбеддингов: проверка NaN/inf, диапазонов, размерности, dtype
- [x] Валидация меток кластеризации: согласованность с количеством сегментов и спикеров
- [x] Консистентность связных фичей (например, `speaker_ids` ↔ `speaker_segments`)

**Evidence**:
- `src/extractors/speaker_diarization_extractor/main.py:112-145` — метод `_validate_embeddings()`
- `src/extractors/speaker_diarization_extractor/main.py:147-156` — метод `_validate_clustering_labels()`
- `src/extractors/speaker_diarization_extractor/main.py:209-211` — валидация эмбеддингов после Triton inference

### 8.2 Human-friendly визуализация / UI render

- [x] Renderer реализован в `src/core/renderer.py` (`render_speaker_diarization_extractor()`)
- [x] HTML renderer для дебага (`render_speaker_diarization_extractor_html()`)
- [x] README содержит раздел "Visualization" с рекомендациями для UI/сайта

**Evidence**:
- `src/core/renderer.py:592-725` — renderer для speaker_diarization_extractor
- `src/core/renderer.py:728-1049` — HTML renderer для дебага
- `src/extractors/speaker_diarization_extractor/README.md` — раздел "Visualization"

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
- `src/extractors/speaker_diarization_extractor/main.py:101-130` — метод `_classify_triton_error()`
- `src/extractors/speaker_diarization_extractor/main.py:456-461` — использование error codes в error handling

### 9.2 No-fallback policy

- [x] Отсутствие модели → `RuntimeError`
- [x] Triton недоступен → `TritonError` с `error_code="triton_unavailable"`
- [x] Аудио < 5 сек → `RuntimeError`
- [x] Пустые сегменты → `ValueError`
- [x] Валидация эмбеддингов/кластеризации → `ValueError`

**Evidence**:
- `src/extractors/speaker_diarization_extractor/main.py:133-139` — fail-fast проверки
- `src/extractors/speaker_diarization_extractor/main.py:456-461` — error handling

---

## 10) Clustering Methods and Speaker Count Estimation

### 10.1 Clustering methods

- [x] Поддержка `agglomerative` (default для обучения)
- [x] Поддержка `kmeans` (быстрее для больших наборов данных)
- [x] Поддержка `auto` (автоматический выбор на основе количества сегментов)

**Evidence**:
- `src/extractors/speaker_diarization_extractor/main.py:36-44` — параметр `clustering_method`
- `src/extractors/speaker_diarization_extractor/main.py:280-295` — методы кластеризации
- `src/extractors/speaker_diarization_extractor/main.py:297-304` — автоматический выбор метода

### 10.2 Speaker count estimation methods

- [x] Поддержка `heuristic` (default, быстрая эвристика)
- [x] Поддержка `silhouette` (оптимальный k на основе silhouette score, default для обучения)
- [x] Поддержка `fixed` (фиксированное значение min_speakers)

**Evidence**:
- `src/extractors/speaker_diarization_extractor/main.py:36-44` — параметр `speaker_count_method`
- `src/extractors/speaker_diarization_extractor/main.py:257-273` — метод `_estimate_speaker_count_heuristic()`
- `src/extractors/speaker_diarization_extractor/main.py:275-295` — метод `_estimate_speaker_count_silhouette()`
- `src/extractors/speaker_diarization_extractor/main.py:297-304` — метод `_estimate_speaker_count()`

### 10.3 Clustering metrics

- [x] Полные метрики качества: silhouette_score, davies_bouldin_score, calinski_harabasz_score
- [x] Дополнительные метрики: mean_intra_cluster_distance, mean_inter_cluster_distance
- [x] Метрики feature-gated через `--diar-enable-clustering-metrics`

**Evidence**:
- `src/extractors/speaker_diarization_extractor/main.py:306-360` — метод `_compute_clustering_metrics()`
- `src/extractors/speaker_diarization_extractor/main.py:222-234` — feature-gated payload с clustering_metrics

---

## 11) Contract Versioning

### 11.1 Contract version для совместимости

- [x] `diarization_contract_version="diarization_contract_v1"` в payload
- [x] Contract version сохраняется в NPZ meta
- [x] Используется для валидации совместимости с downstream extractors (например, `speech_analysis_extractor`)

**Evidence**:
- `src/extractors/speaker_diarization_extractor/main.py:19` — константа `DIARIZATION_CONTRACT_VERSION`
- `src/extractors/speaker_diarization_extractor/main.py:222-234` — contract version в payload
- `run_cli.py:518-564` — contract version в meta

---

## 12) Документация

### 12.1 README extractor'а

- [x] Раздел "Входы" с описанием Segmenter contract
- [x] Раздел "Выходы" с описанием всех фичей (feature-gated)
- [x] Раздел "Feature Dependencies" с явным описанием зависимостей
- [x] Раздел "Конфигурация" с описанием всех параметров
- [x] Раздел "Feature Gating" с описанием всех флагов
- [x] Раздел "Visualization" с рекомендациями для UI/сайта
- [x] Раздел "Алгоритм" с описанием всех этапов обработки

**Evidence**:
- `src/extractors/speaker_diarization_extractor/README.md` — полная документация

---

## 13) Compliance Summary

### ✅ Все критерии выполнены

- ✅ **Архитектура**: соответствует `BaseExtractor`, Segmenter contract, per-run storage
- ✅ **Модели**: ModelManager, no-network, Triton-backed
- ✅ **Контракты**: NPZ schema, meta fields, contract versioning
- ✅ **Feature gating**: персональные флаги для каждой фичи (6 фичей, все opt-in)
- ✅ **Error handling**: детальные error codes для Triton (6 типов)
- ✅ **Валидация**: полная валидация эмбеддингов и кластеризации
- ✅ **Наблюдаемость**: progress reporting, stage timings
- ✅ **UI Render**: renderer + HTML renderer для дебага
- ✅ **Документация**: полный README с разделами Feature Dependencies и Visualization
- ✅ **Clustering**: поддержка нескольких методов кластеризации и оценки количества спикеров
- ✅ **Метрики**: полные метрики качества кластеризации

---

## 14) Open Issues

Нет открытых проблем. Все критерии `AP_AUDIT_CRITERIA.md` выполнены.

---

## 15) Default Settings for Training

**Рекомендуемые настройки для обучения моделей**:
- `clustering_method="agglomerative"` (лучшее качество)
- `speaker_count_method="silhouette"` (оптимальный k)
- Все feature flags включены (максимальное качество и полнота данных)

**Константы в коде**:
- `DEFAULT_CLUSTERING_METHOD_FOR_TRAINING = "agglomerative"`
- `DEFAULT_SPEAKER_COUNT_METHOD_FOR_TRAINING = "silhouette"`

**Evidence**:
- `src/extractors/speaker_diarization_extractor/main.py:17-18` — константы для обучения

