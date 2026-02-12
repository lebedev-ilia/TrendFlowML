# Audit: `key_extractor`

**Дата**: 2026-01-XX  
**Версия компонента**: 2.0.0  
**Статус**: ✅ **DONE** (full audit by `AP_AUDIT_CRITERIA.md`)

---

## Summary

`key_extractor` прошел полный аудит по критериям `AP_AUDIT_CRITERIA.md`. Компонент соответствует всем обязательным требованиям:

- ✅ Segmenter contract (`run_segments()` для `families.key.segments[]`)
- ✅ Feature gating (5 персональных флагов)
- ✅ No-fallback policy (явный выбор метода Essentia/librosa/auto, fail-fast)
- ✅ Full validation (выходные данные, параметры)
- ✅ Detailed error codes (6 типов)
- ✅ Progress reporting (по этапам и сегментам)
- ✅ Contract versioning (`key_contract_v1`)
- ✅ Optional audio normalization
- ✅ Additional ML/analytics metrics (confidence, stability, distribution, quality)
- ✅ Confidence categorization and warnings
- ✅ UI renderer (JSON + HTML для дебага)
- ✅ Integration with chroma_extractor via shared_features

---

## 1. Архитектурное соответствие

### 1.1 Интерфейсы и границы ответственности

✅ **Extractor реализует `BaseExtractor`**:
- `run(input_uri, tmp_path, shared_features)` — для полного аудио
- `run_segments(input_uri, tmp_path, segments, shared_features)` — для сегментов от Segmenter
- Возвращает `ExtractorResult` с полями: `success`, `payload`, `error`, `processing_time`, `device_used`

✅ **Нет скрытых сайд-эффектов**: все операции явные, нет сетевых загрузок, нет записи raw audio в логи

✅ **Требования декларированы**: README содержит раздел "Входы" с описанием required входов

**Evidence**: `src/extractors/key_extractor/main.py:47-530`, `src/extractors/key_extractor/main.py:531-785`

### 1.2 Контракты входа (Segmenter contract)

✅ **Входная единица**: `audio/audio.wav` (Segmenter) + `audio/segments.json` (contract `audio_segments_v1`)

✅ **Segmenter contract**: 
- `run()` использует полный аудио файл
- `run_segments()` использует `families.key.segments[]` из `audio/segments.json`
- Использует `start_sample/end_sample` для загрузки сегментов через `AudioUtils.load_audio_segment()`
- Не генерирует сегменты сами (Segmenter — единственный владелец sampling)

✅ **Fail-fast проверки**: отсутствие `audio/audio.wav` или `families.key.segments[]` → fail-fast

**Evidence**: `src/extractors/key_extractor/main.py:531-785`, `run_cli.py:2256-2257`

### 1.3 No-fallback policy

✅ **Fail-fast на входе**: отсутствие обязательного входа → `raise RuntimeError` с понятным сообщением

✅ **No-fallback для метода**: явный выбор `key_method` ("essentia" | "librosa" | "auto") через CLI
- Если выбран "essentia" и Essentia недоступна → fail-fast с error_code
- Если выбран "librosa" → только librosa метод
- Если выбран "auto" → Essentia с fallback на librosa (но это явная политика, не скрытый fallback)

✅ **Пустой список segments**: `raise ValueError("key | segments is empty (no-fallback)")`

✅ **Минимальная длительность**: fail-fast если аудио < 1 секунды

✅ **Observability**: все ошибки логгируются и отражаются в `manifest.json`

**Evidence**: `src/extractors/key_extractor/main.py:430-530`, `src/extractors/key_extractor/main.py:531-785`

### 1.4 Per-run storage

✅ **Фиксированное имя NPZ**: `key_extractor_features.npz`

✅ **Атомарная запись**: tmp → `os.replace()` (через `_atomic_save_npz`)

✅ **Sub-artifacts (.npy)**: не требуется (key_extractor не генерирует большие массивы)

✅ **Нет абсолютных путей**: в NPZ payload используются только относительные пути

**Evidence**: `run_cli.py:1148-1208`

### 1.5 NPZ schema + meta contract

✅ **Schema version**: `schema_version="audio_npz_v1"`

✅ **Обязательные ключи**: `feature_names`, `feature_values`, `meta`

✅ **Обязательные поля meta**:
- `producer`, `producer_version`, `schema_version`, `created_at`
- `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- `status` ∈ {`ok`, `empty`, `error`}
- `empty_reason` (только если `status="empty"`)
- `device_used`
- `key_contract_version` (для совместимости)

✅ **Стабильные имена фичей**: без случайных suffix/prefix

✅ **Missing values**: NaN для отсутствующих значений, feature gating для опциональных фичей

**Evidence**: `run_cli.py:1148-1208`, `src/extractors/key_extractor/main.py:510-530`

### 1.6 Valid empty outputs

✅ **Canonical empty_reason**: `audio_missing_or_extract_failed`, `video_too_short`, `video_too_long`, `dependency_missing`

✅ **Empty semantics**: при `status="empty"` фичи = NaN или пустые массивы

✅ **Empty не скрывает ошибки**: ошибки → `status="error"` с error_code

**Evidence**: `src/extractors/key_extractor/main.py:430-530`

---

## 2. Model system: no-network + ModelManager

✅ **ML модели не используются**: signal processing + music theory templates (librosa, essentia)

✅ **Нет сетевых загрузок**: все библиотеки локальные

✅ **models_used[]**: пустой (нет моделей)

**Evidence**: `src/extractors/key_extractor/main.py:1-785`, `src/extractors/key_extractor/README.md:43-53`

---

## 3. Segmenter contract: audio/segments.json

✅ **Schema version**: проверяется `schema_version="audio_segments_v1"` (на уровне CLI)

✅ **Family usage**: `families.key.segments[]` для `run_segments()`

✅ **Segment structure**: использует `start_sample/end_sample` для загрузки через `AudioUtils.load_audio_segment()`

✅ **Fail-fast**: отсутствие required family → `raise RuntimeError`

**Evidence**: `run_cli.py:2256-2257`, `src/extractors/key_extractor/main.py:531-785`

---

## 4. Зависимости между extractors

✅ **Опциональная зависимость от chroma_extractor**: может использовать предвычисленный `chroma` из `shared_features` для оптимизации

✅ **Зависимости фичей документированы**: README содержит раздел "Feature Dependencies" с явным описанием зависимостей

**Evidence**: `src/extractors/key_extractor/README.md:64-75`, `src/extractors/key_extractor/main.py:240-280`

---

## 5. Наблюдаемость: progress + stage timings

✅ **Stage-based прогресс**: обновления для этапов: load_audio, normalize_audio, detect_key, aggregate, complete

✅ **Segment-based прогресс**: для `run_segments()` прогресс обновляется каждые 10% сегментов

✅ **Формат прогресса**: JSON-line события через `progress_callback` (без raw audio данных)

✅ **Stage timings**: сохраняются в NPZ meta через `stage_timings_ms` (на уровне orchestrator)

**Evidence**: `src/extractors/key_extractor/main.py:440-530`, `src/extractors/key_extractor/main.py:531-785`, `run_cli.py:2815-2830`

---

## 6. Feature contract: управление выходными фичами

✅ **Feature gating**: 5 персональных флагов:
- `--key-enable-detailed-scores`: детальные оценки (24 значения)
- `--key-enable-top-k`: топ-K альтернативных тональностей
- `--key-enable-time-series`: временные серии (для run_segments)
- `--key-enable-key-changes`: детекция смены тональности
- `--key-enable-stability-metrics`: метрики стабильности

✅ **Default**: все фичи выключены (opt-in), кроме базовых полей (key_name, key_mode, key_confidence)

✅ **Зависимости фичей**: документированы в README (раздел "Feature Dependencies")

✅ **В meta фиксируются**: `features_enabled[]` в NPZ meta

**Evidence**: `src/extractors/key_extractor/main.py:54-80`, `src/extractors/key_extractor/README.md:105-120`

---

## 7. Производительность и ресурсы

✅ **Latency**: ~1.0 секунд для типичного аудио (задокументировано в README)

✅ **CPU RSS peak**: измеряется на уровне orchestrator (через `resource_metrics`)

✅ **GPU VRAM peak**: не используется (CPU-only)

✅ **Batching**: не требуется (signal processing)

**Evidence**: `src/extractors/key_extractor/README.md:200-210`, `run_cli.py:800-900`

---

## 8. Проверка качества выхода

### 8.1 Минимальные sanity-checks

✅ **Диапазоны значений**: проверка на NaN/inf, диапазоны (key_confidence ∈ [0.0, 1.0], key_name ∈ VALID_KEYS, key_mode ∈ VALID_MODES)

✅ **Консистентность**: проверка размеров массивов (key_scores имеет 24 значения), типов, размерностей

✅ **Статистические инварианты**: key_confidence ∈ [0.0, 1.0], key_scores нормализованы к [0, 1]

✅ **Валидация выходных данных**: полная валидация через `_validate_output()` метода

**Evidence**: `src/extractors/key_extractor/main.py:350-410`

### 8.2 Human-friendly визуализация / UI render

✅ **JSON renderer**: `render_key_extractor()` генерирует render-context JSON (без raw audio данных)

✅ **HTML renderer**: `render_key_extractor_html()` для локального дебага с raw данными (только для локального использования)

✅ **README визуализация**: раздел "Visualization" с рекомендациями по визуализации для UI/сайта

**Evidence**: `src/extractors/key_extractor/render.py`, `src/extractors/key_extractor/README.md:250-280`

---

## 9. Документация

### 9.1 README key_extractor

✅ **Обязательные разделы**:
- Input contract (Segmenter contract, families)
- Output contract (NPZ schema, пути, meta)
- Models (нет моделей, signal processing only)
- Feature dependencies (явное описание зависимостей)
- Feature gating (описание всех флагов)
- Configuration (параметры, дефолты)
- Algorithm (описание алгоритма работы)
- Error Handling (детальные error codes)
- Visualization (рекомендации по визуализации)
- Performance characteristics

**Evidence**: `src/extractors/key_extractor/README.md`

---

## 10. Compliance Summary

### ✅ Полное соответствие критериям

| Критерий | Статус | Примечания |
|----------|--------|------------|
| Segmenter contract | ✅ | `run_segments()` для `families.key.segments[]` |
| Feature gating | ✅ | 5 персональных флагов, все opt-in |
| No-fallback policy | ✅ | Явный выбор метода (essentia/librosa/auto), fail-fast |
| Validation | ✅ | Полная валидация выходных данных и параметров |
| Error codes | ✅ | 6 детальных error codes |
| Progress reporting | ✅ | По этапам и сегментам |
| Contract versioning | ✅ | `key_contract_v1` |
| UI renderer | ✅ | JSON + HTML для дебага |
| Documentation | ✅ | Полный README с всеми разделами |
| Confidence categorization | ✅ | Автоматическая категоризация и предупреждения |
| Additional metrics | ✅ | Метрики стабильности, распределения, качества |
| Integration with chroma_extractor | ✅ | Опциональное использование shared_features |

---

## 11. Open Issues

Нет открытых issues. Компонент готов к production использованию.

---

## 12. Implementation Details

### 12.1 Feature Gating

Все фичи контролируются через персональные флаги:
- `enable_detailed_scores`: `key_scores` (24 значения)
- `enable_top_k`: `key_top_k` (топ-K альтернативных тональностей)
- `enable_time_series`: `segment_centers_sec`, `key_names_sequence`, `key_modes_sequence`, `key_confidences_sequence`
- `enable_key_changes`: `key_transitions`, `key_transitions_count`, `key_transitions_rate`
- `enable_stability_metrics`: `key_stability_score`, `key_confidence_mean/std/min/max`, `key_distribution`, `key_diversity`, `key_detection_quality`

### 12.2 No-Fallback Policy

Явный выбор метода через `--key-method essentia|librosa|auto`. Если выбран "essentia" и Essentia недоступна, extractor завершается с ошибкой (fail-fast), без автоматического fallback (кроме режима "auto").

### 12.3 Confidence Categorization

Автоматическая категоризация уверенности:
- `high`: confidence ≥ 0.7
- `medium`: 0.5 ≤ confidence < 0.7
- `low`: 0.3 ≤ confidence < 0.5
- `very_low`: confidence < 0.3

Причины низкой уверенности:
- `normal`: нормальная уверенность
- `multiple_keys`: несколько тональностей с похожими оценками
- `atonal_or_insufficient_content`: атональная музыка или недостаточно контента
- `low_confidence`: общая низкая уверенность

### 12.4 Additional Metrics

Добавлены метрики для ML/аналитики:
- **Confidence metrics**: `key_confidence_mean/std/min/max` (для run_segments)
- **Stability metrics**: `key_stability_score` (доля времени в доминирующей тональности)
- **Distribution metrics**: `key_distribution` (распределение времени по тональностям), `key_diversity` (количество уникальных тональностей)
- **Quality metrics**: `key_detection_quality` (confidence × stability)
- **Change detection**: `key_transitions`, `key_transitions_count`, `key_transitions_rate`

### 12.5 Integration with chroma_extractor

Опциональная интеграция через `shared_features`:
- Если `chroma_extractor` был запущен ранее, `key_extractor` может использовать предвычисленный `chroma`
- Это избегает повторного вычисления хрома и улучшает производительность
- Fallback на собственное вычисление хрома, если `shared_features` не содержит `chroma`

---

## 13. References

- **AP_AUDIT_CRITERIA.md**: `AudioProcessor/docs/audit_v1/AP_AUDIT_CRITERIA.md`
- **Segmenter contract**: `docs/contracts/SEGMENTER_CONTRACT.md`
- **Component README**: `AudioProcessor/src/extractors/key_extractor/README.md`
- **Implementation**: `AudioProcessor/src/extractors/key_extractor/main.py`
- **Renderer**: `AudioProcessor/src/extractors/key_extractor/render.py`

