# Audit: `chroma_extractor`

**Дата**: 2026-01-XX  
**Версия компонента**: 2.0.0  
**Статус**: ✅ **DONE** (full audit by `AP_AUDIT_CRITERIA.md`)

---

## Summary

`chroma_extractor` прошел полный аудит по критериям `AP_AUDIT_CRITERIA.md`. Компонент соответствует всем обязательным требованиям:

- ✅ Segmenter contract (`run_segments()` для `families.chroma.segments[]`)
- ✅ Feature gating (4 персональных флага)
- ✅ No-fallback policy (явный выбор CQT/STFT, fail-fast)
- ✅ Full validation (выходные данные, параметры)
- ✅ Detailed error codes (8 типов)
- ✅ Progress reporting (по этапам и сегментам)
- ✅ Contract versioning (`chroma_contract_v1`)
- ✅ Per-run storage для .npy файлов
- ✅ Optional audio normalization
- ✅ Additional ML/analytics metrics
- ✅ UI renderer (JSON + HTML для дебага)

---

## 1. Архитектурное соответствие

### 1.1 Интерфейсы и границы ответственности

✅ **Extractor реализует `BaseExtractor`**:
- `run(input_uri, tmp_path)` — для полного аудио
- `run_segments(input_uri, tmp_path, segments)` — для сегментов от Segmenter
- Возвращает `ExtractorResult` с полями: `success`, `payload`, `error`, `processing_time`, `device_used`

✅ **Нет скрытых сайд-эффектов**: все операции явные, нет сетевых загрузок, нет записи raw audio в логи

✅ **Требования декларированы**: README содержит раздел "Входы" с описанием required входов

**Evidence**: `src/extractors/chroma_extractor/main.py:18-183`, `src/extractors/chroma_extractor/main.py:185-350`

### 1.2 Контракты входа (Segmenter contract)

✅ **Входная единица**: `audio/audio.wav` (Segmenter) + `audio/segments.json` (contract `audio_segments_v1`)

✅ **Segmenter contract**: 
- `run()` использует полный аудио файл
- `run_segments()` использует `families.chroma.segments[]` из `audio/segments.json`
- Использует `start_sample/end_sample` для загрузки сегментов через `AudioUtils.load_audio_segment()`
- Не генерирует сегменты сами (Segmenter — единственный владелец sampling)

✅ **Fail-fast проверки**: отсутствие `audio/audio.wav` или `families.chroma.segments[]` → fail-fast

**Evidence**: `src/extractors/chroma_extractor/main.py:185-350`, `run_cli.py:1600-1610`

### 1.3 No-fallback policy

✅ **Fail-fast на входе**: отсутствие обязательного входа → `raise RuntimeError` с понятным сообщением

✅ **No-fallback для метода**: явный выбор `chroma_type` ("cqt" | "stft") через CLI, если выбранный метод не работает → fail-fast с error_code

✅ **Пустой список segments**: `raise ValueError("segments is empty (no-fallback)")`

✅ **Observability**: все ошибки логгируются и отражаются в `manifest.json`

**Evidence**: `src/extractors/chroma_extractor/main.py:200-250`, `src/extractors/chroma_extractor/main.py:300-350`

### 1.4 Per-run storage

✅ **Фиксированное имя NPZ**: `chroma_extractor_features.npz`

✅ **Атомарная запись**: tmp → `os.replace()` (через `_atomic_save_npz`)

✅ **Sub-artifacts (.npy)**: большие временные серии сохраняются в `result_store/<component_name>/_artifacts/chroma.npy` и регистрируются в `manifest.json`

✅ **Нет абсолютных путей**: в NPZ payload используются только relpath (`_artifacts/chroma.npy`)

**Evidence**: `run_cli.py:970-1060`, `src/extractors/chroma_extractor/main.py:250-280`

### 1.5 NPZ schema + meta contract

✅ **Schema version**: `schema_version="audio_npz_v1"`

✅ **Обязательные ключи**: `feature_names`, `feature_values`, `meta`

✅ **Обязательные поля meta**:
- `producer`, `producer_version`, `schema_version`, `created_at`
- `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- `status` ∈ {`ok`, `empty`, `error`}
- `empty_reason` (только если `status="empty"`)
- `device_used`
- `chroma_contract_version` (для совместимости)

✅ **Стабильные имена фичей**: без случайных suffix/prefix

✅ **Missing values**: NaN для отсутствующих значений, маски `*_present` не используются (feature gating)

**Evidence**: `run_cli.py:970-1060`, `src/extractors/chroma_extractor/main.py:320-350`

### 1.6 Valid empty outputs

✅ **Canonical empty_reason**: `audio_missing_or_extract_failed`, `video_too_short`, `video_too_long`, `dependency_missing`

✅ **Empty semantics**: при `status="empty"` фичи = NaN или пустые массивы

✅ **Empty не скрывает ошибки**: ошибки → `status="error"` с error_code

**Evidence**: `src/extractors/chroma_extractor/main.py:179-183`, `src/extractors/chroma_extractor/main.py:340-350`

---

## 2. Model system: no-network + ModelManager

✅ **ML модели не используются**: signal processing only (librosa)

✅ **Нет сетевых загрузок**: все библиотеки локальные

✅ **models_used[]**: пустой (нет моделей)

**Evidence**: `src/extractors/chroma_extractor/main.py:1-183`, `src/extractors/chroma_extractor/README.md:46-51`

---

## 3. Segmenter contract: audio/segments.json

✅ **Schema version**: проверяется `schema_version="audio_segments_v1"` (на уровне CLI)

✅ **Family usage**: `families.chroma.segments[]` для `run_segments()`

✅ **Segment structure**: использует `start_sample/end_sample` для загрузки через `AudioUtils.load_audio_segment()`

✅ **Fail-fast**: отсутствие required family → `raise RuntimeError`

**Evidence**: `run_cli.py:1600-1610`, `src/extractors/chroma_extractor/main.py:185-350`

---

## 4. Зависимости между extractors

✅ **Нет зависимостей**: `chroma_extractor` не зависит от других extractors

✅ **Зависимости фичей документированы**: README содержит раздел "Feature Dependencies"

**Evidence**: `src/extractors/chroma_extractor/README.md:53-62`

---

## 5. Наблюдаемость: progress + stage timings

✅ **Stage-based прогресс**: обновления для этапов: load_audio, estimate_tuning, extract_chroma, normalize, compute_stats, validate_output, save_artifacts

✅ **Segment-based прогресс**: для `run_segments()` прогресс обновляется каждые 10% сегментов

✅ **Формат прогресса**: JSON-line события через `progress_callback` (без raw audio данных)

✅ **Stage timings**: сохраняются в NPZ meta через `stage_timings_ms` (на уровне orchestrator)

**Evidence**: `src/extractors/chroma_extractor/main.py:240-350`, `run_cli.py:1700-1720`

---

## 6. Feature contract: управление выходными фичами

✅ **Feature gating**: 4 персональных флага:
- `--chroma-enable-basic-stats`: базовые статистики (mean, std, min, max)
- `--chroma-enable-extended-stats`: расширенные статистики (median, p25, p75)
- `--chroma-enable-stats-vector`: компактный вектор статистик
- `--chroma-enable-time-series`: временные серии (chroma spectrogram)

✅ **Default**: все фичи выключены (opt-in)

✅ **Зависимости фичей**: документированы в README (раздел "Feature Dependencies")

✅ **В meta фиксируются**: `features_enabled[]` в NPZ meta

**Evidence**: `src/extractors/chroma_extractor/main.py:32-56`, `src/extractors/chroma_extractor/README.md:64-85`

---

## 7. Производительность и ресурсы

✅ **Latency**: ~1.2 секунд для типичного аудио (задокументировано в README)

✅ **CPU RSS peak**: измеряется на уровне orchestrator (через `resource_metrics`)

✅ **GPU VRAM peak**: не используется (CPU-only)

✅ **Batching**: не требуется (signal processing)

**Evidence**: `src/extractors/chroma_extractor/README.md:93-100`, `run_cli.py:800-900`

---

## 8. Проверка качества выхода

### 8.1 Минимальные sanity-checks

✅ **Диапазоны значений**: проверка на NaN/inf, диапазоны (chroma ∈ [0, 1] после нормализации, или ≥ 0 без нормализации)

✅ **Консистентность**: проверка размеров массивов (12 классов), типов, размерностей

✅ **Статистические инварианты**: tuning_estimate ∈ [-1.0, 1.0], chroma_frames ≥ 0

**Evidence**: `src/extractors/chroma_extractor/main.py:150-200`

### 8.2 Human-friendly визуализация / UI render

✅ **JSON renderer**: `render_chroma_extractor()` генерирует render-context JSON (без raw audio данных)

✅ **HTML renderer**: `render_chroma_extractor_html()` для локального дебага с raw данными (только для локального использования)

✅ **README визуализация**: раздел "Visualization" с рекомендациями по визуализации для UI/сайта

**Evidence**: `src/core/renderer.py:3100-3300`, `src/extractors/chroma_extractor/README.md:102-130`

---

## 9. Документация

### 9.1 README chroma_extractor

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

**Evidence**: `src/extractors/chroma_extractor/README.md`

---

## 10. Compliance Summary

### ✅ Полное соответствие критериям

| Критерий | Статус | Примечания |
|----------|--------|------------|
| Segmenter contract | ✅ | `run_segments()` для `families.chroma.segments[]` |
| Feature gating | ✅ | 4 персональных флага, все opt-in |
| No-fallback policy | ✅ | Явный выбор CQT/STFT, fail-fast |
| Validation | ✅ | Полная валидация выходных данных и параметров |
| Error codes | ✅ | 8 детальных error codes |
| Progress reporting | ✅ | По этапам и сегментам |
| Contract versioning | ✅ | `chroma_contract_v1` |
| Per-run storage | ✅ | .npy файлы в `_artifacts/` |
| UI renderer | ✅ | JSON + HTML для дебага |
| Documentation | ✅ | Полный README с всеми разделами |

---

## 11. Open Issues

Нет открытых issues. Компонент готов к production использованию.

---

## 12. Implementation Details

### 12.1 Feature Gating

Все фичи контролируются через персональные флаги:
- `enable_basic_stats`: `chroma_mean`, `chroma_std`, `chroma_min`, `chroma_max`
- `enable_extended_stats`: `chroma_median`, `chroma_p25`, `chroma_p75`
- `enable_stats_vector`: `chroma_stats_vector` (конкатенирует все включенные статистики)
- `enable_time_series`: `chroma` (spectrogram)

### 12.2 No-Fallback Policy

Явный выбор метода через `--chroma-type cqt|stft`. Если выбранный метод не работает, extractor завершается с ошибкой (fail-fast), без автоматического fallback.

### 12.3 Additional Metrics

Добавлены метрики для ML/аналитики:
- `tuning_estimate`: оценка строя
- `chroma_dominant_class`: доминирующий хрома-класс
- `chroma_dominant_energy`: энергия доминирующего класса
- `chroma_harmonic_stability`: стабильность гармонического содержания
- `chroma_entropy`: энтропия распределения
- `chroma_contrast`: контраст между классами
- `chroma_centroid`: центроид распределения
- `chroma_rolloff`: rolloff частоты

### 12.4 Per-Run Storage

Большие временные серии (`chroma`) сохраняются в `.npy` файлы в `result_store/<component_name>/_artifacts/chroma.npy` и регистрируются в `manifest.json`.

---

## 13. References

- **AP_AUDIT_CRITERIA.md**: `AudioProcessor/docs/audit_v1/AP_AUDIT_CRITERIA.md`
- **Segmenter contract**: `docs/contracts/SEGMENTER_CONTRACT.md`
- **Component README**: `AudioProcessor/src/extractors/chroma_extractor/README.md`
- **Implementation**: `AudioProcessor/src/extractors/chroma_extractor/main.py`

