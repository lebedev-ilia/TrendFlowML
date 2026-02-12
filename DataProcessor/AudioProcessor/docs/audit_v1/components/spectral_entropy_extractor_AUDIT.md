# Audit: `spectral_entropy_extractor`

**Дата**: 2026-01-XX  
**Версия компонента**: 2.0.0  
**Статус**: ✅ **DONE** (full audit by `AP_AUDIT_CRITERIA.md`)

---

## Summary

`spectral_entropy_extractor` прошел полный аудит по критериям `AP_AUDIT_CRITERIA.md`. Компонент соответствует всем обязательным требованиям:

- ✅ Segmenter contract (`run_segments()` для `families.spectral_entropy.segments[]`)
- ✅ Feature gating (6 персональных флагов)
- ✅ No-fallback policy (librosa only, явно документировано, fail-fast)
- ✅ Full validation (выходные данные, параметры)
- ✅ Detailed error codes (9 типов)
- ✅ Progress reporting (по этапам и сегментам)
- ✅ Contract versioning (`spectral_entropy_contract_v1`)
- ✅ Optional audio normalization
- ✅ Additional ML/analytics metrics (dynamics, distribution, diversity)
- ✅ UI renderer (JSON + HTML для дебага)
- ✅ shared_features support (опционально): готов к переиспользованию спектрограммы/FFT, **но требуется явная публикация STFT/Mel из upstream extractor'а**

---

## 1. Архитектурное соответствие

### 1.1 Интерфейсы и границы ответственности

✅ **Extractor реализует `BaseExtractor`**:
- `run(input_uri, tmp_path, shared_features)` — для полного аудио
- `run_segments(input_uri, tmp_path, segments, shared_features)` — для сегментов от Segmenter
- Возвращает `ExtractorResult` с полями: `success`, `payload`, `error`, `processing_time`, `device_used`

✅ **Нет скрытых сайд-эффектов**: все операции явные, нет сетевых загрузок, нет записи raw audio в логи

✅ **Требования декларированы**: README содержит раздел "Входы" с описанием required входов

**Evidence**: `src/extractors/spectral_entropy_extractor/main.py:49-600`, `src/extractors/spectral_entropy_extractor/main.py:601-900`

### 1.2 Контракты входа (Segmenter contract)

✅ **Входная единица**: `audio/audio.wav` (Segmenter) + `audio/segments.json` (contract `audio_segments_v1`)

✅ **Segmenter contract**: 
- `run()` использует полный аудио файл
- `run_segments()` использует `families.spectral_entropy.segments[]` из `audio/segments.json`
- Использует `start_sample/end_sample` для загрузки сегментов через `AudioUtils.load_audio_segment()`
- Не генерирует сегменты сами (Segmenter — единственный владелец sampling)

✅ **Fail-fast проверки**: отсутствие `audio/audio.wav` или `families.spectral_entropy.segments[]` → fail-fast

**Evidence**: `src/extractors/spectral_entropy_extractor/main.py:601-900`, `run_cli.py:2526-2527`

### 1.3 No-fallback policy

✅ **Fail-fast на входе**: отсутствие обязательного входа → `raise RuntimeError` с понятным сообщением

✅ **No-fallback для метода**: используется только librosa (явно документировано в README)
- Essentia не поддерживается для spectral entropy (явно документировано)
- Все вычисления через librosa

✅ **Пустой список segments**: `raise ValueError("spectral_entropy | segments is empty (no-fallback)")`

✅ **Минимальная длительность**: fail-fast если аудио < 1 секунды (для `run()`) или < 0.1 секунды (для сегментов)

✅ **Валидация параметров**: проверка `n_fft >= 512`, `hop_length <= n_fft`, `n_mels >= 3`, `smoothing_window >= 0`

✅ **Observability**: все ошибки логгируются и отражаются в `manifest.json`

**Evidence**: `src/extractors/spectral_entropy_extractor/main.py:130-180`, `src/extractors/spectral_entropy_extractor/main.py:430-600`, `src/extractors/spectral_entropy_extractor/main.py:601-900`

### 1.4 Per-run storage

✅ **Фиксированное имя NPZ**: `spectral_entropy_extractor_features.npz`

✅ **Атомарная запись**: tmp → `os.replace()` (через `_atomic_save_npz`)

✅ **Sub-artifacts (.npy)**: не требуется (spectral_entropy_extractor не генерирует большие массивы, временные серии сохраняются inline)

✅ **Нет абсолютных путей**: в NPZ payload используются только относительные пути

**Evidence**: `run_cli.py:1332-1450`

### 1.5 NPZ schema + meta contract

✅ **Schema version**: `schema_version="audio_npz_v1"`

✅ **Обязательные ключи**: `feature_names`, `feature_values`, `meta`

✅ **Обязательные поля meta**:
- `producer`, `producer_version`, `schema_version`, `created_at`
- `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- `status` ∈ {`ok`, `empty`, `error`}
- `empty_reason` (только если `status="empty"`)
- `device_used`
- `spectral_entropy_contract_version` (для совместимости)

✅ **Стабильные имена фичей**: без случайных suffix/prefix

✅ **Missing values**: NaN для отсутствующих значений, feature gating для опциональных фичей

**Evidence**: `run_cli.py:1332-1450`, `src/extractors/spectral_entropy_extractor/main.py:520-600`

### 1.6 Valid empty outputs

✅ **Canonical empty_reason**: `audio_missing_or_extract_failed`, `video_too_short`, `video_too_long`, `dependency_missing`

✅ **Empty semantics**: при `status="empty"` фичи = NaN или пустые массивы

✅ **Empty не скрывает ошибки**: ошибки → `status="error"` с error_code

**Evidence**: `src/extractors/spectral_entropy_extractor/main.py:430-600`

---

## 2. Model system: no-network + ModelManager

✅ **ML модели не используются**: signal processing only (librosa)

✅ **Нет сетевых загрузок**: все библиотеки локальные

✅ **models_used[]**: пустой (нет моделей)

**Evidence**: `src/extractors/spectral_entropy_extractor/main.py:1-900`, `src/extractors/spectral_entropy_extractor/README.md:43-49`

---

## 3. Segmenter contract: audio/segments.json

✅ **Schema version**: проверяется `schema_version="audio_segments_v1"` (на уровне CLI)

✅ **Family usage**: `families.spectral_entropy.segments[]` для `run_segments()`

✅ **Segment structure**: использует `start_sample/end_sample` для загрузки через `AudioUtils.load_audio_segment()`

✅ **Fail-fast**: отсутствие required family → `raise RuntimeError`

**Evidence**: `run_cli.py:2526-2527`, `src/extractors/spectral_entropy_extractor/main.py:601-900`

---

## 4. Зависимости между extractors

✅ **Опциональная зависимость от spectral_extractor**: может использовать предвычисленный `stft_magnitude` или `mel_spectrogram` из `shared_features` для оптимизации

✅ **Зависимости фичей документированы**: README содержит раздел "Feature Dependencies" с явным описанием зависимостей

**Evidence**: `src/extractors/spectral_entropy_extractor/README.md:64-75`, `src/extractors/spectral_entropy_extractor/main.py:240-280`

---

## 5. Наблюдаемость: progress + stage timings

✅ **Stage-based прогресс**: обновления для этапов: load_audio, normalize_audio, compute_spectrogram, compute_entropy, compute_flatness, compute_spread, aggregate, complete

✅ **Segment-based прогресс**: для `run_segments()` прогресс обновляется каждые 10% сегментов

✅ **Формат прогресса**: JSON-line события через `progress_callback` (без raw audio данных)

✅ **Stage timings**: сохраняются в NPZ meta через `stage_timings_ms` (на уровне orchestrator)

**Evidence**: `src/extractors/spectral_entropy_extractor/main.py:430-600`, `src/extractors/spectral_entropy_extractor/main.py:601-900`, `run_cli.py:3116-3130`

---

## 6. Feature contract: управление выходными фичами

✅ **Feature gating**: 6 персональных флагов:
- `--spectral-entropy-enable-basic-stats`: базовые статистики (mean, std) для entropy
- `--spectral-entropy-enable-flatness`: метрики flatness (stats + series)
- `--spectral-entropy-enable-spread`: метрики spread (stats + series)
- `--spectral-entropy-enable-time-series`: временные серии для всех метрик
- `--spectral-entropy-enable-extended-stats`: расширенные статистики (min, max, p25, p75)
- `--spectral-entropy-enable-dynamics`: метрики динамики (для run_segments)

✅ **Default**: все фичи выключены (opt-in)

✅ **Зависимости фичей**: документированы в README (раздел "Feature Dependencies")

✅ **В meta фиксируются**: `features_enabled[]` в NPZ meta

**Evidence**: `src/extractors/spectral_entropy_extractor/main.py:54-100`, `src/extractors/spectral_entropy_extractor/README.md:105-120`

---

## 7. Производительность и ресурсы

✅ **Latency**: ~0.9 секунд для типичного аудио (задокументировано в README)

✅ **CPU RSS peak**: измеряется на уровне orchestrator (через `resource_metrics`)

✅ **GPU VRAM peak**: не используется (CPU-only)

✅ **Batching**: не требуется (signal processing)

**Evidence**: `src/extractors/spectral_entropy_extractor/README.md:200-210`, `run_cli.py:800-900`

---

## 8. Проверка качества выхода

### 8.1 Минимальные sanity-checks

✅ **Диапазоны значений**: проверка на NaN/inf, диапазоны (entropy в [0, log2(n_freq_bins)], flatness в [0, 1], spread ≥ 0)

✅ **Консистентность**: проверка размеров массивов, типов, размерностей

✅ **Статистические инварианты**: все метрики в допустимых диапазонах

✅ **Валидация выходных данных**: полная валидация через `_validate_output()` метода

**Evidence**: `src/extractors/spectral_entropy_extractor/main.py:350-410`

### 8.2 Human-friendly визуализация / UI render

✅ **JSON renderer**: `render_spectral_entropy_extractor()` генерирует render-context JSON (без raw audio данных)

✅ **HTML renderer**: `render_spectral_entropy_extractor_html()` для локального дебага с raw данными (только для локального использования)

✅ **README визуализация**: раздел "Visualization" с рекомендациями по визуализации для UI/сайта

**Evidence**: `src/extractors/spectral_entropy_extractor/render.py`, `src/extractors/spectral_entropy_extractor/README.md:250-280`

---

## 9. Документация

### 9.1 README spectral_entropy_extractor

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

**Evidence**: `src/extractors/spectral_entropy_extractor/README.md`

---

## 10. Compliance Summary

### ✅ Полное соответствие критериям

| Критерий | Статус | Примечания |
|----------|--------|------------|
| Segmenter contract | ✅ | `run_segments()` для `families.spectral_entropy.segments[]` |
| Feature gating | ✅ | 6 персональных флагов, все opt-in |
| No-fallback policy | ✅ | Librosa only (явно документировано), fail-fast |
| Validation | ✅ | Полная валидация выходных данных и параметров |
| Error codes | ✅ | 9 детальных error codes |
| Progress reporting | ✅ | По этапам и сегментам |
| Contract versioning | ✅ | `spectral_entropy_contract_v1` |
| UI renderer | ✅ | JSON + HTML для дебага |
| Documentation | ✅ | Полный README с всеми разделами |
| Additional metrics | ✅ | Метрики динамики, распределения, разнообразия |
| Integration with spectral_extractor | ✅ | shared_features support есть; wiring зависит от того, публикует ли upstream extractor STFT/Mel |

---

## 11. Open Issues

Нет открытых issues. Компонент готов к production использованию.

---

## 12. Implementation Details

### 12.1 Feature Gating

Все фичи контролируются через персональные флаги:
- `enable_basic_stats`: `spectral_entropy_stats` (mean, std)
- `enable_flatness`: `spectral_flatness_stats`, `spectral_flatness_series`
- `enable_spread`: `spectral_spread_stats`, `spectral_spread_series`
- `enable_time_series`: `spectral_entropy_series`, `spectral_flatness_series`, `spectral_spread_series`
- `enable_extended_stats`: `min`, `max`, `p25`, `p75` для всех метрик
- `enable_dynamics`: `spectral_entropy_stability`, `spectral_entropy_transitions_count/rate`, `spectral_entropy_distribution`, `spectral_entropy_diversity`

### 12.2 No-Fallback Policy

Используется только librosa (явно документировано в README). Essentia не поддерживается для spectral entropy, так как не имеет прямых аналогов для вычисления энтропии.

### 12.3 Additional Metrics

Добавлены метрики для ML/аналитики:
- **Dynamics metrics**: `spectral_entropy_stability` (variance), `spectral_entropy_transitions_count` (количество переходов), `spectral_entropy_transitions_rate` (частота переходов), `spectral_entropy_distribution` (распределение времени по уровням энтропии), `spectral_entropy_diversity` (разнообразие значений)
- **Additional variance metrics**: `spectral_entropy_variance`, `spectral_flatness_variance`, `spectral_spread_variance`
- **Min/max metrics**: `spectral_entropy_min/max`, `spectral_flatness_min/max`, `spectral_spread_min/max`

### 12.4 Integration with spectral_extractor (optional)

Опциональная интеграция через `shared_features`:
- `spectral_entropy_extractor` умеет переиспользовать `stft_magnitude` или `mel_spectrogram`, **если** upstream extractor публикует их в `shared_features` в рамках одного run.
- На текущий момент это рассматривается как оптимизация, требующая явного контракта публикации shared_features в orchestrator (без persistence).
- Fallback: собственное вычисление спектрограммы, если `shared_features` не содержит нужных данных

### 12.5 Spectral Entropy Computation

Эффективное вычисление спектральной энтропии:
- Нормировка спектра мощности для получения вероятностного распределения
- Вычисление энтропии Шеннона: `entropy = -sum(P * log2(P + eps))`
- Опциональное сглаживание через скользящее среднее
- Поддержка как STFT, так и Mel-спектрограммы

### 12.6 Spectral Flatness and Spread

Дополнительные метрики для комплексного анализа спектра:
- **Spectral Flatness**: отношение геометрического среднего к арифметическому среднему (показывает "шумоподобность")
- **Spectral Spread**: стандартное отклонение частотного индекса (показывает "ширину" спектра)

---

## 13. References

- **AP_AUDIT_CRITERIA.md**: `AudioProcessor/docs/audit_v1/AP_AUDIT_CRITERIA.md`
- **Segmenter contract**: `docs/contracts/SEGMENTER_CONTRACT.md`
- **Component README**: `AudioProcessor/src/extractors/spectral_entropy_extractor/README.md`
- **Implementation**: `AudioProcessor/src/extractors/spectral_entropy_extractor/main.py`
- **Renderer**: `AudioProcessor/src/extractors/spectral_entropy_extractor/render.py`

