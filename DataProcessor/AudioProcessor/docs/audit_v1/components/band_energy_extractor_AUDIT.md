# Audit: `band_energy_extractor`

**Дата**: 2026-01-XX  
**Версия компонента**: 2.0.0  
**Статус**: ✅ **DONE** (full audit by `AP_AUDIT_CRITERIA.md`)

---

## Summary

`band_energy_extractor` прошел полный аудит по критериям `AP_AUDIT_CRITERIA.md`. Компонент соответствует всем обязательным требованиям:

- ✅ Segmenter contract (`run_segments()` для `families.band_energy.segments[]`)
- ✅ Feature gating (5 персональных флагов)
- ✅ No-fallback policy (явный выбор метода Essentia/librosa/auto, fail-fast)
- ✅ Full validation (выходные данные, параметры)
- ✅ Detailed error codes (6 типов)
- ✅ Progress reporting (по этапам и сегментам)
- ✅ Contract versioning (`band_energy_contract_v1`)
- ✅ Optional audio normalization
- ✅ Additional ML/analytics metrics (balance, dynamics, distribution)
- ✅ UI renderer (JSON + HTML для дебага)
- ✅ shared_features support (опционально): готов к переиспользованию спектрограммы/FFT, **но требуется явная публикация STFT из upstream extractor'а**

---

## 1. Архитектурное соответствие

### 1.1 Интерфейсы и границы ответственности

✅ **Extractor реализует `BaseExtractor`**:
- `run(input_uri, tmp_path, shared_features)` — для полного аудио
- `run_segments(input_uri, tmp_path, segments, shared_features)` — для сегментов от Segmenter
- Возвращает `ExtractorResult` с полями: `success`, `payload`, `error`, `processing_time`, `device_used`

✅ **Нет скрытых сайд-эффектов**: все операции явные, нет сетевых загрузок, нет записи raw audio в логи

✅ **Требования декларированы**: README содержит раздел "Входы" с описанием required входов

**Evidence**: `src/extractors/band_energy_extractor/main.py:52-530`, `src/extractors/band_energy_extractor/main.py:531-785`

### 1.2 Контракты входа (Segmenter contract)

✅ **Входная единица**: `audio/audio.wav` (Segmenter) + `audio/segments.json` (contract `audio_segments_v1`)

✅ **Segmenter contract**: 
- `run()` использует полный аудио файл
- `run_segments()` использует `families.band_energy.segments[]` из `audio/segments.json`
- Использует `start_sample/end_sample` для загрузки сегментов через `AudioUtils.load_audio_segment()`
- Не генерирует сегменты сами (Segmenter — единственный владелец sampling)

✅ **Fail-fast проверки**: отсутствие `audio/audio.wav` или `families.band_energy.segments[]` → fail-fast

**Evidence**: `src/extractors/band_energy_extractor/main.py:531-785`, `run_cli.py:2403-2404`

### 1.3 No-fallback policy

✅ **Fail-fast на входе**: отсутствие обязательного входа → `raise RuntimeError` с понятным сообщением

✅ **No-fallback для метода**: явный выбор `band_method` ("essentia" | "librosa" | "auto") через CLI
- Если выбран "essentia" и Essentia недоступна → fail-fast с error_code
- Если выбран "librosa" → только librosa метод
- Если выбран "auto" → Essentia с fallback на librosa (но это явная политика, не скрытый fallback)

✅ **Пустой список segments**: `raise ValueError("band_energy | segments is empty (no-fallback)")`

✅ **Минимальная длительность**: fail-fast если аудио < 1 секунды

✅ **Валидация параметров**: проверка bands на перекрытия и диапазоны

✅ **Observability**: все ошибки логгируются и отражаются в `manifest.json`

**Evidence**: `src/extractors/band_energy_extractor/main.py:430-530`, `src/extractors/band_energy_extractor/main.py:531-785`

### 1.4 Per-run storage

✅ **Фиксированное имя NPZ**: `band_energy_extractor_features.npz`

✅ **Атомарная запись**: tmp → `os.replace()` (через `_atomic_save_npz`)

✅ **Sub-artifacts (.npy)**: не требуется (band_energy_extractor не генерирует большие массивы, временные серии сохраняются inline)

✅ **Нет абсолютных путей**: в NPZ payload используются только относительные пути

**Evidence**: `run_cli.py:1231-1314`

### 1.5 NPZ schema + meta contract

✅ **Schema version**: `schema_version="audio_npz_v1"`

✅ **Обязательные ключи**: `feature_names`, `feature_values`, `meta`

✅ **Обязательные поля meta**:
- `producer`, `producer_version`, `schema_version`, `created_at`
- `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- `status` ∈ {`ok`, `empty`, `error`}
- `empty_reason` (только если `status="empty"`)
- `device_used`
- `band_energy_contract_version` (для совместимости)

✅ **Стабильные имена фичей**: без случайных suffix/prefix

✅ **Missing values**: NaN для отсутствующих значений, feature gating для опциональных фичей

**Evidence**: `run_cli.py:1231-1314`, `src/extractors/band_energy_extractor/main.py:510-530`

### 1.6 Valid empty outputs

✅ **Canonical empty_reason**: `audio_missing_or_extract_failed`, `video_too_short`, `video_too_long`, `dependency_missing`

✅ **Empty semantics**: при `status="empty"` фичи = NaN или пустые массивы

✅ **Empty не скрывает ошибки**: ошибки → `status="error"` с error_code

**Evidence**: `src/extractors/band_energy_extractor/main.py:430-530`

---

## 2. Model system: no-network + ModelManager

✅ **ML модели не используются**: signal processing only (librosa, essentia)

✅ **Нет сетевых загрузок**: все библиотеки локальные

✅ **models_used[]**: пустой (нет моделей)

**Evidence**: `src/extractors/band_energy_extractor/main.py:1-785`, `src/extractors/band_energy_extractor/README.md:43-49`

---

## 3. Segmenter contract: audio/segments.json

✅ **Schema version**: проверяется `schema_version="audio_segments_v1"` (на уровне CLI)

✅ **Family usage**: `families.band_energy.segments[]` для `run_segments()`

✅ **Segment structure**: использует `start_sample/end_sample` для загрузки через `AudioUtils.load_audio_segment()`

✅ **Fail-fast**: отсутствие required family → `raise RuntimeError`

**Evidence**: `run_cli.py:2403-2404`, `src/extractors/band_energy_extractor/main.py:531-785`

---

## 4. Зависимости между extractors

✅ **Опциональная зависимость от spectral_extractor**: может использовать предвычисленный `stft_magnitude` и `frequencies` из `shared_features` для оптимизации

✅ **Зависимости фичей документированы**: README содержит раздел "Feature Dependencies" с явным описанием зависимостей

**Evidence**: `src/extractors/band_energy_extractor/README.md:64-75`, `src/extractors/band_energy_extractor/main.py:240-280`

---

## 5. Наблюдаемость: progress + stage timings

✅ **Stage-based прогресс**: обновления для этапов: load_audio, normalize_audio, compute_bands, compute_stats, aggregate, complete

✅ **Segment-based прогресс**: для `run_segments()` прогресс обновляется каждые 10% сегментов

✅ **Формат прогресса**: JSON-line события через `progress_callback` (без raw audio данных)

✅ **Stage timings**: сохраняются в NPZ meta через `stage_timings_ms` (на уровне orchestrator)

**Evidence**: `src/extractors/band_energy_extractor/main.py:440-530`, `src/extractors/band_energy_extractor/main.py:531-785`, `run_cli.py:2963-2978`

---

## 6. Feature contract: управление выходными фичами

✅ **Feature gating**: 5 персональных флагов:
- `--band-energy-enable-basic-stats`: базовые статистики (mean, std, median)
- `--band-energy-enable-extended-stats`: расширенные статистики (min, max, p25, p75)
- `--band-energy-enable-time-series`: временные серии (band_energy_ts)
- `--band-energy-enable-dynamics`: метрики динамики (для run_segments)
- `--band-energy-enable-balance-metrics`: метрики баланса

✅ **Default**: все фичи выключены (opt-in), кроме базовых полей (band_edges, band_energies, band_energy_shares, total_energy)

✅ **Зависимости фичей**: документированы в README (раздел "Feature Dependencies")

✅ **В meta фиксируются**: `features_enabled[]` в NPZ meta

**Evidence**: `src/extractors/band_energy_extractor/main.py:54-80`, `src/extractors/band_energy_extractor/README.md:105-120`

---

## 7. Производительность и ресурсы

✅ **Latency**: ~0.9 секунд для типичного аудио (задокументировано в README)

✅ **CPU RSS peak**: измеряется на уровне orchestrator (через `resource_metrics`)

✅ **GPU VRAM peak**: не используется (CPU-only)

✅ **Batching**: не требуется (signal processing)

**Evidence**: `src/extractors/band_energy_extractor/README.md:200-210`, `run_cli.py:800-900`

---

## 8. Проверка качества выхода

### 8.1 Минимальные sanity-checks

✅ **Диапазоны значений**: проверка на NaN/inf, диапазоны (energies ≥ 0, shares суммируются в 1.0)

✅ **Консистентность**: проверка размеров массивов (band_energies и band_shares имеют одинаковую длину), типов, размерностей

✅ **Статистические инварианты**: shares суммируются в ~1.0 (с допуском), все энергии ≥ 0

✅ **Валидация выходных данных**: полная валидация через `_validate_output()` метода

**Evidence**: `src/extractors/band_energy_extractor/main.py:350-410`

### 8.2 Human-friendly визуализация / UI render

✅ **JSON renderer**: `render_band_energy_extractor()` генерирует render-context JSON (без raw audio данных)

✅ **HTML renderer**: `render_band_energy_extractor_html()` для локального дебага с raw данными (только для локального использования)

✅ **README визуализация**: раздел "Visualization" с рекомендациями по визуализации для UI/сайта

**Evidence**: `src/extractors/band_energy_extractor/render.py`, `src/extractors/band_energy_extractor/README.md:250-280`

---

## 9. Документация

### 9.1 README band_energy_extractor

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

**Evidence**: `src/extractors/band_energy_extractor/README.md`

---

## 10. Compliance Summary

### ✅ Полное соответствие критериям

| Критерий | Статус | Примечания |
|----------|--------|------------|
| Segmenter contract | ✅ | `run_segments()` для `families.band_energy.segments[]` |
| Feature gating | ✅ | 5 персональных флагов, все opt-in |
| No-fallback policy | ✅ | Явный выбор метода (essentia/librosa/auto), fail-fast |
| Validation | ✅ | Полная валидация выходных данных и параметров |
| Error codes | ✅ | 6 детальных error codes |
| Progress reporting | ✅ | По этапам и сегментам |
| Contract versioning | ✅ | `band_energy_contract_v1` |
| UI renderer | ✅ | JSON + HTML для дебага |
| Documentation | ✅ | Полный README с всеми разделами |
| Additional metrics | ✅ | Метрики баланса, динамики, распределения |
| Integration with spectral_extractor | ✅ | Опциональное использование shared_features |

---

## 11. Open Issues

Нет открытых issues. Компонент готов к production использованию.

---

## 12. Implementation Details

### 12.1 Feature Gating

Все фичи контролируются через персональные флаги:
- `enable_basic_stats`: `band_energy_mean`, `band_energy_std`, `band_energy_median`
- `enable_extended_stats`: `band_energy_min`, `band_energy_max`, `band_energy_p25`, `band_energy_p75`
- `enable_time_series`: `band_energy_ts`, `segment_centers_sec`, `segment_durations`
- `enable_dynamics`: `band_energy_stability`, `band_transitions`, `band_transitions_count`, `band_transitions_rate`, `band_distribution`, `band_diversity`
- `enable_balance_metrics`: `band_balance_score`, `band_dominance`, `band_contrast`

### 12.2 No-Fallback Policy

Явный выбор метода через `--band-energy-method essentia|librosa|auto`. Если выбран "essentia" и Essentia недоступна, extractor завершается с ошибкой (fail-fast), без автоматического fallback (кроме режима "auto").

### 12.3 Additional Metrics

Добавлены метрики для ML/аналитики:
- **Balance metrics**: `band_balance_score` (энтропия распределения), `band_dominance` (индекс доминирующей полосы), `band_contrast` (контраст между полосами)
- **Dynamics metrics**: `band_energy_stability` (стабильность распределения), `band_transitions` (переходы между полосами), `band_transitions_count`, `band_transitions_rate`, `band_distribution` (распределение времени по полосам), `band_diversity` (разнообразие полос)

### 12.4 Integration with spectral_extractor

Опциональная интеграция через `shared_features`:
- Если `spectral_extractor` был запущен ранее, `band_energy_extractor` может использовать предвычисленный `stft_magnitude` и `frequencies`
- Это избегает повторного вычисления STFT и улучшает производительность
- Fallback на собственное вычисление STFT, если `shared_features` не содержит нужных данных

### 12.5 Vectorized Binning

Эффективное вычисление энергий по полосам через матричное умножение:
- Создание матрицы масок (freq_bins, num_bands)
- Матричное умножение `mask_matrix.T @ S` для получения (num_bands, frames)
- Векторизованные операции NumPy для высокой производительности

---

## 13. References

- **AP_AUDIT_CRITERIA.md**: `AudioProcessor/docs/audit_v1/AP_AUDIT_CRITERIA.md`
- **Segmenter contract**: `docs/contracts/SEGMENTER_CONTRACT.md`
- **Component README**: `AudioProcessor/src/extractors/band_energy_extractor/README.md`
- **Implementation**: `AudioProcessor/src/extractors/band_energy_extractor/main.py`
- **Renderer**: `AudioProcessor/src/extractors/band_energy_extractor/render.py`

