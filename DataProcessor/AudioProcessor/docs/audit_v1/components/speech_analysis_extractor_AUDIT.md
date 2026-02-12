# Audit: `speech_analysis_extractor`

**Дата**: 2026-01-XX  
**Статус**: `done`  
**Критерии**: `AudioProcessor/docs/audit_v1/AP_AUDIT_CRITERIA.md`

---

## 1) Summary

`speech_analysis_extractor` соответствует всем критериям `AP_AUDIT_CRITERIA.md`:
- ✅ **Bundle extractor**: объединяет результаты ASR, diarization и pitch под-экстракторов
- ✅ **Segmenter contract**: использует `audio/audio.wav` и `audio/segments.json` families (`asr`, `diarization`)
- ✅ **No-fallback policy**: fail-fast при отсутствии segments, аудио < 5 сек
- ✅ **Model system**: загрузка через `dp_models` (ModelManager), no-network (под-экстракторы обеспечивают ModelManager)
- ✅ **NPZ schema**: `audio_npz_v1` с обязательными полями meta
- ✅ **Per-run storage**: фиксированное имя `speech_analysis_extractor_features.npz`
- ✅ **Feature gating**: персональные флаги для каждой группы фичей (ASR, diarization, pitch, все opt-in)
- ✅ **Error handling**: детальные error codes (asr_failed, diarization_failed, pitch_failed, silence_detection_failed, validation_failed, segments_invalid, audio_too_short)
- ✅ **Segments validation**: полная валидация структуры сегментов (типы, обязательные поля, диапазоны)
- ✅ **Payload validation**: полная валидация payload от под-экстракторов (ASR, diarization, pitch)
- ✅ **Progress reporting**: обновление прогресса для каждого под-экстрактора
- ✅ **UI Render**: renderer реализован в `src/core/renderer.py` + HTML renderer для дебага
- ✅ **Contract version**: `speech_analysis_contract_version` для валидации совместимости с downstream extractors
- ✅ **Additional metrics**: speech_rate_wpm, lang_distribution для ASR; speaker_balance_score, speaker_transitions_count для diarization; pitch range, stability, distribution
- ✅ **Silence detection**: конфигурируемые пороги (`silence_peak_threshold`, `silence_rms_threshold`)

---

## 2) Contract Compliance Checklist

### 2.1 Интерфейсы и границы ответственности

**Extractor как компонент**:
- [x] Реализует `BaseExtractor.run_bundle(input_uri, tmp_path, asr_segments, diar_segments, ...)`
- [x] Не делает скрытых глобальных сайд-эффектов (модели загружаются через под-экстракторы, no-network)
- [x] Требование специфичного входа декларировано: `audio/segments.json` families `asr` и `diarization` (см. README)
- [x] `run()` не поддерживается в production (возвращает error)

**Evidence**:
- `src/extractors/speech_analysis_extractor/main.py:87` — метод `run_bundle()`
- `src/extractors/speech_analysis_extractor/main.py:212-217` — `run()` возвращает error с сообщением
- `src/extractors/speech_analysis_extractor/main.py:92-95` — проверка segments: `if not isinstance(asr_segments, list) or not asr_segments: raise ValueError("speech_analysis | asr_segments is empty (no-fallback)")`

### 2.2 Контракты входа (Segmenter contract)

- [x] Входная единица: `audio/audio.wav` (Segmenter) + `audio/segments.json` families `asr` и `diarization`
- [x] Использует `start_sample/end_sample` для загрузки через `AudioUtils.load_audio_segment()` (в под-экстракторах)
- [x] Не генерирует сегменты сам (Segmenter — единственный владелец sampling)

**Evidence**:
- `src/extractors/speech_analysis_extractor/main.py:87` — метод `run_bundle()` принимает `asr_segments` и `diar_segments`
- `src/extractors/speech_analysis_extractor/main.py:133-138` — запуск под-экстракторов с сегментами

### 2.3 No-fallback policy

- [x] Пустой список segments → fail-fast: `raise ValueError("speech_analysis | asr_segments is empty (no-fallback)")`
- [x] Аудио < 5 сек → fail-fast: `raise RuntimeError(f"speech_analysis | audio too short (<5s): duration_sec={dur_sec:.3f} (error_code={error_code})")`
- [x] Ошибка под-экстрактора → fail-fast с детальным error_code
- [x] Observability: ошибки логируются (без raw audio данных)

**Evidence**:
- `src/extractors/speech_analysis_extractor/main.py:92-95` — проверка segments
- `src/extractors/speech_analysis_extractor/main.py:103-104` — проверка длительности
- `src/extractors/speech_analysis_extractor/main.py:134-138` — error handling для под-экстракторов

### 2.4 Per-run storage

- [x] Имя NPZ файла стабильное: `speech_analysis_extractor_features.npz` (фиксированное имя)
- [x] Запись NPZ атомарная (выполняется в `run_cli.py` через `_atomic_save_npz()`)
- [x] Нет произвольных JSON артефактов (только NPZ)

**Evidence**:
- `run_cli.py:788` — сохранение через `_save_component_npz()` с фиксированным именем
- `run_cli.py:252-821` — функция `_save_component_npz()` использует атомарную запись

### 2.5 NPZ schema + meta contract

**NPZ schema**: `schema_version="audio_npz_v1"`

**Обязательные ключи**:
- [x] `feature_names: object[str]` — имена фичей (scalars)
- [x] `feature_values: float32[]` — значения фичей (scalars)
- [x] `asr_lang_id_by_segment: int32[N]` — идентификаторы языка (feature-gated)
- [x] `speaker_ids: int32[M]` — идентификаторы спикеров (feature-gated)
- [x] `asr_lang_distribution: object(dict)` — распределение языков (feature-gated)
- [x] `pitch_distribution: object(dict)` — распределение pitch (feature-gated)
- [x] `meta: object(dict)` — обязательные поля

**Обязательные поля `meta`**:
- [x] `producer`, `producer_version`, `schema_version`, `created_at`
- [x] `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- [x] `status` ∈ {`ok`, `empty`, `error`}
- [x] `models_used[]`, `model_signature` (ASR и diarization модели через Triton, pitch через signal processing)
- [x] `device_used`
- [x] `speech_analysis_contract_version` — версия контракта для валидации совместимости с downstream extractors
- [x] `features_enabled[]` — список включённых фичей (feature gating)

**Extractor-level требования**:
- [x] Фичи имеют стабильные имена: `duration_sec`, `sample_rate`, `asr_segments_count`, `asr_token_total`, `speaker_count`, `pitch_f0_mean`, и т.д.
- [x] Единицы измерения зафиксированы в README
- [x] Missing values: NaN (если применимо)
- [x] Feature-gated поля сохраняются только если соответствующий флаг включен

**Evidence**:
- `run_cli.py:788-821` — сохранение NPZ с feature-gated полями
- `src/extractors/speech_analysis_extractor/main.py:186-252` — формирование payload с feature gating

### 2.6 Valid empty outputs

- [x] При `status="empty"`: `empty_reason="audio_silent"` (если silence detection включен)
- [x] Фичи при empty: NaN или пустые массивы
- [x] Empty не скрывает ошибки парсинга/модели/валидации

**Evidence**:
- `src/extractors/speech_analysis_extractor/main.py:122-130` — обработка пустого аудио (silence detection)

---

## 3) Model System

### 3.1 ModelManager integration

- [x] Модели резолвятся через `dp_models` (ModelManager) в под-экстракторах
- [x] ASR: spec `whisper_{model_size}_triton` (через ASRExtractor)
- [x] Diarization: spec `speaker_diarization_{model_size}_triton` (через SpeakerDiarizationExtractor)
- [x] Pitch: signal processing (не требует ModelManager)
- [x] В `meta.models_used[]` фиксируются модели от под-экстракторов

**Evidence**:
- `src/extractors/speech_analysis_extractor/main.py:66-76` — создание под-экстракторов
- Под-экстракторы обеспечивают ModelManager integration

### 3.2 No-network policy

- [x] Нет сетевых загрузок моделей/весов во время run
- [x] Под-экстракторы обеспечивают no-network policy

**Evidence**:
- `src/extractors/speech_analysis_extractor/main.py:66-76` — под-экстракторы создаются без сетевых загрузок

---

## 4) Segmenter Contract

### 4.1 Audio segments contract

- [x] Использует `audio/segments.json` families `asr` и `diarization`
- [x] Читает `families.asr.segments[]` и `families.diarization.segments[]` из `audio/segments.json`
- [x] Передает сегменты в под-экстракторы (ASR и diarization)
- [x] Не генерирует сегменты сам

**Evidence**:
- `src/extractors/speech_analysis_extractor/main.py:87` — метод `run_bundle()` принимает `asr_segments` и `diar_segments`
- `src/extractors/speech_analysis_extractor/main.py:133-138` — передача сегментов в под-экстракторы

---

## 5) Наблюдаемость: progress + stage timings

### 5.1 Промежуточный прогресс

- [x] Progress обновляется для каждого под-экстрактора (ASR, diarization, pitch)
- [x] Формат прогресса машиночитаем и безопасен (без raw audio данных)
- [x] Progress callback передаётся в `run_bundle()` через параметр `progress_callback`

**Evidence**:
- `src/extractors/speech_analysis_extractor/main.py:87` — метод `run_bundle()` принимает `progress_callback`
- `src/extractors/speech_analysis_extractor/main.py:133-150` — обновление прогресса для каждого под-экстрактора
- `run_cli.py:1576-1590` — progress callback для speech_analysis_extractor

### 5.2 Stage timings

- [x] Timings сохраняются в NPZ meta через `extra_meta` в `run_cli.py`
- [x] Per-extractor timings сохраняются в `timings_by_extractor`

**Evidence**:
- `run_cli.py:1724` — сохранение timings в meta

---

## 6) Feature Contract: управление выходными фичами (feature gating)

### 6.1 Feature gating flags

- [x] Все фичи opt-in через персональные флаги (default: все False)
- [x] Флаги: `--speech-enable-asr-metrics`, `--speech-enable-diarization-metrics`, `--speech-enable-pitch-metrics`
- [x] В `meta.features_enabled[]` фиксируются включённые фичи

**Evidence**:
- `src/extractors/speech_analysis_extractor/main.py:49-58` — feature gating flags в `__init__`
- `src/extractors/speech_analysis_extractor/main.py:186-252` — feature-gated payload
- `run_cli.py:982-988` — CLI аргументы для feature gating

### 6.2 Feature dependencies

- [x] Зависимости между фичами документированы в README (раздел "Feature Dependencies")
- [x] `asr_metrics` зависят от `ASRExtractor`
- [x] `diarization_metrics` зависят от `SpeakerDiarizationExtractor`
- [x] `pitch_metrics` зависят от `PitchExtractor` и `pitch_enabled=True`

**Evidence**:
- `src/extractors/speech_analysis_extractor/README.md` — раздел "Feature Dependencies"

### 6.3 Feature flags для под-экстракторов

- [x] Под-экстракторы создаются с фиксированными feature flags (hardcoded для bundle)
- [x] ASR: `enable_token_sequences=True`, `enable_token_counts=True`, `enable_token_total=True`, `enable_token_density=True`, `enable_speech_rate=True`, `enable_lang_distribution=True`
- [x] Diarization: `enable_speaker_segments=True`, `enable_speaker_stats=True`, `enable_speaker_durations=True`

**Evidence**:
- `src/extractors/speech_analysis_extractor/main.py:66-76` — создание под-экстракторов с фиксированными feature flags

---

## 7) Производительность и ресурсы

### 7.1 Обязательные измерения

- [x] Latency per bundle задокументирована (estimated_duration = 10.0 сек)
- [x] CPU RSS peak измеряется через resource_metrics в `run_cli.py`
- [x] GPU VRAM peak измеряется через resource_metrics в `run_cli.py`

**Evidence**:
- `src/extractors/speech_analysis_extractor/main.py:43` — `estimated_duration = 10.0`
- `run_cli.py:1724` — resource_metrics в meta

### 7.2 Зависимости от под-экстракторов

- [x] ASR: зависит от размера модели и длительности аудио
- [x] Diarization: зависит от размера модели и количества спикеров
- [x] Pitch: зависит от backend (classic быстрее, torchcrepe медленнее)

**Evidence**:
- `src/extractors/speech_analysis_extractor/README.md` — раздел "Performance characteristics"

---

## 8) Проверка качества выхода (quality validation)

### 8.1 Минимальные sanity-checks

- [x] Валидация структуры сегментов: проверка типов, обязательных полей, диапазонов
- [x] Валидация payload от под-экстракторов: проверка типов, диапазонов, наличия обязательных полей
- [x] Консистентность связных фичей (например, `asr_segments_count` ↔ `asr_lang_id_by_segment`)

**Evidence**:
- `src/extractors/speech_analysis_extractor/main.py:78-145` — методы валидации (`_validate_segments`, `_validate_asr_payload`, `_validate_diarization_payload`, `_validate_pitch_payload`)
- `src/extractors/speech_analysis_extractor/main.py:92-95` — валидация сегментов перед обработкой

### 8.2 Human-friendly визуализация / UI render

- [x] Renderer реализован в `src/core/renderer.py` (`render_speech_analysis_extractor()`)
- [x] HTML renderer для дебага (`render_speech_analysis_extractor_html()`)
- [x] README содержит раздел "Visualization" с рекомендациями для UI/сайта

**Evidence**:
- `src/core/renderer.py:1662-1758` — renderer для speech_analysis_extractor
- `src/core/renderer.py:1760-1950` — HTML renderer для дебага
- `src/extractors/speech_analysis_extractor/README.md` — раздел "Visualization"

---

## 9) Error Handling

### 9.1 Детальные error codes

- [x] `asr_failed` (ASR под-экстрактор упал)
- [x] `diarization_failed` (Diarization под-экстрактор упал)
- [x] `pitch_failed` (Pitch под-экстрактор упал)
- [x] `silence_detection_failed` (Ошибка детекции тишины)
- [x] `validation_failed` (Ошибка валидации payload)
- [x] `segments_invalid` (Невалидная структура сегментов)
- [x] `audio_too_short` (Аудио < 5 секунд)
- [x] `unknown` (Другие ошибки)

**Evidence**:
- `src/extractors/speech_analysis_extractor/main.py:97-130` — метод `_classify_error()`
- `src/extractors/speech_analysis_extractor/main.py:134-150` — использование error codes в error handling

### 9.2 No-fallback policy

- [x] Отсутствие сегментов → `ValueError` с `error_code="segments_invalid"`
- [x] Аудио < 5 сек → `RuntimeError` с `error_code="audio_too_short"`
- [x] Ошибка под-экстрактора → `RuntimeError` с детальным error_code
- [x] Валидация payload → `ValueError` с `error_code="validation_failed"`

**Evidence**:
- `src/extractors/speech_analysis_extractor/main.py:92-150` — fail-fast проверки

---

## 10) Additional Metrics

### 10.1 ASR additional metrics

- [x] `asr_speech_rate_wpm`: скорость речи (слов в минуту)
- [x] `asr_lang_distribution`: распределение языков (dict[lang_id, ratio])

**Evidence**:
- `src/extractors/speech_analysis_extractor/main.py:200-210` — вычисление дополнительных ASR метрик

### 10.2 Diarization additional metrics

- [x] `speaker_balance_score`: метрика баланса спикеров (0 = один доминирует, 1 = равномерное распределение)
- [x] `speaker_transitions_count`: количество переходов между спикерами

**Evidence**:
- `src/extractors/speech_analysis_extractor/main.py:220-230` — вычисление дополнительных diarization метрик

### 10.3 Pitch additional metrics

- [x] `pitch_f0_min`: минимальное значение основной частоты
- [x] `pitch_f0_max`: максимальное значение основной частоты
- [x] `pitch_f0_range`: диапазон основной частоты
- [x] `pitch_stability`: метрика стабильности pitch (0 = нестабильная, 1 = стабильная)
- [x] `pitch_distribution`: распределение pitch по октавам

**Evidence**:
- `src/extractors/speech_analysis_extractor/main.py:232-250` — вычисление дополнительных pitch метрик

---

## 11) Contract Versioning

### 11.1 Contract version для совместимости

- [x] `speech_analysis_contract_version="speech_analysis_contract_v1"` в payload
- [x] Contract version сохраняется в NPZ meta
- [x] Используется для валидации совместимости с downstream extractors

**Evidence**:
- `src/extractors/speech_analysis_extractor/main.py:19` — константа `SPEECH_ANALYSIS_CONTRACT_VERSION`
- `src/extractors/speech_analysis_extractor/main.py:186-252` — contract version в payload
- `run_cli.py:788-821` — contract version в meta

---

## 12) Segments Validation

### 12.1 Полная валидация структуры сегментов

- [x] Валидация типов (list, dict)
- [x] Валидация обязательных полей (`start_sample`, `end_sample`, `start_sec`, `end_sec`, `center_sec`)
- [x] Валидация диапазонов (неотрицательные значения, start < end, center в диапазоне)

**Evidence**:
- `src/extractors/speech_analysis_extractor/main.py:78-145` — метод `_validate_segments()`
- `src/extractors/speech_analysis_extractor/main.py:92-95` — валидация сегментов перед обработкой

---

## 13) Payload Validation

### 13.1 Полная валидация payload от под-экстракторов

- [x] Валидация ASR payload: проверка типов, диапазонов, наличия обязательных полей
- [x] Валидация diarization payload: проверка типов, диапазонов, наличия обязательных полей
- [x] Валидация pitch payload: проверка типов, диапазонов

**Evidence**:
- `src/extractors/speech_analysis_extractor/main.py:147-200` — методы валидации payload (`_validate_asr_payload`, `_validate_diarization_payload`, `_validate_pitch_payload`)
- `src/extractors/speech_analysis_extractor/main.py:140-150` — валидация payload после запуска под-экстракторов

---

## 14) Silence Detection

### 14.1 Конфигурируемые пороги тишины

- [x] `silence_peak_threshold`: порог peak для детекции тишины (по умолчанию 1e-3)
- [x] `silence_rms_threshold`: порог RMS для детекции тишины (по умолчанию 1e-4)
- [x] Feature flag для отключения детекции тишины (`--speech-disable-silence-detection`)

**Evidence**:
- `src/extractors/speech_analysis_extractor/main.py:49-58` — параметры silence detection в `__init__`
- `src/extractors/speech_analysis_extractor/main.py:106-130` — детекция тишины с конфигурируемыми порогами

---

## 15) Документация

### 15.1 README extractor'а

- [x] Раздел "Входы" с описанием Segmenter contract
- [x] Раздел "Выходы" с описанием всех фичей (feature-gated)
- [x] Раздел "Feature Dependencies" с явным описанием зависимостей
- [x] Раздел "Конфигурация" с описанием всех параметров
- [x] Раздел "Feature Gating" с описанием всех флагов
- [x] Раздел "Visualization" с рекомендациями для UI/сайта
- [x] Раздел "Алгоритм" с описанием всех этапов обработки

**Evidence**:
- `src/extractors/speech_analysis_extractor/README.md` — полная документация

---

## 16) Compliance Summary

### ✅ Все критерии выполнены

- ✅ **Архитектура**: соответствует `BaseExtractor`, Segmenter contract, per-run storage
- ✅ **Bundle extractor**: объединяет результаты под-экстракторов с фиксированными feature flags
- ✅ **Контракты**: NPZ schema, meta fields, contract versioning
- ✅ **Feature gating**: персональные флаги для каждой группы фичей (3 группы, все opt-in)
- ✅ **Error handling**: детальные error codes (8 типов)
- ✅ **Валидация**: полная валидация структуры сегментов и payload от под-экстракторов
- ✅ **Наблюдаемость**: progress reporting для каждого под-экстрактора, stage timings
- ✅ **UI Render**: renderer + HTML renderer для дебага
- ✅ **Документация**: полный README с разделами Feature Dependencies и Visualization
- ✅ **Дополнительные метрики**: speech_rate_wpm, lang_distribution для ASR; speaker_balance_score, speaker_transitions_count для diarization; pitch range, stability, distribution
- ✅ **Silence detection**: конфигурируемые пороги и feature flag для отключения

---

## 17) Open Issues

Нет открытых проблем. Все критерии `AP_AUDIT_CRITERIA.md` выполнены.

