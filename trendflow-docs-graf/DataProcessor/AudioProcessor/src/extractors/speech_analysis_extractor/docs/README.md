## `speech_analysis_extractor` (Speech analysis bundle)

### Назначение

Предоставляет компактный "обзор речи" путем комбинирования результатов нескольких экстракторов:
- **ASR (Automatic Speech Recognition)**: токены распознавания речи через inprocess Whisper модель (ModelManager)
- **Speaker Diarization**: определение спикеров через inprocess pyannote.audio + whisperx (ModelManager)
- **Pitch** (опционально): анализ высоты тона через сигнальную обработку

**Важно**: 
- Не сохраняет сырой текст транскрипции
- Не выполняет выравнивание между ASR токенами и спикерами (временные метки Whisper недоступны)
- **Новая архитектура**: компонент использует существующие результаты зависимых компонентов (`asr`, `speaker_diarization`, `pitch`) из `extractor_results`, а не запускает под-экстракторы внутри себя
- Если зависимые компоненты не включены, компонент выдаст ошибку (если зависимости обязательны) или предупреждение (если зависимости опциональны)

**Версия**: 2.1.1  
**Категория**: speech  
**GPU**: опционально (предпочтительно для inprocess ASR/diarization моделей)  
**audit_v3_status**: `in_progress` (schema `speech_analysis_extractor_npz_v1`, см. `docs/audit_v3/components/speech_analysis_extractor_AUDIT_V3_REPORT.md`)

### Входы

- **`audio/audio.wav`** (любой аудио файл, поддерживаемый AudioUtils)
- **`asr_segments`**: список сегментов для ASR анализа (от Segmenter families.asr)
- **`diar_segments`**: список сегментов для diarization (от Segmenter families.diarization)
- **`asr_result`** (опционально): результат от `asr_extractor` (словарь из `extractor_results`, обязателен если `enable_asr_metrics=True`)
- **`diarization_result`** (опционально): результат от `speaker_diarization_extractor` (словарь из `extractor_results`, обязателен если `enable_diarization_metrics=True`)
- **`pitch_result`** (опционально): результат от `pitch_extractor` (словарь из `extractor_results`, используется если `pitch_enabled=True` и `enable_pitch_metrics=True`)

**Требования**:
- Минимальная длительность аудио: **5 секунд** (иначе ошибка)
- Сегменты должны быть предоставлены через `run_bundle()` (метод `run()` не поддерживается)
- Сегменты должны иметь обязательные поля: `start_sample`, `end_sample`, `start_sec`, `end_sec`, `center_sec`
- **Зависимости**: если `enable_asr_metrics=True`, то `asr` extractor должен быть включен и выполнен до `speech_analysis`
- **Зависимости**: если `enable_diarization_metrics=True`, то `speaker_diarization` extractor должен быть включен и выполнен до `speech_analysis`

### Выходы

Схема NPZ: `speech_analysis_extractor_npz_v1` — `docs/SCHEMA.md`, `schemas/speech_analysis_extractor_npz_v1.json`.

#### Audit v4 — заметки по NPZ

- На reference **A** (старый артефакт): **F=15**, **6 NaN** в pitch-полях при **`pitch_enabled=0`** и **`meta.features_enabled`** содержал **`pitch_metrics`** — несогласованность: савер добавлял pitch-колонки из флага, хотя payload их не заполнял. **Исправлено в `main.py`:** `pitch_metrics` в **`_features_enabled`** только если **`pitch_payload is not None`** (после перезапуска pitch-колонки отсутствуют, если pitch не мержился; **F=9** для только ASR на этом профиле).
- **`device_used`** в payload и **`meta`**, не в tabular.
- Observability (audit v4.2): `meta.stage_timings_ms`, `meta.speech_analysis_resource_profile` (env: `AP_SPEECH_ANALYSIS_RESOURCE_PROFILE=1`)

Экстрактор возвращает `ExtractorResult` с payload, содержащим:

#### Обязательные поля (всегда присутствуют)

- `duration_sec`: длительность аудио (секунды)
- `sample_rate`: частота дискретизации (Hz)
- `device_used`: устройство обработки (обычно `"cuda"`)
- `speech_analysis_contract_version`: версия контракта для валидации совместимости (str, `"speech_analysis_contract_v1"`)

#### Feature-gated поля (включаются через флаги)

**`--speech-enable-asr-metrics`**:
- `asr_segments_count`: количество ASR сегментов (int)
- `asr_token_total`: общее количество токенов (float)
- `asr_token_mean`: среднее количество токенов на сегмент (float)
- `asr_token_std`: стандартное отклонение количества токенов (float)
- `asr_token_density_per_sec`: плотность токенов (токенов/секунду) (float)
- `asr_speech_rate_wpm`: скорость речи (слов в минуту) (float)
- `asr_lang_distribution`: распределение языков (dict[lang_id, ratio])
- `asr_lang_id_by_segment`: идентификаторы языка для каждого сегмента (list[int])

**`--speech-enable-diarization-metrics`**:
- `diar_segments_count`: количество сегментов diarization (int)
- `speaker_count`: количество уникальных спикеров (int)
- `dominant_speaker_share`: доля доминирующего спикера (0.0-1.0) (float)
- `speaker_balance_score`: метрика баланса спикеров (0 = один доминирует, 1 = равномерное распределение) (float)
- `speaker_transitions_count`: количество переходов между спикерами (int)
- `speaker_ids`: список идентификаторов спикеров (list[int])

**`--speech-enable-pitch-metrics`** (требует `--speech-analysis-pitch`):
- `pitch_enabled`: включен ли анализ высоты тона (boolean)
- `pitch_f0_mean`: среднее значение основной частоты (Hz) (float)
- `pitch_f0_std`: стандартное отклонение основной частоты (Hz) (float)
- `pitch_f0_min`: минимальное значение основной частоты (Hz) (float)
- `pitch_f0_max`: максимальное значение основной частоты (Hz) (float)
- `pitch_f0_range`: диапазон основной частоты (Hz) (float)
- `pitch_stability`: метрика стабильности pitch (0 = нестабильная, 1 = стабильная) (float)
- `pitch_distribution`: распределение pitch по октавам (dict[octave_id, ratio])

#### Специальные случаи

**Пустое аудио** (status="empty"):
- `status`: `"empty"`
- `empty_reason`: `"audio_missing_or_extract_failed"` (тихое аудио, если silence detection включен)
- `empty_reason`: `"audio_too_short"` (аудио < 5 сек, Audit v3)
- Остальные поля присутствуют (без метрик)

### Feature Dependencies

**Зависимости между фичами**:
- `asr_metrics` требуют результата от `asr_extractor` (обязательная зависимость)
- `diarization_metrics` требуют результата от `speaker_diarization_extractor` (обязательная зависимость)
- `pitch_metrics` могут использовать результат от `pitch_extractor` (опциональная зависимость, fallback на внутренний запуск)

**Зависимости от других extractors** (новая архитектура):
- **`asr_extractor`**: обязательная зависимость, если `enable_asr_metrics=True`. Результат передается через `asr_result` из `extractor_results`.
- **`speaker_diarization_extractor`**: обязательная зависимость, если `enable_diarization_metrics=True`. Результат передается через `diarization_result` из `extractor_results`.
- **`pitch_extractor`**: опциональная зависимость, если `pitch_enabled=True` и `enable_pitch_metrics=True`. Результат передается через `pitch_result` из `extractor_results`. Если результат не предоставлен, используется fallback на внутренний запуск `PitchExtractor`.

**Contract version для совместимости**:
- `speech_analysis_contract_version="speech_analysis_contract_v1"` используется для валидации совместимости с downstream extractors

**Проверка зависимостей на уровне оркестратора**:
- Если `speech_analysis` включен, но `asr` или `speaker_diarization` не включены, оркестратор выдаст ошибку (если `strict_mode=True`) или предупреждение (если `strict_mode=False`)
- Если `auto_add_dependencies=True`, зависимости будут автоматически добавлены в список extractors

### Алгоритм

#### 1. Валидация входных данных

1. Валидация URI (проверка расширения файла)
2. Полная валидация структуры сегментов:
   - Проверка типов (list, dict)
   - Проверка обязательных полей (`start_sample`, `end_sample`, `start_sec`, `end_sec`, `center_sec`)
   - Проверка диапазонов (неотрицательные значения, start < end, center в диапазоне)
3. Проверка длительности: аудио должно быть ≥ 5 секунд

#### 2. Детекция тишины (если включена)

1. Загрузка первого diarization сегмента
2. Вычисление RMS и peak амплитуды
3. Если `peak < silence_peak_threshold` и `rms < silence_rms_threshold` → возвращается `status="empty"`

#### 3. Получение результатов от зависимых компонентов (новая архитектура)

**ASR** (если `enable_asr_metrics=True`):
- Получение результата от `asr_extractor` через `asr_result` из `extractor_results`
- Если `asr_result` не предоставлен, выдается ошибка: `"asr_result is required when enable_asr_metrics=True, but was not provided. Ensure 'asr' extractor is enabled in config."`
- Проверка успешности результата: если `asr_result.success == False`, выдается ошибка
- Извлечение payload: `asr_payload = asr_result.get("payload") or {}`
- Валидация payload от ASR extractor
- Детальное профилирование: время обработки ASR результата (`asr_sec`)
- Progress reporting: обновление прогресса с сообщением "ASR result loaded from dependency"

**Diarization** (если `enable_diarization_metrics=True`):
- Получение результата от `speaker_diarization_extractor` через `diarization_result` из `extractor_results`
- Если `diarization_result` не предоставлен, выдается ошибка: `"diarization_result is required when enable_diarization_metrics=True, but was not provided. Ensure 'speaker_diarization' extractor is enabled in config."`
- Проверка успешности результата: если `diarization_result.success == False`, выдается ошибка
- Извлечение payload: `diar_payload = diarization_result.get("payload") or {}`
- Валидация payload от diarization extractor
- Детальное профилирование: время обработки diarization результата (`diarization_sec`)
- Progress reporting: обновление прогресса с сообщением "Diarization result loaded from dependency"

**Pitch** (если `pitch_enabled=True` и `enable_pitch_metrics=True`):
- Предпочтительно: получение результата от `pitch_extractor` через `pitch_result` из `extractor_results`
- Если `pitch_result` предоставлен и успешен, используется он (progress: "Pitch result loaded from dependency")
- Fallback: если `pitch_result` не предоставлен или неуспешен, запускается `PitchExtractor.run()` внутри `speech_analysis` (progress: "Starting pitch extraction (fallback)")
- Валидация payload от pitch extractor
- Детальное профилирование: время обработки pitch (`pitch_sec`)

#### 4. Агрегация результатов (feature-gated)

Детальное профилирование: время выполнения агрегации (`aggregates_sec`)

**ASR статистики** (если `enable_asr_metrics=True`):
- `token_total`: сумма всех токенов по сегментам
- `token_mean`: среднее количество токенов на сегмент
- `token_std`: стандартное отклонение
- `token_density`: `token_total / duration_sec`
- `speech_rate_wpm`: извлекается из ASR payload
- `lang_distribution`: извлекается из ASR payload

**Diarization статистики** (если `enable_diarization_metrics=True`):
- `speaker_count`: количество уникальных спикеров
- `dominant_share`: доля самого длинного спикера от общей длительности речи
- `speaker_balance_score`: извлекается из diarization payload
- `speaker_transitions_count`: извлекается из diarization payload

**Pitch статистики** (если `enable_pitch_metrics=True` и pitch включен):
- `f0_mean`, `f0_std`: извлекаются из pitch payload
- `f0_min`, `f0_max`: извлекаются из pitch payload (если доступны)
- `f0_range`: вычисляется как `f0_max - f0_min`
- `pitch_stability`: вычисляется как `1 / (1 + cv)`, где `cv = f0_std / f0_mean` (coefficient of variation)
- `pitch_distribution`: вычисляется как распределение по октавам (50-100, 100-200, 200-400, 400-800, 800-1600 Hz)

#### 5. Формирование payload

1. Обязательные поля (duration, sample_rate, device_used, contract_version)
2. Feature-gated поля (ASR, diarization, pitch метрики)
3. Track enabled features для meta (`_features_enabled`)

### Конфигурация

**Параметры конфигурации компонента**:

| Параметр | Тип | Значение по умолчанию | Допустимые значения | Описание | Δ latency | Δ cost |
|----------|-----|----------------------|---------------------|----------|-----------|--------|
| `device` | str | `"auto"` | `"auto"` \| `"cpu"` \| `"cuda"` | Устройство для обработки | 0 ms | 0 ₽ |
| `sample_rate` | int | `16000` | `> 0` | Частота дискретизации | +0.1 ms/frame при увеличении на 1kHz | +0.01 ₽/frame |
| `pitch_enabled` | bool | `false` | `true` \| `false` | Включить анализ высоты тона | +50-200 ms (зависит от backend) | +0.05-0.2 ₽ |
| `pitch_backend` | str | `"classic"` | `"classic"` \| `"torchcrepe"` | Backend для pitch | +50 ms для classic, +200 ms для torchcrepe | +0.05 ₽ для classic, +0.2 ₽ для torchcrepe |
| `silence_peak_threshold` | float | `1e-3` | `> 0` | Порог peak для детекции тишины | 0 ms | 0 ₽ |
| `silence_rms_threshold` | float | `1e-4` | `> 0` | Порог RMS для детекции тишины | 0 ms | 0 ₽ |
| `enable_silence_detection` | bool | `true` | `true` \| `false` | Включить проверку на тишину | +10-50 ms | +0.01 ₽ |
| `enable_asr_metrics` | bool | `false` | `true` \| `false` | Включить ASR метрики (требует asr_extractor) | +10-100 ms (только агрегация) | +0.01 ₽ |
| `enable_diarization_metrics` | bool | `false` | `true` \| `false` | Включить diarization метрики (требует speaker_diarization_extractor) | +10-100 ms (только агрегация) | +0.01 ₽ |
| `enable_pitch_metrics` | bool | `false` | `true` \| `false` | Включить pitch метрики (требует pitch_enabled=true) | +10-50 ms (только агрегация) | +0.01 ₽ |

**Примечание**: Размеры моделей ASR и diarization определяются самими зависимыми компонентами (`asr_extractor`, `speaker_diarization_extractor`), а не этим компонентом. Компонент использует результаты от зависимых компонентов из `extractor_results`. Время обработки зависимых компонентов не включено в Δ latency (указано только время агрегации результатов).

**Источник оценки**: бенчмарк/профилирование на типичных аудио файлах (30-120 секунд)

**Пример конфигурации (минимум)**:
```python
{
    "device": "auto",
    "sample_rate": 16000,
    "enable_silence_detection": true,
}
```

**Пример конфигурации (расширенный)**:
```python
{
    "device": "auto",
    "sample_rate": 16000,
    "pitch_enabled": true,
    "pitch_backend": "classic",
    "silence_peak_threshold": 1e-3,
    "silence_rms_threshold": 1e-4,
    "enable_silence_detection": true,
    "enable_asr_metrics": true,
    "enable_diarization_metrics": true,
    "enable_pitch_metrics": true,
}
```

### Features contract

**Features contract**: компонент имеет явный механизм выбора выходных фич через аргументы/конфиг (feature flags).

**Дефолтный набор фич**: все фичи выключены по умолчанию (opt-in подход). Для включения фич используйте соответствующие флаги.

**Перечень всех возможных фич**:

#### ASR Metrics (`enable_asr_metrics=True`)
- `asr_token_total`: сумма всех токенов по сегментам (int)
  - Формат: per-run
  - Единица: количество токенов
  - Влияние на стоимость: +0.01 ₽ (только агрегация)
- `asr_token_mean`: среднее количество токенов на сегмент (float)
  - Формат: per-run
  - Единица: токенов/сегмент
  - Влияние на стоимость: +0.01 ₽ (только агрегация)
- `asr_token_std`: стандартное отклонение количества токенов (float)
  - Формат: per-run
  - Единица: токенов/сегмент
  - Влияние на стоимость: +0.01 ₽ (только агрегация)
- `asr_token_density_per_sec`: плотность токенов (токенов в секунду) (float)
  - Формат: per-run
  - Единица: токенов/сек
  - Влияние на стоимость: +0.01 ₽ (только агрегация)
- `asr_speech_rate_wpm`: скорость речи (слов в минуту) (float)
  - Формат: per-run
  - Единица: слов/минуту
  - Влияние на стоимость: +0.01 ₽ (только агрегация)
- `asr_lang_distribution`: распределение языков по сегментам (dict[str, float])
  - Формат: per-run
  - Единица: доля (0.0-1.0)
  - Влияние на стоимость: +0.01 ₽ (только агрегация)
- `asr_segments_count`: количество обработанных ASR сегментов (int)
  - Формат: per-run
  - Единица: количество
  - Влияние на стоимость: +0.01 ₽ (только агрегация)

**Зависимости**: требует результат от `asr_extractor` (обязательная зависимость, fail-fast если не предоставлен)

#### Diarization Metrics (`enable_diarization_metrics=True`)
- `diar_speaker_count`: количество уникальных спикеров (int)
  - Формат: per-run
  - Единица: количество
  - Влияние на стоимость: +0.01 ₽ (только агрегация)
- `diar_dominant_speaker_share`: доля самого длинного спикера от общей длительности речи (float)
  - Формат: per-run
  - Единица: доля (0.0-1.0)
  - Влияние на стоимость: +0.01 ₽ (только агрегация)
- `diar_speaker_balance_score`: баланс распределения спикеров (float, 0.0-1.0)
  - Формат: per-run
  - Единица: score (0.0 = один доминирует, 1.0 = равномерное распределение)
  - Влияние на стоимость: +0.01 ₽ (только агрегация)
- `diar_speaker_transitions_count`: количество переходов между спикерами (int)
  - Формат: per-run
  - Единица: количество
  - Влияние на стоимость: +0.01 ₽ (только агрегация)
- `diar_segments_count`: количество обработанных diarization сегментов (int)
  - Формат: per-run
  - Единица: количество
  - Влияние на стоимость: +0.01 ₽ (только агрегация)

**Зависимости**: требует результат от `speaker_diarization_extractor` (обязательная зависимость, fail-fast если не предоставлен)

#### Pitch Metrics (`enable_pitch_metrics=True`, требует `pitch_enabled=true`)
- `pitch_f0_mean`: средняя частота основного тона (float)
  - Формат: per-run
  - Единица: Hz
  - Влияние на стоимость: +0.01 ₽ (только агрегация)
- `pitch_f0_std`: стандартное отклонение частоты основного тона (float)
  - Формат: per-run
  - Единица: Hz
  - Влияние на стоимость: +0.01 ₽ (только агрегация)
- `pitch_f0_min`: минимальная частота основного тона (float)
  - Формат: per-run
  - Единица: Hz
  - Влияние на стоимость: +0.01 ₽ (только агрегация)
- `pitch_f0_max`: максимальная частота основного тона (float)
  - Формат: per-run
  - Единица: Hz
  - Влияние на стоимость: +0.01 ₽ (только агрегация)
- `pitch_f0_range`: диапазон частоты основного тона (float)
  - Формат: per-run
  - Единица: Hz
  - Влияние на стоимость: +0.01 ₽ (только агрегация)
- `pitch_stability`: стабильность высоты тона (float, 0.0-1.0)
  - Формат: per-run
  - Единица: score (0.0 = нестабильная, 1.0 = стабильная)
  - Влияние на стоимость: +0.01 ₽ (только агрегация)
- `pitch_distribution`: распределение по октавам (dict[str, float])
  - Формат: per-run
  - Единица: доля (0.0-1.0)
  - Влияние на стоимость: +0.01 ₽ (только агрегация)

**Зависимости**: предпочтительно результат от `pitch_extractor` (опциональная зависимость, fallback на внутренний запуск если не предоставлен)

#### Feature Gating (персональные флаги)

**По умолчанию в коде**: все фичи отключены (opt-in)  
**По умолчанию в конфиге** (`global_config.yaml`): все фичи отключены (`false`)

- `--speech-enable-asr-metrics` / `enable_asr_metrics`: включить ASR метрики
- `--speech-enable-diarization-metrics` / `enable_diarization_metrics`: включить diarization метрики
- `--speech-enable-pitch-metrics` / `enable_pitch_metrics`: включить pitch метрики (требует `pitch_enabled=true`)
- `--speech-disable-silence-detection` / `disable_silence_detection`: отключить проверку на тишину

**Рекомендации для обучения моделей**:
- Включить все фичи для максимального качества и полноты данных
- Минимальный набор: `enable_asr_metrics` + `enable_diarization_metrics` (для базового анализа речи)

**В meta артефакта фиксируется**: `features_enabled[]` - список включенных фич (для отслеживания того, какие фичи были включены/выданы)

### Архитектура

1. **Инициализация**:
   - Инициализация AudioUtils
   - Настройка параметров тишины
   - Создание `PitchExtractor` (используется только для fallback, если `pitch_result` не предоставлен)
   - **Примечание**: `ASRExtractor` и `SpeakerDiarizationExtractor` больше не создаются внутри компонента

2. **Обработка сегментов** (`run_bundle()`):
   - Валидация входных данных (URI, структура сегментов)
   - Проверка длительности (≥ 5 сек)
   - Детекция тишины (если включена) с профилированием (`silence_detection_sec`)
   - **Получение результатов от зависимых компонентов** (новая архитектура):
     - ASR: получение `asr_result` из `extractor_results`, проверка успешности, извлечение payload
     - Diarization: получение `diarization_result` из `extractor_results`, проверка успешности, извлечение payload
     - Pitch: получение `pitch_result` из `extractor_results` (опционально), fallback на внутренний запуск если не предоставлен
   - Валидация payload от зависимых компонентов
   - Агрегация результатов (feature-gated) с профилированием (`aggregates_sec`)
   - Формирование payload
   - Детальное логирование профилирования всех этапов

3. **Вспомогательные методы**:
   - `_validate_segments()`: полная валидация структуры сегментов
   - `_validate_asr_payload()`: полная валидация payload от ASR extractor
   - `_validate_diarization_payload()`: полная валидация payload от diarization extractor
   - `_validate_pitch_payload()`: полная валидация payload от pitch extractor
   - `_classify_error()`: классификация ошибок с детальными error codes

4. **Обработка ошибок**:
   - ASR failed → `RuntimeError` с `error_code="asr_failed"`
   - Diarization failed → `RuntimeError` с `error_code="diarization_failed"`
   - Pitch failed → `RuntimeError` с `error_code="pitch_failed"`
   - Silence detection failed → `RuntimeError` с `error_code="silence_detection_failed"`
   - Validation failed → `ValueError` с `error_code="validation_failed"`
   - Segments invalid → `ValueError` с `error_code="segments_invalid"`
   - Audio too short (<5s) → `status="empty"`, `empty_reason="audio_too_short"` (Audit v3)

### Обработка ошибок

**Политика NO FALLBACK**:
- Отсутствие сегментов → ошибка
- Аудио < 5 секунд → valid empty (`status="empty"`, `empty_reason="audio_too_short"`, Audit v3)
- Ошибка под-экстрактора → ошибка (fail-fast)
- Валидация payload → ошибка

**Специальные случаи**:
- **Тихое аудио**: возвращается `status="empty"`, `empty_reason="audio_missing_or_extract_failed"` (если silence detection включен)
- **Аудио < 5 сек**: возвращается `status="empty"`, `empty_reason="audio_too_short"`
- **Несоответствие структуры сегментов**: ошибка с описанием

### Особенности

- **Bundle экстрактор**: объединяет результаты нескольких специализированных экстракторов
- **Сегментная обработка**: работает на предварительно сегментированных окнах
- **Компактный вывод**: хранит только сводные статистики, не сырые данные
- **Опциональный pitch**: может быть включен/выключен
- **Детекция тишины**: автоматически определяет полностью тихое аудио (конфигурируемые пороги)
- **No-fallback политика**: требует наличия всех сегментов
- **Новая архитектура зависимостей**: компонент использует существующие результаты зависимых компонентов (`asr`, `speaker_diarization`, `pitch`) из `extractor_results`, а не запускает под-экстракторы внутри себя
  - Если `enable_asr_metrics=True`, требуется результат от `asr_extractor` (обязательная зависимость)
  - Если `enable_diarization_metrics=True`, требуется результат от `speaker_diarization_extractor` (обязательная зависимость)
  - Если `pitch_enabled=True` и `enable_pitch_metrics=True`, предпочтительно использовать результат от `pitch_extractor` (опциональная зависимость, fallback на внутренний запуск)
- **Проверка зависимостей на уровне оркестратора**: если обязательные зависимости не включены, оркестратор выдаст ошибку или предупреждение
- **Feature gating**: все фичи opt-in через персональные флаги (в конфиге по умолчанию все отключены)
- **Contract versioning**: версия контракта для валидации совместимости с downstream extractors
- **Полная валидация**: проверка структуры сегментов и payload от зависимых компонентов
- **Progress reporting**: обновление прогресса с детальными сообщениями и временем выполнения
- **Детальное профилирование**: логирование времени выполнения для каждого этапа (silence, ASR, diarization, pitch, aggregates)
- **Детальные error codes**: классификация ошибок для лучшей диагностики

### Sampling / units-of-processing requirements

**Важно**: `speech_analysis_extractor` **не генерирует сегменты сам** — Segmenter является единственным владельцем sampling.

**Требования к сегментам**:
- Компонент использует два семейства сегментов из `audio/segments.json`:
  - **`families.asr.segments[]`**: длинные sliding windows для ASR анализа (обязательно)
  - **`families.diarization.segments[]`**: фиксированные окна для diarization (обязательно)
- Сегменты должны иметь обязательные поля: `start_sample`, `end_sample`, `start_sec`, `end_sec`, `center_sec`
- Отсутствие обязательных семейств → fail-fast (`raise RuntimeError`)

**Sampling policy (Segmenter contract)**:
- Segmenter строит families по **универсальной нелинейной кривой** (sampling curve):
  - Параметры в `families.<name>.sampling_curve`: `type="ease_out_power"`, `k∈(0,1]`, `linear_until_sec`, `cap_duration_sec`
  - На коротких видео можно близко к 1:1 (секунда→окно), на длинных рост замедляется и упирается в `max_windows`
- См. `docs/contracts/SEGMENTER_CONTRACT.md` для деталей sampling policy

**Минимальные требования**:
- Минимальная длительность аудио: **5 секунд** (иначе `status="empty"`, `empty_reason="audio_too_short"`)
- Минимальное количество сегментов: **1 сегмент** для каждого семейства (иначе ошибка `segments_invalid`)

### Models

**Важно**: `speech_analysis_extractor` не использует модели напрямую, а агрегирует результаты от зависимых компонентов.

**Модели, используемые зависимыми компонентами**:

#### ASR (если `enable_asr_metrics=True`)
- **Whisper** (через `asr_extractor`):
  - **Spec name**: `whisper_{size}_inprocess` (ModelManager), где `size` = `small`/`medium`/`large`
  - **Runtime**: `inprocess` (PyTorch)
  - **Engine**: `torch`
  - **Precision**: `fp16` (на CUDA) или `fp32` (на CPU)
  - **Device**: `cuda` (предпочтительно) или `cpu`
  - **Triton**: ❌ Нет (in-process)
  - **Загрузка**: через `dp_models` (ModelManager), no-network policy

#### Speaker Diarization (если `enable_diarization_metrics=True`)
- **Speaker Diarization** (через `speaker_diarization_extractor`):
  - **Spec name**: `speaker_diarization_{size}_inprocess` (ModelManager), где `size` = `small`/`large`
  - **Runtime**: `inprocess` (PyTorch)
  - **Engine**: `torch`
  - **Precision**: `fp16` (на CUDA) или `fp32` (на CPU)
  - **Device**: `cuda` (предпочтительно) или `cpu`
  - **Triton**: ❌ Нет (in-process)
  - **Загрузка**: через `dp_models` (ModelManager), no-network policy

#### Pitch (если `pitch_enabled=True` и `enable_pitch_metrics=True`)
- **Pitch detection** (через `pitch_extractor` или fallback):
  - **Backend**: `classic` (signal processing) или `torchcrepe` (ML-модель)
  - **Runtime**: `inprocess` (signal processing) или `inprocess` (PyTorch для torchcrepe)
  - **Device**: `cpu` (classic) или `cuda`/`cpu` (torchcrepe)
  - **Triton**: ❌ Нет (in-process или signal processing)
  - **Загрузка**: через `dp_models` (ModelManager) для torchcrepe, no-network policy

**Консистентность ModelManager**:
- Все модели загружаются через `dp_models` (ModelManager)
- Spec `precision/engine/runtime` соответствует фактическим артефактам
- Offline/no-network policy enforced

### Parallelization

**Внутренний параллелизм** (внутри компонента):
- **Нет внутреннего параллелизма**: компонент выполняет только агрегацию результатов от зависимых компонентов
- Обработка происходит последовательно: silence detection → получение результатов зависимых компонентов → агрегация

**Внешний параллелизм** (выше компонента):
- **Batch processing**: компонент batch-safe и может обрабатывать несколько файлов параллельно через `extract_batch()`
  - Каждый файл обрабатывается изолированно через `run_bundle()`
  - Изоляция данных: каждый файл имеет свой `tmp_path` и `artifacts_dir`
  - Результаты зависимых компонентов передаются per-file
- **Video-level parallelism**: компонент может обрабатываться параллельно на разных видео (разные `run_id`)
  - Требования к изоляции: разные `run_id`, разные `result_store` пути
  - Thread-safety: компонент thread-safe (read-only shared state для моделей зависимых компонентов)

**Ограничения**:
- Компонент не поддерживает GPU batching (не требует, так как только агрегирует результаты)
- Параллелизм ограничен зависимыми компонентами (ASR, diarization, pitch)
- Требования к GPU: опционально (через зависимые компоненты)

### Performance characteristics

**Resource costs**:
- **CPU**: низкие (только агрегация данных, ~10-50ms для типичного файла)
- **GPU**: опционально (через зависимые компоненты ASR/diarization)
- **Estimated duration**: ~10.0 секунд для типичного аудио файла (включая время зависимых компонентов)

**Зависимости от под-экстракторов**:
- **ASR**: зависит от размера модели (`small`/`medium`/`large`) и длительности аудио
  - Типичное время: 1-5 секунд для `small` модели на 1 минуту аудио
- **Diarization**: зависит от размера модели (`small`/`large`) и количества спикеров
  - Типичное время: 2-10 секунд для `small` модели на 1 минуту аудио
- **Pitch**: зависит от backend (`classic` быстрее, `torchcrepe` медленнее)
  - Типичное время: 0.5-2 секунды для `classic` backend на 1 минуту аудио

**Stage timings** (сохраняются в `payload.stage_timings_ms`):
- `silence_detection_ms`: время детекции тишины (~10-50ms)
- `asr_ms`: время обработки ASR результата (~10-100ms)
- `diarization_ms`: время обработки diarization результата (~10-100ms)
- `pitch_ms`: время обработки pitch результата (~10-50ms, если включен)
- `aggregates_ms`: время агрегации результатов (~10-50ms)
- `total_ms`: общее время обработки

**Источник данных**: `docs/models_docs/resource_costs/speech_analysis_extractor_costs_v1.json` (если доступен)

### Quality validation & human-friendly inspection

**Минимальные sanity-checks**:
- Диапазоны значений разумны:
  - `asr_token_total` ≥ 0, `asr_token_density_per_sec` ≥ 0
  - `speaker_count` ≥ 0, `dominant_speaker_share` ∈ [0.0, 1.0]
  - `pitch_f0_mean` ∈ [50, 2000] Hz (если включен)
- Консистентность связных фичей:
  - `asr_segments_count` соответствует длине `asr_lang_id_by_segment`
  - `diar_segments_count` соответствует количеству сегментов в `speaker_segments`
  - `speaker_count` соответствует количеству уникальных `speaker_ids`
- Статистические инварианты:
  - `asr_token_mean` ≤ `asr_token_total` (если `asr_segments_count` > 0)
  - `speaker_balance_score` ∈ [0.0, 1.0] (0 = один доминирует, 1 = равномерное распределение)
  - `pitch_stability` ∈ [0.0, 1.0] (0 = нестабильная, 1 = стабильная)

**Human-friendly визуализация**:
- HTML renderer для дебага доступен через `render_speech_analysis_extractor_html()`
- Включает все метрики, распределения, статистики
- Только для локального дебага, не в production артефактах

### Render (dev-only)

**Audit v3**: рендер — мини-дашборд для QA, **offline-only** (vanilla canvas, без CDN).

#### Файлы рендера

- `_render/render_context.json` — JSON контекст (summary, asr_metrics, diarization_metrics, pitch_metrics)
- `_render/render.html` — HTML страница с графиками

#### Как читать выход

- **Key facts** (сверху): `status`, `schema_version`, `empty_reason`, `duration_sec`
- **KPI карточки**: duration_sec, sample_rate, asr_segments_count, asr_token_total, asr_token_density_per_sec, speaker_count, dominant_speaker_share, pitch_f0_mean
- **Графики** (vanilla canvas):
  - **Language distribution** — bar chart распределения языков по сегментам (если asr_metrics включены)
  - **Language ID by segment** — timeline lang_id по сегментам
  - **Speaker distribution** — bar chart количества сегментов на спикера
  - **Pitch distribution by octave** — bar chart распределения F0 по октавам (если pitch включён)

#### Типовые распределения и аномалии

- **Норма**: `asr_token_density_per_sec` 2–8 для речи, `speaker_count` 1–5 для типичного контента, `dominant_speaker_share` 0.3–0.9
- **Аномалии**: `asr_token_total=0` при длинном аудио → тишина или ASR failed; `speaker_count=0` при речи → diarization failed; `pitch_f0_mean` вне 80–400 Hz → артефакты pitch

#### Связь с NPZ

NPZ = source-of-truth. Рендер читает `feature_names`/`feature_values`, `asr_lang_id_by_segment`, `speaker_ids`, `asr_lang_distribution`, `pitch_distribution` из NPZ.

#### Время выполнения

`meta.stage_timings_ms` (или `payload.stage_timings_ms`): `silence_detection_ms`, `asr_ms`, `diarization_ms`, `pitch_ms`, `aggregates_ms`, `total_ms`.

#### Параметры конфига, влияющие на результат

- `enable_asr_metrics`, `enable_diarization_metrics`, `enable_pitch_metrics` — какие фичи попадают в NPZ
- `enable_silence_detection`, `silence_peak_threshold`, `silence_rms_threshold` — empty при тишине
- `pitch_enabled` — требуется для pitch_metrics

### Visualization (рекомендации для UI/сайта)

1. **ASR метрики**: speech_rate_wpm, lang_distribution (bar/pie), lang_id_by_segment timeline, token_total, token_density_per_sec
2. **Diarization метрики**: speaker_count, speaker_balance_score, speaker_transitions_count, speaker_ids timeline, dominant_speaker_share
3. **Pitch метрики**: pitch_f0_mean, pitch_stability, pitch_distribution по октавам, range f0_min–f0_max

**Пример использования HTML renderer**:
```python
from src.extractors.speech_analysis_extractor.utils.render import render_speech_analysis_extractor_html

html_path = render_speech_analysis_extractor_html(
    npz_path="result_store/.../speech_analysis_extractor/speech_analysis_extractor_features.npz",
    output_path="debug_speech_analysis.html"
)
```

### Связанные компоненты

- **`asr_extractor`**: распознавание речи через inprocess Whisper модель (ModelManager). Результат передается через `asr_result` из `extractor_results` (обязательная зависимость, если `enable_asr_metrics=True`).
- **`speaker_diarization_extractor`**: определение спикеров через inprocess pyannote.audio + whisperx (ModelManager). Результат передается через `diarization_result` из `extractor_results` (обязательная зависимость, если `enable_diarization_metrics=True`).
- **`pitch_extractor`**: анализ высоты тона (опционально). Результат передается через `pitch_result` из `extractor_results` (опциональная зависимость, fallback на внутренний запуск).
- **ModelManager** (`dp_models`): управление моделями и спецификациями (используется зависимыми компонентами)
- **AudioUtils**: загрузка и предобработка аудио
- **BaseExtractor**: базовый интерфейс экстрактора
- **Segmenter**: предоставляет сегменты для анализа
- **Dependency Resolver**: управление зависимостями между extractors, автоматическое добавление зависимостей, проверка обязательных/опциональных зависимостей

### Использование

Экстрактор требует вызова через `run_bundle()` с сегментами и результатами зависимых компонентов:

```python
result = extractor.run_bundle(
    input_uri="path/to/audio.wav",
    tmp_path="/tmp",
    asr_segments=[...],  # от Segmenter families.asr
    diar_segments=[...],  # от Segmenter families.diarization
    asr_result=asr_result,  # результат от asr_extractor (обязателен если enable_asr_metrics=True)
    diarization_result=diarization_result,  # результат от speaker_diarization_extractor (обязателен если enable_diarization_metrics=True)
    pitch_result=pitch_result,  # результат от pitch_extractor (опционально, используется если pitch_enabled=True и enable_pitch_metrics=True)
)
```

**Примечание**: В реальном использовании через оркестратор (`extractor_runner.py`), результаты зависимых компонентов автоматически передаются из `extractor_results`. Прямой вызов `run()` вернет ошибку, указывающую на необходимость использования `run_bundle()`.

### Примечания

1. **Сегменты обязательны**: экстрактор требует предварительно сегментированные окна от Segmenter
2. **Минимальная длительность**: аудио < 5 сек → `status="empty"`, `empty_reason="audio_too_short"`
3. **Токены ASR**: полные token IDs сохраняются в NPZ через `asr_extractor`, здесь только статистики
4. **Выравнивание**: не выполняется выравнивание между ASR токенами и спикерами (временные метки Whisper недоступны)
5. **Тихое аудио**: автоматически определяется и возвращается как `empty` с `empty_reason="audio_missing_or_extract_failed"`, а не ошибка (если silence detection включен)
6. **Новая архитектура зависимостей**: компонент использует существующие результаты зависимых компонентов из `extractor_results`, а не запускает под-экстракторы внутри себя
   - Если `enable_asr_metrics=True`, требуется результат от `asr_extractor` (обязательная зависимость)
   - Если `enable_diarization_metrics=True`, требуется результат от `speaker_diarization_extractor` (обязательная зависимость)
   - Если `pitch_enabled=True` и `enable_pitch_metrics=True`, предпочтительно используется результат от `pitch_extractor` (опциональная зависимость, fallback на внутренний запуск)
7. **Проверка зависимостей**: зависимости проверяются во время выполнения через feature flags (fail-fast):
   - Если `enable_asr_metrics=True`, но `asr` extractor не включен или не предоставил результат → ошибка
   - Если `enable_diarization_metrics=True`, но `speaker_diarization` extractor не включен или не предоставил результат → ошибка
   - Если `enable_pitch_metrics=True` и `pitch_enabled=True`, но `pitch` extractor не включен или не предоставил результат → ошибка
8. **NO автоматическое добавление зависимостей**: зависимости НЕ добавляются автоматически через dependency resolver. Пользователь должен явно указать нужные extractors в `--extractors` или в конфиге
9. **Валидация**: полная валидация структуры сегментов и payload от зависимых компонентов
10. **Progress reporting**: обновление прогресса с детальными сообщениями и временем выполнения
11. **Детальное профилирование**: логирование времени выполнения для каждого этапа (silence, ASR, diarization, pitch, aggregates)
12. **Contract versioning**: версия контракта `speech_analysis_contract_v1` используется для валидации совместимости с downstream extractors
13. **Feature flags**: все фичи opt-in через персональные флаги (в конфиге по умолчанию все отключены)
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
