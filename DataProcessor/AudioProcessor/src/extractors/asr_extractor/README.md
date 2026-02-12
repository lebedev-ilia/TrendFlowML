## `asr_extractor` (Audio Tier‑0 baseline, required)

### Назначение

Извлекает **транскрипцию речи** через Whisper ASR (**inprocess**, без сети). Выход — в первую очередь **token IDs** (для downstream), опционально (явным флагом) — **raw text** по сегментам.

**Версия**: 2.2.0  
**Категория**: speech  
**GPU**: preferred (может работать на CPU, но медленнее)

### Входы (строго, no‑fallback)

- **`audio/audio.wav`** (Segmenter)
- **`audio/segments.json`** family: `asr` (длинные окна для ASR)

Если `segments` пустой → **error**.

#### Sampling policy (ASR windows)

`Segmenter` строит family=`asr` как **длинные окна** (обычно 10-30 секунд) для устойчивой транскрипции. Параметры сохраняются в `audio/segments.json`.

### Выходы (per-run storage)

NPZ пишет AudioProcessor в:
- `result_store/<platform_id>/<video_id>/<run_id>/asr_extractor/asr_extractor_features.npz` (**фиксированное имя**)

Схема: `audio_npz_v1` (см. `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`).

#### Полезные поля payload:

**Основные фичи (всегда включены)**:
- `token_ids_by_segment`: список списков token IDs (int32) для каждого сегмента (**feature-gated**, см. ниже)
- `lang_id_by_segment`: список языковых ID для каждого сегмента
- `segment_start_sec`, `segment_end_sec`, `segment_center_sec`: временные границы сегментов
- `segments_count`: количество обработанных сегментов
- `sample_rate`: частота дискретизации (16000 Hz)
- `whisper_model_name`: имя модели Whisper из ModelManager
- `tokenizer_model_name`: имя shared tokenizer (shared_tokenizer_v1)
- `tokenizer_weights_digest`: digest весов tokenizer'а (для воспроизводимости)
- `device_used`: устройство обработки (например `cuda`/`cpu`)
- `asr_text_contract_version`: версия контракта для валидации совместимости с TextProcessor
- `decode_language`: `null` если авто, иначе строка (например `"ru"`)
- `decode_temperature`, `decode_beam_size`, `decode_best_of`: параметры декодинга
- `decode_enable_fallback`: включен ли fallback decode

**Агрегаты и статистики (feature-gated)**:
- `token_counts`: количество токенов по сегментам (list[int])
- `token_total`: общее количество токенов (int)
- `token_density_per_sec`: средняя плотность токенов (tokens/sec, float)
- `speech_rate_wpm`: слова в минуту (приблизительно, float)
- `lang_distribution`: распределение языков по сегментам (dict[lang_id, count])
- `segments_with_speech`: количество сегментов с ненулевыми token IDs (int)
- `avg_segment_duration_sec`: средняя длительность сегмента (float)
- `token_variance`: статистическая дисперсия token counts по сегментам (float)

### Feature dependencies

**Зависимости между фичами**:
- `token_total` зависит от `token_counts` (сумма всех counts)
- `token_density_per_sec` зависит от `token_total` и `segment_duration` (расчёт на основе total)
- `speech_rate_wpm` зависит от `token_total` (приблизительный расчёт: tokens / 1.3 / duration_min)
- `token_variance` зависит от `token_counts` (статистический расчёт)
- `segments_with_speech` зависит от `token_ids_by_segment` (подсчёт непустых sequences)

**Зависимости от других extractors**:
- Нет зависимостей от других extractors (работает независимо)

**Зависимости для downstream**:
- **TextProcessor**:
  - если upstream формирует `VideoDocument` с `doc.asr.segments[].text` — этого достаточно (token IDs не обязательны);
  - если downstream ожидает token IDs — включайте `--asr-enable-token-sequences`.
- **speech_analysis_extractor**: может использовать результаты ASR для анализа речи

### Feature gating

Все фичи контролируются через персональные флаги (CLI аргументы):

**Флаги включения/выключения**:
- `--asr-enable-token-sequences`: `token_ids_by_segment` (sequences)
- `--asr-enable-token-counts`: `token_counts` (per-segment counts)
- `--asr-enable-token-total`: `token_total` (aggregate)
- `--asr-enable-token-density`: `token_density_per_sec`
- `--asr-enable-speech-rate`: `speech_rate_wpm`
- `--asr-enable-lang-distribution`: `lang_distribution`
- `--asr-enable-segments-with-speech`: `segments_with_speech`
- `--asr-enable-avg-segment-duration`: `avg_segment_duration_sec`
- `--asr-enable-token-variance`: `token_variance`

**Флаги декодинга (качество/универсальность)**:
- `--asr-language auto|ru|en|...`: язык (“auto” = автоопределение)
- `--asr-temperature <float>`: температура (0.0 = детерминированно)
- `--asr-beam-size <int>`: beam search (используется когда `temperature==0.0`)
- `--asr-best-of <int>`: sampling best_of (используется когда `temperature>0.0`)
- `--asr-enable-fallback-decode`: включить fallback decode при плохом `avg_logprob` (с явным логированием)
- `--asr-fallback-temperature <float>`
- `--asr-fallback-avg-logprob-threshold <float>`

**Privacy-sensitive output**:
- `--asr-save-segment-text`: сохранить `segment_texts_by_segment` (raw text) + `segment_quality_by_segment` в NPZ payload

**По умолчанию**: все фичи выключены (enable флаги по умолчанию `False`). Включайте только нужные фичи через `--asr-enable-*` флаги.

**Примечание**: если фича A зависит от фичи B, и фича B отключена, фича A также не будет вычисляться (fail-safe).

### Модели

**ML модели используются** (Whisper ASR inprocess):

- **Whisper**:
  - Модели (dp_models): `whisper_small_inprocess`, `whisper_medium_inprocess`, `whisper_large_inprocess`
  - Размер выбирается через параметр `model_size` (по умолчанию: `"small"`)
  - Runtime: `inprocess` (PyTorch)
  - Вход: log-mel spectrogram (80 x N_FRAMES), выровненный до `whisper.audio.N_FRAMES`
  - Выход: DecodingResult (text + quality metrics + tokens в зависимости от версии whisper)

- **Shared Tokenizer** (локально):
  - Модель: `shared_tokenizer_v1` (dp_models)
  - Используется TextProcessor для декодирования token IDs в текст
  - Не хранится в артефактах ASR (только метаданные: имя и digest)

#### ModelManager integration

Экстрактор использует `dp_models.ModelManager` для:
- Разрешения Whisper Triton spec по размеру модели
- Получения Triton runtime params (URL, model name, version, input/output names)
- Разрешения shared tokenizer для метаданных

#### Важно про `beam_size` и `best_of`

`openai-whisper` не позволяет задавать `beam_size` и `best_of` одновременно. Экстрактор автоматически выбирает режим:
- `temperature==0.0` → beam search (`beam_size`), `best_of` игнорируется
- `temperature>0.0` → sampling (`best_of`), `beam_size` игнорируется

### Конфигурация

```python
{
    "model_size": "small",              # "small" | "medium" | "large"
    "sample_rate": 16000,               # Частота дискретизации (Whisper требует 16kHz)
    "device": "auto",                   # "auto" | "cuda" | "cpu" (client-side)
    "triton_batch_size": None,          # None | int (опциональный батчинг)
    # Feature gating flags (все по умолчанию True)
    "enable_token_sequences": True,
    "enable_token_counts": True,
    "enable_token_total": True,
    "enable_token_density": True,
    "enable_speech_rate": True,
    "enable_lang_distribution": True,
    "enable_segments_with_speech": True,
    "enable_avg_segment_duration": True,
    "enable_token_variance": True,
}
```

### Архитектура

1. **Загрузка аудио сегментов**: через `AudioUtils.load_audio_segment()` с ресемплированием до `sample_rate`
2. **Mel**: вычисление log-mel spectrogram (80 mel bins) и pad/trim до `N_FRAMES`
3. **Whisper decode**: `detect_language()` (best-effort) + `decode()` с `DecodingOptions`
4. **Token IDs**:
   - preferred: `decoded_text` → `shared_tokenizer_v1.encode()` (если получилось)
   - fallback: токены whisper (если shared-tokenizer encode недоступен/упал)
5. **Token validation**: валидация диапазонов/типа + `lang_id`
6. **Агрегация**: сбор результатов по всем сегментам и вычисление статистик (feature-gated)
7. **Progress reporting**: обновление прогресса каждые 10% сегментов (если progress_callback установлен)

### Обработка ошибок

**Ошибки**:
- **Пустые segments**: `ValueError("segments is empty (no-fallback)")`
- **ModelManager ошибки**: `RuntimeError` при отсутствии Whisper spec или shared tokenizer
- **Несоответствие sample_rate**: `RuntimeError` если ресемплирование не сработало
- **Token validation ошибки**: `ValueError` если token IDs невалидны (out of range, invalid dtype)

**Retry логика**:
- Retry на уровне orchestrator (`run_cli.py`) для transient ошибок (timeout/connection и т.п.)
- No-fallback policy: ошибки не маскируются, явно репортируются

### Особенности

- **No-network**: Whisper грузится строго локально через `dp_models`
- **No raw text по умолчанию**: raw text сохраняется только при `--asr-save-segment-text`
- **Segments mode only**: `run()` не поддерживается в production, используется только `run_segments()`
- **Shared tokenizer contract**: token IDs относятся к `shared_tokenizer_v1` (если encode успешен)
- **Contract version**: `asr_text_contract_version` для валидации совместимости с TextProcessor
- **Progress reporting**: обновление прогресса каждые 10% сегментов (для длинных видео)

### Performance characteristics

**Resource costs**:
- **Client-side**: минимальные (только HTTP запросы к Triton)
- **Triton-side**: зависит от размера модели (small/medium/large)
- **Estimated duration**: ~8.0 секунд для типичного видео

**Единица обработки**: `audio_window` (Segmenter `families.asr`)

**Оптимизация**:
- Triton batching может уменьшить latency при большом количестве сегментов
- Progress reporting не влияет на производительность (callback-based)

### Параллелизм/батчинг/лимиты

- **Segment parallelism**: whisper inprocess выполняется последовательно по сегментам
- **Batch mode**: поддерживается на уровне AudioProcessor (сбор сегментов из разных видео), но внутри whisper декодинг остаётся последовательным

### Качество: sanity checks

**Валидация token IDs**:
- Проверка dtype (должен быть int32)
- Проверка диапазонов (token IDs должны быть в [0, vocab_size-1])
- Проверка lang_id (должен быть в [-1, 99])
- Проверка special tokens (soft check, не обязательна)

**Статистические инварианты**:
- `token_total >= 0`
- `token_density_per_sec >= 0`
- `speech_rate_wpm >= 0`
- `segments_with_speech <= segments_count`
- `avg_segment_duration_sec > 0` (если есть сегменты)

### Visualization

**Рекомендуемые типы визуализации для UI/сайта**:

1. **Timeline визуализация**:
   - Ось X: время (секунды)
   - Ось Y: token count per segment
   - Интерактивные tooltips: показывать lang_id, start/end time, token count
   - Цветовая кодировка: разные цвета для разных lang_id
   - Zoom и pan для навигации по длинным видео

2. **Distribution графики**:
   - Histogram token counts: распределение количества токенов по сегментам
   - Bar chart lang_distribution: количество сегментов по языкам
   - Box plot token_counts: статистики (min, max, median, quartiles)

3. **Summary карточки**:
   - Token total (большое число)
   - Speech rate WPM (с индикатором нормальности)
   - Segments with speech / total segments (progress bar)
   - Token density (tokens/sec) с трендом

4. **Интерактивные элементы**:
   - Фильтры по lang_id
   - Поиск сегментов с высоким/низким token count
   - Экспорт данных (CSV, JSON)

**HTML renderer для дебага**:
- Функция `render_asr_extractor_html()` в `src/core/renderer.py`
- Опциональное декодирование token IDs в текст (только для локального дебага)
- Timeline с сегментами, статистики, distribution графики
- Использование: `python -c "from src.core.renderer import render_asr_extractor_html; render_asr_extractor_html('path/to/npz', 'output.html', decode_tokenizer=True)"`

### Связанные компоненты

- **TextProcessor**: декодирует token IDs в текст через shared tokenizer
- **Segmenter**: предоставляет ASR windows (`families.asr`)
- **dp_models.ModelManager**: управляет моделями и Triton specs
- **dp_triton**: Triton HTTP client для inference
- **speech_analysis_extractor**: может использовать результаты ASR для анализа речи

### Примечания

1. **Production policy**: экстрактор работает только в режиме segments (не поддерживает `run()`)
2. **Token IDs output**: выход — token IDs, а не raw text, для уменьшения размера артефактов и privacy
3. **Triton dependency**: требует доступный Triton сервер с развернутыми Whisper моделями
4. **Model size tradeoff**: small (быстрее, меньше точность) vs large (медленнее, выше точность)
5. **Contract version**: `asr_text_contract_version` для валидации совместимости с TextProcessor
6. **Feature gating**: все фичи контролируются через персональные флаги для гибкости и оптимизации
