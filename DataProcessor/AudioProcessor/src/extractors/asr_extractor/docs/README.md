## `asr_extractor` (Audio Tier‑0 baseline, required)

### Назначение

Извлекает **транскрипцию речи** через Whisper ASR (**inprocess**, без сети). Выход — в первую очередь **token IDs** (для downstream), опционально (явным флагом) — **raw text** по сегментам.

**Версия**: 2.3.2  
**Категория**: speech  
**GPU**: preferred (может работать на CPU, но медленнее)

**Audit v4:** [`DataProcessor/docs/audit_v4/components/audio_processor/asr_extractor_audit_v4.md`](../../../../../docs/audit_v4/components/audio_processor/asr_extractor_audit_v4.md)

### Каталог полей NPZ (Audit v4)

| Поле | Тип / где | Как получают |
|------|-----------|--------------|
| `token_ids_by_segment[i]` | `object` → `int32[T_i]` | Whisper decode → текст → **`shared_tokenizer_v1.encode`** (без whisper token ids) |
| `segment_*_sec` | `float32[N]` | Границы окон из Segmenter (`families.asr`) |
| `lang_code_by_segment`, `lang_conf_by_segment` | str / float32 | Из `detect_language`: предпочтительно код языка и max prob |
| `lang_id_by_segment` | int32 | **Legacy / best-effort** (часто связано с внутренним Whisper language token/индексом). Для стабильной семантики использовать **`lang_code`** |
| `token_counts`, `token_total` | int32 | Длины последовательностей / сумма |
| `token_density_per_sec` | tabular | `sum(token_counts) / sum(durations)` |
| `speech_rate_wpm` | tabular | Оценка слов/мин: `(sum(counts)/1.3) / (duration_min)` |
| `token_variance` | tabular | `np.var(token_counts)` если сегментов > 1, иначе **0** |
| `lang_distribution` | object dict | Счётчик по непустым `lang_code` |
| `segment_quality_by_segment` | object[N] dict | `avg_logprob`, `compression_ratio`, `no_speech_prob`, `temperature` из decode (без raw text) |
| Агрегаты `asr_quality__*_mean/p50/p90/present_rate` | tabular | Савер: статистики по списку `segment_quality_by_segment` |
| `audio_duration_sec`, `asr_*` sampling | scalars | Из `asr_segments_meta` (Segmenter); `-1` / `NaN` / `""` если неизвестно |
| Идентификация Whisper / tokenizer | **`meta.models_used`** | Не дублируются отдельными обязательными ключами в теле NPZ полей |

### Входы (строго, no‑fallback)

- **`audio/audio.wav`** (Segmenter)
- **`audio/segments.json`** family: `asr` (длинные окна для ASR)

Если `segments` пустой → **error**.

#### Sampling policy (ASR windows)

`Segmenter` строит family=`asr` как **длинные окна** (обычно 10-30 секунд) для устойчивой транскрипции. Параметры сохраняются в `audio/segments.json`.

### Выходы (per-run storage)

NPZ пишет AudioProcessor в:
- `result_store/<platform_id>/<video_id>/<run_id>/asr_extractor/asr_extractor_features.npz` (**фиксированное имя**)

Схема: `asr_extractor_npz_v2` (Audit v3 machine schema; см. `src/extractors/asr_extractor/SCHEMA.md`).

#### Полезные поля (payload → NPZ)

**Сохраняются в NPZ** (часть — feature-gated, см. ниже):

- `token_ids_by_segment` (**feature-gated**)
- `lang_id_by_segment` (**legacy**, best-effort; для продукта лучше `lang_code_by_segment`)
- `lang_code_by_segment`, `lang_conf_by_segment`
- `segment_start_sec`, `segment_end_sec`, `segment_center_sec`
- `segment_quality_by_segment` (числовые метрики; **всегда** в payload сохранения)
- `sample_rate`, `segments_count` (в tabular и/или логике савера)

**В `meta` NPZ:** `device_used`, `models_used` (в т.ч. `whisper_*_inprocess`, `shared_tokenizer_v1`), `schema_version`, `asr_text_contract_version`, `features_enabled`, оркестраторский `stage_timings_ms`, плюс **внутренний профиль ASR** (`asr_stage_timings_ms`, опционально `asr_resource_profile`) — см. §«Этап 2».

**Поля `whisper_model_name`, `tokenizer_*`, decode-параметры** остаются во **внутреннем payload** экстрактора и могут пробрасываться в `meta` через общий meta builder в зависимости от оркестратора; **каноничная** идентификация модели для кэша — **`meta.models_used`**.

**Агрегаты и статистики (feature-gated)**:
- `token_counts`: количество токенов по сегментам (list[int])
- `token_total`: общее количество токенов (int)
- `token_density_per_sec`: средняя плотность токенов (tokens/sec, float)
- `speech_rate_wpm`: слова в минуту (приблизительно, float)
- `lang_distribution`: распределение языков по сегментам (dict[lang_code, count])
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
- `--asr-enable-fallback-decode`: **в Audit v3 запрещён** (fail-fast при попытке включить)

**Privacy-sensitive output**:
- `--asr-save-segment-text`: сохранить `segment_texts_by_segment` (raw text) в NPZ payload (debug-only)

**По умолчанию (CLI):** флаги `--asr-enable-*` с `action="store_true"` → **выключены**, пока не переданы явно. Полные прогоны (e2e) часто включают набор флагов в конфиге/аргументах.

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
    # Feature gating flags (все по умолчанию False)
    "enable_token_sequences": False,
    "enable_token_counts": False,
    "enable_token_total": False,
    "enable_token_density": False,
    "enable_speech_rate": False,
    "enable_lang_distribution": False,
    "enable_segments_with_speech": False,
    "enable_avg_segment_duration": False,
    "enable_token_variance": False,
}
```

### Архитектура

1. **Загрузка аудио сегментов**: через `AudioUtils.load_audio_segment()` с ресемплированием до `sample_rate`
2. **Mel**: вычисление log-mel spectrogram (80 mel bins) и pad/trim до `N_FRAMES`
3. **Whisper decode**: `detect_language()` (best-effort) + `decode()` с `DecodingOptions`
4. **Token IDs (FINAL contract)**:
   - `decoded_text` → `shared_tokenizer_v1.encode()` (через `tokenizers.Tokenizer`, offline)
   - если encode не удался → **error** (fallback на whisper tokens запрещён)
5. **Token validation**: валидация диапазонов/типа + `lang_id`
6. **Агрегация**: сбор результатов по всем сегментам и вычисление статистик (feature-gated)
7. **Progress reporting**: обновление прогресса каждые 10% сегментов (если progress_callback установлен)

### Этап 2 — Профилирование и ресурсы (2.3.0+)

**Цель:** детальные фазы времени внутри экстрактора (без замены оркестраторского `stage_timings_ms` в `meta`) и опциональные снимки RSS/GPU для отладки OOM/скорости.

| Механизм | Описание |
|----------|----------|
| **Лог `INFO`** | После успешного `run_segments` / `extract_batch_segments`: строка `ASR | profiling [<scope>]: …` с фазами в мс. |
| **`meta.asr_stage_timings_ms`** | Словарь числовых полей (мс). `run_segments`: `load_audio_ms`, `infer_ms`, `infer_mel_ms`, `infer_decode_ms` (сумма по сегментам), `aggregates_ms`, `total_ms`, `segments_count`. `extract_batch_segments`: `gather_ms`, `load_preprocess_ms`, `infer_ms`, те же `infer_*`, `aggregates_ms` (сборка результатов по файлам), `total_ms`, `n_input_files`, `n_segments_total`. |
| **`meta.asr_resource_profile`** | Только если включено env ниже: плоские ключи вида `rss_mb_at_start`, `rss_mb_after_load`, `gpu_allocated_mb_at_end`, … |

**Переменная окружения**

| Переменная | Значение | Эффект |
|------------|----------|--------|
| `AP_ASR_RESOURCE_PROFILE` | `1` / `true` / `on` | Дополнительно писать снимки RSS/VRAM (через `psutil` и `torch.cuda`, best-effort) в payload → `meta.asr_resource_profile` и суммарно в лог (`rss_end_mb`, `gpu_alloc_end_mb`). |

Без этого флага накладные расходы минимальны: только `perf_counter` и один `INFO` по завершении прогона.

**Файлы:** реализация таймингов в `main.py`, снимки — `utils/resource_profile.py`; визуализация в render JSON — `utils/render.py` (`summary.asr_stage_timings_ms`, `summary.asr_resource_profile`).

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
- **Процесс**: PyTorch Whisper inprocess (без HTTP к Triton для основного decode)
- **VRAM/RAM**: зависит от `model_size` (small/medium/large); см. `asr_resource_profile` при `AP_ASR_RESOURCE_PROFILE=1`
- **Estimated duration**: ~8.0 с для типичного фрагмента (ориентир `estimated_duration`)

**Единица обработки**: `audio_window` (Segmenter `families.asr`)

**Оптимизация** (релиз **2.3.2**):
- При **`--asr-language`** не `auto` не вызывается **`detect_language`** на каждом окне (тот же decode с заданным языком; качество не ухудшается).
- Кеш целевого числа mel-кадров, один модульный импорт `openai-whisper`, `ascontiguousarray` для numpy→torch без лишней копии.
- В `extract_batch_segments` раздача по файлам через **индексы** (O(сегменты), а не O(файлы×сегменты)).
- **`run_segments`**: окна **подгружаются по одному** (метаданные всех окон в памяти; PCM — не больше одного сегмента).
- **`extract_batch_segments`**: PCM **не хранится для всех сегментов сразу** — чтение аудио перед каждым инференс-батчем (пик RAM ≈ размер батча); в `asr_stage_timings_ms` при желании смотрите `load_meta_only_ms` / `load_audio_lazy_ms`.
- **`AP_ASR_LANG_DETECT_ONCE=1`** при `language=auto`: один **`detect_language` на файл** (batch) или на весь **`run_segments`** (одно входное аудио), следующие окна декодятся с тем же `lang_code` (**риск** при смене языка внутри файла).
- Progress reporting по-прежнему лёгкий (callback).

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
3. **Runtime**: Whisper грузится inprocess через `dp_models` (без Triton, без сетевых загрузок)
4. **Model size tradeoff**: small (быстрее, меньше точность) vs large (медленнее, выше точность)
5. **Contract version**: `asr_text_contract_version` для валидации совместимости с TextProcessor
6. **Feature gating**: все фичи контролируются через персональные флаги для гибкости и оптимизации
