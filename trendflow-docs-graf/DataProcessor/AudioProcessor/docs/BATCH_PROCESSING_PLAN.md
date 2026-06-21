# План адаптации AudioProcessor для батчевой обработки

## Обзор задачи

Адаптировать AudioProcessor и все его компоненты для одновременной обработки нескольких аудио файлов с:
- **Двухуровневой параллельностью**:
  - **Уровень 1**: Параллельная обработка нескольких видео через `ThreadPoolExecutor` (max_video_workers)
  - **Уровень 2**: Параллельная обработка сегментов внутри одного видео (существующая segment_parallelism)
- **Батчингом на GPU** для ML-моделей (ASR, diarization, emotion, CLAP, etc.) с гибридным подходом:
  - Сбор сегментов из всех видео
  - Батчинг с лимитом размера (max_segments_per_gpu_batch)
  - Распределение результатов обратно по видео
- **Сохранением изоляции** данных между файлами
- **Корректной обработкой сегментов** через Segmenter contract
- **Валидацией render файлов** для каждого extractor'а

## Статус реализации

🚧 **В разработке**. План создан на основе опыта разработки TextProcessor batch processing (Stage 0-5).

**Последнее обновление**: 
- ✅ Stage 0 завершена: базовый каркас `run_batch()`, `extract_batch()`, `AudioFileContext` реализованы.
- ✅ Stage 1 завершена: изоляция артефактов реализована - каждый файл имеет свой `artifacts_dir` для каждого extractor'а.
- ✅ Stage 2 завершена: GPU batching для CLAP extractor реализован с гибридным подходом (батчинг сегментов из всех видео).
- ✅ Stage 3 завершена: GPU batching для всех ML-моделей реализован (ASR, speaker_diarization, emotion_diarization, source_separation).
- ✅ Stage 4 завершена: Двухуровневая параллельность и CPU parallelism реализованы (video-level parallelism + GPU batching).
- ✅ Stage 5 завершена: CLI интеграция и production-ready batch processing (основные функции реализованы, render валидация опциональна).

---

## 0. Acceptance Criteria (критерии готовности / DoD)

Эти пункты — **критерии**, по которым можно идти "по компонентам" и рефакторить.  
Формат: сначала делаем **batch-safe** (безопасно для многодокументной обработки), затем **batch-optimized** (ускорение).

### 0.1 Корректность (обязательное)

- **Эквивалентность результатов**: для каждого аудио файла результаты `run_batch([audio_file])` совпадают с `run(audio_file)` (допустимы только тривиальные float-расхождения).
- **Изоляция**:
  - Артефакты (`*.npy`, временные файлы) пишутся **внутрь per-run ResultStore** и **не конфликтуют** между файлами;
  - Нет shared mutable state между файлами внутри extractor'ов (кроме read-only моделей/корпусов);
  - Каждый файл имеет свой `tmp_path` для временных артефактов.
- **Segmenter contract**:
  - Корректная обработка `audio/audio.wav` и `audio/segments.json` для каждого файла;
  - Изоляция сегментов между файлами (нет пересечений в `segments.json`);
  - Корректная работа с `families` в `segments.json` (primary, clap, tempo, asr, diarization, emotion).
- **Детерминизм**: запрещены `glob + mtime`, "последний файл", зависимости от абсолютных путей как source-of-truth.
- **Политика ошибок**:
  - падение одного файла **не валит** весь батч, если extractor не marked required;
  - required extractor → падение файла помечает **этот файл** как error (и/или валит batch — выбрать и зафиксировать контрактом).
- **Наблюдаемость**: логирование/прогресс должны быть **с привязкой к audio file id**.

### 0.2 Производительность (измеримое, но не блокирующее для MVP)

- **GPU batching** даёт ускорение на ML-моделях (ASR, diarization, emotion, CLAP) относительно поштучного прогона.
- **CPU parallelism** даёт ускорение на "чисто CPU" этапах (spectral, quality, mfcc, mel, etc.) без неограниченного роста RAM.
- **Segment batching**: оптимизация обработки сегментов внутри одного файла (микробатчинг).
- Добавлены метрики: wall-time по стадиям/экстракторам, утилизация GPU (best-effort), peak RAM (best-effort).

### 0.3 Render файлы и валидация (критично для AudioProcessor)

- **Корректность render файлов**: каждый extractor должен иметь корректный `render.py` с функцией `render_<extractor_name>()`.
- **Валидация render**: render функции должны корректно обрабатывать NPZ данные и генерировать render-context JSON.
- **Изоляция render**: render контексты для разных файлов не должны пересекаться.
- **Тестирование render**: автоматическая проверка корректности render функций для всех extractors.

---

## 1. Чеклист внедрения (итеративно, стадиями)

### Стадия 0 — "каркас" без оптимизаций (MVP API) ✅

- [x] `BaseExtractor`: добавить `extract_batch(audio_files)` (дефолт — цикл `run`/`run_segments`) и `supports_batch` (дефолт `False`).
- [x] `BaseExtractor`: добавить `extract_batch_segments()` для батчинга сегментов из нескольких видео (для GPU extractors).
- [x] `MainProcessor`: добавить `run_batch(audio_files)` (дефолт — последовательный вызов `process_video()` на каждый файл).
- [x] `MainProcessor`: добавить `AudioFileContext` для изоляции контекста каждого файла (file_id, tmp_path, artifacts_dir, segments.json).
- [ ] Smoke: `run_batch([audio_file])` == `process_video(audio_file)` по базовым полям (`status/error/empty_reason`) + запуск через `DP_MODELS_ROOT`/`PYTHONPATH`.
- [ ] **Специфика AudioProcessor**: поддержка `run_segments()` в batch режиме с двухуровневой параллельностью.

### Стадия 1 — изоляция артефактов и файл-контекст (batch-safe foundation) ✅

- [x] Ввести **AudioFileContext**: `file_id`, `tmp_path`, `artifacts_dir`, ссылки на result_store paths, `segments.json` path.
- [x] Все extractor'ы, которые пишут файлы, должны писать **в свой per-file artifacts_dir** (не общий).
  - Реализовано: `run_batch()` устанавливает `artifacts_dir` для каждого extractor'а для каждого файла отдельно.
- [x] Везде, где имена артефактов фиксированные (`*.npy`), обеспечить, что они фиксированные **внутри per-file artifacts_dir** (иначе конфликт при батче).
  - Реализовано: каждый файл имеет свой `artifacts_dir = <file_artifacts_dir>/<component_name>/_artifacts`.
- [x] Инвариант: артефакты содержат только relpath'и внутри **своего** `_artifacts/`.
  - Реализовано: extractors используют `self.artifacts_dir` для сохранения .npy файлов.
- [ ] **Segmenter contract**: корректная обработка `audio/audio.wav` и `audio/segments.json` для каждого файла.
  - Требуется обновление `run_cli.py` для batch режима (Stage 5).
- [ ] **Render изоляция**: render контексты для разных файлов не пересекаются.
  - Требуется обновление render системы для batch режима (Stage 5).

### Стадия 2 — первый GPU batching PoC (минимум: `CLAPExtractor`) ✅

- [x] Реализовать `CLAPExtractor.extract_batch_segments()` с гибридным батчингом:
  - Сбор сегментов из всех видео
  - Группировка в батчи по `max_segments_per_batch` (если задан)
  - Последовательная обработка батчей
  - Распределение результатов обратно по видео
- [x] Добавлено свойство `supports_batch = True` для CLAP extractor.
- [ ] Добавить micro-bench: `scripts/bench_clap_batch.py` (loop `run_segments()` vs `extract_batch_segments()`).
  - Отложено до тестирования (можно добавить позже).
- [x] **Специфика**: поддержка батчинга сегментов внутри одного файла и между файлами (гибридный подход).

### Стадия 3 — batching переменной длины (hard cases)

- [ ] `ASRExtractor.extract_batch()`: батчирование сегментов ASR из всех файлов → batch inference → распределение обратно → сохранение per-file artifacts.
- [ ] `SpeakerDiarizationExtractor.extract_batch()`: аналогично для diarization.
- [ ] `EmotionDiarizationExtractor.extract_batch()`: аналогично для emotion.
- [ ] `SourceSeparationExtractor.extract_batch()`: аналогично для source separation.
- [ ] **Специфика**: обработка сегментов разной длины, padding для батчинга.

### Стадия 4 — Двухуровневая параллельность и CPU parallelism (по уровням зависимостей)

- [ ] Добавлены параметры двухуровневой параллельности в `MainProcessor.run_batch()`:
  - `max_video_workers`: количество параллельных воркеров для обработки видео (уровень 1)
  - `enable_video_parallel`: включение параллельной обработки нескольких видео
  - `max_segment_workers`: количество параллельных воркеров для сегментов (уровень 2, для CPU extractors)
  - `enable_segment_parallel`: включение параллельной обработки сегментов
  - `enable_gpu_batching`: включение GPU batching для сегментов
  - `max_segments_per_gpu_batch`: лимит размера батча для GPU extractors (null = без лимита)
  - `enable_cpu_parallel`: включение CPU параллелизма для независимых extractors
- [ ] Реализован граф зависимостей (`_build_dependency_levels()`) с топологической сортировкой для группировки extractors по уровням.
- [ ] Использовать существующий `dependency_resolver.py` для определения порядка extractors.
- [ ] **Уровень 1 (видео)**: ThreadPoolExecutor для параллельной обработки нескольких видео (если `enable_video_parallel=True`).
- [ ] **Уровень 2 (сегменты)**: 
  - CPU extractors: существующая `segment_parallelism` через ThreadPoolExecutor внутри `run_segments()`
  - GPU extractors: гибридный батчинг сегментов из всех видео с лимитом размера батча
- [ ] Обработка по уровням зависимостей: extractors одного уровня могут выполняться параллельно/в батче для всех видео одновременно.
- [ ] GPU batch extractors обрабатываются батчем сегментов из всех видео (если `supports_batch=True` и `enable_gpu_batching=True`).
- [ ] CPU extractors обрабатываются параллельно через ThreadPoolExecutor на уровне видео и сегментов (если `enable_cpu_parallel=True`).
- [ ] GPU legacy extractors обрабатываются последовательно для каждого файла.
- [ ] Лимиты: `max_video_workers` контролирует параллельность на уровне видео, `max_segment_workers` - на уровне сегментов.

**Реализованные зависимости** (из `dependency_resolver.py`):
- `key` → `chroma`
- `band_energy` → `spectral`
- `spectral_entropy` → `spectral`

**Feature flags зависимости** (из `dependency_resolver.py`):
- `key`: `enable_top_k` → `enable_detailed_scores`, `enable_key_changes` → `enable_time_series`, `enable_stability_metrics` → `enable_time_series`
- `asr`: `enable_token_total` → `enable_token_counts`, `enable_token_density` → `enable_token_total`, etc.
- `mel`: `enable_stats_vector` → `enable_statistics`

### Стадия 5 — CLI интеграция и production-ready batch processing ✅

- [x] Добавлены CLI аргументы `--audio-input-dir` и `--audio-input-list` для batch режима.
- [x] Интеграция в верхний оркестратор (`DataProcessor/main.py`) с поддержкой batch флагов (через `global_config_parser.get_audio_cli_args()`).
- [x] Конфигурация через `global_config.yaml`:
  - `audio.batch_processing.enabled`: включение batch режима
  - `audio.batch_processing.max_video_workers`: количество параллельных воркеров для видео (null = auto, обычно os.cpu_count())
  - `audio.batch_processing.enable_video_parallel`: включение параллельной обработки нескольких видео
  - `audio.batch_processing.max_segment_workers`: количество параллельных воркеров для сегментов (null = auto, для CPU extractors)
  - `audio.batch_processing.enable_segment_parallel`: включение параллельной обработки сегментов
  - `audio.batch_processing.enable_gpu_batching`: включение GPU batching для сегментов
  - `audio.batch_processing.max_segments_per_gpu_batch`: лимит размера батча для GPU extractors (null = без лимита)
  - `audio.batch_processing.enable_cpu_parallel`: включение CPU параллелизма для независимых extractors
- [x] CLI флаги для тонкой настройки:
  - `--batch-max-workers`: переопределение max_workers
  - `--no-batch-gpu`: отключение GPU batching
  - `--no-batch-cpu-parallel`: отключение CPU параллелизма
  - `--batch-max-segments-per-gpu-batch`: лимит размера батча для GPU extractors
- [x] Изоляция результатов: каждый файл сохраняется в отдельную директорию внутри ResultStore.
- [x] Валидация NPZ файлов для каждого файла в batch режиме.
- [ ] **Render валидация**: автоматическая проверка корректности render функций для всех extractors (опционально, можно отложить).

---

## 2. Специфичные моменты AudioProcessor (уроки из TextProcessor)

### 2.1 Файлы render и их корректность

**Проблема**: Каждый extractor должен иметь корректный `render.py` с функцией `render_<extractor_name>()` для генерации render-context JSON.

**Решение**:
- Автоматическая валидация render функций при запуске batch processing.
- Тестирование render функций на реальных NPZ данных.
- Документирование формата render-context JSON для каждого extractor'а.

**Чеклист**:
- [ ] Все extractors имеют `render.py` файлы.
- [ ] Render функции корректно обрабатывают NPZ данные.
- [ ] Render контексты изолированы между файлами.
- [ ] Автоматические тесты для render функций.

### 2.2 Время и производительность

**Проблема**: AudioProcessor обрабатывает большие аудио файлы и сегменты, что требует оптимизации.

**Решение**:
- GPU batching для ML-моделей (ASR, diarization, emotion, CLAP).
- CPU parallelism для спектральных extractors (spectral, quality, mfcc, mel).
- Segment batching внутри одного файла (микробатчинг).

**Метрики**:
- Wall-time по стадиям/экстракторам.
- Утилизация GPU (best-effort).
- Peak RAM (best-effort).
- Время обработки на файл.

### 2.3 Ошибки и error handling

**Проблема**: Ошибки в одном extractor'е не должны валить весь batch.

**Решение**:
- Каждый extractor обёрнут в try/except, ошибки собираются в `errors_by_extractor`.
- Required extractors (через `required_extractors` параметр) fail-fast при ошибках.
- Optional extractors логируют warning и продолжают run.

**Чеклист**:
- [ ] Error handling для всех extractors.
- [ ] Логирование ошибок с привязкой к file_id.
- [ ] Fail-fast для required extractors.
- [ ] Graceful degradation для optional extractors.

### 2.4 Взаимодействия между компонентами и процессорами

**Проблема**: AudioProcessor зависит от Segmenter (audio/audio.wav, audio/segments.json).

**Решение**:
- Корректная обработка Segmenter contract для каждого файла.
- Изоляция сегментов между файлами.
- Корректная работа с `families` в `segments.json`.

**Чеклист**:
- [ ] Корректная обработка `audio/audio.wav` для каждого файла.
- [ ] Корректная обработка `audio/segments.json` для каждого файла.
- [ ] Изоляция сегментов между файлами.
- [ ] Корректная работа с `families` (primary, clap, tempo, asr, diarization, emotion).

### 2.5 Зависимости

**Проблема**: Extractors имеют зависимости друг от друга (например, `key` → `chroma`, `band_energy` → `spectral`).

**Решение**:
- Использовать существующий `dependency_resolver.py` для определения порядка extractors.
- Топологическая сортировка для группировки extractors по уровням.
- Валидация feature flags зависимостей.

**Чеклист**:
- [ ] Граф зависимостей extractors корректно определен.
- [ ] Топологическая сортировка работает корректно.
- [ ] Feature flags зависимости валидируются.
- [ ] Автоматическое добавление недостающих зависимостей (опционально).

### 2.6 Батчинг и параллелизм

**Проблема**: Нужно оптимизировать обработку для ускорения.

**Решение**:
- GPU batching для ML-моделей (ASR, diarization, emotion, CLAP).
- CPU parallelism для спектральных extractors.
- Segment batching внутри одного файла.

**Чеклист**:
- [ ] GPU batching реализован для ML-моделей.
- [ ] CPU parallelism реализован для спектральных extractors.
- [ ] Segment batching реализован внутри одного файла.
- [ ] Метрики производительности собираются.

### 2.7 Документация

**Проблема**: Нужна полная документация для batch processing.

**Решение**:
- Обновить `README.md` с информацией о batch режиме.
- Создать примеры использования.
- Документировать конфигурацию через `global_config.yaml`.

**Чеклист**:
- [ ] `README.md` обновлен с информацией о batch режиме.
- [ ] Примеры использования созданы.
- [ ] Конфигурация документирована.
- [ ] API документирован.

### 2.8 Глобальный конфиг и флаги

**Проблема**: Нужна конфигурация batch processing через `global_config.yaml`.

**Решение**:
- Добавить секцию `audio.batch_processing` в `global_config.yaml`.
- Парсинг конфигурации в `config_parser.py`.
- Передача параметров в `MainProcessor` и CLI.

**Чеклист**:
- [ ] Секция `audio.batch_processing` добавлена в `global_config.yaml`.
- [ ] Парсинг конфигурации реализован в `config_parser.py`.
- [ ] Параметры передаются в `MainProcessor` и CLI.
- [ ] CLI флаги для тонкой настройки реализованы.

### 2.9 Модели и ModelManager

**Проблема**: Модели должны загружаться только через `dp_models` (ModelManager), без сетевых загрузок.

**Решение**:
- Использовать `get_global_model_manager()` для загрузки моделей.
- Enforce offline/no-network policy.
- Ленивая загрузка моделей (lazy loading).

**Чеклист**:
- [ ] Все модели загружаются через ModelManager.
- [ ] Offline/no-network policy enforced.
- [ ] Ленивая загрузка моделей реализована.
- [ ] Модели изолированы между файлами (read-only shared state).

---

## 3. Матрица готовности по extractor'ам (чеклист)

Легенда:
- **batch-safe**: корректно работает при обработке нескольких файлов (без утечек/конфликтов), допускается внутренний цикл по файлам.
- **batch-optimized**: реализован `extract_batch()` и есть ожидаемое ускорение.
- **artifacts**: пишет ли `*.npy` и требуется ли раздельный artifacts_dir.
- **render**: есть ли корректный `render.py` файл.

| Extractor (class) | Device | Зависимости | batch-safe | batch-optimized | artifacts | render | Критичные риски/заметки |
|---|---|---|---|---|---:|---:|---|
| `CLAPExtractor` | GPU | - | ✅ | ✅ | ✅ | ⬜ | ML-модель, GPU batching реализован (Stage 2) |
| `TempoExtractor` | CPU | - | ⬜ | ⬜ | ✅ | ⬜ | Обработка длинных sliding windows |
| `LoudnessExtractor` | CPU | - | ⬜ | ⬜ | ✅ | ⬜ | Обработка коротких окон |
| `ASRExtractor` | GPU | - | ✅ | ✅ | ✅ | ⬜ | ML-модель, GPU batching реализован (Stage 3) |
| `SpeakerDiarizationExtractor` | GPU | - | ✅ | ✅ | ✅ | ⬜ | ML-модель, GPU batching + per-file кластеризация (Stage 3) |
| `EmotionDiarizationExtractor` | GPU | - | ✅ | ✅ | ✅ | ⬜ | ML-модель, GPU batching реализован (Stage 3) |
| `SourceSeparationExtractor` | GPU | - | ✅ | ✅ | ✅ | ⬜ | ML-модель, GPU batching реализован (Stage 3) |
| `SpeechAnalysisExtractor` | CPU/GPU | ASR, Diarization, Pitch | ⬜ | ⬜ | ✅ | ⬜ | Агрегатор, зависит от других extractors |
| `SpectralExtractor` | CPU | - | ⬜ | ⬜ | ✅ | ⬜ | Спектральный анализ, может быть параллелизован |
| `QualityExtractor` | CPU | - | ⬜ | ⬜ | ✅ | ⬜ | Анализ качества, может быть параллелизован |
| `MFCCExtractor` | CPU/GPU | - | ⬜ | ⬜ | ✅ | ⬜ | MFCC features, может быть параллелизован |
| `MelExtractor` | CPU/GPU | - | ⬜ | ⬜ | ✅ | ⬜ | Mel spectrogram, может быть параллелизован |
| `OnsetExtractor` | CPU | - | ⬜ | ⬜ | ✅ | ⬜ | Onset detection, может быть параллелизован |
| `ChromaExtractor` | CPU | - | ⬜ | ⬜ | ✅ | ⬜ | Chroma features, может быть параллелизован |
| `RhythmicExtractor` | CPU | - | ⬜ | ⬜ | ✅ | ⬜ | Rhythmic analysis, может быть параллелизован |
| `KeyExtractor` | CPU | chroma | ⬜ | ⬜ | ✅ | ⬜ | Key detection, зависит от chroma |
| `BandEnergyExtractor` | CPU | spectral | ⬜ | ⬜ | ✅ | ⬜ | Band energy, зависит от spectral |
| `SpectralEntropyExtractor` | CPU | spectral | ⬜ | ⬜ | ✅ | ⬜ | Spectral entropy, зависит от spectral |
| `VoiceQualityExtractor` | CPU | - | ⬜ | ⬜ | ✅ | ⬜ | Voice quality analysis, может быть параллелизован |
| `HPSSExtractor` | CPU | - | ⬜ | ⬜ | ✅ | ⬜ | HPSS separation, может быть параллелизован |
| `PitchExtractor` | CPU/GPU | - | ⬜ | ⬜ | ✅ | ⬜ | Pitch detection, может быть параллелизован |

---

## 4. Риски и митигация

### 4.1 Риск: Сложность изоляции артефактов
**Митигация**: Использовать per-file artifacts_dir с четкой изоляцией директорий

### 4.2 Риск: Зависимости между extractors
**Митигация**: Использовать существующий `dependency_resolver.py` для определения порядка extractors

### 4.3 Риск: Разные размеры сегментов
**Митигация**: Padding для батчинга, группировка по размерам

### 4.4 Риск: Память GPU
**Митигация**: Динамический размер батча, мониторинг памяти

### 4.5 Риск: Segmenter contract
**Митигация**: Строгая валидация `audio/audio.wav` и `audio/segments.json` для каждого файла

### 4.6 Риск: Render файлы
**Митигация**: Автоматическая валидация render функций, тестирование на реальных данных

---

## 5. Дополнительные улучшения (опционально)

### 5.1 Кеширование
- Общий кеш эмбеддингов для всех файлов в батче
- Кеширование результатов CPU extractors

### 5.2 Асинхронная обработка
- Асинхронная загрузка аудио для следующего батча
- Перекрытие GPU и CPU обработки

### 5.3 Мониторинг
- Метрики производительности в реальном времени
- Алерты при превышении лимитов памяти/времени

---

## 6. Следующие шаги

1. **Начать с Stage 0**: создать базовый каркас `run_batch()` API
2. **Stage 1**: реализовать изоляцию артефактов и файл-контекст
3. **Stage 2**: реализовать GPU batching для CLAP extractor'а
4. **Stage 3**: расширить GPU batching на другие ML-модели
5. **Stage 4**: реализовать CPU parallelism по уровням зависимостей
6. **Stage 5**: интегрировать в CLI и production

---

## 7. Ссылки

- [TextProcessor Batch Processing Plan](../TextProcessor/docs/BATCH_PROCESSING_PLAN.md) — исходный план для TextProcessor
- [AudioProcessor README](../AudioProcessor/README.md) — основная документация AudioProcessor
- [Dependency Resolver](../AudioProcessor/src/core/dependency_resolver.py) — разрешение зависимостей extractors
- [Renderer](../AudioProcessor/src/core/renderer.py) — система render файлов
- [Segmenter Contract](../../docs/contracts/SEGMENTER_CONTRACT.md) — контракт Segmenter
---

## Навигация

[README](README.md) · [Module README](../README.md) · [AudioProcessor](MAIN_INDEX.md) · [DataProcessor](../../docs/MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
