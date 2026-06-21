# AudioProcessor

AudioProcessor — процессор аудио модальности. Он извлекает **аудио признаки** из аудио дорожки видео и сохраняет результат в **per‑run `result_store`**.

**Нормализация (portfolio + prod):** [docs/NORMALIZATION_WAVE2.md](docs/NORMALIZATION_WAVE2.md) · [docs/EXTRACTOR_DEPENDENCIES.md](docs/EXTRACTOR_DEPENDENCIES.md) · [docs/PORTFOLIO_PROGRESS_LOG.md](../docs/PORTFOLIO_PROGRESS_LOG.md)

## Контракт входа

- **Единица обработки**: `audio/audio.wav` (Segmenter) + `audio/segments.json` (contract `audio_segments_v1`).
- **Источник**: Segmenter генерирует `audio/audio.wav` и `audio/segments.json` в `frames_dir`, AudioProcessor читает их:
  - **Single-file mode**: через флаг `--frames-dir` (обязателен)
  - **Batch mode**: через флаг `--audio-input-dir` или `--audio-input-list` (обязателен для batch mode)
- **Требования (no‑fallback)**:
  - Если AudioProcessor включён в профиль как required, отсутствие `audio/audio.wav` или `audio/segments.json` → run должен падать на уровне CLI (`raise RuntimeError`).
  - Отсутствие обязательного family в `segments.json` для required extractor'а → fail-fast (`raise RuntimeError`).
  - Модели должны грузиться **только локально** через `dp_models` (no‑network policy).
- **Batch mode**: Каждый frames_dir должен содержать `audio/audio.wav` и `audio/segments.json`. Файлы без этих файлов пропускаются с предупреждением.

### Формат `audio/segments.json`

См. `docs/contracts/SEGMENTER_CONTRACT.md` (раздел 9.3).

Ключевые поля:
- `schema_version="audio_segments_v1"`
- `sample_rate`, `total_samples`, `audio_duration_sec`, `video_duration_sec`
- `families.<name>.segments[]` для каждого используемого extractor'а

**Стандартные families**:
- `primary`: короткие окна вокруг time‑anchors → `loudness_extractor`
- `clap`: короткие окна на нелинейной кривой → `clap_extractor`
- `tempo`: длинные sliding windows → `tempo_extractor`
- `asr`: длинные sliding windows → `asr_extractor`
- `diarization`: фиксированные окна → `speaker_diarization_extractor`
- `emotion`: перекрывающиеся окна → `emotion_diarization_extractor`
- `source_separation`: длинные окна → `source_separation_extractor`

Каждый сегмент содержит:
- `start_sec`, `end_sec`, `center_sec` (float)
- `start_sample`, `end_sample` (int, индексы в `audio/audio.wav`)

### Legacy Mode

Legacy mode (извлечение аудио из видео внутри AudioProcessor) **удалён**. AudioProcessor работает **только** по Segmenter contract:
- **Single-file mode**: `--frames-dir` обязателен
- **Batch mode**: `--audio-input-dir` или `--audio-input-list` обязательны

## Наблюдаемость оркестратора (профилирование по экстракторам)

Включение: `AP_ORCHESTRATOR_TELEMETRY=1`. События (RAM, при CUDA — `torch` GPU memory, wall-time) попадают в `_reports/scheduler_runtime_report.json`. Подробнее: [`docs/ORCHESTRATOR_TELEMETRY.md`](docs/ORCHESTRATOR_TELEMETRY.md).

## Контракт выхода (result_store)

AudioProcessor пишет **отдельные NPZ артефакты для каждого extractor'а**:

- `result_store/<platform_id>/<video_id>/<run_id>/<component_name>/<component_name>_features.npz`

и апдейтит:

- `result_store/<platform_id>/<video_id>/<run_id>/manifest.json`

### NPZ schema

Схема:
- Legacy (общая): `schema_version="audio_npz_v1"` (см. `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`).
- Audit v3 rollout: audited extractors постепенно переходят на **per-extractor schema_version** (`<extractor>_npz_v*`) + machine schemas в `DataProcessor/AudioProcessor/schemas/`.
- **Audited extractors (v3)**:
  - `clap_extractor_npz_v1` (clap_extractor)
  - `loudness_extractor_npz_v1` (loudness_extractor)
  - `asr_extractor_npz_v2` (asr_extractor)
  - `band_energy_extractor_npz_v1` (band_energy_extractor)
  - `chroma_extractor_npz_v1` (chroma_extractor)
  - `emotion_diarization_extractor_npz_v1` (emotion_diarization_extractor)

Минимальные ключи:
- `feature_names: object[str]` — имена фичей (scalars)
- `feature_values: float32[]` — значения фичей (scalars), выровнены с `feature_names`
- Дополнительные ключи по extractor'у (например, `embedding`, `embedding_sequence`, `tempo_estimates`, `segment_centers_sec`)
- `meta: object(dict)` — run identity + версии + статус + stage timings

Обязательные поля `meta`:
- `producer`, `producer_version`, `schema_version`, `created_at`
- `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- `status` ∈ {`ok`, `empty`, `error`}
- `empty_reason` (только если `status="empty"`, иначе null)
- `models_used[]`, `model_signature` (если используются модели)
- `device_used` (если применимо)
- `scheduler_knobs` (если применимо: `segment_parallelism`, `max_inflight`, `model_batch_size`)
- `stage_timings_ms` (обязательно): тайминги стадий обработки
- `timings_by_extractor` (обязательно): тайминги по extractor'ам

### Privacy / raw audio

По умолчанию AudioProcessor **не сохраняет raw audio** в артефактах (только фичи/статистики).

Raw audio может быть в `_tmp_audio/` для дебага, но не считается source-of-truth.

## Запуск (CLI)

### Single-file mode (одиночная обработка)

Standalone (с Segmenter contract):

```bash
python3 AudioProcessor/run_cli.py \
  --frames-dir /path/to/frames_dir \
  --rs-base /path/to/result_store \
  --platform-id youtube \
  --video-id <video_id> \
  --run-id <run_id> \
  --sampling-policy-version v1 \
  --dataprocessor-version unknown \
  --extractors clap,tempo,loudness
```

**Основные флаги**:
- `--extractors <csv>`: список extractors для запуска (default: `clap,tempo,loudness` — Tier-0 baseline)
- `--frames-dir <path>`: путь к `frames_dir` с `audio/audio.wav` и `audio/segments.json` (Segmenter contract) — **обязателен для single-file mode**
- `--device <cuda|cpu|auto>`: устройство для обработки (default: `auto`)
- `--segment-parallelism <N>`: параллелизм на уровне сегментов для CPU extractors (default: 1)
- `--max-inflight <N>`: максимальное количество сегментов в обработке одновременно (default: `segment_parallelism`)
- `--clap-batch-size <N>`: размер батча для CLAP inference (default: 1)
- `--no-strict-extractors`: graceful degradation вместо fail-fast при ошибках инициализации extractors (для дебага)

### Batch mode (Stage 5: батчевая обработка)

AudioProcessor поддерживает одновременную обработку нескольких аудио файлов с оптимизациями:

- **Двухуровневая параллельность**:
  - **Уровень 1**: Параллельная обработка нескольких видео через `ThreadPoolExecutor` (max_video_workers)
  - **Уровень 2**: Параллельная обработка сегментов внутри одного видео (существующая segment_parallelism)
- **GPU batching**: Батчинг сегментов из всех видео для ML-моделей (ASR, diarization, emotion, CLAP, source_separation)
- **CPU parallelism**: Параллелизация CPU extractors (spectral, quality, mfcc, mel, etc.)
- **Изоляция данных**: Каждый файл имеет свой `artifacts_dir` и `tmp_path`

**Пример 1: Batch mode из директории**

```bash
python3 AudioProcessor/run_cli.py \
  --audio-input-dir /path/to/frames_dirs \
  --rs-base /path/to/result_store \
  --platform-id youtube \
  --run-id <run_id> \
  --sampling-policy-version v1 \
  --dataprocessor-version unknown \
  --extractors clap,tempo,loudness,asr \
  --batch-max-workers 4 \
  --device cuda
```

**Пример 2: Batch mode из списка файлов**

```bash
# Создать список frames_dirs (по одному пути на строку)
echo "/path/to/frames_dir1" > frames_list.txt
echo "/path/to/frames_dir2" >> frames_list.txt
echo "/path/to/frames_dir3" >> frames_list.txt

python3 AudioProcessor/run_cli.py \
  --audio-input-list frames_list.txt \
  --rs-base /path/to/result_store \
  --platform-id youtube \
  --run-id <run_id> \
  --sampling-policy-version v1 \
  --dataprocessor-version unknown \
  --extractors clap,tempo,loudness,asr \
  --batch-max-workers 4 \
  --device cuda
```

**Batch processing флаги**:
- `--audio-input-dir <path>`: директория, содержащая несколько frames_dirs (каждый подкаталог должен содержать `audio/audio.wav` и `audio/segments.json`)
- `--audio-input-list <path>`: путь к текстовому файлу со списком frames_dir путей (по одному пути на строку)
- `--batch-max-workers <N>`: количество параллельных воркеров для обработки видео (null = auto, обычно `os.cpu_count()`)
- `--no-batch-gpu`: отключить GPU batching для batch processing
- `--no-batch-cpu-parallel`: отключить CPU параллелизм для batch processing
- `--batch-max-segments-per-gpu-batch <N>`: лимит размера батча для GPU extractors (null = без лимита)

**Примечания**:
- В batch mode `--frames-dir` не используется (используется `--audio-input-dir` или `--audio-input-list`)
- Каждый файл сохраняется в отдельную директорию: `result_store/<platform_id>/<video_id>/<run_id>/`
- Валидация NPZ файлов выполняется автоматически для каждого файла
- Ошибки в одном файле не валят весь batch (если extractor не marked required)

**Dependency Resolution**:
AudioProcessor автоматически разрешает зависимости между extractors:
- **Автоматическое добавление зависимостей**: если выбран `key`, автоматически добавляется `chroma` (для оптимизации через `shared_features`)
- **Автоматическое упорядочивание**: extractors выполняются в правильном порядке (зависимости перед зависимыми)
- **Валидация feature flags**: проверка зависимостей между feature flags внутри extractors (например, `enable_key_changes` требует `enable_time_series`)
- **Предупреждения и ошибки**: вывод предупреждений при отсутствии опциональных зависимостей, ошибок при отсутствии обязательных (в strict mode)

Примеры зависимостей:
- `key_extractor` → `chroma_extractor` (опционально, для оптимизации)
- `band_energy_extractor` → `spectral_extractor` (опционально, для оптимизации)
- `spectral_entropy_extractor` → `spectral_extractor` (опционально, для оптимизации)

**Дополнительные флаги**:
- `--asr-model-size <small|medium|large>`: размер модели Whisper для ASR (default: `small`)
- `--diarization-model-size <small|medium|large>`: размер модели для speaker diarization (default: `small`)
- `--emotion-model-size <small|medium|large>`: размер модели для emotion diarization (default: `small`)
- `--source-separation-model-size <small|medium|large>`: размер модели для source separation (default: `small`)
- `--speech-analysis-pitch`: включить pitch extraction в `speech_analysis_extractor`
- `--no-strict-extractors`: graceful degradation вместо fail-fast при ошибках инициализации extractors (для дебага)
- `--write-legacy-manifest`: писать legacy manifest.json (deprecated)

### Конфигурация через global_config.yaml

AudioProcessor поддерживает конфигурацию через единый `global_config.yaml`:

```yaml
processors:
  audio:
    enabled: true
    required: false
    device: "auto"  # auto|cpu|cuda
    
    # Scheduler knobs (legacy fallback, если нет индивидуальных настроек для extractors)
    scheduler:
      segment_parallelism: 1
      max_inflight: null  # null = segment_parallelism
      clap_batch_size: 1
    
    # Batch processing configuration (Stage 5)
    batch_processing:
      enabled: true  # Enable batch processing optimizations
      max_video_workers: null  # null = auto, typically os.cpu_count()
      enable_gpu_batching: true  # Use extract_batch() for GPU extractors
      max_segments_per_gpu_batch: null  # Limit batch size (null = no limit)
      enable_cpu_parallel: true  # Parallelize CPU extractors
    
    # Extractors configuration
    extractors:
      clap:
        enabled: true
        # Индивидуальные настройки параллелизма и батчинга для CLAP
        parallelism:
          preprocess_workers: 4  # Количество воркеров для параллельной загрузки/предобработки сегментов
          batch_size: 16  # Размер батча для GPU inference
        render:
          enable_render: true  # Генерировать render-context JSON
          enable_html_render: true  # Генерировать HTML debug страницу
      
      tempo:
        enabled: true
        # Индивидуальные настройки параллелизма для Tempo (CPU extractor)
        parallelism:
          segment_workers: 8  # Количество параллельных воркеров для обработки сегментов
          max_inflight: null  # Максимальное количество одновременно выполняемых задач (null = segment_workers)
        render:
          enable_render: true
          enable_html_render: true
      
      loudness:
        enabled: true
        # Индивидуальные настройки параллелизма для Loudness (CPU extractor)
        parallelism:
          segment_workers: 8
          max_inflight: null
        render:
          enable_render: true
          enable_html_render: true
      
      asr:
        enabled: true
        model_size: "small"  # small|medium|large
        # Decoding controls (Whisper DecodingOptions)
        decode:
          language: "auto"  # "auto" | "ru" | "en" | ... (Whisper language code)
          temperature: 0.0  # 0.0 = deterministic (beam search), >0 = sampling
          beam_size: 5  # Используется при temperature=0.0 (beam search)
          best_of: 1  # Используется при temperature>0.0 (sampling)
        # Output controls (privacy-sensitive, Audit v3)
        output:
          save_segment_text: false  # debug-only: persist per-segment raw text (privacy-sensitive, opt-in)
        # Индивидуальные настройки параллелизма для ASR (GPU extractor inprocess)
        parallelism:
          preprocess_workers: 4  # Количество воркеров для параллельной загрузки/предобработки сегментов
        # Feature flags (Audit v3: explicit gating)
        feature_flags:
          enable_token_sequences: true  # token_ids_by_segment (model_facing, required for TextProcessor)
          enable_token_counts: true  # token_counts per segment (analytics)
          enable_token_total: true  # token_total aggregate (analytics)
          enable_token_density: true  # token_density_per_sec (analytics)
          enable_speech_rate: true  # speech_rate_wpm (analytics)
          enable_lang_distribution: true  # lang_distribution (analytics, by lang_code)
          enable_segments_with_speech: true  # segments_with_speech count (analytics)
          enable_avg_segment_duration: true  # avg_segment_duration_sec (analytics)
          enable_token_variance: true  # token_variance (analytics)
        # Audit v3: schema_version=asr_extractor_npz_v1, strict token contract (shared_tokenizer_v1 only)
        render:
          enable_render: true  # Генерировать render-context JSON
          enable_html_render: true  # Генерировать HTML debug страницу
      
      speaker_diarization:
        enabled: true
        model_size: "small"  # small|large
        batch_size: null  # Размер батча для inference (None = auto, >100 segments → split)
        clustering_method: "agglomerative"  # agglomerative|kmeans|auto
        speaker_count_method: "heuristic"  # heuristic|silhouette|fixed
        silence_peak_threshold: 1e-3
        silence_rms_threshold: 1e-4
        # Индивидуальные настройки параллелизма для Speaker Diarization (GPU extractor in-process)
        parallelism:
          preprocess_workers: 4  # Количество воркеров для параллельной загрузки/предобработки сегментов
        feature_flags:
          enable_speaker_segments: false
          enable_speaker_embeddings: false
          enable_speaker_stats: false
          enable_speaker_durations: false
          enable_clustering_metrics: false
          enable_segment_embeddings: false
          disable_silence_detection: false
        render:
          enable_render: true  # Генерировать render-context JSON
          enable_html_render: true  # Генерировать HTML debug страницу
```

При использовании `global_config.yaml` параметры batch processing автоматически передаются в CLI через `config_parser.get_audio_cli_args()`.

## Models

Все ML‑модели загружаются **только через `dp_models` (ModelManager)**, без сетевых загрузок.

### GPU Models

1. **CLAP** (Contrastive Language-Audio Pre-training)
   - **Spec name**: `laion_clap` (ModelManager)
   - **Runtime**: `inprocess`
   - **Engine**: `torch`
   - **Precision**: `fp32`
   - **Device**: `cuda` (если доступно) или `cpu`
   - **Используется в**: `clap_extractor`

2. **Whisper** (ASR)
   - **Spec name**: `whisper_{size}_inprocess` (ModelManager)
   - **Runtime**: `inprocess` (PyTorch)
   - **Precision**: `fp16` (на CUDA) или `fp32` (на CPU)
   - **Device**: `cuda` (предпочтительно) или `cpu`
   - **Используется в**: `asr_extractor`

3. **Speaker Diarization**
   - **Spec name**: `speaker_diarization_{size}_inprocess` (ModelManager)
   - **Runtime**: `inprocess` (PyTorch)
   - **Engine**: `torch`
   - **Precision**: `fp16` (на CUDA) или `fp32` (на CPU)
   - **Device**: `cuda` (предпочтительно) или `cpu`
   - **Используется в**: `speaker_diarization_extractor`, `speech_analysis_extractor`

4. **Emotion Diarization**
   - **Spec name**: `emotion_diarization_{size}_triton` (ModelManager)
   - **Runtime**: `triton`
   - **Engine**: `onnx` или `tensorrt`
   - **Precision**: `fp16` или `fp32`
   - **Device**: `cuda`
   - **Используется в**: `emotion_diarization_extractor`

5. **Source Separation**
   - **Spec name**: `source_separation_{size}_triton` (ModelManager)
   - **Runtime**: `triton`
   - **Engine**: `onnx` или `tensorrt`
   - **Precision**: `fp16` или `fp32`
   - **Device**: `cuda`
   - **Используется в**: `source_separation_extractor`

### CPU Models

Все signal processing extractors (tempo, loudness, spectral, etc.) работают на CPU без ML‑моделей.

## Sampling Requirements

AudioProcessor **не генерирует сегменты сам** — Segmenter является единственным владельцем sampling.

Segmenter строит families в `audio/segments.json` используя **универсальную нелинейную кривую** (sampling curve):
- Параметры в `families.<name>.sampling_curve`: `type="ease_out_power"`, `k∈(0,1]`, `linear_until_sec`, `cap_duration_sec`
- На коротких видео можно близко к 1:1 (секунда→окно), на длинных рост замедляется и упирается в `max_windows`

Подробности см. `docs/contracts/SEGMENTER_CONTRACT.md` (раздел 9.3).

## Parallelization

### Внутренний параллелизм (single-file mode)

**CPU extractors** (loudness, tempo):
- Параллелизм на уровне сегментов через `segment_parallelism` и `max_inflight`
- Thread-safe: каждый сегмент обрабатывается независимо
- Контролируется через CLI аргументы `--segment-parallelism` и `--max-inflight`

**GPU extractors** (CLAP):
- Батчинг на уровне модели через `clap_batch_size`
- Контролируется через CLI аргумент `--clap-batch-size`
- `segment_parallelism=1` для CLAP (scheduler контролирует batching)

**Triton-backed extractors** (diarization, emotion, source separation):
- Triton сам управляет батчингом
- Extractors отправляют запросы последовательно, Triton батчит на своей стороне

**Whisper ASR (inprocess)**:
- выполняется локально (dp_models), без Triton
- декодинг идёт последовательно по сегментам (Whisper не поддерживает “настоящий” батчинг decode)

### Batch processing параллелизм (Stage 5)

**Двухуровневая параллельность**:
- **Уровень 1 (видео)**: Параллельная обработка нескольких видео через `ThreadPoolExecutor` (контролируется через `--batch-max-workers`)
- **Уровень 2 (сегменты)**: Параллельная обработка сегментов внутри одного видео (существующая `segment_parallelism`)

**GPU batching**:
- Сбор сегментов из всех видео для ML-моделей (ASR, diarization, emotion, CLAP, source_separation)
- Группировка в батчи по `max_segments_per_gpu_batch` (если задан)
- Распределение результатов обратно по видео
- Контролируется через `--enable-gpu-batching` (по умолчанию включено) или `--no-batch-gpu` для отключения

**CPU parallelism**:
- Параллелизация CPU extractors (spectral, quality, mfcc, mel, etc.) через `ThreadPoolExecutor`
- Контролируется через `--enable-cpu-parallel` (по умолчанию включено) или `--no-batch-cpu-parallel` для отключения

**Изоляция данных**:
- Каждый файл имеет свой `artifacts_dir` и `tmp_path`
- Артефакты (`*.npy`, временные файлы) пишутся внутрь per-run ResultStore и не конфликтуют между файлами
- Каждый файл сохраняется в отдельную директорию: `result_store/<platform_id>/<video_id>/<run_id>/`

### Внешний параллелизм

- Можно запускать несколько экземпляров AudioProcessor параллельно на разных видео (разные `run_id`)
- Требования к изоляции: разные `run_id`, разные `result_store` пути
- Ограничения: shared GPU ресурсы (нужно контролировать через scheduler)

### Комбинированный подход

- **Single-file mode**: Внутренний батчинг (CLAP: `batch_size=8`) + внешний запуск на разных GPU (по одному компоненту на GPU)
- **Batch mode**: Двухуровневая параллельность (видео + сегменты) + GPU batching для ML-моделей + CPU parallelism для signal processing

## Performance Characteristics

**Источник данных**: `docs/models_docs/resource_costs/audio_processor_<extractor>_costs_v1.json`

**Единица обработки**: `audio_segment` (окно аудио из `audio/segments.json`)

**Типичные значения (preset="default")**:

| Extractor | Latency per segment | CPU RAM peak | GPU VRAM peak | Notes |
|-----------|---------------------|--------------|---------------|-------|
| clap | ~50 ms | ~500 MB | ~2000 MB | GPU, batch_size=8 |
| tempo | ~100 ms | ~200 MB | 0 MB | CPU, librosa |
| loudness | ~20 ms | ~100 MB | 0 MB | CPU, signal processing |

**Для видео с N сегментами**: Total latency ≈ N × latency_per_segment (с учётом параллелизма)

**Полные данные**: см. `docs/models_docs/resource_costs/audio_processor_<extractor>_costs_v1.json`

## Quality Validation & Human-friendly Inspection

### Автоматическая оценка (sanity checks)

Для каждого extractor'а проверяются:
- Диапазоны значений разумны (no NaN там, где обязаны быть числа; нет inf; нет отрицательных там, где нельзя)
- Консистентность связных фичей (например, `*_present` ↔ значение)
- Статистические инварианты:
  - Tempo BPM ∈ [40, 220]
  - Loudness dBFS ∈ [-∞, 0]
  - Embedding norms > 0
- Для per-segment sequences: монотонность `segment_centers_sec`, согласованность размеров массивов

### Human-friendly визуализация

AudioProcessor генерирует **render-context JSON** для каждого extractor'а:

- `result_store/.../<component_name>/_render/render_context.json`

Этот JSON содержит:
- Timeline данные (сегменты с временными метками и значениями)
- Статистики (mean, std, min, max, distributions)
- Quality flags (warnings, confidence scores)

Render-context может быть использован:
- LLM для генерации текстовых описаний (см. `docs/contracts/LLM_RENDERING.md`)
- Frontend для построения графиков и визуализаций

**Примеры визуализации**:
- **clap_extractor**: Timeline с `segment_center_sec` (+ `segment_mask`) и embedding norms, t-SNE визуализация (опционально)
- **tempo_extractor**: Timeline с `windowed_bpm` по времени, распределение `tempo_estimates`, warnings
- **loudness_extractor**: Timeline с `segment_rms/segment_dbfs/segment_lufs` по времени, распределения

Опционально: HTML страница с интерактивными элементами (timeline, графики) в `_render/render.html`.

## Features

### Extractor Gating

Выбор extractors через `--extractors <csv>`:
- `clap,tempo,loudness` — Tier-0 baseline (default)
- `asr,speaker_diarization` — для speech analysis
- `speech_analysis` — bundle extractor (ASR + diarization + optional pitch)

### Feature Sets (planned)

В будущем будет поддержка feature sets для выбора подмножества фичей внутри extractor'а:
- `baseline`: только scalar фичи (без sequences/embeddings)
- `standard`: scalar + агрегаты (mean, std, norm)
- `full`: все фичи включая sequences/embeddings

### Default Extractor Set

По умолчанию запускаются Tier-0 baseline extractors:
- `clap_extractor` — семантические эмбеддинги CLAP
- `tempo_extractor` — оценка BPM и ритмические фичи
- `loudness_extractor` — метрики громкости (RMS, dBFS, LUFS)

## Архитектура проекта

AudioProcessor имеет модульную архитектуру, разделенную на несколько уровней:

### Структура директорий

```
AudioProcessor/
├── run_cli.py                    # Главная точка входа CLI (681 строка)
├── src/
│   ├── core/                     # Основные модули обработки
│   │   ├── main_processor.py     # MainProcessor - главный координатор extractors
│   │   ├── base_extractor.py     # Базовый класс для всех extractors
│   │   ├── audio_file_context.py # Контекст аудио файла для batch processing
│   │   ├── audio_utils.py        # Утилиты для работы с аудио
│   │   ├── dependency_resolver.py # Разрешение зависимостей между extractors
│   │   ├── renderer.py           # Генерация render-context для визуализации
│   │   │
│   │   ├── cli_args.py           # Парсинг аргументов командной строки
│   │   ├── config_hash.py        # Создание config_hash для идемпотентности
│   │   ├── model_resolver.py     # Разрешение метаданных моделей (ModelManager)
│   │   ├── segments_loader.py    # Загрузка и валидация segments.json
│   │   ├── processor_factory.py  # Фабрика для создания MainProcessor
│   │   │
│   │   ├── extractor_runner.py   # Запуск extractors и обработка результатов
│   │   ├── batch_processor.py    # Batch обработка нескольких файлов
│   │   ├── resource_monitor.py   # Мониторинг CPU/GPU ресурсов
│   │   │
│   │   ├── npz_saver.py          # Координация сохранения NPZ артефактов
│   │   └── npz_savers/           # Специфичные саверы для каждого extractor'а
│   │       ├── clap.py
│   │       ├── tempo.py
│   │       ├── loudness.py
│   │       ├── asr.py
│   │       ├── speaker_diarization.py
│   │       ├── emotion_diarization.py
│   │       ├── source_separation.py
│   │       ├── speech_analysis.py
│   │       ├── spectral.py
│   │       ├── quality.py
│   │       ├── mfcc.py
│   │       ├── mel.py
│   │       ├── onset.py
│   │       ├── chroma.py
│   │       ├── rhythmic.py
│   │       ├── key.py
│   │       ├── band_energy.py
│   │       ├── spectral_entropy.py
│   │       ├── hpss.py
│   │       └── voice_quality.py
│   │
│   ├── utils/                    # Утилиты общего назначения
│   │   ├── cli_utils.py          # Базовые утилиты (время, хеши, атомарные операции)
│   │   ├── progress.py           # Управление прогресс-баром (tqdm/JSON)
│   │   ├── retry.py              # Ретраи с экспоненциальной задержкой и OOM fallback
│   │   ├── meta_builder.py       # Построение метаданных
│   │   └── prof.py               # Профилирование
│   │
│   ├── extractors/               # Реализации extractors
│   │   ├── clap_extractor/
│   │   ├── tempo_extractor/
│   │   ├── loudness_extractor/
│   │   ├── asr_extractor/
│   │   ├── speaker_diarization_extractor/
│   │   ├── emotion_diarization_extractor/
│   │   ├── source_separation_extractor/
│   │   ├── speech_analysis_extractor/
│   │   ├── spectral_extractor/
│   │   ├── quality_extractor/
│   │   ├── mfcc_extractor/
│   │   ├── mel_extractor/
│   │   ├── onset_extractor/
│   │   ├── chroma_extractor/
│   │   ├── rhythmic_extractor/
│   │   ├── key_extractor/
│   │   ├── band_energy_extractor/
│   │   ├── spectral_entropy_extractor/
│   │   ├── hpss_extractor/
│   │   └── voice_quality_extractor/
│   │
│   ├── schemas/                  # Схемы данных
│   │   └── models.py             # Pydantic модели для валидации
│   │
│   └── api/                      # API endpoints (опционально)
│       ├── main.py
│       └── endpoints.py
│
└── docs/                         # Документация
    ├── contracts/                # Контракты и схемы
    └── models_docs/              # Документация по моделям
```

### Основные модули

#### `run_cli.py` (681 строка)
Главная точка входа CLI. Координирует работу всех модулей:
- Парсинг аргументов через `cli_args.py`
- Создание config_hash через `config_hash.py`
- Разрешение моделей через `model_resolver.py`
- Загрузка segments через `segments_loader.py`
- Создание MainProcessor через `processor_factory.py`
- Запуск extractors через `extractor_runner.py` или `batch_processor.py`
- Сохранение NPZ через `npz_saver.py`
- Мониторинг ресурсов через `resource_monitor.py`

#### `src/core/main_processor.py`
Главный координатор extractors. Управляет:
- Инициализацией extractors
- Запуском extractors (single-file и batch режимы)
- Управлением зависимостями между extractors
- Сбором результатов

#### `src/core/extractor_runner.py`
Запуск extractors для single-file режима:
- Последовательный запуск extractors с учетом зависимостей
- Обработка сегментов с параллелизмом
- Прогресс-репортинг
- Обработка ошибок и ретраев

#### `src/core/batch_processor.py`
Batch обработка нескольких файлов:
- Сбор списка frames_dirs
- Создание AudioFileContext для каждого файла
- Обработка результатов batch
- Статистика успешных/неуспешных файлов

#### `src/core/npz_saver.py` + `src/core/npz_savers/`
Система сохранения NPZ артефактов:
- `npz_saver.py` - координация сохранения, делегирование специфичным саверам
- `npz_savers/*.py` - специфичные саверы для каждого extractor'а (16 файлов)
- Каждый савер знает структуру данных своего extractor'а

#### `src/core/cli_args.py`
Парсинг аргументов командной строки:
- `create_argument_parser()` - создание ArgumentParser со всеми аргументами
- `parse_extractors_arg()` - парсинг списка extractors
- Аргументы организованы по extractor'ам (подфункции `_add_*_arguments()`)

#### `src/core/dependency_resolver.py`
Разрешение зависимостей между extractors:
- Автоматическое упорядочивание extractors
- Валидация feature flags
- Автоматическое добавление опциональных зависимостей

#### `src/core/resource_monitor.py`
Мониторинг ресурсов:
- Отслеживание CPU (RSS) и GPU (VRAM) памяти
- Сбор метрик в фоновом потоке
- Возврат пиковых значений

#### `src/utils/`
Утилиты общего назначения:
- `cli_utils.py` - базовые утилиты (время, хеши, атомарные операции)
- `progress.py` - управление прогресс-баром (tqdm с JSON fallback)
- `retry.py` - ретраи с экспоненциальной задержкой и OOM fallback для CLAP

#### `src/core/config_hash.py`
Создание config_hash для идемпотентности:
- Сбор всех параметров конфигурации
- Вычисление SHA256 хеша
- Используется для кеширования результатов

#### `src/core/model_resolver.py`
Разрешение метаданных моделей через ModelManager:
- CLAP, Whisper, Diarization, Emotion, Source Separation
- Сбор информации о версиях, весах, runtime, engine

#### `src/core/segments_loader.py`
Загрузка и валидация segments.json:
- Проверка наличия файлов
- Валидация schema_version
- Извлечение сегментов для каждого extractor'а
- Проверка обязательных families

#### `src/core/processor_factory.py`
Фабрика для создания MainProcessor:
- Создание MainProcessor с правильными параметрами из args
- Упрощает создание processor'а в run_cli.py

### Статистика рефакторинга

Проект был рефакторирован для улучшения читаемости и поддержки:
- **Исходный размер `run_cli.py`**: 4108 строк
- **Финальный размер `run_cli.py`**: 681 строка
- **Сокращение**: 3426 строк (~83%)
- **Создано модулей**: 24+ модулей

Все основные блоки логики вынесены в специализированные модули, что значительно упростило поддержку и тестирование.

## Orchestrator (run_cli.py)

**Error handling**:
- Строгий fail-fast: если extractor указан в `--extractors`, но не смог инициализироваться → `status="error"`, run завершается
- Опционально: `--no-strict-extractors` для graceful degradation (для дебага)
- Retry политики для Triton (503, 504, connection timeout) — 2 попытки с exponential backoff
- Автоматический fallback при OOM (уменьшение batch_size, максимум 2 попытки)
- **Batch mode**: Ошибки в одном файле не валят весь batch (если extractor не marked required). Каждый файл обрабатывается независимо, результаты сохраняются отдельно.

**Batch processing результаты**:
- Каждый файл сохраняется в отдельную директорию: `result_store/<platform_id>/<video_id>/<run_id>/`
- Валидация NPZ файлов выполняется автоматически для каждого файла
- Batch summary сохраняется в `per_extractor_report["batch_summary"]`:
  - `total_files`: общее количество файлов
  - `successful`: количество успешно обработанных файлов
  - `failed`: количество файлов с ошибками
  - `results`: список результатов для каждого файла

**Progress reporting**:
- Stage-based прогресс в stdout (JSON-lines):
  - `load_input` (5%)
  - `run_extractors` (10-80%, обновляется по мере завершения extractors)
  - `save_npz` (80%)
  - `validate_artifact` (85%, per component)
  - `update_manifest` (95%, per component)
  - `complete` (100%)
- Формат: `{"platform_id": "...", "video_id": "...", "run_id": "...", "component": "...", "stage_id": "...", "stage_name": "...", "progress_pct": N, "ts": "..."}`

**Stage timings**:
- Сохраняются в NPZ meta: `stage_timings_ms` (dict) и `timings_by_extractor` (dict)
- Используются для анализа производительности и оптимизации

**Models_used collection**:
- Автоматический сбор `models_used` из всех extractors через `dp_models`
- Сохраняется в NPZ meta для воспроизводимости

## Виртуальные Extractors

Некоторые extractors являются "виртуальными" — они публикуют результаты других extractors:

- `pitch` → реализуется внутри `speech_analysis_extractor`
- `asr` → может быть виртуальным (результат `speech_analysis_extractor`) или реальным (`asr_extractor`)
- `speaker_diarization` → может быть виртуальным (результат `speech_analysis_extractor`) или реальным (`speaker_diarization_extractor`)

Виртуальные extractors создают отдельные NPZ артефакты для упрощения downstream обработки.

## Batch Processing (Stage 5)

AudioProcessor поддерживает батчевую обработку нескольких аудио файлов с оптимизациями производительности. Подробности реализации см. в `docs/BATCH_PROCESSING_PLAN.md`.

### Основные возможности

- **Двухуровневая параллельность**: параллельная обработка нескольких видео + параллельная обработка сегментов внутри видео
- **GPU batching**: батчинг сегментов из всех видео для ML-моделей (ASR, diarization, emotion, CLAP, source_separation)
- **CPU parallelism**: параллелизация CPU extractors (spectral, quality, mfcc, mel, etc.)
- **Изоляция данных**: каждый файл имеет свой `artifacts_dir` и `tmp_path`
- **Валидация**: автоматическая валидация NPZ файлов для каждого файла

### Статус реализации

- ✅ Stage 0: Базовый каркас `run_batch()`, `extract_batch()`, `AudioFileContext`
- ✅ Stage 1: Изоляция артефактов
- ✅ Stage 2: GPU batching для CLAP extractor
- ✅ Stage 3: GPU batching для всех ML-моделей
- ✅ Stage 4: Двухуровневая параллельность и CPU parallelism
- ✅ Stage 5: CLI интеграция и production-ready batch processing

### Примеры использования

См. раздел "Batch mode (Stage 5: батчевая обработка)" выше.

## Ссылки

- **AudioProcessor Extractors Index**: `docs/MAIN_INDEX.md` — индекс всех extractors с кратким описанием и ссылками на README
- **Batch Processing Plan**: `docs/BATCH_PROCESSING_PLAN.md`
- **Audit criteria**: `docs/audit_v1/AP_AUDIT_CRITERIA.md`
- **Segmenter contract**: `docs/contracts/SEGMENTER_CONTRACT.md`
- **Artifacts and schemas**: `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`
- **Model system rules**: `docs/models_docs/MODEL_SYSTEM_RULES.md`
- **Extractor READMEs**: `src/extractors/<name>/README.md`
- **DataProcessor Main Index**: `../docs/MAIN_INDEX.md` — главный индекс всей документации DataProcessor
---

## Навигация

[AudioProcessor](docs/MAIN_INDEX.md) · [DataProcessor](../docs/MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
