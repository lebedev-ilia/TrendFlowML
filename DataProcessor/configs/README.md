# Global Configuration для DataProcessor

Единый глобальный конфиг для всех процессоров (AudioProcessor, TextProcessor, VisualProcessor) и их компонентов.

## Использование

### Запуск с глобальным конфигом

```bash
python3 DataProcessor/main.py \
  --video-path /path/to/video.mp4 \
  --global-config configs/global_config.yaml \
  --output /path/to/output \
  --rs-base /path/to/result_store
```

### Структура конфига

Глобальный конфиг (`global_config.yaml`) включает:

1. **Глобальные настройки** (`global`):
   - `platform_id`, `sampling_policy_version`, `dataprocessor_version`
   - Scheduler knobs (resource limits, parallelism)

2. **Настройки процессоров** (`processors`):
   - **audio**: все extractors с их параметрами и feature flags
   - **text**: все extractors с их параметрами и feature flags
   - **visual**: inline конфиг со всеми core_providers и modules (не ссылается на внешний файл)

### Пример конфига

```yaml
version: "1.0.0"

global:
  platform_id: "youtube"
  sampling_policy_version: "v1"
  dataprocessor_version: "unknown"

processors:
  audio:
    enabled: true
    required: false
    device: "auto"
    scheduler:
      segment_parallelism: 1
      max_inflight: null
      clap_batch_size: 1
    extractors:
      clap:
        enabled: true
      tempo:
        enabled: true
      key:
        enabled: true
        sample_rate: 22050
        feature_flags:
          enable_time_series: true
          enable_stability_metrics: true
  text:
    enabled: false
    required: false
    input_json: null
    feature_flags:
      enable_embeddings: false
    extractors:
      lexico_static_features:
        enabled: true
        feature_flags:
          enable_title: true
          enable_description: true
      # ... все остальные extractors
  visual:
    enabled: true
    required: true
    inline_config:
      core_providers:
        core_clip: true
        core_depth_midas: true
        # ... все core_providers
      modules:
        cut_detection: true
        scene_classification: true
        # ... все modules
      # ... конфигурации для каждого компонента
```

## Приоритет настроек

1. **Глобальный конфиг** (`--global-config`) — основной источник настроек
2. **Profile** (`--profile-path`) — может переопределить enabled/required для процессоров
3. **CLI аргументы** — используются как fallback, если глобальный конфиг не указан

## Валидация

Глобальный конфиг автоматически валидируется при загрузке:
- Проверка обязательных полей
- Проверка существования файлов (visual config)
- Проверка структуры конфига

## TextProcessor Extractors

Все extractors TextProcessor поддерживаются в конфиге:
- `lexico_static_features`, `tags_extractor`, `asr_text_proxy_audio_features` (базовые)
- `title_embedder`, `description_embedder`, `hashtag_embedder` (embedders, требуют `--enable-embeddings`)
- `transcript_chunk_embedder`, `comments_embedder` (embedders)
- `transcript_aggregator`, `comments_aggregator` (агрегаторы)
- `cosine_metrics_extractor`, `embedding_pair_topk_extractor`, `embedding_stats_extractor`
- `embedding_shift_indicator_extractor`, `embedding_source_id_extractor`
- `speaker_turn_embeddings_aggregator`, `title_to_hashtag_cosine_extractor`
- `topk_similar_titles_extractor`, `title_embedding_cluster_entropy_extractor`
- `semantic_cluster_extractor`, `qa_embedding_pairs_extractor`, `semantics_topics_keyphrases`

Каждый extractor может иметь:
- Параметры алгоритма (model_name, batch_size, device, etc.)
- Feature flags (enable_* флаги для включения дополнительных фичей)

## AudioProcessor Extractors

Все extractors AudioProcessor поддерживаются в конфиге:
- `clap`, `tempo`, `loudness` (Tier-0 baseline)
- `asr`, `speaker_diarization`, `emotion_diarization`, `source_separation`
- `speech_analysis`, `spectral`, `quality`, `mfcc`, `mel`, `onset`
- `chroma`, `rhythmic`, `voice_quality`, `hpss`
- `key`, `band_energy`, `spectral_entropy`

Каждый extractor может иметь:
- Параметры алгоритма (sample_rate, n_fft, model_size, etc.)
- Feature flags (enable_* флаги для включения дополнительных фичей)

## VisualProcessor Components

Все компоненты VisualProcessor описаны inline в глобальном конфиге:

**Core Providers**:
- `core_clip`, `core_depth_midas`, `core_optical_flow`, `core_object_detections`
- `core_face_landmarks`, `content_domain`, `franchise_recognition`, `ocr_extractor`
- `brand_semantics`, `car_semantics`, `face_identity`, `place_semantics`

**Modules**:
- `cut_detection`, `scene_classification`, `video_pacing`, `uniqueness`
- `shot_quality`, `story_structure`, `detalize_face`, `emotion_face`
- `behavioral`, `optical_flow`, `action_recognition`, `color_light`
- `frames_composition`, `high_level_semantic`, `micro_emotion`
- `similarity_metrics`, `text_scoring`

Каждый компонент имеет свою секцию конфигурации с параметрами.

### Batch Processing для VisualProcessor

VisualProcessor поддерживает batch processing для оптимизации обработки нескольких видео:

```yaml
visual:
  batch_processing:
    enabled: true  # Enable batch processing optimizations
    max_video_workers: 4  # Number of parallel workers for video-level processing (null = auto)
    enable_video_parallel: false  # Enable parallel processing of multiple videos (Stage 4+)
    max_frames_per_gpu_batch: 32  # Limit batch size for GPU components (null = no limit)
    enable_gpu_batching: true  # Enable GPU batching for frames from multiple videos (Stage 2+)
    enable_cpu_parallel: false  # Enable CPU parallelism for independent components (Stage 4+)
```

**Статус реализации**:
- ✅ Stage 0: Базовый каркас (`run_batch()`, `VideoContext`, `process_batch()`)
- ✅ Stage 1: Изоляция артефактов (per-video `rs_path`)
- ✅ Stage 2: GPU batching для `core_clip` (гибридный подход)
- 🚧 Stage 3-5: В разработке

**Особенности**:
- **GPU batching**: кадры из всех видео обрабатываются батчами для ML-моделей (например, `core_clip`)
- **Изоляция артефактов**: каждый видео имеет свой `rs_path` для предотвращения конфликтов
- **Гибридный подход**: батчинг кадров внутри одного видео и между видео

## Dependency Resolution

AudioProcessor автоматически разрешает зависимости между extractors:
- Если включен `key`, автоматически добавляется `chroma` (для оптимизации)
- Если включен `band_energy` или `spectral_entropy`, автоматически добавляется `spectral`
- Extractors выполняются в правильном порядке (зависимости перед зависимыми)

## Обратная совместимость

Если `--global-config` не указан, DataProcessor работает в режиме обратной совместимости:
- Используются CLI аргументы (`--audio-extractors`, `--audio-device`, etc.)
- Поведение идентично предыдущим версиям

