# Главный индекс документации dp_models

Этот документ служит единой точкой входа для навигации по всей документации и структуре модуля `dp_models` — единой системы управления моделями для DataProcessor. Каждый раздел содержит краткое описание компонентов и ссылки на файлы.

---

## Основные компоненты

### manager.py
**Краткое описание**: Реализация `ModelManager` — центрального менеджера моделей для всего DataProcessor. Обеспечивает единый интерфейс для загрузки моделей (in-process и Triton), валидацию локальных артефактов, управление провайдерами, вычисление weights_digest, поддержку offline режима (no-network enforcement), device policy (auto/cpu/cuda), thread-safe кэширование загруженных моделей. Используется всеми компонентами AudioProcessor, TextProcessor, VisualProcessor для доступа к ML-моделям.

**Расположение**: `dp_models/manager.py`

### catalog.py
**Краткое описание**: Реализация `ModelCatalog` — каталога декларативных спецификаций моделей. Загружает и индексирует YAML/JSON спецификации из `spec_catalog/`, обеспечивает поиск по `model_name` и `role`, валидацию уникальности имен, группировку по ролям. Используется ModelManager для резолвинга моделей по имени или роли.

**Расположение**: `dp_models/catalog.py`

### specs.py
**Краткое описание**: Определяет структуру `ModelSpec` и `LocalArtifact` — декларативных спецификаций моделей. Содержит парсинг YAML/JSON спецификаций, расширение переменных окружения (${VAR}), валидацию полей (model_name, model_version, role, runtime, engine, precision, device_policy, local_artifacts, weights_digest, runtime_params). Обеспечивает каноническое представление модели для reproducibility.

**Расположение**: `dp_models/specs.py`

### errors.py
**Краткое описание**: Определяет `ModelManagerError` — структурированное исключение для всех ошибок ModelManager. Содержит поля: message, error_code, details. Используется для fail-fast обработки ошибок (weights_missing, model_spec_invalid, models_root_missing, model_catalog_missing) с детальной диагностикой.

**Расположение**: `dp_models/errors.py`

### signatures.py
**Краткое описание**: Утилиты для работы с метаданными моделей и вычисления детерминированных подписей. Содержит функции `model_used()` для создания канонической записи в `models_used[]`, `compute_model_signature()` для вычисления SHA256 подписи списка моделей, `_canonicalize_models_used()` для нормализации. Обеспечивает стабильную сортировку и детерминированное хеширование для reproducibility. Интегрирован с `common/meta_builder.py`.

**Расположение**: `dp_models/signatures.py`

### digests.py
**Краткое описание**: Утилиты для вычисления SHA256 дайджестов файлов и директорий. Содержит функции `sha256_file()` для файлов, `sha256_dir()` для директорий с опциями игнорирования (.git, __pycache__, .tmp, .lock), ограничением на количество файлов (max_files=50k). Используется для вычисления `weights_digest` в ModelSpec и валидации целостности артефактов.

**Расположение**: `dp_models/digests.py`

### offline.py
**Краткое описание**: Утилиты для обеспечения offline режима (no-network enforcement). Содержит функции `enforce_offline_env()` для установки переменных окружения (HF_HOME, TORCH_HOME, DP_CLIP_WEIGHTS_DIR, SENTENCE_TRANSFORMERS_HOME), `pin_cache_env()` для привязки кэшей к models_root, `network_guard()` context manager для строгой блокировки сетевых соединений в тестах (monkeypatch socket.connect). Обеспечивает no-network policy для production runtime.

**Расположение**: `dp_models/offline.py`

### __init__.py
**Краткое описание**: Публичный API модуля `dp_models`. Экспортирует `ModelManager`, `ModelManagerError`, `get_global_model_manager()`. Используется компонентами для импорта ModelManager.

**Расположение**: `dp_models/__init__.py`

---

## Провайдеры моделей

### providers/base.py
**Краткое описание**: Базовый класс `ModelProvider` и `ResolvedModel` для всех провайдеров моделей. Определяет интерфейс провайдера (resolve(), load()), структуру ResolvedModel (model_name, model_version, weights_digest, runtime, engine, device, model_object, metadata), `ProviderRegistry` для регистрации провайдеров по engine. Все провайдеры наследуются от базового класса.

**Расположение**: `dp_models/providers/base.py`

### providers/sentence_transformers.py
**Краткое описание**: Провайдер для моделей SentenceTransformers (embeddings). Загружает модели из локальных директорий под `${SENTENCE_TRANSFORMERS_HOME}`, поддерживает no-network policy (запрет HuggingFace downloads), вычисляет weights_digest от директории модели. Используется TextProcessor extractors (title_embedder, description_embedder, comments_embedder, hashtag_embedder, transcript_chunk_embedder).

**Расположение**: `dp_models/providers/sentence_transformers.py`

### providers/torchscript.py
**Краткое описание**: Провайдер для TorchScript моделей (`.pt`/`.pth` файлы). Загружает модели через `torch.jit.load()`, поддерживает device policy (cpu/cuda), вычисляет weights_digest от файла. Используется для оптимизированных моделей (например, экспортированные через torch.jit.script).

**Расположение**: `dp_models/providers/torchscript.py`

### providers/torch_state_dict.py
**Краткое описание**: Провайдер для PyTorch checkpoints (state_dict). Загружает checkpoint через `torch.load()`, применяет через `load_state_dict()` к модели, созданной через factory функцию из `runtime_params.factory`. Поддерживает device policy, вычисляет weights_digest от checkpoint файла. Используется для моделей, требующих кастомной архитектуры (например, Places365, EmoNet).

**Расположение**: `dp_models/providers/torch_state_dict.py`

### providers/speechbrain.py
**Краткое описание**: Провайдер для SpeechBrain моделей (audio processing). Загружает модели через SpeechBrain API из локальных директорий, поддерживает no-network policy, вычисляет weights_digest. Используется AudioProcessor extractors (emotion_diarization, speaker_diarization).

**Расположение**: `dp_models/providers/speechbrain.py`

### providers/pyannote.py
**Краткое описание**: Провайдер для pyannote.audio моделей (speaker diarization). Загружает модели через pyannote API из локальных директорий, поддерживает no-network policy, вычисляет weights_digest. Используется AudioProcessor extractor (speaker_diarization_extractor).

**Расположение**: `dp_models/providers/pyannote.py`

### providers/triton_http.py
**Краткое описание**: Провайдер для моделей, развернутых в Triton Inference Server. Не загружает модель локально, а делает HTTP запросы к Triton API. Поддерживает dynamic batching, вычисляет weights_digest как "provided_by_deploy" (фиксируется в deployment). Используется для GPU моделей с высоким throughput (CLIP, MiDaS, RAFT, YOLO).

**Расположение**: `dp_models/providers/triton_http.py`

### providers/__init__.py
**Краткое описание**: Регистрация всех провайдеров в `ProviderRegistry`. Экспортирует все провайдеры и обеспечивает автоматическую регистрацию при импорте.

**Расположение**: `dp_models/providers/__init__.py`

---

## Фабрики моделей

### factories/audio.py
**Краткое описание**: Factory функции для создания архитектур аудио моделей из state_dict checkpoints. Содержит функции для создания моделей через `runtime_params.factory` в ModelSpec. Используется провайдером `torch_state_dict` для инициализации архитектуры перед загрузкой весов.

**Расположение**: `dp_models/factories/audio.py`

### factories/vision.py
**Краткое описание**: Factory функции для создания архитектур визуальных моделей из state_dict checkpoints. Содержит функции для Places365, EmoNet и других vision моделей. Используется провайдером `torch_state_dict` для инициализации архитектуры перед загрузкой весов.

**Расположение**: `dp_models/factories/vision.py`

### factories/__init__.py
**Краткое описание**: Экспорт factory функций для использования в провайдерах.

**Расположение**: `dp_models/factories/__init__.py`

---

## Каталог спецификаций моделей

### spec_catalog/README.md
**Краткое описание**: Документация каталога декларативных спецификаций моделей. Описывает правила (no-network, no-fallback, добавление новых моделей через YAML/JSON), структуру bundled layout, поддерживаемые in-process engines (sentence-transformers, torchscript, torch-state-dict). Содержит ссылки на спецификации по категориям (audio, text, vision).

**Полный документ**: `dp_models/spec_catalog/README.md`

### spec_catalog/audio/
**Краткое описание**: YAML спецификации для всех аудио моделей. Содержит спецификации для Whisper (small/medium/large, inprocess/triton), Source Separation (small/medium/large, inprocess/triton), Speaker Diarization (small/large, inprocess/triton), Emotion Diarization (small/large, inprocess/triton), LAION CLAP, Pyannote Speaker Diarization. Каждая спецификация определяет model_name, model_version, role, runtime, engine, local_artifacts, weights_digest, runtime_params.

**Расположение**: `dp_models/spec_catalog/audio/`

### spec_catalog/text/
**Краткое описание**: YAML спецификации для всех текстовых моделей. Содержит спецификации для embedding моделей (intfloat_multilingual-e5-large, sentence-transformers_all-MiniLM-L6-v2), семантических баз (semantic_clusters_v1, similar_titles_corpus_v1, topics_taxonomy_v1), shared tokenizer (shared_tokenizer_v1). Каждая спецификация определяет model_name, model_version, role, runtime, engine, local_artifacts, weights_digest.

**Расположение**: `dp_models/spec_catalog/text/`

### spec_catalog/vision/
**Краткое описание**: YAML спецификации для всех визуальных моделей. Содержит спецификации для CLIP (разные размеры, inprocess/triton), MiDaS depth (разные размеры, inprocess/triton), RAFT optical flow (inprocess/triton), YOLO object detection (inprocess/triton), Places365 (разные архитектуры), EmoNet, Action Recognition (SlowFast), Scene Classification, Face Landmarks (MediaPipe). Каждая спецификация определяет model_name, model_version, role, runtime, engine, local_artifacts, weights_digest, runtime_params.

**Расположение**: `dp_models/spec_catalog/vision/`

---

## Бандл моделей (DP_MODELS_ROOT)

### bundled_models/README.md
**Краткое описание**: Документация offline бандла моделей для TrendFlowML/DataProcessor. Описывает структуру DP_MODELS_ROOT (torch_cache для TORCH_HOME, hf_cache для HF_HOME/TRANSFORMERS_CACHE, clip_cache для DP_CLIP_WEIGHTS_DIR, visual/ для локальных артефактов vision моделей, audio/ для локальных артефактов audio моделей). Содержит инструкции по настройке DP_MODELS_ROOT для offline режима, созданию бандла через Colab и копированию на workers.

**Полный документ**: `dp_models/bundled_models/README.md`

### bundled_models/audio/
**Краткое описание**: Локальные артефакты аудио моделей. Содержит директории: `emotion_diarization/wavlm_large/` (SpeechBrain модель), `laion_clap/clap_ckpt.pt` (CLAP checkpoint), `pyannote_speaker_diarization/` (pyannote модель), `source_separation/large.pt` (Demucs checkpoint), `whisper/small.pt` (Whisper checkpoint). Все артефакты ссылаются из `spec_catalog/audio/*.yaml` через `local_artifacts`.

**Расположение**: `dp_models/bundled_models/audio/`

### bundled_models/text/
**Краткое описание**: Локальные артефакты текстовых моделей. Содержит директории: `embeddings/all-MiniLM-L6-v2/` (SentenceTransformers), `embeddings/intfloat_multilingual-e5-large/` (SentenceTransformers), `semantic_clusters_v1/` (семантические кластеры), `shared_tokenizer_v1/` (общий токенизатор), `similar_titles_v1/` (корпус похожих заголовков), `topics_v1/` (таксономия топиков). Все артефакты ссылаются из `spec_catalog/text/*.yaml` через `local_artifacts`.

**Расположение**: `dp_models/bundled_models/text/`

### bundled_models/visual/
**Краткое описание**: Локальные артефакты визуальных моделей. Содержит директории: `action_recognition/` (SlowFast), `clip/` (CLIP weights), `emonet/` (EmoNet), `places365/` (Places365 checkpoints), `yolo/` (YOLO weights). Все артефакты ссылаются из `spec_catalog/vision/*.yaml` через `local_artifacts`.

**Расположение**: `dp_models/bundled_models/visual/`

### bundled_models/semantics/
**Краткое описание**: Локальные базы для semantic heads (offline, no-network). Содержит директории: `brands/v1/` (логотипы брендов), `cars/v1/` (car makes/models/segments), `celebs/v1/` (celebrity identities), `content_domain/v1/` (game/anime/cartoon/etc.), `franchises/v1/` (games/anime/cartoons), `places/v1/` (places/landmarks). Каждая база содержит `manifest.json` с db_name, db_version, db_digest, files[], и основной индекс (brands.jsonl, cars.jsonl, etc.). Используется core_object_detections и semantic heads для идентификации объектов.

**Расположение**: `dp_models/bundled_models/semantics/`

**Полный документ**: `dp_models/bundled_models/semantics/README.md`

### bundled_models/hf_cache/
**Краткое описание**: HuggingFace кэш (HF_HOME/TRANSFORMERS_CACHE). Содержит загруженные модели из HuggingFace Hub (bert-base-uncased, facebook/bart-base, jonatasgrosman/wav2vec2-large-xlsr-53-russian, microsoft/wavlm-large, openai/clip-vit-base-patch32, pyannote/speaker-diarization-community-1, roberta-base, speechbrain/emotion-diarization-wavlm-large, speechbrain/spkrec-ecapa-voxceleb, Systran/faster-whisper-small). Используется SentenceTransformers, Transformers, SpeechBrain для загрузки моделей в offline режиме.

**Расположение**: `dp_models/bundled_models/hf_cache/`

### bundled_models/torch_cache/
**Краткое описание**: PyTorch кэш (TORCH_HOME). Содержит загруженные модели через torch.hub (MiDaS, RAFT, torchvision weights, etc.). Используется для кэширования torch.hub моделей и torchvision pretrained weights в offline режиме.

**Расположение**: `dp_models/bundled_models/torch_cache/`

### bundled_models/clip_cache/
**Краткое описание**: OpenAI CLIP кэш (DP_CLIP_WEIGHTS_DIR). Содержит загруженные CLIP weights (ViT-B-32.pt, ViT-L-14.pt). Используется core_clip для загрузки CLIP моделей в offline режиме.

**Расположение**: `dp_models/bundled_models/clip_cache/`

### bundled_models/cache/
**Краткое описание**: Дополнительный кэш для специфичных моделей. Содержит `core_clip_text_embeddings/` с предвычисленными текстовыми эмбеддингами для разных размеров (size_224, size_336). Используется для оптимизации CLIP inference.

**Расположение**: `dp_models/bundled_models/cache/`

### bundled_models/BUNDLE_REPORT.json
**Краткое описание**: JSON отчет о содержимом бандла моделей. Содержит инвентаризацию всех артефактов, их размеры, дайджесты, версии. Используется для валидации полноты бандла и отслеживания изменений.

**Расположение**: `dp_models/bundled_models/BUNDLE_REPORT.json`

---

## Дополнительные компоненты

### emonet/
**Краткое описание**: Реализация EmoNet — модели для детекции эмоций на лицах. Содержит архитектуру модели, pretrained weights, evaluation скрипты, демо скрипты (demo.py, demo_video.py), тесты (test.py), данные (pickles/, pretrained/), документацию (README.md, LICENSE.txt). Используется VisualProcessor modules (emotion_face, micro_emotion) для анализа эмоций.

**Расположение**: `dp_models/emonet/`

### visual/places365/
**Краткое описание**: Утилиты для Places365 — модели классификации сцен. Содержит `categories_places365.txt` с списком категорий Places365. Используется VisualProcessor core provider (core_places365) для классификации сцен.

**Расположение**: `dp_models/visual/places365/`

---

## Использование компонентами DataProcessor

### AudioProcessor Extractors
**Краткое описание**: AudioProcessor extractors используют ModelManager для загрузки аудио моделей. Основные примеры: `asr_extractor` (Whisper через ModelManager), `speaker_diarization_extractor` (pyannote.audio), `emotion_diarization_extractor` (SpeechBrain), `source_separation_extractor` (Demucs), `clap_extractor` (LAION CLAP). Все extractors получают ModelManager через `get_global_model_manager()`, резолвят модели по `model_name` или `role`, используют `ResolvedModel.handle` для inference. Поддерживают offline режим (no-network), фиксируют `models_used[]` в артефактах.

**Примеры использования**:
- `AudioProcessor/src/extractors/asr_extractor/main.py` — Whisper ASR
- `AudioProcessor/src/extractors/speaker_diarization_extractor/main.py` — Speaker diarization
- `AudioProcessor/src/extractors/emotion_diarization_extractor/main.py` — Emotion diarization
- `AudioProcessor/src/extractors/source_separation_extractor/main.py` — Source separation
- `AudioProcessor/src/extractors/clap_extractor/__init__.py` — CLAP embeddings

**Расположение**: `AudioProcessor/src/extractors/*/main.py`

### TextProcessor Extractors
**Краткое описание**: TextProcessor extractors используют ModelManager для загрузки embedding моделей и семантических баз. Основные примеры: `title_embedder`, `description_embedder`, `comments_embedder`, `hashtag_embedder`, `transcript_chunk_embedder` (SentenceTransformers через ModelManager), `semantic_cluster_extractor` (семантические кластеры), `topics_extractor` (таксономия топиков). Используют `get_model_with_meta()` helper из `model_registry.py` для упрощенного доступа к моделям, поддерживают батчинг, кэширование, GPU/CPU device policy.

**Примеры использования**:
- `TextProcessor/src/extractors/title_embedder/main.py` — Title embeddings
- `TextProcessor/src/extractors/transcript_chunk_embedder/main.py` — Transcript embeddings
- `TextProcessor/src/extractors/semantic_cluster_extractor/main.py` — Semantic clusters
- `TextProcessor/src/core/model_registry.py` — Helper для доступа к моделям

**Расположение**: `TextProcessor/src/extractors/*/main.py`

### VisualProcessor Core Providers
**Краткое описание**: VisualProcessor core providers используют ModelManager для загрузки vision моделей. Основные примеры: `core_clip` (CLIP embeddings, inprocess/triton), `core_depth_midas` (MiDaS depth, inprocess/triton), `core_optical_flow` (RAFT optical flow, inprocess/triton), `core_object_detections` (YOLO, inprocess/triton), `core_face_landmarks` (MediaPipe). Поддерживают оба runtime (inprocess и triton), используют `get_global_model_manager()` для резолвинга моделей, фиксируют `models_used[]` в артефактах.

**Примеры использования**:
- `VisualProcessor/core/model_process/core_clip/main.py` — CLIP embeddings
- `VisualProcessor/core/model_process/core_depth_midas/main.py` — MiDaS depth
- `VisualProcessor/core/model_process/core_optical_flow/main.py` — RAFT optical flow
- `VisualProcessor/core/model_process/core_object_detections/main.py` — YOLO detections

**Расположение**: `VisualProcessor/core/model_process/*/main.py`

### VisualProcessor Modules
**Краткое описание**: VisualProcessor modules используют ModelManager для загрузки специализированных моделей. Основные примеры: `action_recognition` (SlowFast), `scene_classification` (Places365), `emotion_face` (EmoNet), `cut_detection` (использует core providers), `story_structure` (использует core providers). Модули могут использовать как прямые модели через ModelManager, так и результаты core providers (downstream dependencies).

**Примеры использования**:
- `VisualProcessor/modules/action_recognition/action_recognition_slowfast.py` — Action recognition
- `VisualProcessor/modules/scene_classification/scene_classification.py` — Scene classification
- `VisualProcessor/modules/emotion_face/core/video_processor.py` — Emotion detection
- `VisualProcessor/modules/cut_detection/cut_detection.py` — Cut detection

**Расположение**: `VisualProcessor/modules/*/main.py`

### Core Identity Modules
**Краткое описание**: Semantic heads в `core_identity используют семантические базы из `bundled_models/semantics/`. Модули: `brand_semantics`, `car_semantics`, `face_identity`, `place_semantics`, `content_domain`, `franchise_recognition`. Используют ModelManager для загрузки баз данных (manifest.json, индексы), фиксируют `db_digest` в метаданных для reproducibility.

**Расположение**: `VisualProcessor/core/model_process/core_identity/*/main.py`

---

## API и паттерны использования

### get_global_model_manager()
**Краткое описание**: Глобальный singleton ModelManager, доступный через `from dp_models import get_global_model_manager`. Инициализируется один раз при первом вызове, использует `DP_MODELS_ROOT` из окружения, автоматически применяет offline enforcement. Используется всеми компонентами для единообразного доступа к моделям.

**Пример использования**:
```python
from dp_models import get_global_model_manager

mm = get_global_model_manager()
resolved = mm.get(model_name="whisper_small_inprocess")
model = resolved.handle
```

**Расположение**: `dp_models/manager.py`

### ModelManager.get()
**Краткое описание**: Основной метод для получения загруженной модели. Принимает `model_name` или `role` (с опциональным `preferred_name`), возвращает `ResolvedModel` с handle, metadata, models_used_entry. Поддерживает LRU кэширование, автоматический device selection, weights_digest вычисление.

**Сигнатура**: `get(*, model_name: Optional[str] = None, role: Optional[str] = None, preferred_name: Optional[str] = None) -> ResolvedModel`

**Расположение**: `dp_models/manager.py`

### ModelManager.get_spec()
**Краткое описание**: Метод для получения ModelSpec без загрузки модели. Используется для валидации наличия модели, получения метаданных, проверки артефактов. Возвращает `ModelSpec` с полной информацией о модели (model_name, model_version, role, runtime, engine, local_artifacts, weights_digest).

**Сигнатура**: `get_spec(*, model_name: Optional[str] = None, role: Optional[str] = None, preferred_name: Optional[str] = None) -> ModelSpec`

**Расположение**: `dp_models/manager.py`

### ResolvedModel
**Краткое описание**: Результат резолвинга модели через ModelManager. Содержит: `spec` (ModelSpec), `device`, `precision`, `runtime`, `engine`, `weights_digest`, `resolved_artifacts` (mapping paths), `handle` (загруженная модель), `models_used_entry` (каноническая запись для артефактов). Используется компонентами для доступа к модели и фиксации метаданных.

**Расположение**: `dp_models/providers/base.py`

---

## Связанная документация

### DataProcessor/docs/models_docs/
**Краткое описание**: Документация по моделям и ML-системе DataProcessor. Содержит MODEL_MANAGER_PLAN.md (план реализации ModelManager), MODEL_MANAGER_Q.md (Q&A по ModelManager), MODEL_INVENTORY.md (инвентаризация моделей), BASELINE_MODELS.md (baseline набор моделей), GPU_VS_CPU_PERFORMANCE.md (анализ производительности), EMBEDDING_UNIFICATION_STRATEGY.md (стратегия унификации эмбеддингов), SEMANTIC_HEADS_CONTRACTS_QA.md (контракты semantic heads), SCHEMA_SEMANTIC_HEADS_NPZ.md (схемы NPZ), SEMANTIC_BASES_BUILD_GUIDE.md (гайд по сборке баз), и другие документы.

**Полный документ**: [DataProcessor/docs/models_docs/README.md](../docs/models_docs/README.md)

### DataProcessor/common/meta_builder.py
**Краткое описание**: Общие утилиты для работы с метаданными моделей, используемые во всех процессорах. Содержит функции для канонизации списка используемых моделей (`model_used()`), вычисления детерминированной подписи моделей (`compute_model_signature()`), применения метаданных моделей к мета-словарю (`apply_models_meta()`). Интегрирован с `dp_models/signatures.py` для единообразия.

**Расположение**: `DataProcessor/common/meta_builder.py`

### AudioProcessor/README.md
**Краткое описание**: Основная документация AudioProcessor. Описывает использование ModelManager в extractors, контракты моделей, offline режим, batch processing. Содержит примеры использования ModelManager для загрузки Whisper, CLAP, Speaker Diarization моделей.

**Полный документ**: [AudioProcessor/README.md](../AudioProcessor/README.md)

### AudioProcessor/docs/MAIN_INDEX.md
**Краткое описание**: Индекс документации всех extractors AudioProcessor. Содержит краткие описания всех 21 extractor'а, включая информацию о моделях, используемых через ModelManager.

**Полный документ**: [AudioProcessor/docs/MAIN_INDEX.md](../AudioProcessor/docs/MAIN_INDEX.md)

---

