# Главный индекс скриптов DataProcessor

Этот документ служит единой точкой входа для навигации по всем скриптам и утилитам проекта. Каждый раздел содержит краткое описание скриптов и их назначение.

**Главный индекс DataProcessor**: [../docs/MAIN_INDEX.md](../docs/MAIN_INDEX.md) · **Vault root**: [../../docs/MAIN_INDEX.md](../../docs/MAIN_INDEX.md)

Старт нормализации для портфолио: [../docs/PORTFOLIO_NORMALIZATION_PLAN.md](../docs/PORTFOLIO_NORMALIZATION_PLAN.md)

---

## Утилиты и тесты

### dp_models_selftest.py
**Краткое описание**: Набор unit-тестов для проверки функциональности ModelManager и связанных утилит. Содержит тесты для SHA256 digest вычислений (стабильность), вычисления model signature (канонический порядок моделей), проверки безопасности путей ModelManager (защита от path traversal атак). Используется для валидации корректности работы dp_models системы перед использованием в production.

**Расположение**: `scripts/dp_models_selftest.py`

### venv_doctor.py
**Краткое описание**: Диагностический скрипт для проверки окружения разработки. Проверяет наличие необходимых бинарных файлов (ffmpeg, ffprobe), корректность настройки виртуального окружения Python, наличие обязательных импортов для работы DataProcessor. Выводит предупреждения и ошибки с подсказками по установке недостающих зависимостей. Используется для быстрой диагностики проблем окружения перед запуском обработки.

**Расположение**: `scripts/venv_doctor.py`

### storage_smoke_test.py
**Краткое описание**: Smoke test для проверки работы storage адаптера (FileSystem или S3/MinIO). Выполняет базовые операции: запись файла, чтение, проверка существования, листинг директории. Поддерживает оба режима через переменные окружения (`TREND_STORAGE_BACKEND=fs` или `s3`). Используется для валидации конфигурации хранилища перед запуском production пайплайна.

**Расположение**: `scripts/storage_smoke_test.py`

---

## Загрузка и сохранение моделей

### download_whisper_models.py
**Краткое описание**: Скрипт для загрузки и сохранения моделей Whisper (ASR) в DP_MODELS_ROOT. Поддерживает загрузку моделей разных размеров (tiny, base, small, medium, large, large-v2, large-v3). Сохраняет модели в каноническую структуру `dp_models/bundled_models/audio/whisper/<size>.pt`. Используется для первоначальной настройки окружения и подготовки offline-кэша моделей Whisper.

**Расположение**: `scripts/download_whisper_models.py`

### download_emotion_diarization_models.py
**Краткое описание**: Скрипт для загрузки моделей emotion diarization в DP_MODELS_ROOT. Загружает предобученные модели для определения эмоций в аудио из HuggingFace или других источников. Сохраняет модели в структурированном виде для использования AudioProcessor. Используется для подготовки моделей emotion_diarization extractor.

**Расположение**: `scripts/download_emotion_diarization_models.py`

### prepare_hf_cache.sh
**Краткое описание**: Подготовка HuggingFace cache для emotion_diarization (WavLM). Добавляет недостающий `preprocessor_config.json` в snapshots microsoft/wavlm-large при неполной загрузке. Проверяет bundled_models/hf_cache/hub и ~/.cache/huggingface/hub. Запускать перед первым smoke-тестом с emotion_diarization.

**Расположение**: `scripts/prepare_hf_cache.sh`

### hf_artifacts_sync.py
**Краткое описание**: Универсальный двунаправленный sync артефактов между локальным репозиторием и HuggingFace Hub по манифесту `configs/hf_artifacts_manifest.json`. Поддерживает режимы `upload` и `download`, dry-run, создание репозитория, а также восстановление файлов в исходные пути внутри проекта.

**Расположение**: `scripts/hf_artifacts_sync.py`

### hf_upload_all.sh
**Краткое описание**: Удобная обертка для массовой выгрузки артефактов в HuggingFace Hub согласно манифесту. Автоматически вычисляет корень репозитория и передает параметры в `hf_artifacts_sync.py upload`.

**Расположение**: `scripts/hf_upload_all.sh`

### hf_download_all.sh
**Краткое описание**: Удобная обертка для массовой загрузки артефактов из HuggingFace Hub согласно манифесту. Восстанавливает данные по тем же путям в репозитории через `hf_artifacts_sync.py download`.

**Расположение**: `scripts/hf_download_all.sh`

### download_source_separation_models.py
**Краткое описание**: Скрипт для загрузки моделей source separation (разделение источников звука) в DP_MODELS_ROOT. Загружает модели для извлечения вокала, инструментов и других источников из аудио. Сохраняет модели для использования source_separation extractor в AudioProcessor.

**Расположение**: `scripts/download_source_separation_models.py`

### save_sentence_transformer_model.py
**Краткое описание**: Скрипт для сохранения моделей sentence transformers (эмбеддинги текста) в DP_MODELS_ROOT. Загружает модели из HuggingFace (например, multilingual-e5-large) и сохраняет их в канонической структуре для использования TextProcessor. Обеспечивает offline-доступ к моделям эмбеддингов.

**Расположение**: `scripts/save_sentence_transformer_model.py`

### save_pyannote_diarization_model.py
**Краткое описание**: Скрипт для сохранения моделей pyannote diarization (speaker diarization) в DP_MODELS_ROOT. Загружает предобученные модели pyannote.audio для определения спикеров в аудио и сохраняет их для использования speaker_diarization extractor в AudioProcessor.

**Расположение**: `scripts/save_pyannote_diarization_model.py`

### fix_source_separation_model.py
**Краткое описание**: Утилита для исправления проблем с моделями source separation. Исправляет несовместимости версий, проблемы с форматом файлов или структурой моделей. Используется для восстановления работоспособности моделей после обновлений зависимостей или миграций.

**Расположение**: `scripts/fix_source_separation_model.py`

---

## Baseline демо и проверка качества

### baseline/demo_clap_extractor_quality.py
**Краткое описание**: Демо-скрипт для визуализации качества работы clap_extractor. Генерирует HTML отчет с timeline embedding norms, cosine similarity между сегментами, валидацией NPZ артефактов. Принимает путь к result_store run'а, анализирует clap_extractor артефакты, выводит статистику и графики для оценки качества извлечения CLAP эмбеддингов.

**Расположение**: `scripts/baseline/demo_clap_extractor_quality.py`

### baseline/demo_tempo_extractor_quality.py
**Краткое описание**: Демо-скрипт для визуализации качества работы tempo_extractor. Генерирует HTML отчет с анализом извлеченных BPM значений, валидацией артефактов, статистикой и графиками. Используется для проверки корректности определения темпа музыки в аудио.

**Расположение**: `scripts/baseline/demo_tempo_extractor_quality.py`

### baseline/demo_loudness_extractor_quality.py
**Краткое описание**: Демо-скрипт для визуализации качества работы loudness_extractor. Генерирует HTML отчет с анализом извлеченных значений громкости (LUFS), timeline графиками, валидацией NPZ артефактов. Используется для проверки корректности измерения громкости аудио.

**Расположение**: `scripts/baseline/demo_loudness_extractor_quality.py`

### baseline/demo_cut_detection_quality.py
**Краткое описание**: Демо-скрипт для визуализации качества работы cut_detection модуля. Генерирует HTML отчет с визуализацией обнаруженных срезов, timeline с метками переходов, статистикой переходов, сравнением с ground truth (если доступно). Используется для оценки качества детекции смены сцен и кадров.

**Расположение**: `scripts/baseline/demo_cut_detection_quality.py`

### baseline/demo_scene_classification_quality.py
**Краткое описание**: Демо-скрипт для визуализации качества работы scene_classification модуля. Генерирует HTML отчет с анализом классификации сцен, распределением классов, timeline с метками сцен, метриками точности. Используется для оценки качества классификации сцен в видео.

**Расположение**: `scripts/baseline/demo_scene_classification_quality.py`

### baseline/demo_shot_quality_quality.py
**Краткое описание**: Демо-скрипт для визуализации качества работы shot_quality модуля. Генерирует HTML отчет с анализом оценок качества кадров, распределением scores, timeline графиками, корреляцией с другими метриками. Используется для оценки качества определения технического качества съемки.

**Расположение**: `scripts/baseline/demo_shot_quality_quality.py`

### baseline/demo_similarity_metrics_quality.py
**Краткое описание**: Демо-скрипт для визуализации качества работы similarity_metrics модуля. Генерирует HTML отчет с анализом метрик схожести между кадрами/сегментами, heatmap матриц схожести, timeline графиками. Используется для оценки качества вычисления метрик схожести контента.

**Расположение**: `scripts/baseline/demo_similarity_metrics_quality.py`

### baseline/demo_uniqueness_quality.py
**Краткое описание**: Демо-скрипт для визуализации качества работы uniqueness модуля. Генерирует HTML отчет с анализом метрик уникальности контента, распределением scores, timeline графиками, сравнением с референсными значениями. Используется для оценки качества определения уникальности визуального контента.

**Расположение**: `scripts/baseline/demo_uniqueness_quality.py`

### baseline/demo_video_pacing_quality.py
**Краткое описание**: Демо-скрипт для визуализации качества работы video_pacing модуля. Генерирует HTML отчет с анализом темпа видео, метриками pacing, timeline графиками изменений темпа, корреляцией с другими метриками. Используется для оценки качества определения темпа и ритма видео.

**Расположение**: `scripts/baseline/demo_video_pacing_quality.py`

### baseline/build_similarity_reference_pack.py
**Краткое описание**: Скрипт для построения референсного пакета данных для тестирования similarity_metrics. Собирает набор референсных видео/кадров с известными метриками схожести, создает ground truth датасет для валидации качества работы similarity extractors. Используется для подготовки тестовых данных и бенчмарков.

**Расположение**: `scripts/baseline/build_similarity_reference_pack.py`

---

## Оптимизация моделей (ONNX экспорт)

### model_opt/bootstrap_models_root.py
**Краткое описание**: One-time bootstrap скрипт для загрузки всех необходимых baseline моделей в DP_MODELS_ROOT. Позволяет выполнить сетевые запросы один раз при настройке, заполняет DP_MODELS_ROOT закэшированными артефактами, затем система работает полностью offline. Загружает torch_cache, hf_cache, visual/places365, visual/clip и другие необходимые модели. Используется для первоначальной настройки окружения разработки.

**Расположение**: `scripts/model_opt/bootstrap_models_root.py`

### model_opt/export_openai_clip_onnx.py
**Краткое описание**: Скрипт для экспорта OpenAI CLIP (image encoder + text encoder) в ONNX формат для Triton Inference Server. Поддерживает fixed-shape (baseline) и dynamic batching режимы. Экспортирует image encoder (float32 [1,3,S,S] -> float32 [1,D]) и text encoder (int64 [1,77] -> float32 [1,77,D]). Использует DP_MODELS_ROOT для offline-режима. Используется для подготовки CLIP моделей для Triton deployment.

**Расположение**: `scripts/model_opt/export_openai_clip_onnx.py`

### model_opt/export_midas_onnx.py
**Краткое описание**: Скрипт для экспорта MiDaS depth estimation модели в ONNX формат для Triton. Экспортирует модель оценки глубины с поддержкой различных входных разрешений. Поддерживает fixed-shape и dynamic batching. Используется для подготовки core_depth_midas модели для GPU inference через Triton.

**Расположение**: `scripts/model_opt/export_midas_onnx.py`

### model_opt/export_raft_onnx.py
**Краткое описание**: Скрипт для экспорта RAFT optical flow модели в ONNX формат для Triton. Экспортирует модель вычисления оптического потока с поддержкой двух входных кадров (prev_frame + cur_frame). Поддерживает fixed-shape и dynamic batching. Используется для подготовки core_optical_flow модели для GPU inference через Triton.

**Расположение**: `scripts/model_opt/export_raft_onnx.py`

### model_opt/export_ultralytics_yolo11_onnx.py
**Краткое описание**: Скрипт для экспорта Ultralytics YOLO11 детектора объектов в ONNX формат для Triton. Экспортирует модель детекции объектов с поддержкой различных входных разрешений (640x640, 1280x1280 и др.). Поддерживает fixed-shape и dynamic batching. Используется для подготовки core_object_detections модели для GPU inference через Triton.

**Расположение**: `scripts/model_opt/export_ultralytics_yolo11_onnx.py`

### model_opt/export_places365_onnx.py
**Краткое описание**: Скрипт для экспорта Places365 scene classification модели в ONNX формат для Triton. Экспортирует модель классификации сцен с поддержкой 365 классов мест. Поддерживает fixed-shape и dynamic batching. Используется для подготовки scene classification модели для GPU inference через Triton.

**Расположение**: `scripts/model_opt/export_places365_onnx.py`

### model_opt/export_mediapipe_to_onnx.py
**Краткое описание**: Скрипт для экспорта MediaPipe моделей (face landmarks, pose и др.) в ONNX формат. Экспортирует MediaPipe графы в ONNX для использования в Triton или других inference engines. Используется для подготовки MediaPipe-based моделей для GPU deployment.

**Расположение**: `scripts/model_opt/export_mediapipe_to_onnx.py`

### model_opt/patch_onnx_dynamic_batch.py
**Краткое описание**: Утилита для патчинга ONNX моделей для поддержки dynamic batching. Модифицирует ONNX граф, добавляя динамические размеры батча, обновляет input/output shapes. Используется для конвертации fixed-shape ONNX моделей в dynamic batching версии для Triton.

**Расположение**: `scripts/model_opt/patch_onnx_dynamic_batch.py`

### model_opt/quantize_onnx_dynamic.py
**Краткое описание**: Скрипт для квантования ONNX моделей с dynamic batching. Применяет quantization (INT8, FP16) к ONNX моделям для уменьшения размера и ускорения inference. Сохраняет квантованные модели с сохранением динамических размеров батча. Используется для оптимизации моделей для production deployment.

**Расположение**: `scripts/model_opt/quantize_onnx_dynamic.py`

### model_opt/onnx_opt.py
**Краткое описание**: Утилита для оптимизации ONNX моделей. Применяет различные оптимизации (graph optimization, constant folding, operator fusion и др.) для уменьшения размера модели и ускорения inference. Используется для финальной оптимизации ONNX моделей перед deployment.

**Расположение**: `scripts/model_opt/onnx_opt.py`

### model_opt/info.py
**Краткое описание**: Утилита для инспекции ONNX моделей. Выводит информацию о структуре модели: входы/выходы, shapes, типы данных, opset версии, список операторов. Используется для отладки и проверки корректности экспортированных ONNX моделей.

**Расположение**: `scripts/model_opt/info.py`

---

## Preflight проверки

### preflight/check_semantic_bases.py
**Краткое описание**: Preflight валидатор для проверки наличия необходимых offline баз данных и галерей перед запуском пайплайна. Читает VisualProcessor/config.yaml, определяет включенные core providers, валидирует их требуемые входы. Проверяет существование db package директорий, наличие обязательных файлов, консистентность ID с gallery_index.json, наличие требуемых model specs в конфиге. Выполняет fail-fast проверку перед запуском обработки. Используется для валидации окружения перед production запуском.

**Расположение**: `scripts/preflight/check_semantic_bases.py`

---

## Семантические кластеры

### sem_clust_v1/build_semantic_clusters_v1.py
**Краткое описание**: Скрипт для построения semantic_clusters_v1 — набора файлов для классификации эмбеддингов по семантическим кластерам. Принимает файл с эмбеддингами (.npy), применяет PCA для снижения размерности, выполняет кластеризацию (K-means или другой алгоритм), генерирует выходные файлы: pca.npy (PCA матрица), centroids.npy (центроиды кластеров), clusters.jsonl (словарь кластеров). Используется для построения офлайн-базы семантических кластеров для классификации контента.

**Расположение**: `scripts/sem_clust_v1/build_semantic_clusters_v1.py`

### sem_clust_v1/BUILD_SEMANTIC_CLUSTERS.md
**Краткое описание**: Документация по процессу создания semantic_clusters_v1. Содержит описание требований (эмбеддинги, Python пакеты), инструкции по подготовке эмбеддингов, шаги запуска скрипта, описание выходных файлов, примеры использования. Используется как руководство для построения семантических кластеров.

**Полный документ**: `scripts/sem_clust_v1/BUILD_SEMANTIC_CLUSTERS.md`

### sem_clust_v1/build_similar_titles_corpus_v1.py
**Краткое описание**: Скрипт для построения корпуса похожих заголовков для обучения/валидации semantic clusters. Собирает заголовки видео, группирует их по семантической схожести, создает корпус для последующего построения кластеров. Используется для подготовки данных для build_semantic_clusters_v1.

**Расположение**: `scripts/sem_clust_v1/build_similar_titles_corpus_v1.py`

### sem_clust_v1/BUILD_SIMILAR_TITLES_CORPUS.md
**Краткое описание**: Документация по процессу создания корпуса похожих заголовков. Содержит инструкции по сбору данных, группировке заголовков, формату выходных данных. Используется как руководство для подготовки корпуса.

**Полный документ**: `scripts/sem_clust_v1/BUILD_SIMILAR_TITLES_CORPUS.md`

### sem_clust_v1/build_topics_taxonomy_v1.py
**Краткое описание**: Скрипт для построения таксономии тем (topics taxonomy) для семантической классификации. Создает иерархическую структуру тем, категорий и подкатегорий контента. Используется для организации семантических кластеров в структурированную таксономию.

**Расположение**: `scripts/sem_clust_v1/build_topics_taxonomy_v1.py`

### sem_clust_v1/BUILD_TOPICS_TAXONOMY.md
**Краткое описание**: Документация по процессу создания таксономии тем. Содержит описание структуры таксономии, инструкции по построению, примеры категорий. Используется как руководство для создания topics taxonomy.

**Полный документ**: `scripts/sem_clust_v1/BUILD_TOPICS_TAXONOMY.md`

### sem_clust_v1/create_title_emb_for_cluster_v1.py
**Краткое описание**: Скрипт для создания эмбеддингов заголовков для использования в semantic clusters. Принимает корпус заголовков, генерирует эмбеддинги с помощью sentence transformer модели, сохраняет их в формате для последующего построения кластеров. Используется для подготовки входных данных для build_semantic_clusters_v1.

**Расположение**: `scripts/sem_clust_v1/create_title_emb_for_cluster_v1.py`

### sem_clust_v1/data_*.json
**Краткое описание**: Промежуточные данные для построения semantic clusters. Содержат собранные заголовки, метаданные, результаты промежуточных этапов обработки. Используются как входные/выходные данные для различных этапов построения кластеров.

**Расположение**: `scripts/sem_clust_v1/data_*.json`

### sem_clust_v1/title_embeddings.npy
**Краткое описание**: Файл с предвычисленными эмбеддингами заголовков. Содержит numpy массив формы (N, D) с эмбеддингами для N заголовков. Используется как вход для build_semantic_clusters_v1.

**Расположение**: `scripts/sem_clust_v1/title_embeddings.npy`

---

