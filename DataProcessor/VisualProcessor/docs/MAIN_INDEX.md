# Главный индекс документации VisualProcessor

Этот документ служит единой точкой входа для навигации по всей документации VisualProcessor. Каждый раздел содержит краткое описание документов и ссылки на полные версии.

---

## Документация

### PRODUCTION_READINESS_AND_SCALE_PLAN.md
**Краткое описание**: Production‑чеклист и план доводки VisualProcessor до массовой обработки (100k видео): выбранная целевая инфраструктура (K8s + Triton + очередь + storage), приоритеты (P0–P2), DoD для каждого шага, рекомендации по GPU sizing и эксплуатации.

**Полный документ**: [docs/PRODUCTION_READINESS_AND_SCALE_PLAN.md](PRODUCTION_READINESS_AND_SCALE_PLAN.md)

### BATCH_PROCESSING_PLAN.md
**Краткое описание**: План адаптации VisualProcessor для батчевой обработки. Описывает двухуровневую параллельность (видео + кадры), GPU batching для ML-моделей (CLIP, object detection, optical flow, depth, face landmarks, identity), CPU parallelism для независимых компонентов, изоляцию данных, валидацию, этапы реализации (Stage 0-5), примеры использования, производительность и оптимизации. Статус: Stage 0-2 завершены (базовый каркас, изоляция артефактов, GPU batching для core_clip), Stage 3-5 в разработке.

**Полный документ**: [docs/BATCH_PROCESSING_PLAN.md](BATCH_PROCESSING_PLAN.md)

### LAST_FULL_RUN_LOG.md
**Краткое описание**: Лог последнего полного запуска VisualProcessor. Содержит примеры вывода команд, статусы выполнения компонентов (core providers и modules), тайминги, использование памяти (RAM/VRAM), диагностическую информацию для отладки и валидации пайплайна, stage timings для каждого компонента.

**Полный документ**: [docs/LAST_FULL_RUN_LOG.md](LAST_FULL_RUN_LOG.md)

### Production Schemas (NPZ contracts)
**Краткое описание**: Машиночитаемые схемы артефактов VisualProcessor (NPZ) для **строгой типизации** и **версионирования** по `meta.schema_version`. Включает реестр JSON схем (`VisualProcessor/schemas/*.json`), runtime‑валидацию ключей/dtype/shape (fail‑fast для известных схем) и human‑friendly `SCHEMA.md` рядом с компонентами.

**Входная точка**: [`VisualProcessor/schemas/README.md`](../schemas/README.md)

---

## Core Components

VisualProcessor содержит базовые провайдеры (core components), которые извлекают низкоуровневые признаки из кадров видео. Core components организованы по уровням зависимостей и поддерживают batch processing, GPU ускорение через Triton и детерминированное кеширование.

### Tier-0: Baseline Core Providers (независимые)

#### CoreCLIP (core_clip)
**Краткое описание**: Вычисляет CLIP эмбеддинги для выборки кадров (union-domain) и сохраняет их в NPZ. Дополнительно сохраняет text embeddings для фиксированных prompt-наборов (shot_quality, scene_aesthetic, scene_luxury, scene_atmosphere, cut_detection_transition, popularity_topic, places365), чтобы downstream компоненты могли делать zero-shot scoring без загрузки CLIP весов. Версия 1.0.0, категория embeddings, GPU (Triton). Поддерживает batch processing, кеширование text embeddings на диск, GPU ускорение через Triton, возвращает нормализованные векторы.

**Полный документ**: [core/model_process/core_clip/README.md](../core/model_process/core_clip/README.md)

#### CoreObjectDetections (core_object_detections)
**Краткое описание**: Детекция объектов (YOLO) на primary выборке кадров (union-domain). Извлекает `boxes/scores/class_ids/valid_mask` + нормализованную геометрию (`boxes_norm/centers_norm/areas_frac`) и frame-level агрегаты (`person_count`, `*_area_frac` и т.д.) и пишет `detections.npz` (schema `core_object_detections_npz_v2`). **Tracking удалён** в baseline Audit v3; downstream semantic heads используют surrogate `track_ids` (per-detection). Runtime: `ultralytics` (in-process, локальные веса через `DP_MODELS_ROOT`) или `triton` (через ModelManager spec). Поддерживает batch processing и строгий no-fallback по `frame_indices`.

**Полный документ**: [core/model_process/core_object_detections/README.md](../core/model_process/core_object_detections/README.md)

#### CoreOpticalFlow (core_optical_flow)
**Краткое описание**: Оптический поток (RAFT) для оценки движения между кадрами. Извлекает flow vectors, motion norms, temporal motion features. Версия 1.0.0, категория optical flow, GPU (Triton). Поддерживает batch processing, GPU ускорение через Triton, используется модулями (cut_detection, story_structure) для анализа движения.

**Полный документ**: [core/model_process/core_optical_flow/README.md](../core/model_process/core_optical_flow/README.md)

#### CoreDepthMiDaS (core_depth_midas)
**Краткое описание**: Оценка глубины (MiDaS) для кадров видео. Извлекает depth maps, depth statistics, используется модулями (shot_quality) для анализа композиции. Версия 1.0.0, категория depth estimation, GPU (Triton). Поддерживает batch processing, GPU ускорение через Triton.

**Полный документ**: [core/model_process/core_depth_midas/README.md](../core/model_process/core_depth_midas/README.md)

#### CoreFaceLandmarks (core_face_landmarks)
**Краткое описание**: Landmarks лиц (MediaPipe FaceMesh) для анализа лиц на primary sampling группе. Работает строго в union-domain и использует **person-mask gating** от `core_object_detections` (no-fallback, строгий `frame_indices` match), чтобы не запускать FaceMesh без людей. Пишет **raw+filtered** landmarks (`*_landmarks_raw` + `*_landmarks`) и QA-диагностику (`person_present`, `face_mesh_ran`). Версия Audit v3: `schema_version=core_face_landmarks_npz_v2`, `producer_version=2.1`. CPU (MediaPipe, isolated venv). Рендер: offline mini-dashboard + `_render/assets/` (privacy banner).

**Полный документ**: [core/model_process/core_face_landmarks/README.md](../core/model_process/core_face_landmarks/README.md)

#### OCRExtractor (ocr_extractor)
**Краткое описание**: OCR текст из кадров видео по bbox-кропам `text_region` из `core_object_detections`. Поддерживает выбор движка: **`ppocr_rec_onnx` (рекомендовано, offline через dp_models + ONNXRuntime)** и `tesseract` (CLI). Версия Audit v3: `producer_version=0.2`, `schema_version=ocr_extractor_npz_v2`. Поддерживает batch processing; по умолчанию raw OCR text не сохраняется (privacy), в dev можно включить `retain_raw_ocr_text=true`.

**Полный документ**: [core/model_process/ocr_extractor/README.md](../core/model_process/ocr_extractor/README.md)

### Tier-1: Semantic Heads (зависят от Tier-0)

#### BrandSemantics (core_identity/brand_semantics)
**Краткое описание**: Распознавание брендов/логотипов (semantic head) через Embedding Service retrieval. Использует bbox proposals из `core_object_detections` (настраиваемые `proposal_classes`, дефолт `logo_region,text_region`), выбирает best crop per-track и делает поиск top‑K **без threshold-gating** (threshold используется только для `*_is_confident_top1`). Audit v3 версия пишет `brand_semantics_npz_v2` с NaN/-1 policy, per-detection output, deterministic label-space (`semantic_label_names` + `semantic_object_ids`) и `db_digest` (через `GET /categories/brand/labels`). Рендер: offline mini-dashboard (`render.html`) + `_render/assets/` (crop примеры top/anti-top).

**Полный документ**: [core/model_process/core_identity/brand_semantics/README.md](../core/model_process/core_identity/brand_semantics/README.md)

#### CarSemantics (core_identity/car_semantics)
**Краткое описание**: Семантика автомобилей через retrieval в Embedding Service по bbox proposals из `core_object_detections` (default `proposal_classes="car"`). Audit v3 версия пишет `car_semantics_npz_v2` с **K=5**, NaN/-1 policy, deterministic label-space (`semantic_label_names` + `semantic_object_ids`) и `db_digest` (через `GET /categories/car/labels`). Threshold не режет top‑K (используется только для `*_is_confident_top1`). Рендер: offline mini-dashboard (`render.html`) + `_render/assets/` (кропы примеров).

**Полный документ**: [core/model_process/core_identity/car_semantics/README.md](../core/model_process/core_identity/car_semantics/README.md)

#### FaceIdentity (core_identity/face_identity)
**Краткое описание**: Идентификация известных людей (celebrity retrieval) по face embeddings через Embedding Service. Использует ArcFace для извлечения face embeddings, сравнивает с базой известных людей через Embedding Service. Версия 1.0.0, категория semantic head, GPU (ArcFace + Embedding Service). Зависит от core_object_detections (frame_indices) и core_face_landmarks (face bbox из landmarks). Поддерживает batch processing, интеграцию с Embedding Service.

**Полный документ**: [core/model_process/core_identity/face_identity/README.md](../core/model_process/core_identity/face_identity/README.md)

#### PlaceSemantics (core_identity/place_semantics)
**Краткое описание**: Распознавание мест и лэндмарков через retrieval по CLIP frame embeddings. Использует frame embeddings из core_clip для cosine similarity с gallery embeddings мест через Embedding Service. Версия 1.0.0, категория semantic head, GPU (Triton + Embedding Service). Зависит от core_object_detections (frame_indices) и core_clip (frame embeddings). Поддерживает batch processing, интеграцию с Embedding Service.

**Полный документ**: [core/model_process/core_identity/place_semantics/README.md](../core/model_process/core_identity/place_semantics/README.md)

#### ContentDomain (core_identity/content_domain)
**Краткое описание**: Классификация домена контента (игра/аниме/мульт/реал/скрин-рекординг) через **CLIP text‑retrieval** поверх `core_clip` frame embeddings. Использует offline domain DB (`domains.jsonl` + manifest + optional thresholds) и Triton (`clip_text`) для вычисления text embeddings. Audit v3 версия пишет `content_domain_npz_v2` (producer_version=0.2) с `db_digest`, NaN/-1 policy при `A<K`, `meta_json` и offline mini-dashboard render (`render.html` без CDN).

**Полный документ**: [core/model_process/core_identity/content_domain/README.md](../core/model_process/core_identity/content_domain/README.md)

#### FranchiseRecognition (core_identity/franchise_recognition)
**Краткое описание**: Распознавание франшиз и названий через CLIP text encoder. Использует CLIP через Triton для идентификации конкретных франшиз/тайтлов. Версия 1.0.0, категория semantic head, GPU (Triton). Поддерживает batch processing.

**Полный документ**: [core/model_process/core_identity/franchise_recognition/README.md](../core/model_process/core_identity/franchise_recognition/README.md)

---

## Modules

VisualProcessor содержит высокоуровневые модули (modules), которые анализируют признаки из core components и извлекают сложные метрики и признаки.

### Tier-0: Baseline Modules (зависят от core components)

#### CutDetection (cut_detection)
**Краткое описание**: Детекция склеек (hard cuts и soft transitions: fade/dissolve + motion transitions) на выборке кадров. Производит shot boundary timeline для downstream модулей (shot_quality, video_pacing) и богатый набор editing/pacing features. **Audit v3**: `schema_version=cut_detection_npz_v1` + обязательный model-facing `cut_detection_model_facing_npz_v1`, offline render (без CDN). Hard dep: core_optical_flow (no-fallback). Quality deps (soft): core_face_landmarks/core_object_detections (для jump-cuts; при отсутствии — warning и ухудшение качества). Поддерживает batch processing.

**Полный документ**: [modules/cut_detection/README.md](../modules/cut_detection/README.md)

#### ShotQuality (shot_quality)
**Краткое описание**: Техническое качество кадров и шотов (sharpness/noise/exposure/contrast/color/compression + depth/object/face ROI proxies) + CLIP zero-shot `quality_probs`. Consumer-only модуль: не запускает модели, использует `core_*` и `cut_detection` для shot boundaries (предпочтительно `cut_detection_model_facing`). **Audit v3**: `schema_version=shot_quality_npz_v3`, `producer_version=2.0.2`, `frame_feature_present_ratio`, `shot_frame_feature_present_ratio`, shot-level агрегаты по `quality_probs`, `meta.impl_meta` (debug), `meta.ui_payload`, `meta.stage_timings_ms`, offline render (без CDN). CPU-only. Hard deps: `core_clip/core_depth_midas/core_object_detections/core_face_landmarks/cut_detection` (no-fallback, aligned indices).

**Полный документ**: [modules/shot_quality/README.md](../modules/shot_quality/README.md)

#### VideoPacing (video_pacing)
**Краткое описание**: Темп/монтаж (pacing metrics): частота склеек, длительности шотов, motion/semantic/color change rate на оси Segmenter. **Audit v3**: `schema_version=video_pacing_npz_v3`, `producer_version=2.0.1`, strict `union_timestamps_sec` time-axis (no-fallback), **tabular** `feature_names/feature_values` (включая flattened histogram bins), `meta.stage_timings_ms` + config highlights + `meta.ui_payload`, offline render (без CDN). CPU-only. Hard deps: `cut_detection/core_optical_flow/core_clip`.

**Полный документ**: [modules/video_pacing/README.md](../modules/video_pacing/README.md)

#### SceneClassification (scene_classification)
**Краткое описание**: Классификация и сегментация сцен (Places365). Дает per-frame distribution (top‑K + entropy/gap) и группирует кадры в сцены по hard shot boundaries (`cut_detection`) с CLIP‑семантикой строго из `core_clip`. **Audit v3**: `schema_version=scene_classification_npz_v2`, `producer_version=2.0.1`, Segmenter-owned axis, no-network (ModelManager/Triton), offline render (без CDN), `meta.ui_payload` + `meta.stage_timings_ms`. GPU/CPU. Зависит от core_clip и cut_detection (hard, no-fallback). Поддерживает batch processing.

**Полный документ**: [modules/scene_classification/README.md](../modules/scene_classification/README.md)

### Tier-1: Advanced Modules (зависят от Tier-0 modules)

#### StoryStructure (story_structure)
**Краткое описание**: Структура истории (Tier‑0 baseline): story/energy proxies, hook/climax markers, face proxies + (опционально) text topic-shift (OCR→CLIP text). **Audit v3**: `schema_version=story_structure_npz_v3`, `producer_version=3.0.2`, Segmenter-owned axis (`frame_indices`) + time axis строго из `union_timestamps_sec` (no-fallback), **tabular** `feature_names/feature_values` вместо object‑dict, `meta.stage_timings_ms` + config highlights + `meta.ui_payload`, **`frame_feature_present_ratio`**, offline render (без CDN). CPU-only. Hard deps: `core_clip/core_optical_flow/core_face_landmarks`.

**Полный документ**: [modules/story_structure/README.md](../modules/story_structure/README.md)

#### EmotionFace (emotion_face)
**Краткое описание**: Эмоции на лицах (facial emotion recognition). Анализирует эмоциональное состояние лиц в кадрах (EmoNet), выдаёт axis-aligned time-series + keyframes. **Audit v3**: `schema_version=emotion_face_npz_v3`, `producer_version=2.0.2`, axis aligned to Segmenter + `face_present`/`processed_mask`, top-level model-facing arrays (valence/arousal/probs/…), no-network (ModelManager; `emo_path` только debug override), offline render (без CDN). GPU/CPU. Зависит от core_face_landmarks (hard, no-fallback). Поддерживает batch processing.

**Полный документ**: [modules/emotion_face/README.md](../modules/emotion_face/README.md)

#### DetalizeFace (detalize_face)
**Краткое описание**: Детальный анализ лиц (facial feature proxies). Вычисляет производные метрики по лицам на основе `core_face_landmarks` (per-frame masks/count + optional `primary_*` heuristic curves) + per-track агрегаты, без запуска собственных ML моделей. **Audit v3**: `schema_version=detalize_face_npz_v3`, `producer_version=2.0.2`, axis aligned to Segmenter + masks + model-facing `primary_compact_features (N,40)` и `aggregated` (tabular stats), offline render (без CDN). CPU-only. Зависит от core_face_landmarks (hard, no-fallback). Поддерживает batch processing.

**Полный документ**: [modules/detalize_face/README.md](../modules/detalize_face/README.md)

#### Behavioral (behavioral)
**Краткое описание**: Поведенческий анализ (gestures, poses, body language). Анализирует жесты, позы, язык тела для оценки поведения людей в кадрах. **Audit v3**: `schema_version=behavioral_npz_v1`, `producer_version=2.0.1`, offline render (без CDN). CPU. Зависит от core_face_landmarks. Поддерживает batch processing.

**Полный документ**: [modules/behavioral/README.md](../modules/behavioral/README.md)

#### ActionRecognition (action_recognition)
**Краткое описание**: Распознавание действий (SlowFast action recognition). Использует SlowFast модель для классификации действий в видео. Версия 1.0.0, категория action analysis, GPU. Зависит от core_object_detections. Поддерживает batch processing.

**Полный документ**: [modules/action_recognition/README.md](../modules/action_recognition/README.md)

#### ColorLight (color_light)
**Краткое описание**: Анализ цвета и света (palette, lighting, mood). Строит scene-level/video-level метрики и выдаёт **стабильный model-facing compact** `frame_compact_features (M,16)` + имена + `aggregated` на оси Segmenter (union-domain). **Audit v3**: `schema_version=color_light_npz_v2`, `producer_version=2.0.2`, hard dep `scene_classification` (no-fallback), no-resampling (strict Segmenter axis), `meta.stage_timings_ms` + `models_used/model_signature`, offline render (без CDN), `store_debug_objects` для отключения тяжёлых `frames/scenes`. CPU-only. Поддерживает batch processing.

**Полный документ**: [modules/color_light/README.md](../modules/color_light/README.md)

#### FramesComposition (frames_composition)
**Краткое описание**: Композиция кадров (balance, thirds/anchors, symmetry, negative space, complexity, leading lines). Выдаёт `frame_feature_values (N,D)` + `feature_values (F)` в табличном формате (model_facing) на оси Segmenter. **Audit v3**: `schema_version=frames_composition_npz_v1`, `producer_version=2.0.1`, hard deps `core_object_detections/core_face_landmarks/core_depth_midas` (aligned frame_indices, no-fallback; depth must be ok), valid empty `no_faces_in_video`, `meta.stage_timings_ms` + `models_used/model_signature`, **`frame_feature_present_ratio`**, offline render (без CDN). CPU-only. Поддерживает batch processing.

**Полный документ**: [modules/frames_composition/README.md](../modules/frames_composition/README.md)

#### SimilarityMetrics (similarity_metrics)
**Краткое описание**: Метрики схожести видео: intra-video coherence (покадровые кривые на `core_clip`) и reference similarity (опционально, dp_models reference set) по нескольким модальностям. **Audit v3**: `schema_version=similarity_metrics_npz_v3`, `producer_version=2.0.2`, **фиксированный stable** `feature_names/feature_values` (per‑modality mean_topn/max/p10 + present flags + uniqueness), strict Segmenter axis + strict match с `core_clip.frame_indices`, `meta.ui_payload`, `meta.stage_timings_ms`, offline render (без CDN). CPU-only. Hard dep: `core_clip`.

**Полный документ**: [modules/similarity_metrics/README.md](../modules/similarity_metrics/README.md)

#### Uniqueness (uniqueness)
**Краткое описание**: Уникальность (intra-video baseline): повторяемость/разнообразие по sampled кадрам только на `core_clip` embeddings. **Audit v3**: `schema_version=uniqueness_npz_v4`, `producer_version=1.0.2`, Segmenter-owned axis + `union_timestamps_sec` (no-fallback), **tabular** `feature_names/feature_values` (добавлены mean/p95 агрегаты, effective_unique_* и quality auto-threshold), `meta.stage_timings_ms` + `meta.ui_payload` (top repeats + anti-top unique), offline render (без CDN). CPU-only. Hard dep: `core_clip`.

**Полный документ**: [modules/uniqueness/README.md](../modules/uniqueness/README.md)

#### TextScoring (text_scoring)
**Краткое описание**: Оценка текста (consumer OCR). Анализирует присутствие/плотность текста, синхронизацию с активностью, CTA и читаемость. **Audit v3**: `schema_version=text_scoring_npz_v2`, `producer_version=2.0.1`, Segmenter-owned axis + `union_timestamps_sec` (no-fallback), **tabular** `feature_names/feature_values`, optional debug `ocr_raw/ocr_unique_elements` (через `store_debug_objects`), `meta.stage_timings_ms` + `meta.ui_payload`, offline render (без CDN). CPU-only. Optional dep: `core_face_landmarks` (если `use_face_data=true`).

**Полный документ**: [modules/text_scoring/README.md](../modules/text_scoring/README.md)

#### HighLevelSemantic (high_level_semantic)
**Краткое описание**: Высокоуровневая семантика (semantic summary, high-level features). Агрегирует высокоуровневые семантические признаки из `core_clip` + `cut_detection` + `emotion_face` (и опционально text/audio processors) и выравнивает их по union‑оси Segmenter. **Audit v3**: `schema_version=high_level_semantic_npz_v2`, `producer_version=2.0.2`, strict schema JSON + `SCHEMA.md`, no-network (не грузит CLIP веса), offline render (без CDN), `meta.stage_timings_ms` + config highlights, best-effort `models_used/model_signature` из upstream артефактов, `frame_feature_present_ratio`. CPU-only. Поддерживает batch processing.

**Полный документ**: [modules/high_level_semantic/README.md](../modules/high_level_semantic/README.md)

#### MicroEmotion (micro_emotion)
**Краткое описание**: Микро-эмоции (micro-expressions, Action Units). Анализирует AU/landmarks/pose/gaze и micro-expressions через OpenFace (Docker), выдаёт per-frame вектора (model_facing) + **tabular scalar aggregates** `feature_names/feature_values` + events stream. **Audit v3**: `schema_version=micro_emotion_npz_v3`, `producer_version=2.0.2`, Segmenter-owned axis + face-gating через core_face_landmarks, `meta.stage_timings_ms` + config highlights, offline render (без CDN). GPU required. Hard dep: core_face_landmarks. Поддерживает batch processing.

**Полный документ**: [modules/micro_emotion/README.md](../modules/micro_emotion/README.md)

#### OpticalFlow (optical_flow)
**Краткое описание**: Анализ оптического потока (flow analysis, motion patterns). **Consumer-only** модуль: читает `core_optical_flow/flow.npz` и выдаёт axis-aligned кривую `motion_norm_per_sec_mean (N,)` + **model-facing compact** `frame_feature_values (N,D)` + агрегаты `feature_names/feature_values`. **Audit v3**: `schema_version=optical_flow_npz_v3`, `producer_version=2.0.2`, `meta.stage_timings_ms`, offline render (без CDN), best-effort `models_used/model_signature` из `core_optical_flow`. CPU-only. Hard dep: `core_optical_flow`. Поддерживает batch processing.

**Полный документ**: [modules/optical_flow/README.md](../modules/optical_flow/README.md)

---

## Архитектура и Core

### main.py
**Краткое описание**: Главный оркестратор VisualProcessor. Класс `run()` и `run_batch()` управляют списком core providers и modules, последовательно применяют их к видео, поддерживают конфигурацию устройств (CPU/GPU), batch processing с параллелизмом по уровням зависимостей, топологическую сортировку компонентов (DAG), обработку ошибок и прогресс-репортинг. Реализует registry компонентов для ленивой загрузки, поддерживает required/optional компоненты, artifacts_dir per-run, GPU slot management, stage timings.

**Расположение**: `VisualProcessor/main.py`

### modules/base_module.py
**Краткое описание**: Базовый интерфейс для всех modules. Абстрактный класс `BaseModule` определяет контракт `run()` и опциональный `process_batch()` для batch processing. Свойство `supports_batch` указывает на оптимизированную batch реализацию. Все modules наследуются от этого класса. Поддерживает сохранение результатов в NPZ формате, валидацию артефактов, render system.

**Расположение**: `VisualProcessor/modules/base_module.py`

### utils/frame_manager.py
**Краткое описание**: Менеджер кадров для эффективной загрузки кадров из frames_dir. Класс `FrameManager` обеспечивает ленивую загрузку кадров, кеширование в памяти, поддержку chunk_size и cache_size, работу с metadata.json. Используется всеми компонентами для доступа к кадрам видео.

**Расположение**: `VisualProcessor/utils/frame_manager.py`

### utils/video_context.py
**Краткое описание**: Контекст видео для batch processing. Класс `VideoContext` изолирует контекст каждого видео (video_id, frames_dir, rs_path, metadata.json) для batch обработки. Обеспечивает изоляцию артефактов между видео, корректную обработку метаданных для каждого видео.

**Расположение**: `VisualProcessor/utils/video_context.py`

### utils/results_store.py
**Краткое описание**: Утилиты для работы с result_store. Содержит функции для сохранения NPZ артефактов, обновления manifest.json, работы с per-run директориями. Используется компонентами для сохранения результатов в правильном формате.

**Расположение**: `VisualProcessor/utils/results_store.py`

### utils/manifest.py
**Краткое описание**: Утилиты для работы с manifest.json. Классы `RunManifest` и `ManifestComponent` обеспечивают обновление манифеста run'а с информацией о компонентах, статусах, таймингах. Используется для отслеживания прогресса обработки.

**Расположение**: `VisualProcessor/utils/manifest.py`

### utils/artifact_validator.py
**Краткое описание**: Валидатор NPZ артефактов. Функция `validate_npz()` проверяет корректность NPZ файлов, структуру данных, наличие обязательных полей meta, соответствие схеме. Используется для валидации артефактов перед сохранением.

**Расположение**: `VisualProcessor/utils/artifact_validator.py`

### utils/quality_validator.py
**Краткое описание**: Валидатор качества результатов. Содержит функции для проверки качества извлечённых признаков, валидации метрик, проверки корректности данных. Используется для отладки и тестирования компонентов.

**Расположение**: `VisualProcessor/utils/quality_validator.py`

### utils/renderer.py
**Краткое описание**: Рендерер для генерации HTML/JSON визуализаций результатов VisualProcessor. Создает human-readable представления component results, метрик, эмбеддингов (опционально), статистик. Используется для отладки и визуализации результатов обработки через render system.

**Расположение**: `VisualProcessor/utils/renderer.py`

### utils/meta_builder.py
**Краткое описание**: Утилиты для работы с метаданными моделей. Содержит функции для канонизации списка используемых моделей (`model_used()`), вычисления детерминированной подписи моделей (`compute_model_signature()`), применения метаданных моделей к мета-словарю (`apply_models_meta()`). Обеспечивает стабильную сортировку и детерминированное хеширование для reproducibility. Интегрирован с общим `DataProcessor/common/meta_builder.py`.

**Расположение**: `VisualProcessor/utils/meta_builder.py`

### utils/logger.py
**Краткое описание**: Утилиты для логирования. Содержит функции для настройки логирования, создания logger'ов с префиксами компонентов, форматирования логов. Используется всеми компонентами для единообразного логирования.

**Расположение**: `VisualProcessor/utils/logger.py`

### utils/resource_probe.py
**Краткое описание**: Утилиты для мониторинга ресурсов. Содержит функции для получения информации о памяти (RAM/VRAM), CPU, GPU, мониторинга использования ресурсов в реальном времени. Используется для профилирования и оптимизации производительности.

**Расположение**: `VisualProcessor/utils/resource_probe.py`

### utils/batch_utils.py
**Краткое описание**: Утилиты для batch processing. Содержит функции для обработки batch результатов, сбора video contexts, распределения результатов обратно по видео. Используется для реализации batch processing в компонентах.

**Расположение**: `VisualProcessor/utils/batch_utils.py`

### utils/batching.py
**Краткое описание**: Утилиты для батчинга кадров. Содержит функции для группировки кадров в батчи, распределения батчей по компонентам, обработки batch inference. Используется для GPU batching в компонентах.

**Расположение**: `VisualProcessor/utils/batching.py`

---

## Batch Processing Utilities

VisualProcessor содержит специализированные утилиты для batch processing каждого компонента:

### Core Components Batch Utilities

- **`utils/core_clip_batch.py`**: Batch processing для core_clip с гибридным батчингом кадров из всех видео
- **`utils/core_object_detections_batch.py`**: Batch processing для core_object_detections
- **`utils/core_optical_flow_batch.py`**: Batch processing для core_optical_flow
- **`utils/core_depth_midas_batch.py`**: Batch processing для core_depth_midas
- **`utils/core_face_landmarks_batch.py`**: Batch processing для core_face_landmarks
- **`utils/ocr_extractor_batch.py`**: Batch processing для ocr_extractor

### Semantic Heads Batch Utilities

- **`utils/brand_semantics_batch.py`**: Batch processing для brand_semantics
- **`utils/car_semantics_batch.py`**: Batch processing для car_semantics
- **`utils/face_identity_batch.py`**: Batch processing для face_identity
- **`utils/place_semantics_batch.py`**: Batch processing для place_semantics
- **`utils/content_domain_batch.py`**: Batch processing для content_domain
- **`utils/franchise_recognition_batch.py`**: Batch processing для franchise_recognition

### Modules Batch Utilities

- **`utils/cut_detection_batch.py`**: Batch processing для cut_detection
- **`utils/scene_classification_batch.py`**: Batch processing для scene_classification
- **`utils/video_pacing_batch.py`**: Batch processing для video_pacing
- **`utils/emotion_face_batch.py`**: Batch processing для emotion_face
- **`utils/action_recognition_batch.py`**: Batch processing для action_recognition

---

## Дополнительная документация

### core/model_process/README_RUN_ALL_CORE.md
**Краткое описание**: Документация по скрипту запуска всех core components с Triton. Описывает использование `run_all_core_components.py` для прогона всех компонентов core с использованием моделей Triton и генерации HTML отчета с метриками производительности и демонстрацией качества. Содержит параметры запуска, структуру отчета, требования, модели и runtime.

**Полный документ**: [core/model_process/README_RUN_ALL_CORE.md](../core/model_process/README_RUN_ALL_CORE.md)

### core/model_process/REQUIREMENTS.md
**Краткое описание**: Документация по зависимостям между Core Providers. Описывает дерево зависимостей (Tier-0: независимые, Tier-1: semantic heads), детальное описание зависимостей для каждого компонента, важные замечания (CLIP через Triton vs core_clip, shared sampling group, порядок выполнения). Источник правды для зависимостей между компонентами.

**Полный документ**: [core/model_process/REQUIREMENTS.md](../core/model_process/REQUIREMENTS.md)

### core/model_process/core_identity/brand_semantics/BRAND_DATABASE_GUIDE.md
**Краткое описание**: Руководство по заполнению базы брендов для brand_semantics. Описывает процесс сбора данных (локальная база known_brands/), синхронизации с Embedding Service, использование скрипта add_brand.py для интерактивного добавления логотипов из видео/фото, синхронизацию через sync_known_brands_to_embedding_service.py.

**Полный документ**: [core/model_process/core_identity/brand_semantics/BRAND_DATABASE_GUIDE.md](../core/model_process/core_identity/brand_semantics/BRAND_DATABASE_GUIDE.md)

### modules/cut_detection/SCHEMA_MODEL_FACING.md
**Краткое описание**: Схема model-facing NPZ артефакта для cut_detection. Определяет структуру данных для downstream ML-моделей (popularity prediction), содержит frame-level и shot-level features, метаданные, временные метки. Используется для передачи данных в ML-модели без дополнительной обработки.

**Полный документ**: [modules/cut_detection/SCHEMA_MODEL_FACING.md](../modules/cut_detection/SCHEMA_MODEL_FACING.md)

---

## FEATURES_DESCRIPTION.md

Многие модули содержат файлы `FEATURES_DESCRIPTION.md` с детальным описанием извлекаемых признаков:

- **`modules/action_recognition/FEATURES_DESCRIPTION.md`**: Описание признаков action recognition
- **`modules/behavioral/FEATURES_DESCRIPTION.md`**: Описание признаков behavioral analysis
- **`modules/color_light/FEATURES_DESCRIPTION.md`**: Описание признаков color and light analysis
- **`modules/cut_detection/FEATURES_DESCRIPTION.md`**: Описание признаков cut detection
- **`modules/detalize_face/FEATURES_DESCRIPTION.md`**: Описание признаков detalize face
- **`modules/emotion_face/FEATURES_DESCRIPTION.md`**: Описание признаков emotion face
- **`modules/frames_composition/FEATURES_DESCRIPTION.md`**: Описание признаков frames composition
- **`modules/high_level_semantic/FEATURES_DESCRIPTION.md`**: Описание признаков high-level semantic
- **`modules/micro_emotion/FEATURES_DESCRIPTION.md`**: Описание признаков micro emotion
- **`modules/optical_flow/FEATURES_DESCRIPTION.md`**: Описание признаков optical flow analysis
- **`modules/scene_classification/FEATURES_DESCRIPTION.md`**: Описание признаков scene classification
- **`modules/shot_quality/FEATURES_DESCRIPTION.md`**: Описание признаков shot quality
- **`modules/similarity_metrics/FEATURES_DESCRIPTION.md`**: Описание признаков similarity metrics
- **`modules/text_scoring/FEATURES_DESCRIPTION.md`**: Описание признаков text scoring

---

## CLI и Entry Points

### main.py
**Краткое описание**: CLI entry point для VisualProcessor. Парсит аргументы командной строки, загружает конфигурацию из YAML файла, инициализирует обработку одного или нескольких видео (single-file и batch mode), сохраняет результаты в result_store. Поддерживает интеграцию с верхним оркестратором DataProcessor через CLI аргументы, конфигурацию через global_config.yaml.

**Расположение**: `VisualProcessor/main.py`

### core/model_process/run_all_core_components.py
**Краткое описание**: Скрипт для запуска всех core components с Triton. Позволяет прогнать все компоненты core с использованием моделей Triton и сгенерировать HTML отчет с метриками производительности и демонстрацией качества. Поддерживает выбор компонентов, параметры Triton моделей, batch size, мониторинг памяти.

**Расположение**: `VisualProcessor/core/model_process/run_all_core_components.py`

---

## Интеграция с DataProcessor

VisualProcessor интегрирован в общий пайплайн DataProcessor:

- **Конфигурация**: через `DataProcessor/configs/global_config.yaml` и `config_parser.py`
- **Orchestration**: через `DataProcessor/main.py` с поддержкой batch processing флагов
- **Storage**: результаты сохраняются в per-run result_store (`dp_results/<platform_id>/<video_id>/<run_id>/<component_name>/`)
- **State management**: через `DataProcessor/state/` для отслеживания прогресса
- **Models**: через `dp_models` для offline моделей (no-network policy), Triton для GPU inference
- **Artifacts**: NPZ файлы сохраняются в `result_store/<component_name>/` per-run
- **Render**: render-context JSON и HTML визуализации сохраняются в `result_store/<component_name>/_render/` per-run

---

## Структура проекта

VisualProcessor организован в модульную структуру:

- **`core/model_process/`**: Core components (core_clip, core_object_detections, core_optical_flow, core_depth_midas, core_face_landmarks, ocr_extractor, core_identity/*)
- **`modules/`**: Все высокоуровневые модули, каждый в отдельной директории с `main.py`, `README.md`, `render.py`, `FEATURES_DESCRIPTION.md`
- **`utils/`**: Вспомогательные утилиты (frame_manager, video_context, results_store, manifest, artifact_validator, batch utilities)
- **`docs/`**: Документация (MAIN_INDEX.md, BATCH_PROCESSING_PLAN.md, LAST_FULL_RUN_LOG.md)
- **`README.md`**: Основная документация VisualProcessor

---

## Статистика

- **Всего core components**: 12
  - **Tier-0 (независимые)**: 6 (core_clip, core_object_detections, core_optical_flow, core_depth_midas, core_face_landmarks, ocr_extractor)
  - **Tier-1 (semantic heads)**: 6 (brand_semantics, car_semantics, face_identity, place_semantics, content_domain, franchise_recognition)
- **Всего modules**: 17
  - **Tier-0 (baseline)**: 4 (cut_detection, shot_quality, video_pacing, scene_classification)
  - **Tier-1 (advanced)**: 13 (story_structure, emotion_face, detalize_face, behavioral, action_recognition, color_light, frames_composition, similarity_metrics, uniqueness, text_scoring, high_level_semantic, micro_emotion, optical_flow)
- **GPU components**: 14 (core_clip, core_object_detections, core_optical_flow, core_depth_midas, brand_semantics, car_semantics, face_identity, place_semantics, content_domain, franchise_recognition, action_recognition, emotion_face, scene_classification, micro_emotion)
- **CPU-only components**: 15 (core_face_landmarks, ocr_extractor, cut_detection, shot_quality, video_pacing, story_structure, detalize_face, behavioral, color_light, frames_composition, similarity_metrics, uniqueness, text_scoring, high_level_semantic, optical_flow)

