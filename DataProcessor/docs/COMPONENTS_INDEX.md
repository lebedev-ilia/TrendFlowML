## Components index (single source-of-truth = README рядом с кодом)

Правило проекта:
- **README каждого компонента лежит рядом с кодом** (single source-of-truth).
- В `docs/` храним только **ссылки/индексы**.

### Processors / Orchestrators

- **Segmenter**: `Segmenter/README.md`
- **VisualProcessor**: `VisualProcessor/README.md`
- **AudioProcessor**: `AudioProcessor/README.md` (если есть) | Audit: `AudioProcessor/docs/audit_v1/AP_AUDIT_CRITERIA.md`
- **TextProcessor**: `TextProcessor/README.md`

### Visual core providers (Tier‑0)

- **core_clip**: `VisualProcessor/core/model_process/core_clip/README.md`
- **content_domain**: `VisualProcessor/core/model_process/content_domain/README.md`
- **franchise_recognition**: `VisualProcessor/core/model_process/franchise_recognition/README.md`
- **ocr_extractor**: `VisualProcessor/core/model_process/ocr_extractor/README.md`
- **core_face_landmarks**: `VisualProcessor/core/model_process/core_face_landmarks/README.md`
- **core_depth_midas**: `VisualProcessor/core/model_process/depth_midas/README.md`
- **core_object_detections**: `VisualProcessor/core/model_process/object_detections/README.md`
- **core_optical_flow**: `VisualProcessor/core/model_process/core_optical_flow/README.md`

### Visual modules

- **cut_detection**: `VisualProcessor/modules/cut_detection/README.md`
- **shot_quality**: `VisualProcessor/modules/shot_quality/README.md`
- **scene_classification**: `VisualProcessor/modules/scene_classification/README.md`
- **video_pacing**: `VisualProcessor/modules/video_pacing/README.md`
- **uniqueness**: `VisualProcessor/modules/uniqueness/README.md`
- **story_structure**: `VisualProcessor/modules/story_structure/README.md`
- **optical_flow**: `VisualProcessor/modules/optical_flow/README.md`
- **micro_emotion**: `VisualProcessor/modules/micro_emotion/README.md`

### Audio extractors (Tier‑0 baseline)

- **clap_extractor**: `AudioProcessor/src/extractors/clap_extractor/README.md`
- **tempo_extractor**: `AudioProcessor/src/extractors/tempo_extractor/README.md`
- **loudness_extractor**: `AudioProcessor/src/extractors/loudness_extractor/README.md`

### Other

- **DatasetBuilder**: `DatasetBuilder/README.md`
- **BatchRunner**: `BatchRunner/README.md`
- **Docs entry**: `docs/README.md`


