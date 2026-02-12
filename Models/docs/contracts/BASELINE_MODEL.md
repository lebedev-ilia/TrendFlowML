## Baseline model (Boosting)

### Роль

Baseline — “контрольная точка качества” и production fallback (degraded-mode).

### Inputs (v1.0)

Baseline использует:

**1) Visual modules (7)**:
- `cut_detection`
- `optical_flow`
- `scene_classification`
- `shot_quality`
- `story_structure`
- `uniqueness`
- `video_pacing`

**2) Audio extractors (3)**:
- `clap_extractor`
- `loudness_extractor`
- `tempo_extractor`

**3) Snapshot_0 fields** (см. `TARGETS_SPLITS_METRICS.md`).

**4) Required core providers**:
- `core_brand_semantics`
- `core_car_semantics`
- `core_clip`
- `core_face_identity`
- `core_face_landmarks`
- `core_optical_flow`
- `core_place_semantics`
- `core_depth_midas`
- `core_object_detections`

### Outputs (v1.0)

- 2 модели: `views` и `likes`
- каждая multi-output: горизонты 7/14/21 (7d masked)

### Freeze policy (важно)

После начала baseline dataset collection:
- **запрещено менять** алгоритмы/выходы компонент, которые дали baseline фичи
- улучшения делаем через **новые компоненты/новые ветки артефактов**
- обязательно bump `feature_schema_version`

Режимы схемы:
- `feature_schema_version=v0`: допускаются изменения (логируем всё)
- `feature_schema_version=v1`: frozen (после этого baseline “финально” обучаем и закрепляем в проде)

### Resources envelope

Ориентиры окружения (для планирования latency/cost):
- GPU: 16GB (допускаем 24/32)
- RAM: 16GB
- CPU: 4 cores


