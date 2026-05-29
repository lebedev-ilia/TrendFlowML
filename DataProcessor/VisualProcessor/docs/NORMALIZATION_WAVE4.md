# VisualProcessor — Wave 4 Normalization

Этап нормализации `VisualProcessor` (portfolio + production).  
Управляющий план: [../../docs/PORTFOLIO_NORMALIZATION_PLAN.md](../../docs/PORTFOLIO_NORMALIZATION_PLAN.md)  
Журнал: [../../docs/PORTFOLIO_PROGRESS_LOG.md](../../docs/PORTFOLIO_PROGRESS_LOG.md)  
Шаблоны: [../../AudioProcessor/docs/NORMALIZATION_WAVE2.md](../../AudioProcessor/docs/NORMALIZATION_WAVE2.md), [../../TextProcessor/docs/NORMALIZATION_WAVE3.md](../../TextProcessor/docs/NORMALIZATION_WAVE3.md)

---

## Статус: `done` (документация и навигация; 2026-05-28)

---

## 1. Структура модуля (as-is)

| Зона | Путь | Роль |
|------|------|------|
| Entry | `main.py` | CLI orchestrator |
| **Core** | `core/model_process/*` | GPU/Triton providers, semantic heads |
| **Modules** | `modules/*` | Feature modules (pacing, cuts, face, …) |
| Schemas | `schemas/*.json` | Machine NPZ schemas |
| Docs | `docs/MAIN_INDEX.md` | Индекс компонентов |
| Utils | `utils/` | Renderer, shared helpers |

### Runtime / не source (локально в дереве VP)

| Путь | Policy | Action |
|------|--------|--------|
| `VisualProcessor/result_store/` | generated | Не путать с canonical `dp_results/` |
| `VisualProcessor/state/` | generated | Runtime state events (если пишется локально) |
| `__pycache__/` | generated | ignore |

**Prod:** canonical result store — `dp_results/` или S3 via `storage/` (см. [TOP_LEVEL_LAYOUT.md](../../docs/TOP_LEVEL_LAYOUT.md)).

---

## 2. Core (`core/model_process`) — inventory v1

| Component | Тип | Triton / GPU |
|-----------|-----|--------------|
| `core_clip` | embeddings | да |
| `core_depth_midas` | depth | да |
| `core_face_landmarks` | landmarks | MediaPipe / ONNX |
| `core_object_detections` | detections | YOLO / Triton |
| `core_optical_flow` | flow | RAFT / Triton |
| `ocr_extractor` | OCR | ONNX |
| `core_identity/*` | semantic heads | embeddings + offline DB |

**Identity sub-modules:** `brand_semantics`, `car_semantics`, `content_domain`, `franchise_recognition`, `place_semantics`, `core_face_identity`, …

---

## 3. Modules — inventory v1

| Module | Категория |
|--------|-----------|
| `cut_detection` | Tier-0 / pacing |
| `scene_classification` | scene |
| `shot_quality` | quality |
| `video_pacing` | pacing |
| `color_light` | scene-dependent |
| `optical_flow` | motion (module-level) |
| `action_recognition` | semantics |
| `behavioral` | behavior |
| `emotion_face` | face |
| `micro_emotion` | face (OpenFace) |
| `detalize_face` | face |
| `frames_composition` | composition |
| `high_level_semantic` | semantic |
| `similarity_metrics` | similarity |
| `uniqueness` | similarity |
| `story_structure` | structure |
| `text_scoring` | text-on-frame |
| `failing_module` | test / dev only? |

---

## 4. Doc layout (целевой)

По образцу audited компонентов:

```text
<component>/
  README.md
  SCHEMA.md (или schemas/*.json + SCHEMA.md)
  docs/FEATURE_DESCRIPTION.md
```

**Задача Wave 4:** сверить все core + modules на наличие README/SCHEMA/FEATURE_DESCRIPTION и единых ссылок в `docs/MAIN_INDEX.md`.

---

## 5. Upstream / cross-processor

| Upstream | Consumers (примеры) |
|----------|---------------------|
| **Segmenter** | все VP components (`frame_indices`, `frames_dir`) |
| `core_object_detections` | `core_face_landmarks`, `action_recognition`, semantics |
| `core_clip` | `scene_classification`, `high_level_semantic`, … |
| `cut_detection` | `scene_classification`, `video_pacing` |
| `core_face_landmarks` | `emotion_face`, `detalize_face`, `micro_emotion` |
| `scene_classification` | `color_light` |

Детальная карта — в `EXTRACTOR_DEPENDENCIES.md` (создать в Wave 4).

---

## 6. DoD Wave 4

- [x] Inventory core + modules с tier/required/optional
- [x] [EXTRACTOR_DEPENDENCIES.md](EXTRACTOR_DEPENDENCIES.md)
- [x] Doc coverage audit (29 components; `failing_module` exempt)
- [x] Runtime dirs — в EXTRACTOR_DEPENDENCIES + `.gitignore`
- [x] Prod smoke / e2e checklist (§6 EXTRACTOR_DEPENDENCIES)
- [x] Ссылки Wave 4 в `VisualProcessor/README.md`, `docs/MAIN_INDEX.md`

---

## 7. Следующий шаг

- `Wave 5`: API, orchestration, monitoring, scripts — единые runbooks
