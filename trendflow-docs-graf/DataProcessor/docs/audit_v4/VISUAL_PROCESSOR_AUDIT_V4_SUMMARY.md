# Audit v4 — VisualProcessor: сводка по компонентам

**Дата среза:** 2026-04-14. Источник статусов и путей: [`RUN_LOG.md`](RUN_LOG.md). Уровень отчёта по видео-компонентам: в основном **L2 (A+B, 5 прогонов)**. **L3** и **§8** (`AUDIT_4_CRITERIA_AND_PLAN.md`) нигде не закрыты — статус «passed» не применяется.

## Легенда

| Поле | Смысл |
|------|--------|
| **Оценка** | субъективная сводка зрелости по текущему аудиту: контракт NPZ, инварианты NaN/масок, покрытие B, наличие tooling/stats, очевидные блокеры |
| **Риск** | что может сломать downstream или воспроизводимость, если не учесть |

Шкала **оценки** (кратко): **A** — инварианты и L2 стабильны, блокеров нет; **B** — L2 есть, но есть известные edge-case / вырожденный B / доля NaN; **C** — неполный L2 или известный блокер в данных или коде.

---

## Core (`VisualProcessor/core/model_process`)

| Компонент | Уровень | Статус (RUN_LOG) | Stats JSON (L2) | Оценка | Риск |
|-----------|---------|------------------|-----------------|--------|------|
| `core_clip` | L2 | in_progress (v4 L2) | `storage/audit_v4/core_clip_l2/core_clip_audit_v4_stats.json` | B | Top-K / similarity-скоры не вероятности; NaN на idx 0 у cosine — ожидаемо |
| `core_depth_midas` | L2 | in_progress (v4 L2) | `storage/audit_v4/core_depth_midas_l2/core_depth_midas_audit_v4_stats.json` | B | Зависимость от preview/оси кадров |
| `core_face_landmarks` | L2 | in_progress (v4 L2) | `storage/audit_v4/core_face_landmarks_l2/core_face_landmarks_audit_v4_stats.json` | B | Сочетание `face_mesh_ran` и отсутствия face — гейтинг |
| `core_object_detections` | L2 | in_progress (v4 L2) | `storage/audit_v4/core_object_detections_l2/core_object_detections_audit_v4_stats.json` | A− | Только `valid_mask` как источник истины для слотов |
| `core_optical_flow` | L2 | in_progress (v4 L2) | `storage/audit_v4/core_optical_flow_l2/core_optical_flow_audit_v4_stats.json` | A− | NaN на idx 0 для flow-рядов — явная политика |
| `ocr_extractor` | L2 | in_progress (v4 L2) | `storage/audit_v4/ocr_extractor_l2/ocr_extractor_audit_v4_stats.json` | B | Privacy defaults скрывают сырой текст — ок для продукта, не для отладки OCR |

**Отчёты:** [`components/visual_processor/core/`](components/visual_processor/core/). **Engineering 4.2:** [`components/audit_4_2/visual_processor/core/`](components/audit_4_2/visual_processor/core/).

---

## Modules (`VisualProcessor/modules`)

| Компонент | Уровень | Статус (RUN_LOG) | Stats JSON (L2) | Оценка | Риск |
|-----------|---------|------------------|-----------------|--------|------|
| `action_recognition` | L2 | in_progress (v4 L2) | `storage/audit_v4/action_recognition_l2/action_recognition_audit_v4_stats.json` | C | На B `num_clips=1` — метрики динамики не валидируются; **операционно (2026-04-22):** E2E `exit 4` устраняется санитарией `PYTORCH_CUDA_ALLOC_CONF` в `action_recognition/main.py`; отладка цепочки YOLO→SlowFast — `VisualProcessor/.vp_venv` + `configs/audit_v3/visual/visual_minimal_object_detections_action_recognition.yaml` ([RUN_LOG.md](RUN_LOG.md)) |
| `behavioral` | L2 | in_progress (v4 L2) | `storage/audit_v4/behavioral_l2/behavioral_audit_v4_stats.json` | B | Мало кадров с landmarks; вторичные NaN при present |
| `color_light` | L2 | in_progress (v4 L2) | `storage/audit_v4/color_light_l2/color_light_audit_v4_stats.json` | B | Стабильный набор NaN-ключей в video_features |
| `cut_detection` | L2 | in_progress (v4 L2) | `storage/audit_v4/cut_detection_l2/cut_detection_audit_v4_stats.json` | B | Динамические имена NPZ; deep-ветка не активна на выборке |
| `detalize_face` | L2 | in_progress (v4 L2) | `storage/audit_v4/detalize_face_l2/detalize_face_audit_v4_stats.json` | B | Очень низкая доля `primary_valid`; компактные нули |
| `emotion_face` | L2 | in_progress (v4 L2) | `storage/audit_v4/emotion_face_l2/emotion_face_audit_v4_stats.json` | B | Редкие `face_present` / `processed_mask` |
| `frames_composition` | L2 | in_progress (v4 L2) | `storage/audit_v4/frames_composition_l2/frames_composition_audit_v4_stats.json` | B | Корреляции признаков — мультиколлинеарность |
| `high_level_semantic` | L2 | in_progress (v4 L2) | `storage/audit_v4/high_level_semantic_l2/high_level_semantic_audit_v4_stats.json` | B | Часть аудио-признаков NaN; `T=0` без text_processor |
| `micro_emotion` | L2 | **blocked** | `storage/audit_v4/micro_emotion_l2/micro_emotion_audit_v4_stats.json` (4 NPZ) | C | Ошибка PCA на одном B-run; нет полного набора из 5 OK |
| `optical_flow` | L2 | in_progress (v4 L2) | `storage/audit_v4/optical_flow_l2/optical_flow_audit_v4_stats.json` | B | Высокая доля missing в кривых/матрице — семантика полей |
| `scene_classification` | L2 | in_progress (v4 L2) | `storage/audit_v4/scene_classification_l2/scene_classification_audit_v4_stats.json` | B | Top-K не суммируется в 1 — ожидаемый «срез» |
| `shot_quality` | L2 | in_progress (v4 L2) | `storage/audit_v4/shot_quality_l2/shot_quality_audit_v4_stats.json` | B | Часть фичей all-NaN на всей выборке |
| `similarity_metrics` | L2 | in_progress (v4 L2) | `storage/audit_v4/similarity_metrics_l2/similarity_metrics_audit_v4_stats.json` | B | Нет reference на выборке — ограниченная finite-зона |
| `story_structure` | L2 | in_progress (v4 L2) | `storage/audit_v4/story_structure_l2/story_structure_audit_v4_stats.json` | B | Экстремальные ratio; нет topic_shift curve на 5 run |
| `text_scoring` | L2 | in_progress (v4 L2) | `storage/audit_v4/text_scoring_l2/text_scoring_audit_v4_stats.json` | B | OCR raw пуст из privacy — ожидаемо |
| `uniqueness` | L2 | in_progress (v4 L2) | `storage/audit_v4/uniqueness_l2/uniqueness_audit_v4_stats.json` | B | Контентно-зависимые repetition metrics |
| `video_pacing` | L2 | in_progress (v4 L2) | `storage/audit_v4/video_pacing_l2/video_pacing_audit_v4_stats.json` | B | Optional блоки дают стабильные NaN |

**Отчёты:** [`components/visual_processor/modules/`](components/visual_processor/modules/). **Engineering 4.2:** [`components/audit_4_2/visual_processor/modules/`](components/audit_4_2/visual_processor/modules/).

---

## Общий итог

- **22** визуальных компонента в контуре Audit v4 (6 core + 1 OCR + 15 модулей с полным L2 в RUN_LOG; `micro_emotion` с пометкой **blocked** из-за одного failed run).
- Для **всех** актуален следующий шаг плана: набор **C**, golden / §4.8, и формальный **L3** при необходимости.
- Наиболее явный блокер: **`micro_emotion`** (PCA / недостаточно признаков на одном видео). Наиболее «слабая» валидация смысла метрик: **`action_recognition`** (вырожденная временная ось на текущих5 run).
- Структура документов 4.2: [`components/audit_4_2/README.md`](components/audit_4_2/README.md).
---

## Навигация

[Audit v4 hub](components/audit_4_2/README.md) · [DataProcessor](../MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
