# Component Graph — Index

Декларативный DAG компонентов DataProcessor: [component_graph.yaml](component_graph.yaml)  
Runtime loader: `DataProcessor/dag/component_graph.py`

---

## Stages (версия 0.1)

| Stage | Назначение | Узлов (approx) |
|-------|------------|----------------|
| `baseline` | Segmenter + visual core/modules + audio tier-0 | полный visual DAG + clap/loudness/tempo |
| `audio_extended` | segmenter + ASR, diarization, emotion, source_separation, speech_analysis | 6 |
| `text_processor_tier0` | tags → lexico / embedders (часть text) | 5 |
| `v1`, `v2` | зарезервированы | 0 |

**Важно:** hard deps проверяются **внутри одного stage**. Cross-stage зависимости не валидируются автоматически.

---

## Что в `baseline` (visual)

**Core providers:** `core_clip`, `core_object_detections`, `core_depth_midas`, `core_optical_flow`, `core_face_landmarks`, `ocr_extractor`, `content_domain`, `franchise_recognition`

**Identity:** `brand_semantics`, `car_semantics`, `place_semantics`, `face_identity`

**Modules:** `cut_detection`, `shot_quality`, `scene_classification`, `video_pacing`, `uniqueness`, `story_structure`, `color_light`, `action_recognition`, `behavioral`, `emotion_face`, `detalize_face`, `micro_emotion`, `frames_composition`, `high_level_semantic`, `optical_flow`, `similarity_metrics`, `text_scoring`

**Не в DAG (намеренно):** `failing_module` — test utility

---

## Text / Audio вне полного baseline

Полный порядок text (22) — [TextProcessor/docs/EXTRACTOR_DEPENDENCIES.md](../../TextProcessor/docs/EXTRACTOR_DEPENDENCIES.md)  
Расширенный audio — stage `audio_extended`

**Prod TODO:** stage `text_processor_full` со всеми 22 узлами и deps в одном stage.

---

## Проверка графа

```bash
cd "/media/ilya/Новый том/TrendFlowML/DataProcessor"
python3 - <<'PY'
import yaml
from pathlib import Path
from dag.component_graph import ComponentGraph

path = Path("docs/reference/component_graph.yaml")
data = yaml.safe_load(path.read_text(encoding="utf-8"))
for stage in ("baseline", "audio_extended", "text_processor_tier0"):
    g = ComponentGraph.from_yaml_dict(data, stage=stage)
    print(f"{stage}: {len(g.nodes)} nodes, topo_ok")
PY
```

---

## Связанные документы

- [VisualProcessor/docs/EXTRACTOR_DEPENDENCIES.md](../../VisualProcessor/docs/EXTRACTOR_DEPENDENCIES.md)
- [AudioProcessor/docs/EXTRACTOR_DEPENDENCIES.md](../../AudioProcessor/docs/EXTRACTOR_DEPENDENCIES.md)
- [TextProcessor/docs/EXTRACTOR_DEPENDENCIES.md](../../TextProcessor/docs/EXTRACTOR_DEPENDENCIES.md)
---

## Навигация

[DataProcessor](../MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
