# DataProcessor — Portfolio Demo Runbook

Пошаговые сценарии для **живой демонстрации** (портфолио / собеседование / prod sanity).  
Требует: Linux, Python 3.10+, `ffmpeg`, GPU опционально (Triton для visual core).

Связано: [PORTFOLIO_INTERVIEW_GUIDE.md](PORTFOLIO_INTERVIEW_GUIDE.md) · [../README.md](../README.md)

---

## 0. Подготовка (один раз)

```bash
cd "/media/ilya/Новый том/TrendFlowML"

# env
cp DataProcessor/env.example .env
# отредактировать: DP_MODELS_ROOT=/abs/path/to/DataProcessor/dp_models/bundled_models

export DP_MODELS_ROOT="/abs/path/to/DataProcessor/dp_models/bundled_models"

# модели (если bundled_models пустой)
# export HF_TOKEN=...
# ./DataProcessor/scripts/hf_download_all.sh

# HF cache для emotion_diarization (audio smoke)
./DataProcessor/scripts/prepare_hf_cache.sh
```

**Тестовое видео:** `example/example_videos/video1.mp4` (или свой файл 5–120 с).

**Проверка DAG:**

```bash
cd DataProcessor && python3 - <<'PY'
import yaml
from pathlib import Path
from dag.component_graph import ComponentGraph
data = yaml.safe_load(Path("docs/reference/component_graph.yaml").read_text())
g = ComponentGraph.from_yaml_dict(data, stage="baseline")
print("baseline:", len(g.nodes), "components")
PY
```

---

## Demo A — Visual minimal (5–10 мин, GPU + Triton)

Показывает: Segmenter contract → object detections → action recognition.

**Нужно:** Triton на `http://localhost:8000`, visual venv `DataProcessor/VisualProcessor/.vp_venv`

```bash
cd DataProcessor

# 1) Triton (отдельный терминал) — см. triton/README.md
# docker compose --profile triton up -d   # из DataProcessor/

# 2) Прогон
VisualProcessor/.vp_venv/bin/python VisualProcessor/main.py \
  --cfg-path configs/audit_v3/visual/visual_minimal_object_detections_action_recognition.yaml

# 3) Артефакты
ls -la ../storage/result_store_ar_minimal/youtube/*/ar_minimal_cli_001/
```

**Рассказ:** hard dep `core_object_detections` → `action_recognition`, NPZ + schema_version в meta.

---

## Demo B — Audio tier-0 smoke (10–15 мин, CPU/GPU)

Показывает: 21 extractor, Segmenter audio contract, result_store.

```bash
cd "/media/ilya/Новый том/TrendFlowML"
chmod +x DataProcessor/scripts/run_smoke_all_components.sh
./DataProcessor/scripts/run_smoke_all_components.sh

./DataProcessor/scripts/validate_smoke_results.sh
# ожидание: 21/21
```

**Артефакты:** `DataProcessor/dp_results/smoke_test/youtube/smoke_*`

**Рассказ:** families в `audio/segments.json`, fail-fast без segments.

---

## Demo C — Text one extractor (5 мин, CPU)

Показывает: VideoDocument + ModelManager e5-large + изолированный extractor.

```bash
cd DataProcessor/TextProcessor
export DP_MODELS_ROOT=/abs/path/to/dp_models/bundled_models

./.tp_venv/bin/python scripts/smoke_each_extractor_audit_v3.py \
  --scenario-index 0
# 22 прогона подряд — долго; для демо достаточно одного extractor в скрипте или --help
```

Сценарии: `example/text_audit_v3_smoke/scenarios/README.md`

---

## Demo D — Full pipeline CLI (15+ мин, тяжёлый)

Показывает: `main.py` orchestrator, multimodal run.

```bash
cd "/media/ilya/Новый том/TrendFlowML"

python3 DataProcessor/main.py \
  --video-path "example/example_videos/video1.mp4" \
  --global-config DataProcessor/configs/global_config.yaml \
  --run-audio \
  --run-visual \
  --platform-id youtube \
  --video-id portfolio_demo \
  --run-id run_1 \
  --output-dir "DataProcessor/dp_results"
```

**Перед прогоном:** в `global_config.yaml` включить нужные processors (`audio.enabled`, visual profile).

**Показать после:**

```bash
RUN="DataProcessor/dp_results/youtube/portfolio_demo/run_1"
cat "$RUN/manifest.json" | python3 -m json.tool | head -60
ls "$RUN"
```

---

## Demo E — E2E stack (prod narrative, 20+ мин)

Показывает: Backend + Redis + DataProcessor API + monitoring.

```bash
cd "/media/ilya/Новый том/TrendFlowML"
source backend/scripts/e2e_env.sh
backend/scripts/start_e2e_stack.sh
```

Док: `backend/docs/E2E_RUNBOOK.md`, Grafana: `DataProcessor/monitoring/README.md`

---

## Что показать в NPZ (30 сек)

```bash
python3 - <<'PY'
import numpy as np
from pathlib import Path
import sys
p = Path(sys.argv[1])
d = np.load(p, allow_pickle=True)
print("keys:", sorted(d.files)[:20])
if "meta" in d.files:
    print("meta keys:", list(d["meta"].item().keys())[:15])
PY
"$RUN/clap_extractor/clap_extractor_features.npz"
```

Акцент: `feature_names` / `feature_values`, `meta.schema_version`, `models_used`.

---

## Troubleshooting

| Симптом | Действие |
|---------|----------|
| `segments.json missing families.*` | Segmenter не создал family для extractor → fail-fast OK |
| Triton connection refused | Запустить triton profile / проверить `TRITON_HTTP_URL` |
| `DP_MODELS_ROOT` not found | `hf_download_all` или указать bundled_models |
| Text smoke: e5 model missing | Проверить `text/embeddings/intfloat_multilingual-e5-large` |
| OOM on visual | Уменьшить batch в config / короче видео |

---

## Рекомендуемый набор для 20-мин интервью

1. **Demo A** (visual slice) — 7 мин  
2. Открыть `cut_detection/README.md` + NPZ meta — 3 мин  
3. **Demo B** validate 21/21 (можно заранее прогнать) — 2 мин  
4. Архитектура: `README.md` + `TOP_LEVEL_LAYOUT.md` — 5 мин  
5. Tech debt + prod path — 3 мин  
