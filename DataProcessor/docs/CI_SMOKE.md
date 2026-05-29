# DataProcessor — CI Smoke (P3.3)

Workflow: [`.github/workflows/dataprocessor-smoke.yml`](../../.github/workflows/dataprocessor-smoke.yml)

---

## Что проверяет CI (без GPU и без `bundled_models`)

| Step | Проверка |
|------|----------|
| component_graph | stages `baseline`, `audio_extended`, `text_processor_tier0`, `text_processor_full` — validate + topo |
| bash -n | `run_smoke_all_components.sh`, `validate_smoke_results.sh` |
| py_compile | `dp_models_selftest.py`, `dag/component_graph.py` |
| dp_models_selftest | unit tests ModelManager (без весов) |

---

## Что остаётся локально (P0 gate)

| Прогон | Команда | Машина |
|--------|---------|--------|
| Audio 21 | `./DataProcessor/scripts/run_smoke_all_components.sh` | models + CPU; emotion — GPU |
| Text 22 | `TextProcessor/.tp_venv` + `smoke_each_extractor_audit_v3.py --scenario-index 0` | e5-large bundle |
| Visual minimal | Segmenter + `visual_minimal_*.yaml` | GPU для AR |
| Лёгкий demo | `configs/portfolio_demo.yaml` | tier-0 audio |

См. [PRODUCTION_HARDENING_PLAN.md](PRODUCTION_HARDENING_PLAN.md), [ENV_ALIGNMENT.md](ENV_ALIGNMENT.md).

---

## Локальный pre-push (рекомендуется)

```bash
cd DataProcessor
python3 - <<'PY'
import yaml
from pathlib import Path
from dag.component_graph import ComponentGraph
data = yaml.safe_load(Path("docs/reference/component_graph.yaml").read_text())
for stage in ("baseline", "audio_extended", "text_processor_tier0", "text_processor_full"):
    ComponentGraph.from_yaml_dict(data, stage=stage)
    print("OK", stage)
PY
bash -n scripts/run_smoke_all_components.sh
```
