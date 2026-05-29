# DataProcessor — Wave 5: API, Orchestration, Monitoring, Scripts

Этап нормализации operational-слоя (portfolio + production).  
Журнал: [PORTFOLIO_PROGRESS_LOG.md](PORTFOLIO_PROGRESS_LOG.md)  
План: [PORTFOLIO_NORMALIZATION_PLAN.md](PORTFOLIO_NORMALIZATION_PLAN.md)

---

## Статус: `done` (2026-05-28)

---

## 1. Canonical entry points (куда идти за запуском)

| Сценарий | Canonical doc / path |
|----------|----------------------|
| Локальный полный пайплайн (CLI) | `DataProcessor/main.py` + `configs/global_config.yaml` |
| HTTP API + worker | [api/README.md](../api/README.md), [api/docs/INDEX.md](../api/docs/INDEX.md) |
| Celery queue | `dp_queue/` (см. [TOP_LEVEL_LAYOUT.md](TOP_LEVEL_LAYOUT.md)) |
| Component DAG | `dag/component_graph.py` + [reference/component_graph.yaml](reference/component_graph.yaml) |
| E2E stack (backend + DP + Triton) | `backend/docs/E2E_RUNBOOK.md`, `backend/scripts/start_e2e_stack.sh` |
| Observability | [monitoring/README.md](../monitoring/README.md), [audit_v4/OBSERVABILITY_STACK_LOCAL_E2E.md](audit_v4/OBSERVABILITY_STACK_LOCAL_E2E.md) |
| Скрипты (индекс) | [scripts/MAIN_INDEX.md](../scripts/MAIN_INDEX.md) |
| Контракты | [contracts/CONTRACTS_OVERVIEW.md](contracts/CONTRACTS_OVERVIEW.md) |

**Prod-правило:** один сценарий = одна canonical doc entry (не дублировать в 3 README).

---

## 2. `api/` — структура

| Путь | Роль |
|------|------|
| `main.py` | FastAPI app |
| `worker.py` | Background worker |
| `endpoints/` | REST handlers |
| `services/` | processor, metrics, redis |
| `docs/` | API architecture, checklist, implementation stages |
| `tests/` | API tests |

**Ключевые docs:**
- [DATAPROCESSOR_API_ARCHITECTURE.md](../api/docs/DATAPROCESSOR_API_ARCHITECTURE.md)
- [API_DEVELOPMENT_CHECKLIST.md](../api/docs/API_DEVELOPMENT_CHECKLIST.md)
- [ENDPOINTS.md](../api/docs/ENDPOINTS.md)
- [ENVIRONMENT_VARIABLES.md](../api/docs/ENVIRONMENT_VARIABLES.md)

**Action:** в `docs/MAIN_INDEX.md` добавить блок «Operational / API» со ссылкой сюда.

---

## 3. `dag/` + orchestration

| Файл | Роль |
|------|------|
| `dag/component_graph.py` | Runtime DAG loader |
| `docs/reference/component_graph.yaml` | Declarative MVP DAG (baseline) |

Processors (Audio/Text/Visual) имеют свои dependency docs:
- [AudioProcessor/docs/EXTRACTOR_DEPENDENCIES.md](../AudioProcessor/docs/EXTRACTOR_DEPENDENCIES.md)
- [TextProcessor/docs/EXTRACTOR_DEPENDENCIES.md](../TextProcessor/docs/EXTRACTOR_DEPENDENCIES.md)
- [VisualProcessor/docs/EXTRACTOR_DEPENDENCIES.md](../VisualProcessor/docs/EXTRACTOR_DEPENDENCIES.md)

**Action:** при изменении baseline — синхронизировать yaml + processor docs.

---

## 4. `monitoring/`

| Компонент | Путь |
|-----------|------|
| Prometheus | `monitoring/prometheus/` |
| Grafana | `monitoring/grafana/dashboards/dataprocessor-overview.json` |
| Docker | `DataProcessor/docker-compose.yml` (services prometheus/grafana/jaeger) |

См. [monitoring/README.md](../monitoring/README.md), [monitoring/METRICS_REFERENCE.md](../monitoring/METRICS_REFERENCE.md)

---

## 5. `scripts/` — классификация

Уже индексировано в [scripts/MAIN_INDEX.md](../scripts/MAIN_INDEX.md). Группы для prod-навигации:

| Класс | Примеры | Когда использовать |
|-------|---------|-------------------|
| **setup** | `prepare_hf_cache.sh`, `download_*_models.py`, `hf_download_all.sh` | Первичная настройка / восстановление models |
| **preflight** | `preflight/check_semantic_bases.py`, `preflight_triton.py` | Перед production/batch run |
| **smoke / QA** | `run_smoke_all_components.sh`, `validate_smoke_results.sh`, `run_all_quality_checks.sh` | CI / pre-release |
| **model_opt** | `export_*_onnx.py`, `quantize_onnx_dynamic.py` | Triton deployment |
| **baseline demo** | `scripts/baseline/demo_*_quality.py` | Качество одного компонента |
| **one-off / migration** | `reorganize_youtube_results.sh`, `fix_source_separation_model.py` | Только по runbook |
| **processor runners** | `run_text_processor_from_global_config.py`, `run_visual_full_test.py` | Dev / audit |

**Action:** не добавлять новые скрипты без строки в `scripts/MAIN_INDEX.md`.

---

## 6. Production operational checklist (Wave 5)

| # | Шаг |
|---|-----|
| 1 | `env.example` / [ENVIRONMENT_VARIABLES.md](../api/docs/ENVIRONMENT_VARIABLES.md) — все required vars заданы |
| 2 | Storage: `TREND_STORAGE_BACKEND` fs vs s3 ([storage/MAIN_INDEX.md](../storage/MAIN_INDEX.md)) |
| 3 | Redis + worker (если API mode) |
| 4 | Triton health + model repo |
| 5 | Prometheus scrape targets up |
| 6 | Smoke: Audio 21 + Text 22 + Visual core baseline |
| 7 | E2E: `backend/scripts/start_e2e_stack.sh` (перед релизом) |

---

## 7. DoD Wave 5

- [x] Создан этот документ (operational map)
- [x] Ссылка Wave 5 в `docs/MAIN_INDEX.md`
- [x] `env.example` очищен (дубль `DP_MODELS_ROOT`, локальный путь); drift зафиксирован ниже
- [ ] Пометить legacy/duplicate run instructions (если найдены)
- [x] Wave 6: [README.md](../README.md), [PORTFOLIO_INTERVIEW_GUIDE.md](PORTFOLIO_INTERVIEW_GUIDE.md)

---

## 8. Известные зоны дублирования (для последующей чистки)

- `docs/MAIN_INDEX.md` дублирует блоки DATAPROCESSOR_API_ARCHITECTURE (2×) — косметика
- `api/docs/IMPLEMENTATION/*` — исторический лог; не entry для операций
- `scripts/reorganize_youtube_results_*` — one-off migration

## 9. Env drift (API vs storage vs env.example)

| Переменная | Где используется | Заметка |
|------------|------------------|---------|
| `TREND_STORAGE_BACKEND`, `TREND_FS_ROOT` | `storage/settings.py` | CLI / main pipeline |
| `STORAGE_TYPE`, `STORAGE_ROOT` | API (`api/`) | HTTP worker path |
| `S3_*`, `AWS_*` | storage S3 + boto3 | Общие для MinIO |
| `CELERY_*` | `dp_queue/` | Queue mode |

**Prod:** при E2E задавать оба слоя согласованно (см. `backend/scripts/e2e_env.sh`). Полный список API vars: [ENVIRONMENT_VARIABLES.md](../api/docs/ENVIRONMENT_VARIABLES.md).
