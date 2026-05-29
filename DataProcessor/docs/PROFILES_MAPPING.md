# Profiles & Config Mapping (P2.2)

Как **Backend profile JSON**, **global_config YAML** и **visual cfg_path** сходятся в один run DataProcessor.

Связано: [backend/docs/PROFILES.md](../../backend/docs/PROFILES.md) · [configs/README.md](../configs/README.md) · [ENV_ALIGNMENT.md](ENV_ALIGNMENT.md)

---

## 1. Три слоя конфигурации

| Слой | Формат | Где живёт | Кто читает |
|------|--------|-----------|------------|
| **Analysis profile** | JSON (`processors`, `visual.cfg_path`) | Postgres `analysis_profiles` + seed из `DataProcessor/profiles/*.yaml` | Backend → payload в DataProcessor API |
| **global_config** | YAML (`processors.audio/text/visual` + extractors) | `DataProcessor/configs/*.yaml` | `main.py --global-config` (override деталей extractors) |
| **Visual cfg** | YAML (core_providers + modules) | `configs/audit_v3/visual/*.yaml` | `VisualProcessor/main.py` или inline в `global_config` |

**Правило:** profile JSON задаёт **включение процессоров** (`enabled` / `required`).  
Если задан `global_config_path`, детали extractors берутся из YAML; иначе — из дефолтов orchestrator + `visual.cfg_path`.

---

## 2. Platform → run (Backend E2E / ingestion)

```
Fetcher finalize
    → Backend POST /api/v1/runs/{id}/trigger-processing
    → Celery process_ingestion_run
    → build IngestionPayloadFromFetcher (dataprocessor_adapter.py)
    → POST DataProcessor /api/v1/process
    → Redis stream → dataprocessor-worker
    → subprocess DataProcessor/main.py
    → storage/result_store/... + manifest.json
```

### Какой profile уходит в DP

| Условие | `profile_config` | Processors |
|---------|------------------|------------|
| **Дефолт E2E** (нет override) | `_default_ingestion_profile_config` | segmenter ✅; audio/text/visual ❌ |
| **`TF_BACKEND_DATAPROCESSOR_GLOBAL_CONFIG_PATH`** или marker `storage/e2e_full_max/active_global_config` | `_ingestion_profile_with_global_config_yaml` + `global_config_path` в payload | segmenter ✅; audio/text/visual ✅ (детали — в YAML) |
| **Analysis job** (не ingestion smoke) | JSON из `analysis_profiles` по `processing_config_id` | по профилю пользователя |

Toggle для tier-0 audio E2E:

```bash
export E2E_USE_PORTFOLIO_DEMO_CONFIG=1
# → TF_BACKEND_DATAPROCESSOR_GLOBAL_CONFIG_PATH=.../configs/portfolio_demo.yaml
./backend/scripts/start_e2e_stack.sh
```

См. [E2E_PREFLIGHT.md](E2E_PREFLIGHT.md).

---

## 3. Seed-профили (`DataProcessor/profiles/`)

| Файл | Назначение |
|------|------------|
| [profiles/config.yaml](../profiles/config.yaml) | Публичный seed: visual only (`visual.cfg_path: configs/visual_config.yaml`), audio/text off |

На старте Backend API: `seed_public_profiles(db, dataproc_root/profiles/)` — создаёт public записи в БД по stem имени файла.

**Монорепо:** `dataproc_root` = `DataProcessor/` (см. `backend/app/config.py` → `resolve_paths`).

---

## 4. YAML-конфиги по сценарию

| Сценарий | Конфиг | Profile / override |
|----------|--------|-------------------|
| E2E smoke (segmenter only) | — | Backend default ingestion profile |
| E2E audio tier-0 | [portfolio_demo.yaml](../configs/portfolio_demo.yaml) | `E2E_USE_PORTFOLIO_DEMO_CONFIG=1` |
| E2E full multimodal | [global_config.yaml](../configs/global_config.yaml) | `e2e_full_max_run.py` или marker `active_global_config` |
| Audio regression 21/21 | [audit_v3/audio/profile_*.yaml](../configs/audit_v3/audio/) | CLI `--global-config` per component |
| Visual YOLO → AR | [visual_minimal_object_detections_action_recognition.yaml](../configs/audit_v3/visual/visual_minimal_object_detections_action_recognition.yaml) | `VisualProcessor/main.py --cfg-path` |
| Prod-like local demo | [portfolio_demo.yaml](../configs/portfolio_demo.yaml) | CLI или E2E toggle |

Полный индекс: [configs/README.md](../configs/README.md).

---

## 5. Backend defaults (важные пути)

| Setting | Env | Default (монорепо) |
|---------|-----|-------------------|
| `dataproc_root` | `TF_BACKEND_DATAPROC_ROOT` | `<repo>/DataProcessor` |
| `visual_cfg_default` | `TF_BACKEND_VISUAL_CFG_DEFAULT` | `configs/audit_v3/visual/visual_core_5_only.yaml` |
| `dataprocessor_global_config_path` | `TF_BACKEND_DATAPROCESSOR_GLOBAL_CONFIG_PATH` | unset → segmenter-only E2E |

Resolver global_config (приоритет):  
1) `storage/e2e_full_max/active_global_config` (первая строка — абс. путь)  
2) `TF_BACKEND_DATAPROCESSOR_GLOBAL_CONFIG_PATH`

Код: `backend/app/services/dataprocessor_adapter.py` → `resolve_dataprocessor_global_config_path`.

---

## 6. `config_hash`

SHA-256 от JSON profile с сортировкой ключей (`compute_config_hash`).  
Прокидывается в manifest для воспроизводимости.  
Ingestion E2E использует фиксированные строки (`ingestion-e2e-segmenter-only`, `ingestion-e2e-full-max-global-yaml`) — не путать с user profiles.

---

## 7. Связанные документы

- [API_WORKER_RUNBOOK.md](API_WORKER_RUNBOOK.md) — POST /process → worker → result_store
- [GPU_MACHINE_RUNBOOK.md](GPU_MACHINE_RUNBOOK.md) — закрытие GPU-only smoke на второй машине
- [PORTFOLIO_DEMO_RUNBOOK.md](PORTFOLIO_DEMO_RUNBOOK.md) — пошаговые demo-сценарии
