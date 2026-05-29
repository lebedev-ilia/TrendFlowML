# DataProcessor — E2E Preflight & Runbook (P3.4)

Полный стек: **Backend → Fetcher → DataProcessor API/worker**.  
Скрипты в `backend/scripts/`. Env: [ENV_ALIGNMENT.md](ENV_ALIGNMENT.md) · `backend/scripts/e2e_env.sh`

---

## 1. Preflight (перед `start_e2e_stack.sh`)

| # | Проверка | Команда / критерий |
|---|----------|-------------------|
| 1 | Docker | `docker info` OK |
| 2 | Порты свободны | 8000 Fetcher, 8001 Backend, 8002 DP API, 5433 Postgres, 6379 Redis, 9000 MinIO |
| 3 | Models | `DP_MODELS_ROOT` → `bundled_models` (см. P0.1) |
| 4 | Storage sync | `STORAGE_ROOT` == `TREND_FS_ROOT` (e2e_env.sh) |
| 5 | Visual (optional) | Triton `8010` или `--with-triton-docker` в full-max run |

---

## 2. Быстрый старт (полный stack)

```bash
cd /path/to/TrendFlowML

# env (один shell)
source backend/scripts/e2e_env.sh

# инфра + сервисы (первый раз — долго)
./backend/scripts/start_e2e_stack.sh --with-infra

# остановка
./backend/scripts/stop_e2e_stack.sh
```

Логи: `backend/.e2e/logs/latest/`

---

## 3. DataProcessor-only smoke (без полного Fetcher cycle)

### A) Лёгкий tier-0 (рекомендуется после краша ПК)

```bash
export DP_MODELS_ROOT="/path/to/DataProcessor/dp_models/bundled_models"
python3 DataProcessor/main.py \
  --video-path example/example_videos/-Q6fnPIybEI.mp4 \
  --global-config DataProcessor/configs/portfolio_demo.yaml \
  --platform-id youtube --video-id e2e_smoke --run-id run_1 \
  --rs-base DataProcessor/dp_results/portfolio_demo \
  --no-run-visual
```

### B) E2E script (нужен готовый Fetcher run_id + поднятый stack)

```bash
cd backend
source scripts/e2e_env.sh
.venv/bin/python scripts/e2e_dataprocessor_audio_smoke.py --fetcher-run-id <uuid>
```

### C) Full-max (тяжёлый, GPU/Triton)

```bash
cd backend
source scripts/e2e_env.sh
.venv/bin/python scripts/e2e_full_max_run.py --help
```

---

## 4. Критерий «P3.4 green»

- [ ] `start_e2e_stack.sh --with-infra` — все PID живы, health endpoints OK
- [ ] Минимум один прогон: Fetcher finalize → DP worker → `manifest.json` status success
- [ ] Запись в [PORTFOLIO_PROGRESS_LOG.md](PORTFOLIO_PROGRESS_LOG.md) (run_id, дата, ветка)

---

## 5. Связанные документы

- [PRODUCTION_HARDENING_PLAN.md](PRODUCTION_HARDENING_PLAN.md)
- [CI_SMOKE.md](CI_SMOKE.md) — CI без infra
- [PORTFOLIO_DEMO_RUNBOOK.md](PORTFOLIO_DEMO_RUNBOOK.md)
- `docs/GITHUB_WORKFLOW_TWO_DEVICES.md` — sync веток
