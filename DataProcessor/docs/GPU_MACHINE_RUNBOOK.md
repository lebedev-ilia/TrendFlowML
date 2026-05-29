# GPU Machine Runbook (Phase 8 closure)

Чеклист для **второй машины с CUDA** (ноутбук `fetcher-dev` или GPU-ПК): закрыть блокеры CPU-only прогона и подготовить full multimodal E2E.

Предпосылка на CPU-ПК уже зелёная: segmenter + audio tier-0 E2E, 20/21 audio NPZ, visual detections NPZ.

Связано: [PRODUCTION_HARDENING_PLAN.md](PRODUCTION_HARDENING_PLAN.md) · [ENV_ALIGNMENT.md](ENV_ALIGNMENT.md) · [E2E_PREFLIGHT.md](E2E_PREFLIGHT.md)

---

## 0. Sync

```bash
git checkout system-testing
git pull origin system-testing
# ожидаемые коммиты Phase 8: 6554494 (E2E audio), 908a017 (smoke gates)
```

---

## 1. One-time setup

```bash
export REPO_ROOT="$(pwd)"   # абсолютный путь без опечаток
export DP_MODELS_ROOT="${REPO_ROOT}/DataProcessor/dp_models/bundled_models"
export TORCH_HOME="${DP_MODELS_ROOT}/torch_cache"
export HF_HOME="${DP_MODELS_ROOT}/hf_cache"

# Проверка CUDA
nvidia-smi

# Preflight models
./DataProcessor/scripts/prepare_hf_cache.sh
DataProcessor/.data_venv/bin/python3 DataProcessor/scripts/dp_models_selftest.py
```

**AudioProcessor/.env:** если старый путь — переопределить `DP_MODELS_ROOT` в shell (см. Entry 018).

---

## 2. Audio 21/21 (GPU gate)

```bash
cd "${REPO_ROOT}"
./DataProcessor/scripts/run_smoke_all_components.sh
./DataProcessor/scripts/validate_smoke_results.sh
```

**DoD:** `21/21` валидных NPZ (включая `emotion_diarization`).

Ожидаемое время: ~30–40 мин (полный прогон).

---

## 3. Visual minimal + AR (GPU gate)

```bash
cd DataProcessor

# Segmenter (если frames_dir ещё нет)
# ... или использовать существующий storage/frames_dir/-Q6fnPIybEI/video/

VisualProcessor/.vp_venv/bin/python VisualProcessor/main.py \
  --cfg-path configs/audit_v3/visual/visual_minimal_object_detections_action_recognition.yaml

cd ..
./DataProcessor/scripts/validate_visual_minimal.sh
```

**DoD:** detections ✅ + `action_recognition.npz` ✅ (без SKIP).

**Известный фикс:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` из `e2e_env.sh` — санитария в `action_recognition/main.py` (см. audit_v4 RUN_LOG).

---

## 4. Triton preflight (перед visual core / full-max)

```bash
# Поднять Triton (порт 8010 на хосте, не 8000–8002)
./backend/scripts/e2e_triton_docker.sh   # или docker compose из DataProcessor/triton/

export TRITON_HTTP_URL=http://127.0.0.1:8010
DataProcessor/.data_venv/bin/python3 DataProcessor/scripts/preflight_triton.py \
  --base-url "${TRITON_HTTP_URL}" --preset core_low
```

---

## 5. Full E2E multimodal

```bash
./backend/scripts/setup_e2e_infra.sh
./backend/scripts/start_e2e_stack.sh --with-infra

# Вариант A: global_config через full-max runner
cd backend && source scripts/e2e_env.sh
.venv/bin/python scripts/e2e_full_max_run.py --help

# Вариант B: явный global_config
export TF_BACKEND_DATAPROCESSOR_GLOBAL_CONFIG_PATH="${REPO_ROOT}/DataProcessor/configs/global_config.yaml"
./backend/scripts/stop_e2e_stack.sh --quiet
./backend/scripts/start_e2e_stack.sh --no-stop
.venv/bin/python scripts/e2e_run_to_complete.py \
  --source-url "https://www.youtube.com/watch?v=-Q6fnPIybEI" \
  --with-dataprocessor --timeout 7200
```

**DoD:** ingestion `completed`, DP processors success в status, manifest в `storage/result_store/`.

---

## 6. Запись результатов

Добавить Entry в [PORTFOLIO_PROGRESS_LOG.md](PORTFOLIO_PROGRESS_LOG.md):

- machine / GPU model
- audio validate: N/21
- visual validate: AR ok/fail
- E2E run_id + duration
- blockers (если остались)

---

## 7. Быстрый tier-0 (без full-max)

Если нужен только smoke E2E с audio без visual:

```bash
export E2E_USE_PORTFOLIO_DEMO_CONFIG=1
./backend/scripts/start_e2e_stack.sh --no-stop
# ... e2e_run_to_complete.py (~4 min)
```

Уже проверено на основном ПК (Entry 025).

---

## Связанные документы

- [PROFILES_MAPPING.md](PROFILES_MAPPING.md) — как profile ↔ global_config
- [API_WORKER_RUNBOOK.md](API_WORKER_RUNBOOK.md) — worker path
- [PORTFOLIO_DEMO_RUNBOOK.md](PORTFOLIO_DEMO_RUNBOOK.md) — demo A/B/C
