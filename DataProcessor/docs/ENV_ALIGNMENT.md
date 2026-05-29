# DataProcessor — Environment Alignment (P1.2)

Единая шпаргалка: **два слоя storage** + models + E2E.  
Связано: [NORMALIZATION_WAVE5.md](NORMALIZATION_WAVE5.md) §9 · [env.example](../env.example) · [api/docs/ENVIRONMENT_VARIABLES.md](../api/docs/ENVIRONMENT_VARIABLES.md)

---

## 1. Два слоя storage (не смешивать)

| Слой | Переменные | Код | Когда |
|------|------------|-----|-------|
| **CLI / main.py / storage adapter** | `TREND_STORAGE_BACKEND`, `TREND_FS_ROOT` | `storage/settings.py` | Локальный прогон, orchestrator |
| **HTTP API / worker** | `STORAGE_TYPE`, `STORAGE_ROOT` | `api/` | Queue mode, E2E |

**Prod-правило:** для одного run оба слоя должны указывать **один и тот же каталог** (или один S3 bucket).

Пример E2E (из `backend/scripts/e2e_env.sh`):

```bash
export E2E_STORAGE_ROOT="/path/to/TrendFlowML/storage"
export STORAGE_TYPE=fs
export STORAGE_ROOT="${E2E_STORAGE_ROOT}"
export TREND_STORAGE_BACKEND=fs
export TREND_FS_ROOT="${STORAGE_ROOT}"
```

---

## 2. Локальная разработка (основной ПК, без Docker)

Скопируйте в shell или `.env` (gitignored):

```bash
# Repo root — подставьте свой абсолютный путь
export REPO_ROOT="/media/ilya/Новый том/TrendFlowML"

# Models (обязательно для smoke)
export DP_MODELS_ROOT="${REPO_ROOT}/DataProcessor/dp_models/bundled_models"
export TORCH_HOME="${DP_MODELS_ROOT}/torch_cache"
export HF_HOME="${DP_MODELS_ROOT}/hf_cache"

# Storage — CLI path
export TREND_STORAGE_BACKEND=fs
export TREND_FS_ROOT="${REPO_ROOT}/storage"

# Если поднимаете API локально — дублируйте root
export STORAGE_TYPE=fs
export STORAGE_ROOT="${REPO_ROOT}/storage"
```

**Не использовать** старый путь из `AudioProcessor/.env` (`Рабочий стол/...`) на другом диске.

E2E: `backend/scripts/e2e_env.sh` задаёт те же переменные по умолчанию (`DP_MODELS_ROOT` → `DataProcessor/dp_models/bundled_models`).

---

## 3. Docker Compose (DataProcessor stack)

`env.example` ориентирован на **docker network** (`redis:`, `minio:`, `triton:`).  
На хосте без compose — см. §2.

| Переменная | Docker | Local host |
|------------|--------|------------|
| `CELERY_BROKER_URL` | `redis://redis:6379/0` | `redis://localhost:6379/0` |
| `S3_ENDPOINT` | `http://minio:9000` | `http://localhost:9000` |
| `TRITON_HTTP_URL` | `http://triton:8000` | `http://127.0.0.1:8010` (E2E) |
| `DP_MODELS_ROOT` | `/app/models` | абс. путь к `bundled_models` |

---

## 4. Матрица «кто читает что»

| Переменная | CLI | API | Worker | Segmenter |
|------------|-----|-----|--------|-----------|
| `DP_MODELS_ROOT` | ✓ | ✓ | ✓ | — |
| `TREND_STORAGE_*` | ✓ | частично | ✓ | ✓ (frames) |
| `STORAGE_*` | — | ✓ | ✓ | — |
| `TRITON_HTTP_URL` | visual | ✓ | ✓ | — |
| `REDIS_URL` / `CELERY_*` | queue | ✓ | ✓ | — |

---

## 5. Быстрая проверка

```bash
cd "${REPO_ROOT}/DataProcessor"
test -d "$DP_MODELS_ROOT/audio/whisper" && echo "models OK"
python3 - <<'PY'
import os
from storage.settings import load_storage_settings
s = load_storage_settings()
print("storage backend:", s.backend, "root:", getattr(s, "fs_root", None))
PY
```

---

## 6. Checklist перед E2E / prod-like run

- [ ] `DP_MODELS_ROOT` существует, `prepare_hf_cache.sh` при audio emotion
- [ ] `STORAGE_ROOT` == `TREND_FS_ROOT` (или оба S3 с одним bucket)
- [ ] `TRITON_HTTP_URL` задан в том же shell, что API/worker (если visual)
- [ ] `source backend/scripts/e2e_env.sh` для полного стека
