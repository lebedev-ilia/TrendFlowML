# Runbook: сбор баз `core_identity` → Embedding Service

**Зачем:** не пропустить env, выполнить sync **в безопасном порядке** (сначала короткие пути, потом тяжёлый **InsightFace**).

**Где лежат оффлайн-базы (часто в `.gitignore` — у каждой машины свой набор):**

| Категория | Каталог |
|-----------|---------|
| `car` | `VisualProcessor/core/model_process/core_identity/car_semantics/known_cars/` |
| `brand` | `.../brand_semantics/known_brands/` (подпапки `car/`, `electronic/`, … с лейф-брендом внутри) |
| `face` | `.../face_identity/known_people/` |

---

## 0. Предпосылки (один раз перед сессией)

1. **Postgres** с БД `embeddings` и схемой (как в E2E: `setup_e2e_infra.sh` или ваш стенд). Если **`connection refused` на `localhost:5433`** — поднять контейнер:  
   `(cd Fetcher && docker compose up -d postgres)`  
   дождаться `pg_isready` (или аналога).
2. **Triton** с CLIP для `car`/`brand` (скрипты sync вызывают `EmbeddingManager` → `CLIPExtractor` → **Triton**). В E2E чаще всего `TRITON_BASE_URL`/`TRITON_HTTP_URL` = `http://127.0.0.1:8010`. Ошибка вида `Triton server not ready` — Triton **не** запущен или другой порт/URL. Поднять, например: `backend/scripts/e2e_triton_docker.sh` (см. репо), либо выставить `TRITON_BASE_URL` на **живой** endpoint с `/v2/health/ready`.
3. **Embedding Service** не обязан быть запущен **как HTTP-сервер** для этих трёх sync-скриптов: они поднимают `EmbeddingManager` **локально** (Postgres + Triton + FAISS). Удобно взять те же `POSTGRES_*`, что в E2E:  
   `source backend/scripts/e2e_env.sh`  
   (host `localhost`, порт **5433**, user `fetcher`, БД `embeddings` — как после `setup_e2e_infra.sh`).
4. Python: **`DataProcessor/.data_venv`**, `PYTHONPATH` включает корень `DataProcessor`.

**Проверка без sync:**

```bash
curl -s -o /dev/null -w "%{http_code}\n" "${EMBEDDING_SERVICE_URL:-http://127.0.0.1:8005}/health"
```

---

## 1. Быстрый инвентарь локальных папок

Из **корня репо** (или `DataProcessor/`):

```bash
DP="DataProcessor/VisualProcessor/core/model_process/core_identity"
for what in "car_semantics/known_cars" "brand_semantics/known_brands" "face_identity/known_people"; do
  echo "=== $what ==="
  find "$DP/$what" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l
  find "$DP/$what" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.webp" \) 2>/dev/null | wc -l
done
```

Ожидаемо: **для каждой** подпапки (машина/бренд/человек) **≥1** изображение, иначе sync пропустит лейбл.

---

## 2. Синхронизация: порядок запуска

Все команды — из каталога **`DataProcessor/`**, venv:

```bash
cd DataProcessor
source .data_venv/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH}"
```

### 2.1 Машины (`car`) — **первыми** (коротко, проверка пайплайна Triton+ES)

```bash
python VisualProcessor/core/model_process/core_identity/car_semantics/utils/sync_known_cars_to_embedding_service.py
```

### 2.2 Бренды (`brand`)

```bash
python VisualProcessor/core/model_process/core_identity/brand_semantics/utils/sync_known_brands_to_embedding_service.py
```

### 2.3 Лица (`face`, InsightFace, дольше по CPU/GPU)

```bash
python VisualProcessor/core/model_process/core_identity/face_identity/utils/sync_known_people_to_embedding_service.py
```

### 2.4 Места (`place`) и франшизы (`franchise`)

Отдельного `sync_*.py` в репо **нет** — добавляйте через **HTTP API** `POST .../objects/add` (см. README компонентов). Начните с **5–8** `place` и **5–8** `franchise` с разными картинками.

### 2.5 `content_domain`

Оффлайн-пак **без** ES; убедитесь, что путь существует:  
`dp_models/bundled_models/semantics/content_domain/v1/`.

---

## 3. Повторный запуск и дубликаты

Скрипты делают **`add_from_embedding`** по каждой папке. **Повторный** прогон без очистки категории может **добавлять** новые UUID (дубликаты логических брендов/машин/людей). Для **чистого** переноса:

- либо **очистить** таблицу/категорию в dev (только в своей среде),
- либо один раз **прогнать** sync на пустой категории после миграций.

В проде согласовать политику с владельцем БД.

---

## 4. Контроль в ES после sync

- HTTP: список labels по категории (как в клиенте / swagger `embedding_service`).
- SQL (пример): `SELECT category, COUNT(*) FROM embeddings GROUP BY category;`

---

## 5. Ссылка

План объёмов и разнообразия: [CORE_IDENTITY_MINIMAL_BASES_BATCH_70.md](CORE_IDENTITY_MINIMAL_BASES_BATCH_70.md).
