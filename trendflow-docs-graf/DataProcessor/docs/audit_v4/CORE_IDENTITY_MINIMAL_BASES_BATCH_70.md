# Минимальные разнообразные базы `core_identity` для прогона **70** видео

**Цель:** не «прод-качество каталога», а **достаточно разных** опорных записей, чтобы на батче из **~70** `video_id` компоненты **регулярно** выходили из чистого *empty* (где контент это позволяет) — и при этом **реалистично** проверить схемы NPZ, `db_digest`, top‑K, рендеры, отчёты.

**Инварианты среды:** [Embedding Service](../../embedding_service/README.md) (Postgres `embeddings` + FAISS), **Triton** с нужными CLIP/моделями, для face-sync — **InsightFace** в окружении, куда смотрит sync-скрипт. Локальный E2E: `EMBEDDING_SERVICE_URL`, БД `embeddings` (см. [setup_e2e_infra.sh](../../../backend/scripts/setup_e2e_infra.sh) — **seed** для smoke).

**Пошаговый runbook (команды, порядок sync):** [CORE_IDENTITY_BASES_RUNBOOK.md](CORE_IDENTITY_BASES_RUNBOOK.md)

**Принцип разнообразия:** по каждой категории — **несколько** разных «типов» сущностей (бренд: техника/одежда/еда; место: indoor/outdoor/архитектура; авто: разные сегменты/марки; лицо: разные ракурсы/люди; франшиза: жанры/медиа), чтобы **матрица контента** ([COVERAGE_MATRIX_60PLUS.md](COVERAGE_MATRIX_60PLUS.md), [VIDEO_REGISTRY_60PLUS.yaml](VIDEO_REGISTRY_60PLUS.yaml)) пересекалась с опорной базой **хотя бы у нескольких** роликов.

---

## 1. `face_identity` (категория `face`)

| Что | Минимум | Разнообразие | Инструмент |
|-----|--------|--------------|------------|
| Локальные кропы | **5–10** папок в `known_people/<id>/` по нескольку фото | Разные люди; при возможности **разные** ракурсы/освещение | `face_identity/utils/sync_known_people_to_embedding_service.py` |
| В ES | После sync — **≥5** уникальных UUID в `face` | Не дублировать одно лицо под разными именами без нужды | Проверка: `get_labels` / API категории |

*Уже в E2E-seed* — одна placeholder-строка; для батча **добавить** реальные `known_people` + sync.

---

## 2. `brand_semantics` (категория `brand`)

| Что | Минимум | Разнообразие | Инструмент |
|-----|--------|--------------|------------|
| Кропы логотипов/бренд-объектов | **5–8** брендов в `brand_semantics/known_brands/` (по нескольку jpg) | Разные отрасли, чтобы YouTube-контент **иногда** матчился | `brand_semantics/utils/add_brand.py` → `utils/sync_known_brands_to_embedding_service.py` |
| Triton | Модель **CLIP image 336** (как в sync) | | См. комментарий в начале [sync_known_brands_to_embedding_service.py](../../VisualProcessor/core/model_process/core_identity/brand_semantics/utils/sync_known_brands_to_embedding_service.py) |

---

## 3. `place_semantics` (категория `place`, обычно `clip_448` / как в ES)

| Что | Минимум | Разнообразие | Как |
|-----|--------|--------------|-----|
| Записи в ES | **8–15** мест (лучше **12+**) | Смесь: город/природа/интерьер/достопримечательности | **API** `POST /objects/add` с репрезентативным кадром/изображением (см. [place_semantics README](../../VisualProcessor/core/model_process/core_identity/place_semantics/README.md)) |
| *Empty* | Ожидаемо, если в кадре **нет** матча по косинусу | | [W4](BATCH_60PLUS_WAIVERS.md) — не путать с «нет базы» |

Отдельного `sync_*` в репо для place нет: опирайтесь на **HTTP API** Embedding Service + ручной/скриптовый батч из референс-картинок.

---

## 4. `car_semantics` (категория `car`)

| Что | Минимум | Разнообразие | Инструмент |
|-----|--------|--------------|------------|
| Кропы машин | **5–8** лейблов в `car_semantics/known_cars/` | Разные сегменты (седан/SUV/спорт) и **разные** make | `car_semantics/add_car.py` → `utils/sync_known_cars_to_embedding_service.py` |
| *Empty* | `no_car_proposals` при отсутствии детекций | | [W6](BATCH_60PLUS_WAIVERS.md) |

---

## 5. `franchise_recognition` (категория `franchise`)

| Что | Минимум | Разнообразие | Как |
|-----|--------|--------------|-----|
| Объекты | **5–10** франшиз (логотип/кадр-ключ) | Игры, кино, стриминг, анима — разные **визуальные** сигнатуры | [franchise_recognition README](../../VisualProcessor/core/model_process/core_identity/franchise_recognition/README.md) (примеры curl/Python) — `POST .../objects/add` |
| Режим | С **≥1** записью в базе срабатывает **оптимизированный** путь (массовое получение эмбеддингов) | | |

*E2E-seed* уже кладёт placeholder franchise в БД; для осмысленного батча **заменить/дополнить** реальными сущностями.

---

## 6. `content_domain` (категория **не** обязана совпадать с ES)

| Что | Минимум | Разнообразие | Как |
|-----|--------|--------------|-----|
| Офлайн-набор | Пак **v1** из репо | **Несколько** доменов в `domains.jsonl` уже дают вариативность | Путь: `DataProcessor/dp_models/bundled_models/semantics/content_domain/v1` (см. [content_domain README](../../VisualProcessor/core/model_process/core_identity/content_domain/README.md)) |
| Triton | CLIP text/image как в `main` | | ES для `content_domain` — **опционально** (в README — будущее расширение) |

Здесь «база» = **файлы + Triton**, а не наполнение Embedding Service (если не делаете отдельный эксперимент).

---

## 7. Сводная таблица (ориентир объёма)

| Компонент | Категория ES | Порядок величины записей | Главный артефакт в репо |
|-----------|-------------|-------------------------|-------------------------|
| face_identity | `face` | 5–10+ | `known_people/` + sync |
| brand_semantics | `brand` | 5–8+ | `known_brands/` + sync |
| place_semantics | `place` | 8–15 | API add (нет единого sync) |
| car_semantics | `car` | 5–8+ | `known_cars/` + sync |
| franchise_recognition | `franchise` | 5–10+ | API / сценарии из README |
| content_domain | — | bundle v1 | `dp_models/.../content_domain/v1` |

---

## 8. Связь с **70** видео

1. После наполнения — **пересчитать** `db_digest` не обязателен вручную (считает `main.py`), но зафиксировать в **отчёте пилота**, что label-space **достаточен** (не 1–2 id на всю категорию).
2. В [VIDEO_REGISTRY_60PLUS.yaml](VIDEO_REGISTRY_60PLUS.yaml) **метки** (`tags` / `notes`): часть роликов с **ожидаемыми** визуальными якорями (бренды, авто, интерьеры, известные лица) — чтобы и база, и контент **пересекались**; **edge**-слоты (нет лица, нет авто) — **намеренно** для *empty* ([чеклист 3.6](CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md)).
3. Не гнаться за **полнотой** справочника: цель — **достаточно**, чтобы в отчёте L2/манифесте **разные** `processor` / `empty_reason` / top‑K **не** свелись к одной причине.

---

## 9. Проверка перед батчем

- [ ] `GET` health Embedding Service, список категорий / counts по `category` (по вкусу — SQL `COUNT(*)`).
- [ ] Один **короткий** прогон **full-max** на 1 `video_id` с типичным брендом/местом/… и сверка `manifest` / NPZ.
- [ ] Triton/CLIP/InsightFace — те же **endpoint** и `embedding_model`, что в sync-скриптах (иначе `db_digest` и поиск **расходятся**).

---

*См. также: [CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md](CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md) (NPZ, waivers W4–W6), [METRICS_LABELS_INVENTORY_60PLUS.md](METRICS_LABELS_INVENTORY_60PLUS.md) (сквозные метрики `pipeline`/`main_py`).*
---

## Навигация

[Audit v4 hub](components/audit_4_2/README.md) · [DataProcessor](../MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
