## Как собирать semantic базы (offline, no-network) — v1 guide

Этот документ отвечает на “как собрать базы”, которые будут использовать semantic heads:
- brands/logos (500)
- cars (make/model/segment/body_type/price buckets)
- celebs (500)
- places/landmarks (gallery retrieval)

Контракты и решения см.:
- `docs/models_docs/SEMANTIC_HEADS_CONTRACTS_QA.md` (Resolved decisions v0.1)
- `dp_models/bundled_models/semantics/README.md`

---

### 0) Общие правила (важно)

- **No-network в runtime**: базы должны лежать локально в `dp_models/bundled_models/...`.
- **Reproducibility**: каждая база — “artifact package” с `manifest.json` и `db_digest`.
- **Stable IDs**: id присваиваем один раз и дальше только добавляем новые (не переупорядочиваем).
- **English canon**: `name`/`semantic_label_names` в EN, но допускаем `aliases_ru` + `prompts_ru`.

Путь пакета:

`dp_models/bundled_models/semantics/<domain>/<version>/`

После сборки запускаем генерацию `manifest.json` + `db_digest`:

```bash
python dp_models/bundled_models/semantics/_tools/build_db_manifest.py \
  --package-dir dp_models/bundled_models/semantics/<domain>/<version> \
  --db-name <domain> \
  --db-version <version>
```

### 0.1 Preflight (fail-fast) перед запуском пайплайна

Если в `VisualProcessor/config.yaml` включены semantic heads, можно заранее проверить,
что все required базы/галереи/поля в конфиге присутствуют:

```bash
python3 scripts/preflight/check_semantic_bases.py --cfg-path VisualProcessor/config.yaml
```

---

## 1) Brands base (v1=500)

### 1.1 Что это за “бренд” в нашей системе

- 1 id = 1 бренд (например `nike`, `bmw`).
- Внутри бренда держим:
  - алиасы/локализации
  - prompts (RU/EN)
  - (опционально) изображения-прототипы логотипов

### 1.2 Формат пакета

`dp_models/bundled_models/semantics/brands/v1/`

- `brands.jsonl` (обязательный):
  - каждая строка JSON:
    - `id: int`
    - `name_en: str` (канон)
    - `aliases_en: [str]` (опционально)
    - `aliases_ru: [str]` (опционально)
    - `prompts_en: [str]` (опционально)
    - `prompts_ru: [str]` (опционально)
    - `category: str` (опционально; одежда/косметика/еда/техника/авто/…)

- `prototypes/<id>/...png|jpg` (опционально):
  - 1–10 изображений логотипа/эмблемы/wordmark

### 1.5 Сбор `gallery_embeddings.npy` для брендов (prototype retrieval, опционально)

Структура:

`dp_models/bundled_models/semantics/brands/v1/prototypes/<brand_id>/*.jpg|png|webp`

Сборка gallery (нужен Triton + DP_MODELS_ROOT):

```bash
python3 dp_models/bundled_models/semantics/_tools/build_clip_image_gallery.py \
  --package-dir dp_models/bundled_models/semantics/brands/v1 \
  --index-jsonl brands.jsonl \
  --prototypes-dir prototypes \
  --clip-image-model-spec clip_image_224_triton
```

Скрипт создаст:
- `gallery_embeddings.npy` (A,D) L2-normalized
- `gallery_index.json` (id->row_index)

Policy: если `core_brand_semantics` включён как required head, prototype gallery считается **обязательной** (no-fallback).

После этого пересобери `manifest.json`/`db_digest`:

```bash
python3 dp_models/bundled_models/semantics/_tools/build_db_manifest.py \
  --package-dir dp_models/bundled_models/semantics/brands/v1 \
  --db-name brands \
  --db-version v1
```

### 1.3 Как собирать список (без “слабых эвристик”)

Рекомендованный подход:
- собрать исходный список брендов (по доменам) → нормализовать названия
- руками убрать:
  - дубликаты/омонимы (например “Apple” как фрукт vs бренд)
  - слишком общие слова
- на каждый бренд сделать 2–6 prompts EN + 2–6 prompts RU (multi-lingual)

Пример prompts (для одного бренда):
- EN: `"nike logo"`, `"nike swoosh logo"`, `"nike brand mark"`
- RU: `"логотип nike"`, `"эмблема nike"`

### 1.4 QA чеклист

- нет повторяющихся `id`
- `name_en` уникален (или явно задано, как различать)
- `aliases` не содержат мусор/слишком общие слова
- prompts не слишком широкие (“logo” без бренда запрещено)

---

## 2) Cars base (v1)

### 2.1 Что выдаёт `core_car_semantics`

Top‑3 по:
- make
- model
- segment (8 buckets)
- body_type
- price_bucket (8 buckets)

### 2.2 Формат пакета

`dp_models/bundled_models/semantics/cars/v1/`

- `makes.jsonl`:
  - `{id, name_en, aliases_en?, aliases_ru?}`
- `models.jsonl`:
  - `{id, make_id, name_en, aliases_en?, aliases_ru?}`
- `taxonomy.json`:
  - `segment_buckets` (8)
  - `body_type_buckets`
  - `price_buckets` (8) + границы

Опционально:
- `prototypes/make/<make_id>/*.jpg`
- `prototypes/model/<model_id>/*.jpg`

### 2.5 Сбор make/model galleries (prototype retrieval, опционально)

Структура:
- `dp_models/bundled_models/semantics/cars/v1/prototypes/make/<make_id>/*.jpg|png|webp`
- `dp_models/bundled_models/semantics/cars/v1/prototypes/model/<model_id>/*.jpg|png|webp`

Сбор make gallery:

```bash
python3 dp_models/bundled_models/semantics/_tools/build_clip_image_gallery.py \
  --package-dir dp_models/bundled_models/semantics/cars/v1 \
  --index-jsonl makes.jsonl \
  --prototypes-dir prototypes/make \
  --clip-image-model-spec clip_image_224_triton \
  --out-embeddings make_gallery_embeddings.npy \
  --out-index make_gallery_index.json
```

Сбор model gallery:

```bash
python3 dp_models/bundled_models/semantics/_tools/build_clip_image_gallery.py \
  --package-dir dp_models/bundled_models/semantics/cars/v1 \
  --index-jsonl models.jsonl \
  --prototypes-dir prototypes/model \
  --clip-image-model-spec clip_image_224_triton \
  --out-embeddings model_gallery_embeddings.npy \
  --out-index model_gallery_index.json
```

Policy: если `core_car_semantics` включён, **make gallery обязательна** (no-fallback). Model gallery может быть добавлена позже.

После добавления бинарников обязательно пересобери `manifest.json`/`db_digest`:

```bash
python3 dp_models/bundled_models/semantics/_tools/build_db_manifest.py \
  --package-dir dp_models/bundled_models/semantics/cars/v1 \
  --db-name cars \
  --db-version v1
```

### 2.3 Откуда брать список makes/models

Runtime не скачивает сеть, но **на этапе подготовки** можно использовать открытые источники и затем “заморозить” результат в bundle:
- выгрузки/дампы каталогов авто (публичные)
- экспортированные таблицы (Kaggle и т.п.) → сохраняем локально как исходник, но в bundle кладём уже очищенную `jsonl`

### 2.4 Bucket’ы (принято)

- `segment_id`: 8 buckets
- `price_bucket_id`: 8 buckets (по лог-шкале, границы задаём в `taxonomy.json`)

---

## 3) Celebs base (v1=500)

### 3.1 Приватность

- В базе НЕ храним персональные данные сверх минимально нужного (id + name/aliases).
- В runtime не сохраняем face crops в result_store.

### 3.2 Формат пакета

`dp_models/bundled_models/semantics/celebs/v1/`

- `celebs.jsonl`:
  - `{id, name_en, aliases_en?, aliases_ru?}`
- `gallery_embeddings.npy` (optional на v1, но рекомендуется):
  - матрица (N, D) float32, где N=500
- `gallery_index.json`:
  - `{id -> row_index}` mapping (если нужно)

### 3.3 Как собирать

Рекомендованный MVP:
- выбрать 500 “популярных людей” под домен (музыка/спорт/медиа)
- собрать 5–20 референс фото на человека (offline)
- прогнать через face embedding модель → усреднить → получить gallery embeddings

### 3.4 Сбор `gallery_embeddings.npy` из prototype face crops (рекомендуемый способ)

Структура:

`dp_models/bundled_models/semantics/celebs/v1/prototypes/<celebrity_id>/*.jpg|png|webp`

Сборка gallery (нужен Triton + DP_MODELS_ROOT + face-embed модель в ModelManager):

```bash
python3 dp_models/bundled_models/semantics/_tools/build_face_embedding_gallery.py \
  --package-dir dp_models/bundled_models/semantics/celebs/v1 \
  --index-jsonl celebs.jsonl \
  --prototypes-dir prototypes \
  --face-embed-model-spec <FACE_EMBED_SPEC_NAME>   # пока модели нет — этот шаг пропускаем
```

Скрипт создаст:
- `gallery_embeddings.npy` (A,D) L2-normalized
- `gallery_index.json` (id->row_index)

После этого обязательно пересобери `manifest.json`/`db_digest`:

```bash
python3 dp_models/bundled_models/semantics/_tools/build_db_manifest.py \
  --package-dir dp_models/bundled_models/semantics/celebs/v1 \
  --db-name celebs \
  --db-version v1
```

---

## 4) Places/Landmarks base (v1)

Подход: retrieval по gallery embeddings (CLIP).

`dp_models/bundled_models/semantics/places/v1/`

- `places.jsonl`:
  - `{id, name_en, aliases_en?, aliases_ru?, type?}`
- `gallery_embeddings.npy`:
  - CLIP embeddings (N, 512) float32 L2-normalized

Как собирать:
- собрать список мест (N) + набор эталонных изображений
- получить CLIP embeddings offline (через Triton CLIP image encoder) → усреднить per place

### 4.1 Сбор `gallery_embeddings.npy` из prototype images (рекомендуемый способ)

Структура:

`dp_models/bundled_models/semantics/places/v1/prototypes/<place_id>/*.jpg|png|webp`

Дальше собрать gallery (нужен Triton + DP_MODELS_ROOT):

```bash
python3 dp_models/bundled_models/semantics/_tools/build_clip_image_gallery.py \
  --package-dir dp_models/bundled_models/semantics/places/v1 \
  --index-jsonl places.jsonl \
  --prototypes-dir prototypes \
  --clip-image-model-spec clip_image_224_triton
```

Скрипт создаст:
- `gallery_embeddings.npy` (A,D) L2-normalized
- `gallery_index.json` (id->row_index)

После этого обязательно пересобери `manifest.json`/`db_digest`:

```bash
python3 dp_models/bundled_models/semantics/_tools/build_db_manifest.py \
  --package-dir dp_models/bundled_models/semantics/places/v1 \
  --db-name places \
  --db-version v1
```

---

## 5) Golden set (30 видео)

Набор нужен для regression/quality panel, независимо от полноты разметки:
- мульт/аниме
- лица/крупные планы
- бренды/логотипы
- авто
- текст/субтитры
- night/low-light

Мы храним список `video_id` + короткие заметки “что должно находиться”.

---

## 6) LLM-assisted base building: распределение задач (LLM vs human-only)

Цель: нейросеть быстро делает **черновик** базы (список + prompts/aliases), а человек делает **QA и утверждение**
source-of-truth (иначе будут галлюцинации/дубли/ошибки).

### 6.1 Общие правила

- **LLM никогда не является source-of-truth**: её output = “черновик”.
- **Human-only всегда**:
  - дедуп/слияние сущностей,
  - удаление сомнительных/вымышленных элементов,
  - фиксация stable ids (не менять существующие, только добавлять),
  - финальная сборка `manifest.json` + `db_digest`.

### 6.2 Brands base (500)

#### LLM делает
- список 500 брендов (EN canonical)
- `aliases_en`, `aliases_ru`
- `prompts_en`, `prompts_ru` (2–6), только “logo/emblem/brand mark” + имя бренда
- `category`

#### Human-only делает
- удалить несуществующие/сомнительные бренды
- слить дубли (например “Adidas” vs “Adidas Originals”, если не хотим отдельный id)
- разобрать омонимы (“Apple” как бренд vs фрукт)
- поправить RU написания и “официальные” варианты
- (опционально) собрать `prototypes/<id>/...` (реальные изображения логотипов) — только из источников

#### Команда для LLM (генерация `brands.jsonl`)

Скопируй-вставь в нейросеть:

> Сгенерируй файл `brands.jsonl` (500 строк). Формат каждой строки строго JSON (1 объект = 1 строка), поля:  
> `id` (0..499), `name_en`, `aliases_en` (list), `aliases_ru` (list), `prompts_en` (list), `prompts_ru` (list), `category`.  
> Правила:  
> - `name_en` уникален; `id` уникален, contiguous 0..499.  
> - Prompts должны быть конкретными (например “nike logo”, “bmw car emblem”), запрещены общие prompts типа “logo”, “brand”, “emblem” без имени бренда.  
> - Aliases не должны быть общими словами.  
> - Категории: clothing, cosmetics, food, tech, auto, luxury, sports, retail, fintech, other.  
> - Не придумывай факты про популярность; просто дай разумный глобальный набор.  
> Выведи ТОЛЬКО содержимое `brands.jsonl`, без комментариев и без markdown.

### 6.3 Cars base (make/model + buckets)

#### LLM делает (безопасно)
- `taxonomy.json`:
  - `segment_buckets` (8 buckets)
  - `body_type_buckets` (минимум: sedan, suv, crossover, hatchback, coupe, wagon, pickup, van)
  - `price_buckets_usd` (8 buckets) с границами и описаниями

#### LLM делает (черновик, требует проверки)
- `makes.jsonl` (например 80–120 марок)
- `models.jsonl` (например 500–1500 моделей)

#### Human-only делает
- подтверждение списков makes/models по структурированному датасету (иначе будут ошибки/вымышленные модели)
- фиксация правил для сегментов/цен под выбранный рынок (RU/US/EU)
- опционально: сбор `prototypes/` и gallery embeddings

#### Команда для LLM (генерация `taxonomy.json`)

> Сгенерируй `taxonomy.json` для car semantics. Поля:  
> `segment_buckets`: список из 8 объектов `{id,name_en,description,heuristics_keywords}`  
> `body_type_buckets`: список объектов `{id,name_en,description,keywords}` (минимум: sedan,suv,crossover,hatchback,coupe,wagon,pickup,van)  
> `price_buckets_usd`: список из 8 объектов `{id,name_en,usd_min,usd_max,notes}` с лог-шкалой.  
> Выведи ТОЛЬКО JSON.

#### Команда для LLM (черновик `makes.jsonl` + `models.jsonl`)

> Сгенерируй `makes.jsonl` на ~120 марок авто: строки JSON `{id,name_en,aliases_en,aliases_ru}`.  
> Затем `models.jsonl` на ~1000 моделей: `{id,make_id,name_en,aliases_en,aliases_ru}`.  
> Правила: ids contiguous с 0, без дублей по name_en внутри make.  
> Это черновик: допускаются пропуски, но не выдумывай “фантазийные” модели.

### 6.4 Celebs base (500)

#### LLM делает
- `celebs.jsonl` (500): `{id,name_en,aliases_en,aliases_ru,category}`

#### Human-only делает
- проверка существования/написания
- сбор offline референс‑фото/источников
- построение gallery embeddings (face embedding модель) и лицензии/право хранения

#### Команда для LLM (генерация `celebs.jsonl`)

> Сгенерируй `celebs.jsonl` (500 строк): `{id,name_en,aliases_en,aliases_ru,category}`.  
> Правила: ids 0..499; без дублей по name_en; aliases без мусора.  
> Выведи ТОЛЬКО `celebs.jsonl` без markdown.

### 6.5 Places/Landmarks base (retrieval)

#### LLM делает
- `places.jsonl` (например 500): `{id,name_en,aliases_en,aliases_ru,type}`

#### Human-only делает
- подтверждение реальности/уникальности мест
- сбор offline эталонных изображений и построение gallery CLIP embeddings

#### Команда для LLM (генерация `places.jsonl`)

> Сгенерируй `places.jsonl` (N=500 для старта): `{id,name_en,aliases_en,aliases_ru,type}`.  
> Типы: city, landmark, nature, venue, building, region, other.  
> ids contiguous 0..N-1, name_en уникален.  
> Выведи ТОЛЬКО `places.jsonl`.

### 6.6 Human QA checklist (для любой базы)

- **Dedup** по `name_en` + по алиасам (одни и те же сущности под разными именами).
- **Запрет общих слов** в aliases/prompts (например `logo` без бренда).
- **Stable ids**: не менять существующие id, только добавлять новые в конец.
- После QA: обязательно пересобрать `manifest.json` + `db_digest`:

```bash
python dp_models/bundled_models/semantics/_tools/build_db_manifest.py \
  --package-dir dp_models/bundled_models/semantics/<domain>/<version> \
  --db-name <domain> \
  --db-version <version>
```
---

## Навигация

[README](README.md) · [DataProcessor](../MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
