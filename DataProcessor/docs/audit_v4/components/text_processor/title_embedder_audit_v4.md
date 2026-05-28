# Audit v4 — `title_embedder` (TextProcessor)

**Дата:** 2026-04-14  
**Уровень отчёта (план §3.1):** **L2** (целевые наборы **A+B**, **5** путей из `result_store`; фактически данные компонента только на subset из-за ошибок `text_processor`, см. `dataset_quality`).  
**L2 stats (JSON):** `storage/audit_v4/title_embedder_l2/title_embedder_audit_v4_stats.json`  
**Engineering log 4.2:** `DataProcessor/docs/audit_v4/components/audit_4_2/text_processor/title_embedder_engineering_log_v4_2.md`  
**Артефакт (табличный срез):** `…/text_processor/text_features.npz`  
**Срез компонента:** **16** ключей `tp_titleemb_*` — [`title_embedder_output_v1`](../../../../TextProcessor/schemas/title_embedder_output_v1.json); **`allow_extra_keys: false`**. Плотный вектор: `text_processor/_artifacts/title_embedding.npy`.  
**Чтение NPZ:** строки совпадают с именами из массива **`feature_names`**, значения — **`feature_values`** (не отдельные ключи на признак).  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано · **◐** частично / только **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика + семантика + вердикт | ✓ | Этот документ |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты | ✓ | Machine JSON + [`SCHEMA.md`](../../../../TextProcessor/src/extractors/title_embedder/SCHEMA.md) |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, e2e | ✓ | `text_npz_v1` + `.npy` |
| Путь под `run_id` | ✓ | `text_processor/…` |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | Title **present**, **D=1024**, **CUDA**; **`tp_titleemb_fp16=0`** при **CUDA** (см. код: fp16 только при **`cuda` + флаге**) |
| **B** | ◐ | В L2 набор **B** формально включён (4 доп. видео), но `text_processor` падает на **3/5** путях → компонентный срез есть только на subset (см. §2.1 и JSON `dataset_quality`) |
| **C** | ✗ | TODO: пустой title, **`require_title`**, **`compute_embedding=false`**, CPU + cache hit, batch encode |

### 2.1. L2: `result_store` и блокировка

L2 запущен на стандартных **5** путях A+B (см. `paths` в JSON). Результат:

- на **2/5** путях `meta.status=ok` и присутствуют и `tp_titleemb_*` (**16** ключей), и `text_processor/_artifacts/title_embedding.npy` (shape **(1024,)**, float32, \(L2\approx1\));
- на **3/5** путях `meta.status=error`, `feature_names` пустой → компонентный срез отсутствует, артефакт `title_embedding.npy` не записан; причина — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

Итог: **L2 blocked** для `title_embedder` (нужны **5/5** OK `text_processor`).

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи ↔ схема | ✓ | Все **16** полей из `fields`; лишних `tp_titleemb_*` **нет** |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| NaN на **A** | ✓ | **Нет** — все скаляры конечны |
| Ветки с NaN (документация) | ◐ | Пустой title / `compute_embedding=false` / `compute_raw_norm=false` → **NaN** в части полей ([`main.py`](../../../../TextProcessor/src/extractors/title_embedder/main.py)) |

#### §4.7 — Трактовка

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Наблюдения → выводы | ✓ | §5–6 |

#### §4.8 — Golden на **A**

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Сигнатура | ✗ | TODO |

#### §5.3 — Сверка с Models

| Вопрос | Ответ |
|--------|--------|
| Модель | Sentence-transformers через `get_model_with_meta` ([`main.py`](../../../../TextProcessor/src/extractors/title_embedder/main.py)); на **A** размерность **1024** → профиль совместим с **e5-large**-классом |
| Параметр **`emit_extra_metrics`** | В конструкторе есть ([`main.py`](../../../../TextProcessor/src/extractors/title_embedder/main.py)), в **`features_flat`** **не используется** (как у ряда соседних embedder’ов) |

#### §6–§8 (кратко)

L3 DoD — **✗**.

---

## 1. Мета

Реализация: **1.2.0**.

---

## 2. Наблюдения на наборе **A** (исторические L1-заметки)

### 2.1 Табличный срез (сжато)

| Группа | Значение |
|--------|----------|
| Статус | `tp_titleemb_present` **1**, `tp_titleemb_title_present` **1** |
| Размерность | `tp_titleemb_dim` **1024** |
| Нормы | `tp_titleemb_l2_norm` **1**; `tp_titleemb_norm_raw` **1** |
| Устройство / кеш | `tp_titleemb_device_cuda` **1**, `tp_titleemb_fp16` **0**, `tp_titleemb_cache_enabled` **0**, `tp_titleemb_cache_hit` **0** |
| Политика | `tp_titleemb_require_title_enabled` **0**, `tp_titleemb_compute_enabled` **1**, `tp_titleemb_write_artifact_enabled` **1**, `tp_titleemb_artifact_written` **1**, `tp_titleemb_compute_raw_norm` **1** |
| Тайминг | `tp_titleemb_encode_ms` **≈255.9** |
| Идентификатор весов | `tp_titleemb_model_digest_u24` **11259398** |

### 2.2 Вектор

`title_embedding.npy`: форма **(1024,)**, **float32**, **L2 = 1** (согласовано с `tp_titleemb_l2_norm`).

### 2.3 HTML

`text_processor/_render/title_embedder_report.html`.

---

## 3. Вердикт

**Плюсы**

- Полное совпадение **16** имён с жёсткой схемой (**`allow_extra_keys: false`**).
- На **A** основной путь: title есть, эмбеддинг и артефакт записаны, нормы и размерность согласованы с **`.npy`**.

**Минусы / внимание**

- Потребители сырого NPZ должны уметь читать **`feature_names`/`feature_values`**, а не полагаться на отдельные ключи `npz[k]`.
- **`emit_extra_metrics`** в README заявлен, но в **`features_flat`** не влияет — для L2 либо подключить, либо убрать из документации конфигурации.
- Для сравнения с **`description_embedder`** на том же run: там **`tp_descemb_norm_raw`** слегка **> 1**; у title на **A** **ровно 1** — ожидаемо при разных текстах/пути нормализации, но стоит иметь **B/C** с коротким title и включённым кешем.

---

## 4. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Схема ↔ NPZ ↔ код | **9** | 16/16; вектор вне `features_flat` по контракту |
| Полнота эмпирики на **A** | **8** | Один happy-path; нет пустого title / batch |
| Документированность ветвлений | **8** | README объёмный; расхождение **`emit_extra_metrics`** |
| Готовность к модели / продукту | **8** | `digest`, нормы, L2-вектор для косинусов |

**Итог L1: ~8.2 / 10** (условно: **B/C** для кеша, CPU, **`require_title`**, **`compute_embedding=false`**).
