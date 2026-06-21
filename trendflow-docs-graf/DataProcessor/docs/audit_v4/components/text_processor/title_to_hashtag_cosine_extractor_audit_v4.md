# Audit v4 — `title_to_hashtag_cosine_extractor` (TextProcessor)

**Дата:** 2026-04-14  
**Уровень отчёта (план §3.1):** **L2** (целевые наборы **A+B**, **5** путей из `result_store`; фактически данные компонента только на subset из-за ошибок `text_processor`, см. `dataset_quality`).  
**L2 stats (JSON):** `storage/audit_v4/title_to_hashtag_cosine_extractor_l2/title_to_hashtag_cosine_extractor_audit_v4_stats.json`  
**Engineering log 4.2:** `DataProcessor/docs/audit_v4/components/audit_4_2/text_processor/title_to_hashtag_cosine_extractor_engineering_log_v4_2.md`  
**Артефакт (табличный срез):** `…/text_processor/text_features.npz`  
**Срез компонента:** **11** ключей `tp_titlehashcos_*` — [`title_to_hashtag_cosine_extractor_output_v1`](../../../../TextProcessor/schemas/title_to_hashtag_cosine_extractor_output_v1.json); **`allow_extra_keys: false`**. Вход: **`doc.tp_artifacts.embeddings.title` / `hashtag`** → **`title_embedding.npy`**, **`hashtag_embedding.npy`** в `_artifacts/` ([`main.py`](../../../../TextProcessor/src/extractors/title_to_hashtag_cosine_extractor/main.py)).  
**Чтение NPZ:** **`feature_names`** / **`feature_values`**.  
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
| Ссылки на контракты | ✓ | Machine JSON + [`SCHEMA.md`](../../../../TextProcessor/src/extractors/title_to_hashtag_cosine_extractor/SCHEMA.md) |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, e2e | ✓ | `text_npz_v1` + **`tp_artifacts`** |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | Оба эмбеддинга есть; **`present=1`**, cosine **конечен** |
| **B** | ◐ | В L2 набор **B** формально включён (4 доп. видео), но `text_processor` падает на **3/5** путях → компонентный срез есть только на subset (см. §2.1 и JSON `dataset_quality`). Целевые вариации B: только title или только hashtag (**`present=0`**, cosine **NaN**) |
| **C** | ✗ | **`unsafe` relpath**, **`dim_mismatch`**, **zero norm**, **`require_*`** fail-fast |

### 2.1. L2: `result_store` и блокировка

L2 запущен на стандартных **5** путях A+B (см. `paths` в JSON). Результат:

- на **2/5** путях `meta.status=ok` и присутствует полный `tp_titlehashcos_*` (**11** ключей); upstream `title_embedding.npy` и `hashtag_embedding.npy` присутствуют (shape **(1024,)**). Для `present=1` cosine, пересчитанный из `.npy`, совпадает с `tp_titlehashcos_cosine` (см. `per_file[*].consistency.abs_diff` в JSON);
- на **3/5** путях `meta.status=error`, `feature_names` пустой → компонентный срез отсутствует, upstream артефакты не записаны; причина — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

Итог: **L2 blocked** для `title_to_hashtag_cosine_extractor` (нужны **5/5** OK `text_processor`).

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи ↔ схема | ✓ | **11** имён, без лишних |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| На **A** | ✓ | **`tp_titlehashcos_cosine`** **конечен** при **`present=1`** |
| Ветки с NaN | ◐ | Нет обоих векторов / ранний выход → **`cosine`** **NaN** ([`main.py`](../../../../TextProcessor/src/extractors/title_to_hashtag_cosine_extractor/main.py) ~226–227, шаблон ~130–133) |

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
| Обучаемая модель | **Нет** — косинус по уже посчитанным векторам |

#### §6–§8 (кратко)

L3 DoD — **✗**.

---

## 1. Мета

Реализация: **1.2.0**.

---

## 2. Наблюдения на наборе **A** (исторические L1-заметки)

### 2.1 Табличный срез

| Поле | Значение |
|------|----------|
| **`tp_titlehashcos_present`** | **1** |
| **`tp_titlehashcos_cosine`** | **≈0.847** |
| **`tp_titlehashcos_title_present` / `hashtag_present`** | **1** / **1** |
| **`require_*_enabled`** | **0** / **0** |
| Флаги ошибок | все **0** (**unsafe** / **missing** / **dim_mismatch** / **zero_norm**) |

### 2.2 Upstream

Зависит от порядка пайплайна: **`title_embedder`**, **`hashtag_embedder`** (на **A** оба дают **(1024,)**, L2-norm в файлах совместим с повторной L2-нормировкой в экстракторе).

### 2.3 HTML

`text_processor/_render/title_to_hashtag_cosine_extractor_report.html`.

---

## 3. Вердикт

**Плюсы**

- Минимальный контракт (**11** ключей), полное совпадение со схемой и NPZ.
- Чёткое разделение флагов: **unsafe** vs **missing/bad_file** vs **dim** vs **zero norm**.

**Минусы / внимание**

- При отсутствии одного из векторов cosine остаётся **NaN** — потребители таблицы должны смотреть **`present`** и флаги.

---

## 4. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Схема ↔ NPZ ↔ код | **9** | Жёсткий список ключей |
| Полнота эмпирики на **A** | **8** | Только happy path |
| Документированность ветвлений | **8** | README + SCHEMA |
| Готовность к модели / продукту | **8** | Сигнал согласованности title vs aggregate hashtag emb |

**Итог L1: ~8.3 / 10** (условно: **B/C**, **§4.8**).
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
