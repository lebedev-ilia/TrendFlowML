# Audit v4 — `topk_similar_titles_extractor` (TextProcessor)

**Дата:** 2026-04-14  
**Уровень отчёта (план §3.1):** **L2** (целевые наборы **A+B**, **5** путей из `result_store`; фактически данные компонента только на subset из-за ошибок `text_processor`, см. `dataset_quality`).  
**L2 stats (JSON):** `storage/audit_v4/topk_similar_titles_extractor_l2/topk_similar_titles_extractor_audit_v4_stats.json`  
**Engineering log 4.2:** `DataProcessor/docs/audit_v4/components/audit_4_2/text_processor/topk_similar_titles_extractor_engineering_log_v4_2.md`  
**Артефакт (табличный срез):** `…/text_processor/text_features.npz`  
**Срез компонента:** **29** ключей `tp_topktitles_*` — [`topk_similar_titles_extractor_output_v1`](../../../../TextProcessor/schemas/topk_similar_titles_extractor_output_v1.json); **`allow_extra_keys: false`**. Корпус — **`dp_models`** (`similar_titles_corpus_v1` по умолчанию); списки id/scores — в **`result.topk_similar_corpus_titles`**, не в таблице NPZ ([`main.py`](../../../../TextProcessor/src/extractors/topk_similar_titles_extractor/main.py)).  
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
| Ссылки на контракты | ✓ | Machine JSON + [`SCHEMA.md`](../../../../TextProcessor/src/extractors/topk_similar_titles_extractor/SCHEMA.md) |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, e2e | ✓ | `text_npz_v1` + payload |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | **`enabled=1`**, title emb **1024** = **`corpus dim`**, **`present=1`**; **FAISS** модуля **нет** → **`backend_faiss=0`**, точный **numpy** top‑K; корпус **18 409** &lt; **`max_corpus_for_numpy`** **100 000** |
| **B** | ◐ | В L2 набор **B** формально включён (4 доп. видео), но `text_processor` падает на **3/5** путях → компонентный срез есть только на subset (см. §2.1 и JSON `dataset_quality`). Целевые вариации B: **`enabled=false`**, **`export_topk_mode`** **`none` / `ids_only`**, **FAISS** доступен (HNSW vs numpy) |
| **C** | ✗ | Большой корпус без FAISS / **`allow_numpy_large_corpus`**, **`require_faiss`**, отсутствие title emb / **dim mismatch** / **NaN** в query |

### 2.1. L2: `result_store` и блокировка

L2 запущен на стандартных **5** путях A+B (см. `paths` в JSON). Результат:

- на **2/5** путях `meta.status=ok` и присутствует полный `tp_topktitles_*` (**29** ключей); upstream `text_processor/_artifacts/title_embedding.npy` присутствует (shape **(1024,)**);
- на **3/5** путях `meta.status=error`, `feature_names` пустой → компонентный срез отсутствует, upstream артефакт не записан; причина — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

Итог: **L2 blocked** для `topk_similar_titles_extractor` (нужны **5/5** OK `text_processor`).

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи ↔ схема | ✓ | **29** имён, лишних нет |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| На **A** | ✓ | **`top1_score`**, **`topk_mean_score`**, **`export_k_used`** **конечны** при **`present=1`** |
| Ветки с NaN | ◐ | Шаблон (**`_base_features_flat`**) — **`export_k_used`**, scores **NaN** до успешного поиска; при **`enabled=false`** scores остаются **NaN** ([`main.py`](../../../../TextProcessor/src/extractors/topk_similar_titles_extractor/main.py) ~318–350, ~410–412) |

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
| Корпус | Статические эмбеддинги + **ids** через **`ModelManager.resolve`** |
| Обучаемый инференс | **Нет** — только поиск по готовой матрице |

#### §6–§8 (кратко)

L3 DoD — **✗**.

---

## 1. Мета

Класс **`TopKSimilarCorpusTitlesExtractor`**, версия **1.3.0**.

---

## 2. Наблюдения на наборе **A** (исторические L1-заметки)

### 2.1 Табличный срез (сжато)

| Группа | Значение |
|--------|----------|
| Статус | **`tp_topktitles_present`** **1**, **`disabled_by_policy`** **0**, **`enabled`** **1** |
| Поиск | **`k`** **5**, **`export_k_used`** **5**, **`export_k_truncated_flag`** **0**; **`export_topk_mode_ids_and_scores`** **1** |
| Корпус | **`corpus_size`** **18 409**, **`dim`** **1024** |
| Бэкенд | **`faiss_available`** **0**, **`backend_faiss`** **0** (метка в коде — **`numpy_cosine`**) |
| Политики | **`require_faiss_enabled`** **0**, **`require_faiss_above_corpus_size`** **200 000**, **`allow_numpy_large_corpus`** **0**, **`max_corpus_for_numpy`** **100 000** |
| Кеш | **`cache_enabled`** **1**, **`cache_ttl_s`** **3600**, **`cache_max_entries`** **2** |
| Качество top‑K | **`top1_score`** **≈0.930**, **`topk_mean_score`** **≈0.902** |
| Флаги | **`unsafe` / `title_embed_missing` / `dim_mismatch` / `zero_norm` / `nan_inf`** — все **0** |

### 2.2 Upstream / порядок

Требуется **`title_embedder`** → **`title_embedding.npy`** и реестр **`tp_artifacts.embeddings.title`**.

### 2.3 HTML

`text_processor/_render/topk_similar_titles_extractor_report.html`.

---

## 3. Вердикт

**Плюсы**

- Полное совпадение **29** скаляров со схемой; на **A** основной путь без флагов ошибок.
- Явные поля **faiss_available** vs **backend_faiss** (факт индекса HNSW vs матрица корпуса).

**Минусы / внимание**

- При установленном **FAISS** top‑K через **HNSW** — **приближённый**; без **FAISS** на **A** — **точная** сортировка по косинусу — сравнение прогонов должно учитывать backend.
- Переменная длина **`topk_similar_ids` / scores** только в **`result`**, не в **`features_flat`**.

---

## 4. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Схема ↔ NPZ ↔ код | **9** | Жёсткий список; payload описан в schema description |
| Полнота эмпирики на **A** | **8** | Один happy path + numpy backend |
| Документированность ветвлений | **8** | README подробный |
| Готовность к модели / продукту | **8** | Сводные scores + экспорт режимов |

**Итог L1: ~8.2 / 10** (условно: **B/C**, **§4.8**, FAISS-on vs off).
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
