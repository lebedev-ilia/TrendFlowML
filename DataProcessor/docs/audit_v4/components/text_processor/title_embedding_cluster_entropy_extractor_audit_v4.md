# Audit v4 — `title_embedding_cluster_entropy_extractor` (TextProcessor)

**Дата:** 2026-04-14  
**Уровень отчёта (план §3.1):** **L2** (целевые наборы **A+B**, **5** путей из `result_store`; фактически данные компонента только на subset из-за ошибок `text_processor`, см. `dataset_quality`).  
**L2 stats (JSON):** `storage/audit_v4/title_embedding_cluster_entropy_extractor_l2/title_embedding_cluster_entropy_extractor_audit_v4_stats.json`  
**Engineering log 4.2:** `DataProcessor/docs/audit_v4/components/audit_4_2/text_processor/title_embedding_cluster_entropy_extractor_engineering_log_v4_2.md`  
**Артефакт (табличный срез):** `…/text_processor/text_features.npz`  
**Срез компонента:** **24** ключа `tp_titleclent_*` — [`title_embedding_cluster_entropy_extractor_output_v1`](../../../../TextProcessor/schemas/title_embedding_cluster_entropy_extractor_output_v1.json); **`allow_extra_keys: false`**. Вход: **`title_embedding.npy`** из реестра **`doc.tp_artifacts.embeddings.title`** (обычно после **`title_embedder`**); PCA/центроиды — **`dp_models`** ([`main.py`](../../../../TextProcessor/src/extractors/title_embedding_cluster_entropy_extractor/main.py)).  
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
| Ссылки на контракты | ✓ | Machine JSON + [`SCHEMA.md`](../../../../TextProcessor/src/extractors/title_embedding_cluster_entropy_extractor/SCHEMA.md) |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, e2e | ✓ | `text_npz_v1` + цепочка **`tp_artifacts`** |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | Title-эмбеддинг **present**, **`tp_titleclent_present=1`**; **`use_faiss_enabled=1`**, **`backend_faiss=0`** (модуль **faiss** недоступен → **numpy** cosine **`reduced @ centroids.T`**); **`emit_extra_metrics=false`** → **NaN** у `n_clusters` / `model_*_dim` / `margin_top2` / `compute_ms` |
| **B** | ◐ | В L2 набор **B** формально включён (4 доп. видео), но `text_processor` падает на **3/5** путях → компонентный срез есть только на subset (см. §2.1 и JSON `dataset_quality`). Целевые вариации B: **`emit_extra_metrics=true`**, **`export_topk_distribution=true`**, **`top_k_slots` > 8 (clamp)** |
| **C** | ✗ | Нет **`relpath`** / файл пропал, **`dim_mismatch`**, **`require_title_embedding`**, **`require_faiss=true`** без FAISS |

### 2.1. L2: `result_store` и блокировка

L2 запущен на стандартных **5** путях A+B (см. `paths` в JSON). Результат:

- на **2/5** путях `meta.status=ok` и присутствует полный `tp_titleclent_*` (**24** ключа); upstream `text_processor/_artifacts/title_embedding.npy` присутствует (shape **(1024,)**);
- на **3/5** путях `meta.status=error`, `feature_names` пустой → компонентный срез отсутствует, upstream артефакт не записан; причина — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

Итог: **L2 blocked** для `title_embedding_cluster_entropy_extractor` (нужны **5/5** OK `text_processor`).

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи ↔ схема | ✓ | **24** имён, без лишних **`tp_titleclent_*`** |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ожидаемые NaN на **A** | ✓ | **`emit_extra_metrics=false`** ([`main.py`](../../../../TextProcessor/src/extractors/title_embedding_cluster_entropy_extractor/main.py) ~446–451): **`tp_titleclent_n_clusters`**, **`tp_titleclent_model_orig_dim`**, **`tp_titleclent_model_reduced_dim`**, **`tp_titleclent_margin_top2`**, **`tp_titleclent_compute_ms`** |

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
| Внешние артефакты | **`clusters_spec_name`** → PCA + центроиды через **`ModelManager`** |
| Обучаемая модель в рантайме | **Нет** — только матричные операции и softmax по top‑K сходствам |

#### §6–§8 (кратко)

L3 DoD — **✗**.

---

## 1. Мета

Реализация: **1.3.0**.

---

## 2. Наблюдения на наборе **A** (исторические L1-заметки)

### 2.1 Табличный срез (сжато)

| Поле | Значение |
|------|----------|
| Статус | `tp_titleclent_present` **1**, `tp_titleclent_title_present` **1**, `tp_titleclent_dim_mismatch_flag` **0** |
| Top‑K | **requested/slots/used** = **5** / **5** / **5**; **`distinct_clusters_topk`** **5**; cap **`schema_top_k_slots_max`** **8**; **`top_k_slots_clamped`** **0** |
| Софтмакс | **`temperature`** **≈0.1** |
| Энтропия | **`entropy_raw`** **≈1.545**, **`entropy_norm`** **≈0.960**, **`perplexity`** **≈4.69** |
| FAISS | **`use_faiss_enabled`** **1**, **`require_faiss_enabled`** **0**, **`backend_faiss`** **0** (fallback **numpy**) |
| Конфиг (флаги) | **`emit_extra_metrics_enabled`** **0**, **`export_topk_distribution_enabled`** **0**, **`require_title_embedding_enabled`** **0** |
| Заблокировано extra | **`n_clusters`**, **`model_orig_dim`**, **`model_reduced_dim`**, **`margin_top2`**, **`compute_ms`** → **NaN** |

### 2.2 Связанный артефакт

Цепочка: **`title_embedding.npy`** **(1024,)**, согласован с upstream **`title_embedder`** на **A** (см. [`title_embedder_audit_v4.md`](title_embedder_audit_v4.md)).

### 2.3 HTML

`text_processor/_render/title_embedding_cluster_entropy_extractor_report.html`.

### 2.4 Payload

В **`result`** оркестратора (не в таблице **752**): вложенный блок **`title_cluster_entropy_meta`** с **`clusters_spec_*`**, **`backend`**: **`numpy_cosine`** на **A** при **`backend_faiss=0`**.

---

## 3. Вердикт

**Плюсы**

- Жёсткий набор из **24** ключей совпадает со схемой; ветки **empty / mismatch** дают стабильный шаблон (**`main.py`** ~276–299).
- На **A** числовое ядро (энтропия, perplexity, top‑K) конечно и согласовано с **`present=1`**.

**Минусы / внимание**

- **`tp_titleclent_use_faiss_enabled`** отражает **конфиг**, а **`tp_titleclent_backend_faiss`** — фактическое построение индекса; при отсутствии пакета **faiss** они **расходятся** (как в **`semantic_cluster_extractor`** на **A**).
- Пять полей **NaN** при **`emit_extra_metrics=false`** — ожидаемо, но downstream должен фильтровать или включать **`emit_extra_metrics`** для диагностики.

---

## 4. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Схема ↔ NPZ ↔ код | **9** | 24/24; описание extra/NaN в machine schema |
| Полнота эмпирики на **A** | **8** | Happy path + numpy fallback; нет **faiss** / **export_topk** |
| Документированность ветвлений | **8** | README v3; флаги FAISS стоит явнее связать с **backend** |
| Готовность к модели / продукту | **8** | Энтропия по таксономии, зависимость от порядка экстракторов |

**Итог L1: ~8.2 / 10** (условно: **B/C**, **§4.8**, сравнение **faiss** vs **numpy** на тех же весах).
