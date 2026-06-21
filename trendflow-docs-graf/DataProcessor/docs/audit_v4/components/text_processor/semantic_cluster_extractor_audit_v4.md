# Audit v4 — `semantic_cluster_extractor` (TextProcessor)

**Дата отчёта:** 2026-04-14 (дополнение L2 + L1)  
**Уровень отчёта (план §3.1):** **L1 — draft** (исторический набор **A**) + попытка **L2** (A+B) — **заблокирована** на текущем `result_store`.  
**Артефакт (канон L1):** `storage/result_store/youtube/-Q6fnPIybEI/4c3bf25b-e300-47b3-915e-4699c72ab190/text_processor/text_features.npz`  
**Артефакт (воспроизводимый A для L2-скрипта):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/text_processor/text_features.npz`  
**Срез компонента:** **31** ключ `tp_semclust_*` — [`semantic_cluster_extractor_output_v1`](../../../../TextProcessor/schemas/semantic_cluster_extractor_output_v1.json). Эмбеддинги: `doc.tp_artifacts["embeddings"]` → `.npy` в `_artifacts/`; PCA/центроиды — **`dp_models`** ([`main.py`](../../../../TextProcessor/src/extractors/semantic_cluster_extractor/main.py)).  
**Статистика L2 (инструмент):** `storage/audit_v4/semantic_cluster_extractor_l2/semantic_cluster_extractor_audit_v4_stats.json`  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт L2: `TextProcessor/src/extractors/semantic_cluster_extractor/scripts/audit_v4_npz_stats.py` (`--seed 0`)  
**Engineering log 4.2:** [`../audit_4_2/text_processor/semantic_cluster_extractor_engineering_log_v4_2.md`](../audit_4_2/text_processor/semantic_cluster_extractor_engineering_log_v4_2.md)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано · **◐** частично / только **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика + семантика + вердикт | ✓ | Этот документ |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты | ✓ | Machine JSON + [`SCHEMA.md`](../../../../TextProcessor/src/extractors/semantic_cluster_extractor/SCHEMA.md) |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, e2e | ✓ | `text_npz_v1` |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | **Primary = title**, эмбеддинги **title/description/hashtag** доступны, **`present=1`**, ближайший кластер найден |
| **B** | ✗ | **`emit_extra_metrics=true`** (размерности, **`margin_top2`**, **`compute_ms`**) |
| **C** | ✗ | Fallback на другой слот, **`require_faiss=true`** при отсутствии FAISS, пропуски эмбеддингов / **`dim_mismatch`** |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи ↔ схема | ✓ | **31** имя, `allow_extra_keys: false` |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ожидаемые NaN на **A** | ✓ | **`emit_extra_metrics=false`** → **`_apply_extra_block`** обнуляет «доп. блок»: **`tp_semclust_n_clusters`**, **`model_*_dim`**, **`embedding_dim`**, **`margin_top2`**, **`compute_ms`** — **NaN** ([`main.py`](../../../../TextProcessor/src/extractors/semantic_cluster_extractor/main.py) ~L352–358) — согласовано с описанием machine schema |

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
| Обучаемая модель в шаге | **Нет** — ближайший центроид по предрасчитанным **`dp_models`** PCA+centroids; вход — уже готовые эмбеддинги title/description/hashtag |

#### §6–§8 (кратко)

L3 DoD — **✗**.

---

## 1. Мета

Реализация: **1.3.0**.

---

## 2.1. L2: `result_store` и блокировка

Скрипт `TextProcessor/src/extractors/semantic_cluster_extractor/scripts/audit_v4_npz_stats.py` (выход: `storage/audit_v4/semantic_cluster_extractor_l2/semantic_cluster_extractor_audit_v4_stats.json`) берёт 5 путей A+B и выделяет `tp_semclust_*`.

**Факт по `storage/result_store/youtube` (2026-04-14):** из 5 путей **2** файла `meta.status=ok` и содержат полный срез (**31** ключ); **3** файла `meta.status=error` и не содержат табличного слоя (пустой `feature_names`).

Причина блокировки — сбой всего `text_processor` на части mock-run (часто OOM в эмбеддерах до выполнения downstream шагов), а не логика `semantic_cluster_extractor`.

---

## 2. Наблюдения на наборе **A**

### 2.1 Классификация

| Поле | Значение |
|------|----------|
| **`tp_semclust_present`** | **1** |
| **`tp_semclust_id`** | **25** |
| **`tp_semclust_similarity`** | **≈ 0.786** |
| **`tp_semclust_distance`** | **≈ 0.214** (в коде базируется на косинусе L2-нормированных векторов) |

### 2.2 Конфиг vs фактический бэкенд

| Поле | Значение |
|------|----------|
| **`tp_semclust_config_primary_title`** | **1** |
| **`tp_semclust_source_title`** | **1**, **`fallback_used`** **0** |
| **`tp_semclust_use_faiss_enabled`** | **1** |
| **`tp_semclust_backend_faiss`** | **0** |
| **`tp_semclust_require_faiss_enabled`** | **0** |

При **`use_faiss=True`** индекс FAISS создаётся только если модуль **`faiss`** импортировался (~L240–248); иначе **`_faiss_index`** остаётся **`None`**, поиск идёт по **numpy** (см. **`meta.backend`**: `faiss_ip` vs `numpy_cosine` в **`_semantic_meta`** ~L303–310). На **A** среде **FAISS не загружен**, **`require_faiss=false`** — корректный fallback.

### 2.3 Слоты эмбеддингов

**`tp_semclust_*_present`**: title/description/hashtag — **1**; флаги **`*_embed_missing_flag`** — **0**; **`dim_mismatch`**, **`unsafe_relpath`** — **0**.

### 2.4 HTML

`text_processor/_render/semantic_cluster_extractor_report.html`.

---

## 3. Вердикт

**Плюсы**

- Полное совпадение **31** ключа со схемой; на **A** основной продуктовый срез (**id**, **similarity**, **distance**) заполнен.
- Явные флаги источника и отсутствия эмбеддингов упрощают диагностику пайплайна.

**Минусы / внимание**

- **`tp_semclust_use_faiss_enabled`** и **`tp_semclust_backend_faiss`** могут **расходиться** — потребителям аналитики смотреть **`backend_faiss`**, а не только «wanted FAISS».
- Без **`emit_extra_metrics=true`** нет числовой телеметрии по размерностям и **`margin_top2`** — для L2-аудита полезен отдельный прогон.

---

## 4. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Схема ↔ NPZ ↔ код | **9** | Множества имён совпали |
| Полнота эмпирики на **A** | **8** | Кластер найден; extra-блок намеренно **NaN** |
| Документированность ветвлений | **8** | FAISS optional vs фактический backend стоит держать в голове |
| Готовность к модели / продукту | **9** | **`present`**, **id**, **similarity** однозначны |

**Итог L1: ~8.3 / 10** (условно: **B/C** для FAISS-only и негативных путей).
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
