# Audit v4 — `comments_aggregator` (TextProcessor)

**Дата отчёта:** 2026-04-14 (дополнение L2 + L1)  
**Уровень отчёта (план §3.1):** **L1 — draft** (исторический набор **A**) + попытка **L2** (A+B) — **заблокирована** на текущем `result_store` (см. §2.1).  
**Артефакт (канон L1):** `storage/result_store/youtube/-Q6fnPIybEI/4c3bf25b-e300-47b3-915e-4699c72ab190/text_processor/text_features.npz`  
**Артефакт (воспроизводимый A для L2-скрипта):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/text_processor/text_features.npz`  
**Срез компонента:** ровно **39** ключей (`tp_commentsagg_*`, legacy `tp_comments_agg_*`, `tp_cagg_*`) в `feature_names` / `feature_values` — контракт [`comments_aggregator_output_v1`](../../../../TextProcessor/schemas/comments_aggregator_output_v1.json). Доп. массивы: `comments_embeddings.npy`, `comments_agg_mean.npy`, `comments_agg_median.npy`, `comments_selected_indices.npy`.  
**Статистика L2 (инструмент):** `storage/audit_v4/comments_aggregator_l2/comments_aggregator_audit_v4_stats.json`  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт L2: `TextProcessor/src/extractors/comments_aggregator/scripts/audit_v4_npz_stats.py` (`--seed 0`)  
**Engineering log 4.2:** [`../audit_4_2/text_processor/comments_aggregator_engineering_log_v4_2.md`](../audit_4_2/text_processor/comments_aggregator_engineering_log_v4_2.md)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано · **◐** частично / только **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика + семантика + вердикт | ✓ | Этот документ |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты | ✓ | Machine JSON + [`SCHEMA.md`](../../../../TextProcessor/src/extractors/comments_aggregator/SCHEMA.md); upstream: **comments_embedder** (эмбеддинги в `_artifacts`) |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, e2e | ✓ | Merge в `text_npz_v1` + `.npy` агрегаты |
| Путь под `run_id` | ✓ | `text_processor/…` |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | Один reference run, **5** комментариев, **D=1024** |
| **B** | ✗ | Другие **N**, веса likes/authority/recency |
| **C** | ✗ | Нет эмбеддингов, `require_comment_embeddings=true`, неверный relpath |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи ↔ схема (`allow_extra_keys: false`) | ✓ | Множества совпали |
| `float32` в `feature_values` | ✓ |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| NaN там, где ожидается по контракту | ✓ | `*_mean_std`, `*_median_std` — **NaN** при `tp_commentsagg_compute_std_enabled=0`; `tp_commentsagg_agg_*_ms` — **NaN** при `emit_extra_metrics=false` (по умолчанию) |

#### §4.3 — Распределения

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Перцентили tabular | ◐ | Один ряд значений на документ |

#### §4.7 — Трактовка

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Наблюдения → выводы | ✓ | §5–6 |

#### §4.8 — Golden на **A**

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Сигнатура | ✗ | TODO |

#### §4.12 — Anti-leakage

| Вопрос | Ответ |
|--------|--------|
| Утечка future comments? | **N/A** в рамках одного снимка документа; политика сбора комментариев — вне экстрактора |
| Небезопасные пути | **0** `tp_commentsagg_unsafe_relpath_flag` на **A** |

#### §5.3 — Сверка с Models

| Вопрос | Ответ |
|--------|--------|
| Зависимость от embedding-модели | Да: **intfloat/multilingual-e5-large** (и digest) резолвится в **`__init__`** [`main.py`](../../../../TextProcessor/src/extractors/comments_aggregator/main.py); сами векторы считает **comments_embedder** |
| Tabular в Baseline v1 | **Нет** / косвенно: агрегаты — кандидаты product-facing |

#### §6–§8 (кратко)

L3 DoD / автоматический regression — **✗** (L1).

---

## 1. Мета `text_features.npz` (фрагмент)

Как у [`asr_text_proxy_audio_features`](asr_text_proxy_audio_features_audit_v4.md): **`meta.schema_version`:** `text_npz_v1`; **`meta.models_used`:** `[]` на уровне агрегатора (модель фиксируется при инициализации экстрактора и в цепочке embedder).

Реализация **`comments_aggregator`:** **1.3.0**.

---

## 2.1. L2: `result_store` и блокировка

Скрипт `TextProcessor/src/extractors/comments_aggregator/scripts/audit_v4_npz_stats.py` (выход: `storage/audit_v4/comments_aggregator_l2/comments_aggregator_audit_v4_stats.json`) берёт 5 путей A+B (как у Visual L2), выделяет 39 табличных ключей и проверяет наличие `.npy` артефактов.

**Факт по `storage/result_store/youtube` (2026-04-14):** из 5 путей **2** файла `meta.status=ok` и содержат полный срез (**39** ключей), артефакты mean/median/indices существуют; **3** файла `meta.status=error` и не содержат табличного слоя (пустой `feature_names`), артефакты отсутствуют.

Причина блокировки — сбой всего `text_processor` (часто OOM в эмбеддерах до выполнения downstream шагов), а не логика `comments_aggregator`.

---

## 2. Наблюдения на наборе **A**

### 2.1 Табличный срез (39 ключей)

| Группа | Значение (кратко) |
|--------|-------------------|
| Присутствие | `tp_commentsagg_present` = **`1`**, `tp_commentsagg_count` = **`5`**, `tp_commentsagg_dim` = **`1024`** |
| Флаги вычислений | mean/median **on** (`1`), **std off** (`0`); зеркала `tp_comments_agg_compute_*` согласованы |
| Артефакты | `tp_commentsagg_artifact_mean_written` = **`1`**, median **`1`**, `write_artifacts_enabled` **`1`** |
| Веса | `tp_commentsagg_weights_applied` **`0`**, маски весов **`0`** (равные веса на **A**) |
| Выравнивание весов | `tp_commentsagg_weights_align_present` **`1`**, `weights_align_shape_ok` **`1`** |
| Ошибки | `dim_mismatch` **`0`**, `unsafe_relpath` **`0`** |
| NaN (ожидаемо) | `tp_commentsagg_mean_std`, `median_std` и legacy/`tp_cagg_*` копии; `tp_commentsagg_agg_mean_ms`, `agg_median_ms` |

### 2.2 Бинарные артефакты

| Файл | Форма | Примечание |
|------|-------|------------|
| `comments_embeddings.npy` | **(5, 1024)** `float32` | все конечны |
| `comments_agg_mean.npy` | **(1024,)** | **L2 ≈ 1** |
| `comments_agg_median.npy` | **(1024,)** | **L2 ≈ 1** |
| `comments_selected_indices.npy` | **(5,)** `int32` | `[0,1,2,3,4]` |

### 2.3 HTML

`text_processor/_render/comments_aggregator_report.html`.

---

## 3. Сверка с кодом

1. Порядок ключей **`_FEATURES_FLAT_KEYS`** в [`main.py`](../../../../TextProcessor/src/extractors/comments_aggregator/main.py) совпадает со списком полей JSON-схемы (39 позиций).
2. Три семейства имён — дубли для совместимости; на **A** численно совпадают попарно (present, count, dim, std, флаги весов).
3. **`mean_std` / `median_std`**: при `compute_std=False` код возвращает **`nan`** для std — совпадает с NPZ.

---

## 4. Вердикт

**Плюсы**

- Жёсткий **machine schema** и полное совпадение множества ключей с NPZ на **A**.
- Явные флаги гейтов (**compute_**, **weights_**, **artifact_**, safety).
- Агрегированные векторы **L2-нормированы**; формы `(N,D)` / `(D,)` согласованы.

**Минусы / внимание**

- Три префикса увеличивают шум для потребителей — нужна дисциплина «канон = `tp_commentsagg_*`».
- **`emit_extra_metrics`** по умолчанию даёт **NaN** в `*_agg_*_ms` — потребители должны читать флаги конфигурации или документацию.
- L1 не покрывает **0 комментариев**, несовпадение индексов/весов, path traversal (только флаг `unsafe`).

---

## 5. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Схема ↔ NPZ ↔ код | **9** | Полное совпадение на **A** |
| Интерпретируемость для downstream | **8** | Legacy-ключи; NaN по std/timing ожидаемы, но требуют документа |
| Артефакты и численная согласованность | **9** | L2≈1, формы ок |
| Покрытие edge cases эмпирикой | **6** | Только успешный путь с **N=5** |

**Итог L1: ~8.2 / 10** (округляя **8/10** до B/C и §4.8).
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
