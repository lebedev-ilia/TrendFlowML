# Audit v4 — `embedding_source_id_extractor` (TextProcessor)

**Дата отчёта:** 2026-04-14 (дополнение L2 + L1)  
**Уровень отчёта (план §3.1):** **L1 — draft** (исторический набор **A**) + попытка **L2** (A+B) — **заблокирована** на текущем `result_store`.  
**Артефакт (канон L1):** `storage/result_store/youtube/-Q6fnPIybEI/4c3bf25b-e300-47b3-915e-4699c72ab190/text_processor/text_features.npz`  
**Артефакт (воспроизводимый A для L2-скрипта):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/text_processor/text_features.npz`  
**Артефакт (tabular):** `…/text_processor/text_features.npz` — **13** ключей `tp_embid_*` ([`embedding_source_id_extractor_output_v1`](../../../../TextProcessor/schemas/embedding_source_id_extractor_output_v1.json)).  
**Вложенный контракт (не в machine JSON):** `payload["embedding_source_id"]` в том же NPZ + [`SCHEMA.md`](../../../../TextProcessor/src/extractors/embedding_source_id_extractor/SCHEMA.md).  
**Статистика L2 (инструмент):** `storage/audit_v4/embedding_source_id_extractor_l2/embedding_source_id_extractor_audit_v4_stats.json`  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт L2: `TextProcessor/src/extractors/embedding_source_id_extractor/scripts/audit_v4_npz_stats.py` (`--seed 0`)  
**Engineering log 4.2:** [`../audit_4_2/text_processor/embedding_source_id_extractor_engineering_log_v4_2.md`](../audit_4_2/text_processor/embedding_source_id_extractor_engineering_log_v4_2.md)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано · **◐** частично / только **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика + семантика + вердикт | ✓ | Этот документ |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты | ✓ | machine JSON + SCHEMA для nested |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, e2e | ✓ | `text_npz_v1` + `payload` |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | Политика **`transcript_first`**, primary **`transcript_combined_mean`** |
| **B** | ✗ | `title_first`, `strict_missing_primary`, пустой `tp_artifacts` |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `tp_embid_*` ↔ схема | ✓ | **13** имён, `allow_extra_keys: false` |
| Nested dict | ◐ | Валидирован вручную на **A** (ключи + `vector_id`) |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Все `tp_embid_*` на **A** | ✓ | **Конечны** (0/1 и флаги) |

#### §4.7 — Трактовка

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Наблюдения → выводы | ✓ | §5–6 |

#### §4.8 — Golden на **A**

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Сигнатура | ✗ | TODO (хэш **`vector_id`**) |

#### §5.3 — Сверка с Models

| Вопрос | Ответ |
|--------|--------|
| Модель в шаге | **Нет** — метаданные + хэш вектора |

#### §6–§8 (кратко)

L3 DoD — **✗**.

---

## 1. Мета

Реализация: **1.3.0** ([`main.py`](../../../../TextProcessor/src/extractors/embedding_source_id_extractor/main.py)).

---

## 2.1. L2: `result_store` и блокировка

Скрипт `TextProcessor/src/extractors/embedding_source_id_extractor/scripts/audit_v4_npz_stats.py` (выход: `storage/audit_v4/embedding_source_id_extractor_l2/embedding_source_id_extractor_audit_v4_stats.json`) берёт 5 путей A+B и проверяет:

- табличный срез `tp_embid_*` (**13** ключей),
- наличие nested `payload["embedding_source_id"]`,
- совпадение `vector_id` с вычисленным sha256 по `float32` байтам файла `text_processor/_artifacts/<embedding_relpath>`.

**Факт по `storage/result_store/youtube` (2026-04-14):** из 5 путей **2** файла `meta.status=ok` и содержат полный `tp_embid_*` + nested `embedding_source_id` + успешную сверку `vector_id`; **3** файла `meta.status=error` и не содержат табличного слоя, а nested `embedding_source_id` отсутствует.

Причина блокировки — сбой всего `text_processor` на части mock-run (часто OOM в эмбеддерах до выполнения downstream шагов), а не логика `embedding_source_id_extractor`.

---

## 2. Наблюдения на наборе **A**

### 2.1 `features_flat` (`tp_embid_*`)

| Поле | Значение |
|------|----------|
| `tp_embid_present` | **1** |
| `tp_embid_strict_missing_primary_enabled` | **1** |
| Политика | **`tp_embid_policy_transcript_first=1`**, остальные policy-флаги **0** |
| Primary one-hot | **`tp_embid_primary_is_transcript=1`**, title/description **0** |
| Флаги ошибок | `unsafe_relpath`, `primary_embed_missing`, `nan_inf` — **0** |

Сверка множеств имён JSON ↔ NPZ: **полное совпадение**.

### 2.2 `payload["embedding_source_id"]`

```json
{
  "vector_id": "15f24f63b5f0dbc3932e65af",
  "vector_store_uri": "faiss://semantic_titles_v1",
  "embedding_relpath": "transcript_combined_agg_mean.npy",
  "model_version": "unknown",
  "primary_source": "transcript_combined_mean"
}
```

### 2.3 Сверка **`vector_id`** с артефактом

Файл: `text_processor/_artifacts/transcript_combined_agg_mean.npy`, форма **(1024,)**, `float32`. Хэш по алгоритму [`_vector_id_from_values`](../../../../TextProcessor/src/extractors/embedding_source_id_extractor/main.py) даёт **`15f24f63b5f0dbc3932e65af`** — **совпадает** с `payload`.

### 2.4 HTML

`text_processor/_render/embedding_source_id_extractor_report.html`.

### 2.5 Заметка по render summary

В **`render_context.json`** поле **`summary.primary_source`** может отображаться укороченным (**\"transcript\"**), тогда как в **`embedding_source_id`** — каноническое **`transcript_combined_mean`**: для продукта опираться на **`payload` / result**, не только на summary рендера.

---

## 3. Вердикт

**Плюсы**

- Компактный жёсткий **machine schema** (13 ключей) и полное совпадение с NPZ.
- **Детерминированный** `vector_id` от тензора, проверенный на файле агрегата транскрипта.
- Privacy: в nested объекте **`embedding_relpath`**, не абсолютные пути.

**Минусы / внимание**

- Nested **`embedding_source_id`** не валидируется тем же JSON — нужен дисциплинарный SCHEMA/тесты.
- L1 не покрывает **другие политики** и падения при **`strict_missing_primary`**.

---

## 4. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Схема ↔ NPZ | **9.5** | Идеальное совпадение |
| Nested + `vector_id` | **9** | Эмпирически сверен хэш |
| Документированность | **8** | Расхождение summary vs payload |
| Edge coverage | **6** | Один policy path на **A** |

**Итог L1: ~8.4 / 10** (до **B/C** и §4.8).
