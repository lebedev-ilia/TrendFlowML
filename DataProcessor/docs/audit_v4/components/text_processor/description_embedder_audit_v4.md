# Audit v4 — `description_embedder` (TextProcessor)

**Дата отчёта:** 2026-04-14 (дополнение L2 + L1)  
**Уровень отчёта (план §3.1):** **L1 — draft** (исторический набор **A**) + попытка **L2** (A+B) — **заблокирована** на текущем `result_store` (см. §2.1).  
**Артефакт (канон L1):** `storage/result_store/youtube/-Q6fnPIybEI/4c3bf25b-e300-47b3-915e-4699c72ab190/text_processor/text_features.npz`  
**Артефакт (воспроизводимый A для L2-скрипта):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/text_processor/text_features.npz`  
**Срез компонента:** **19** ключей `tp_descemb_*` — [`description_embedder_output_v1`](../../../../TextProcessor/schemas/description_embedder_output_v1.json). Артефакт: `text_processor/_artifacts/description_embedding.npy`.  
**Статистика L2 (инструмент):** `storage/audit_v4/description_embedder_l2/description_embedder_audit_v4_stats.json`  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт L2: `TextProcessor/src/extractors/description_embedder/scripts/audit_v4_npz_stats.py` (`--seed 0`)  
**Engineering log 4.2:** [`../audit_4_2/text_processor/description_embedder_engineering_log_v4_2.md`](../audit_4_2/text_processor/description_embedder_engineering_log_v4_2.md)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано · **◐** частично / только **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика + семантика + вердикт | ✓ | Этот документ |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты | ✓ | Machine JSON + [`SCHEMA.md`](../../../../TextProcessor/src/extractors/description_embedder/SCHEMA.md) |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, e2e | ✓ | `text_npz_v1` + `.npy` |
| Путь под `run_id` | ✓ | `text_processor/…` |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | Описание непустое, **1** chunk, **D=1024**, **CUDA** + **fp16** |
| **B** | ✗ | Много чанков, CPU, кеш hit |
| **C** | ✗ | Пустое описание, `compute_embedding=false` |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи ↔ схема | ✓ | **19** имён, `allow_extra_keys: false` |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| NaN на **A** | ✓ | **Нет** среди фактических значений (все скаляры конечны) |
| `tp_descemb_norm_raw` | ✓ | **~1.00015** (до финальной L2-нормировки) |

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
| Модель | Sentence-transformers через `get_model_with_meta` + токенизатор **`shared_tokenizer_v1`** ([`main.py`](../../../../TextProcessor/src/extractors/description_embedder/main.py)) |
| Baseline | Нет явного списка `tp_descemb_*` в [`BASELINE_MODEL.md`](../../../../../Models/docs/contracts/BASELINE_MODEL.md) |

#### §6–§8 (кратко)

L3 DoD — **✗**.

---

## 1. Мета

Реализация: **1.2.0**.

---

## 2.1. L2: `result_store` и блокировка

Скрипт `TextProcessor/src/extractors/description_embedder/scripts/audit_v4_npz_stats.py` (выход: `storage/audit_v4/description_embedder_l2/description_embedder_audit_v4_stats.json`) берёт 5 путей A+B (как у Visual L2), выделяет `tp_descemb_*` и проверяет `description_embedding.npy`.

**Факт по `storage/result_store/youtube` (2026-04-14):** из 5 путей **2** файла `meta.status=ok` и содержат полный срез (**19** ключей) и артефакт; **3** файла `meta.status=error` и не содержат табличного слоя (пустой `feature_names`), артефакт отсутствует.

Причина блокировки — сбой всего `text_processor` (часто OOM в эмбеддерах до выполнения downstream шагов), а не логика `description_embedder`.

---

## 2. Наблюдения на наборе **A**

### 2.1 Табличный срез (сжато)

| Группа | Значение |
|--------|----------|
| Статус | `tp_descemb_present` **1**, `tp_descemb_description_present` **1** |
| Размерность | `tp_descemb_dim` **1024** |
| Нормы | `tp_descemb_l2_norm` **1**; `tp_descemb_norm_raw` **≈1.00015** |
| Чанки | `tp_descemb_n_chunks` **1**, `tp_descemb_avg_chunk_tokens` **39** |
| Пуллинг | `tp_descemb_pooling_length_weighted` **1** |
| Аппарат | `tp_descemb_device_cuda` **1**, `tp_descemb_fp16` **1** |
| Артефакт | `tp_descemb_artifact_written` **1**, `write_artifact_enabled` **1** |
| Кеш | `cache_enabled` **0**, `cache_hit` **0** |
| Тайминги (мс) | `chunk_ms` **~0.37**, `encode_ms` **~132**, `pool_ms` **~51.8** |
| Digest | `tp_descemb_model_digest_u24` **11259398** (uint24 из hex digest) |

Сверка множеств имён JSON ↔ NPZ: **без расхождений**.

### 2.2 Файл `description_embedding.npy`

| Проверка | Результат |
|----------|-----------|
| Форма | **(1024,)** `float32` |
| L2 | **1.0** |

### 2.3 HTML

`text_processor/_render/description_embedder_report.html`.

### 2.4 Заметка по коду

Параметр **`emit_extra_metrics`** есть в [`__init__`](../../../../TextProcessor/src/extractors/description_embedder/main.py), но **нигде не используется** — тайминги и вспомогательные поля заполняются на успешном пути всегда. Для выравнивания с `comments_embedder` / документацией имеет смысл либо задействовать флаг, либо удалить его (вне L1).

---

## 3. Вердикт

**Плюсы**

- Строгое совпадение **19** ключей со схемой.
- Согласованность **таблица ↔ `.npy`** (размерность, L2).
- Явные флаги **presence**, **chunk stats**, **CUDA/fp16**.

**Минусы / внимание**

- Мёртвый **`emit_extra_metrics`** — путаница для читателей кода.
- L1 не покрывает **пустое описание**, **multi-chunk**, **cache hit**.

---

## 4. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Схема ↔ NPZ ↔ артефакт | **9.5** | Полное совпадение на **A** |
| Чистота контракта / кода | **7.5** | `emit_extra_metrics` не используется |
| Численное качество | **9** | L2=1, norm_raw осмыслен |
| Edge coverage | **6** | Один короткий путь (1 chunk) |

**Итог L1: ~8.2 / 10** (до **B/C** и правки флага/доков).
