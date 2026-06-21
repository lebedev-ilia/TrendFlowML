# Audit v4 — `hashtag_embedder` (TextProcessor)

**Дата отчёта:** 2026-04-14 (дополнение L2 + L1)  
**Уровень отчёта (план §3.1):** **L1 — draft** (исторический набор **A**) + попытка **L2** (A+B) — **заблокирована** на текущем `result_store`.  
**Артефакт (канон L1):** `storage/result_store/youtube/-Q6fnPIybEI/4c3bf25b-e300-47b3-915e-4699c72ab190/text_processor/text_features.npz`  
**Артефакт (воспроизводимый A для L2-скрипта):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/text_processor/text_features.npz`  
**Срез компонента:** **23** ключей `tp_hashemb_*` — [`hashtag_embedder_output_v1`](../../../../TextProcessor/schemas/hashtag_embedder_output_v1.json). Плотный вектор: `text_processor/_artifacts/hashtag_embedding.npy` (не в `features_flat`, см. machine schema). Registry: `doc.tp_artifacts["embeddings"]["hashtag"]`.  
**Статистика L2 (инструмент):** `storage/audit_v4/hashtag_embedder_l2/hashtag_embedder_audit_v4_stats.json`  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт L2: `TextProcessor/src/extractors/hashtag_embedder/scripts/audit_v4_npz_stats.py` (`--seed 0`)  
**Engineering log 4.2:** [`../audit_4_2/text_processor/hashtag_embedder_engineering_log_v4_2.md`](../audit_4_2/text_processor/hashtag_embedder_engineering_log_v4_2.md)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано · **◐** частично / только **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика + семантика + вердикт | ✓ | Этот документ |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты | ✓ | Machine JSON + [`SCHEMA.md`](../../../../TextProcessor/src/extractors/hashtag_embedder/SCHEMA.md) |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, e2e | ✓ | `text_npz_v1` + `.npy` |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | **`doc.hashtags`**: **3** → **3** уникальных после каноникализации; эмбеддинг **L2=1**, **D=1024**, **CUDA** + **fp16** |
| **B** | ✗ | `extract_batch`, `use_frequencies=true`, `aggregation` ≠ mean, `cache_enabled` + hit |
| **C** | ✗ | Пустые хештеги, `require_hashtags=true`, `write_artifact=false` |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи ↔ схема | ✓ | **23** имён, `allow_extra_keys: false` |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| NaN на **A** | ✓ | **Нет** — все **`tp_hashemb_*`** конечны |
| Пустой путь в коде | ✓ | При отсутствии тегов шаблон выставляет **`dim`/`l2`/часть счётчиков** в **NaN** / **0** по веткам [`main.py`](../../../../TextProcessor/src/extractors/hashtag_embedder/main.py) — на **A** не воспроизведено |

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
| Модель | Sentence-transformers через **`get_model_with_meta`** ([`main.py`](../../../../TextProcessor/src/extractors/hashtag_embedder/main.py)); на **A** **`tp_hashemb_dim=1024`**, **`tp_hashemb_model_digest_u24`** заполнен |

#### §6–§8 (кратко)

L3 DoD — **✗**.

---

## 1. Мета

Реализация: **1.2.0**.

---

## 2.1. L2: `result_store` и блокировка

Скрипт `TextProcessor/src/extractors/hashtag_embedder/scripts/audit_v4_npz_stats.py` (выход: `storage/audit_v4/hashtag_embedder_l2/hashtag_embedder_audit_v4_stats.json`) берёт 5 путей A+B и проверяет:

- табличный срез `tp_hashemb_*` (**23** ключа),
- артефакт `text_processor/_artifacts/hashtag_embedding.npy` (shape/dtype/L2),
- согласование `tp_hashemb_dim` и размерности `.npy`.

**Факт по `storage/result_store/youtube` (2026-04-14):** из 5 путей **2** файла `meta.status=ok` и содержат полный `tp_hashemb_*` + присутствует `hashtag_embedding.npy` (**1024**, `float32`); **3** файла `meta.status=error` и не содержат табличного слоя, артефакт отсутствует.

Причина блокировки — сбой всего `text_processor` на части mock-run (часто OOM в эмбеддерах до выполнения downstream шагов), а не логика `hashtag_embedder`.

---

## 2. Наблюдения на наборе **A**

### 2.1 Табличный срез

| Группа | Значение |
|--------|----------|
| Статус | **`tp_hashemb_present` 1**, **`tp_hashemb_compute_enabled` 1** |
| Теги | **`n_input_tags` 3**, **`n_unique_tags` 3**, **`tag_count` 3**, **`n_tags_truncated` 0** |
| Вектор (мета) | **`dim` 1024**, **`l2_norm` 1** |
| Политика | **`require_hashtags_enabled` 0**, **`disabled_by_policy_hint` 0** |
| Артефакт | **`write_artifact_enabled` 1**, **`artifact_written` 1** |
| Устройство | **`device_cuda` 1**, **`fp16` 1** |
| Агрегация | **`agg_mean` 1**, **`agg_max`/`agg_logsumexp`/`use_frequencies` 0** |
| Кеш | **`cache_enabled` 0**, **`cache_hit` 0** |
| Тайминги | **`encode_ms` ≈ 49**, **`agg_ms` ≈ 0.09** — конечны (флаг **`emit_extra_metrics`** в [`main.py`](../../../../TextProcessor/src/extractors/hashtag_embedder/main.py) **не влияет** на запись таймингов — только хранится в `__init__`) |

### 2.2 Файл `hashtag_embedding.npy`

Форма **`(1024,)`**, **`float32`**, **L2 = 1.0** — согласовано с **`tp_hashemb_*`**.

### 2.3 HTML

`text_processor/_render/hashtag_embedder_report.html`.

---

## 3. Вердикт

**Плюсы**

- Полное совпадение множества **23** ключей со схемой; на **A** — «счастливый» путь: теги есть, эмбеддинг и `.npy` согласованы.
- Детерминированные счётчики тегов и флаги агрегации/устройства читаемы для аналитики.

**Минусы / внимание**

- Параметр **`emit_extra_metrics`** задан в конструкторе, но **нигде не используется** при формировании `features_flat` — риск расхождения ожиданий YAML/операторов с другими embedder’ами.
- В возврате **`system.peaks.gpu_peak_mb`** в коде зафиксирован **0** при **`device_cuda=1`** (косметика для телеметрии пика GPU).

---

## 4. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Схема ↔ NPZ ↔ код | **9** | Множества имён совпали; вектор вне таблицы — по контракту |
| Полнота эмпирики на **A** | **9** | Заполнен основной сценарий |
| Документированность ветвлений | **7** | Мёртвый **`emit_extra_metrics`** |
| Готовность к модели / продукту | **9** | **`present`**, **L2=1**, размерность явно |

**Итог L1: ~8.3 / 10** (условно: **B/C** и golden — отдельно).
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
