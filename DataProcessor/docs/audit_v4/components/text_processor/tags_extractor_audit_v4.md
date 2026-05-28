# Audit v4 — `tags_extractor` (TextProcessor)

**Дата отчёта:** 2026-04-14 (дополнение L2 + L1)  
**Уровень отчёта (план §3.1):** **L1 — draft** (исторический набор **A**) + попытка **L2** (A+B) — **заблокирована** на текущем `result_store`.  
**Артефакт (канон L1):** `storage/result_store/youtube/-Q6fnPIybEI/4c3bf25b-e300-47b3-915e-4699c72ab190/text_processor/text_features.npz`  
**Артефакт (воспроизводимый A для L2-скрипта):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/text_processor/text_features.npz`  
**Срез компонента:** **28** базовых ключей из [`tags_extractor_output_v1`](../../../../TextProcessor/schemas/tags_extractor_output_v1.json) + **15** слотов **`tp_tags_top{i}_{present,hash01,len}`** при **`top_k_slots=5`** (`i=1..5`). Machine schema: **`allow_extra_keys: true`** (доп. слоты при увеличении **`top_k_slots`**).  
**Статистика L2 (инструмент):** `storage/audit_v4/tags_extractor_l2/tags_extractor_audit_v4_stats.json`  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт L2: `TextProcessor/src/extractors/tags_extractor/scripts/audit_v4_npz_stats.py` (`--seed 0`)  
**Engineering log 4.2:** [`../audit_4_2/text_processor/tags_extractor_engineering_log_v4_2.md`](../audit_4_2/text_processor/tags_extractor_engineering_log_v4_2.md)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано · **◐** частично / только **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика + семантика + вердикт | ✓ | Этот документ |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты | ✓ | Machine JSON + [`SCHEMA.md`](../../../../TextProcessor/src/extractors/tags_extractor/SCHEMA.md) |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, e2e | ✓ | `text_npz_v1` |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | Title+description **present**; **3** уникальных хештега из title; слоты **`top4`/`top5`** пустые |
| **B** | ✗ | **`export_hashtags_mode_raw/hashed`**, **`merge_json_hashtags`** с JSON, усечения, **`require_title`**, пустые поля |
| **C** | ✗ | **`hashtags_disabled_by_policy`**, **`enable_extract_hashtags=false`** |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Базовые ключи ↔ схема | ✓ | Все **28** полей из `fields` присутствуют в NPZ |
| Доп. ключи | ✓ | Разрешены контрактом; на **A** **15** ключей **`tp_tags_top*_*`** |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ожидаемые NaN на **A** | ✓ | Для незаполненных слотов (**`present=0`**) **`hash01`/`len`** — **NaN** ([`main.py`](../../../../TextProcessor/src/extractors/tags_extractor/main.py) ~359–361); при **0** тегов **`avg_len`/`max_len`** могли бы быть **NaN** — на **A** теги есть |

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
| Обучаемая модель | **Нет** — парсинг и нормализация строк |

#### §6–§8 (кратко)

L3 DoD — **✗**.

---

## 1. Мета

Реализация: **1.2.0**.

---

## 2.1. L2: `result_store` и блокировка

Скрипт `TextProcessor/src/extractors/tags_extractor/scripts/audit_v4_npz_stats.py` (выход: `storage/audit_v4/tags_extractor_l2/tags_extractor_audit_v4_stats.json`) берёт 5 путей A+B и проверяет:

- полный срез `tp_tags_*`,
- базовые поля (28 ключей из machine schema) + наличие/консистентность слотов `top1..top5` (`present/hash01/len`),
- NaN-консистентность: при `top{i}_present=0` поля `hash01/len` должны быть NaN.

**Факт по `storage/result_store/youtube` (2026-04-14):** из 5 путей **2** файла `meta.status=ok` и содержат полный набор `tp_tags_*` для `top_k_slots=5`: **43** ключа (= 28 базовых + 15 slot‑ключей). **3** пути имеют `meta.status=error` и не содержат табличного слоя (пустой `feature_names`).

Причина блокировки — сбой всего `text_processor` на части mock-run (часто OOM в эмбеддерах до выполнения downstream шагов), а не логика `tags_extractor`.

---

## 2. Наблюдения на наборе **A**

### 2.1 Наличие текстов и политика

| Поле | Значение |
|------|----------|
| **`tp_tags_title_present` / `description_present`** | **1** / **1** |
| **`tp_tags_group_extract_enabled`** | **1** |
| **`tp_tags_hashtags_disabled_by_policy`** | **0** |
| **`tp_tags_export_hashtags_mode_none`** | **1** (сырой список тегов в NPZ не экспортируется режимом таблицы; downstream — **`doc.hashtags`** / мутации) |
| **`tp_tags_require_title_enabled`** | **1** |

### 2.2 Хештеги

| Поле | Значение |
|------|----------|
| **`hashtag_unique_count`** | **3** |
| **`title_hashtag_found_count`** | **3**, **description** — **0** |
| **`hashtag_avg_len` / `max_len`** | **5** / **7** |
| Слоты **top1–top3** | **`present=1`**, **`hash01`/`len`** конечны |
| Слоты **top4–top5** | **`present=0`**, **`hash01`/`len`** — **NaN** |
| **`tp_tags_topk_slots`** | **5** |

### 2.3 HTML

`text_processor/_render/tags_extractor_report.html`.

---

## 3. Вердикт

**Плюсы**

- Все обязательные поля схемы присутствуют; динамические слоты согласованы с **`allow_extra_keys: true`** и **`top_k_slots`**.
- На **A** виден заполненный путь: title даёт теги, description без inline `#`, счётчики и плотность согласованы.

**Минусы / внимание**

- Потребители должны помнить про **NaN** в **hash/len** при **`present=0`** (а не **0.0** для hash).
- Для L2 имеет смысл прогнать **`export_hashtags_mode_hashed`** только в таблице vs **raw** в debug, **`merge_json_hashtags`**, пустой title при **`require_title`**.

---

## 4. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Схема ↔ NPZ ↔ код | **9** | База + слоты по контракту |
| Полнота эмпирики на **A** | **9** | Основной путь + пустые слоты |
| Документированность ветвлений | **8** | Режимы export/mutate в README |
| Готовность к модели / продукту | **8** | **`hashtag_unique_count`**, слоты hash/len |

**Итог L1: ~8.2 / 10** (условно: **B/C** для политик и edge).
