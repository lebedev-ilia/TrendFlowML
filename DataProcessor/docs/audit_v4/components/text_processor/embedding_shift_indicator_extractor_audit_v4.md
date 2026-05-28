# Audit v4 — `embedding_shift_indicator_extractor` (TextProcessor)

**Дата отчёта:** 2026-04-14 (дополнение L2 + L1)  
**Уровень отчёта (план §3.1):** **L1 — draft** (исторический набор **A**) + попытка **L2** (A+B) — **заблокирована** на текущем `result_store`.  
**Артефакт (канон L1):** `storage/result_store/youtube/-Q6fnPIybEI/4c3bf25b-e300-47b3-915e-4699c72ab190/text_processor/text_features.npz`  
**Артефакт (воспроизводимый A для L2-скрипта):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/text_processor/text_features.npz`  
**Срез компонента:** **27** ключей `tp_embshift_*` — [`embedding_shift_indicator_extractor_output_v1`](../../../../TextProcessor/schemas/embedding_shift_indicator_extractor_output_v1.json). Матрица чанков — по `doc.tp_artifacts` → `.npy` upstream (`transcript_chunk_embedder`).  
**Статистика L2 (инструмент):** `storage/audit_v4/embedding_shift_indicator_extractor_l2/embedding_shift_indicator_extractor_audit_v4_stats.json`  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт L2: `TextProcessor/src/extractors/embedding_shift_indicator_extractor/scripts/audit_v4_npz_stats.py` (`--seed 0`)  
**Engineering log 4.2:** [`../audit_4_2/text_processor/embedding_shift_indicator_extractor_engineering_log_v4_2.md`](../audit_4_2/text_processor/embedding_shift_indicator_extractor_engineering_log_v4_2.md)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано · **◐** частично / только **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика + семантика + вердикт | ✓ | Этот документ |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты | ✓ | Machine JSON + [`SCHEMA.md`](../../../../TextProcessor/src/extractors/embedding_shift_indicator_extractor/SCHEMA.md) |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, e2e | ✓ | `text_npz_v1` |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | **Деградированный** путь: `n_chunks=1` &lt; `require_min_chunks=2` → метрики сдвига не считаются |
| **B** | ✗ | ≥2 чанка, `compute_extra_cosines=true` |
| **C** | ✗ | Нет файла чанков, `require_transcript_chunks=true` |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи ↔ схема | ✓ | **27** имён, `allow_extra_keys: false` |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ожидаемые NaN на **A** | ✓ | После раннего выхода при **`n_chunks < require_min_chunks`**: **`tp_embshift_cosine_begin_end`**, **`shift_flag`**, **`margin`**, доп. косинусы — **NaN**; **`tp_embshift_present=0`** |
| Тайминги | ✓ | **`emit_extra_metrics=false`** → **`load_ms`**, **`compute_ms`** остаются **NaN** (даже если загрузка была) |

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
| Обучаемая модель в шаге | **Нет** — косинусы по уже посчитанным чанк-эмбеддингам |

#### §6–§8 (кратко)

L3 DoD — **✗**.

---

## 1. Мета

Реализация: **1.3.0** ([`main.py`](../../../../TextProcessor/src/extractors/embedding_shift_indicator_extractor/main.py)).

---

## 2.1. L2: `result_store` и блокировка

Скрипт `TextProcessor/src/extractors/embedding_shift_indicator_extractor/scripts/audit_v4_npz_stats.py` (выход: `storage/audit_v4/embedding_shift_indicator_extractor_l2/embedding_shift_indicator_extractor_audit_v4_stats.json`) берёт 5 путей A+B и выделяет `tp_embshift_*`.

**Факт по `storage/result_store/youtube` (2026-04-14):** из 5 путей **2** файла `meta.status=ok` и содержат полный срез (**27** ключей); **3** файла `meta.status=error` и не содержат табличного слоя (пустой `feature_names`).

Причина блокировки — сбой всего `text_processor` на части mock-run (часто OOM в эмбеддерах до выполнения downstream шагов), а не логика `embedding_shift_indicator_extractor`.

---

## 2. Наблюдения на наборе **A**

### 2.1 Путь данных

| Поле | Значение |
|------|----------|
| `tp_embshift_enabled` | **1** |
| `tp_embshift_disabled_by_policy` | **0** |
| `tp_embshift_source_used_whisper` | **1** |
| `tp_embshift_source_used_youtube_auto` | **0** |
| `tp_embshift_used_legacy_key_flag` | **0** |
| Флаги ошибок загрузки | `unsafe_relpath`, `chunk_embed_missing`, `dim_mismatch`, `nan_inf`, `zero_norm` — **0** |

### 2.2 Объём чанков vs порог

| Поле | Значение |
|------|----------|
| `tp_embshift_n_chunks` | **1** |
| `tp_embshift_n_window_chunks` | **1** (после загрузки; `win=min(n_window_chunks, max(1, n_chunks//2))` → при **1** чанке **1**) |
| `tp_embshift_dim` | **1024** |
| `tp_embshift_require_min_chunks` | **2** |
| `tp_embshift_present` | **0** — вычисление основной метрики **не выполнялось** ( [`extract` early return](../../../../TextProcessor/src/extractors/embedding_shift_indicator_extractor/main.py) при `n_chunks < require_min_chunks` при **`require_transcript_chunks=false`**) |

### 2.3 Метрики сдвига

`tp_embshift_cosine_begin_end`, `tp_embshift_shift_flag`, `tp_embshift_margin`, `tp_embshift_cosine_first_last`, `tp_embshift_mean_cosine_last_to_start_window` — **NaN** на **A** (согласовано с `present=0`). **`compute_extra_cosines_enabled=0`**.

### 2.4 HTML

`text_processor/_render/embedding_shift_indicator_extractor_report.html`.

---

## 3. Вердикт

**Плюсы**

- Полное совпадение **27** ключей с machine schema при типичной **валидации пустоты** (мало чанков).
- Источник (**whisper**) и **`n_chunks`**/`dim` всё равно заполняются до early return — полезно для телеметрии.
- Поведение при **`require_transcript_chunks=false`** предсказуемо (без исключения).

**Минусы / внимание**

- На reference **A** компонент **не демонстрирует** заполненные `cosine_*` / `shift_flag` — для L1 по «семантике сдвига» нужен run с **≥ `require_min_chunks`** чанками.
- Потребители должны трактовать **`tp_embshift_present`** как «метрика сдвига посчитана», а не «файл чанков найден».

---

## 4. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Схема ↔ NPZ ↔ код | **9** | Множества имён совпали |
| Полнота эмпирики на **A** | **6** | Только ранний выход |
| Документированность ветвлений | **8** | Чёткий порог чанков |
| Готовность к модели / продукту | **8** | Нужны runs с 2+ чанками |

**Итог L1: ~8.0 / 10** (условно: контракт **9/10**, но «счастливый» путь на **A** не показан — добирать **B**).
