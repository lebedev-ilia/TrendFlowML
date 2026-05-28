# Audit v4 — `embedding_stats_extractor` (TextProcessor)

**Дата отчёта:** 2026-04-14 (дополнение L2 + L1)  
**Уровень отчёта (план §3.1):** **L1 — draft** (исторический набор **A**) + попытка **L2** (A+B) — **заблокирована** на текущем `result_store`.  
**Артефакт (канон L1):** `storage/result_store/youtube/-Q6fnPIybEI/4c3bf25b-e300-47b3-915e-4699c72ab190/text_processor/text_features.npz`  
**Артефакт (воспроизводимый A для L2-скрипта):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/text_processor/text_features.npz`  
**Срез компонента:** **39** ключей `tp_embstats_*` — [`embedding_stats_extractor_output_v1`](../../../../TextProcessor/schemas/embedding_stats_extractor_output_v1.json). Матрица чанков — `doc.tp_artifacts` → `.npy` (`transcript_chunk_embedder`).  
**Статистика L2 (инструмент):** `storage/audit_v4/embedding_stats_extractor_l2/embedding_stats_extractor_audit_v4_stats.json`  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт L2: `TextProcessor/src/extractors/embedding_stats_extractor/scripts/audit_v4_npz_stats.py` (`--seed 0`)  
**Engineering log 4.2:** [`../audit_4_2/text_processor/embedding_stats_extractor_engineering_log_v4_2.md`](../audit_4_2/text_processor/embedding_stats_extractor_engineering_log_v4_2.md)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано · **◐** частично / только **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика + семантика + вердикт | ✓ | Этот документ |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты | ✓ | Machine JSON + [`SCHEMA.md`](../../../../TextProcessor/src/extractors/embedding_stats_extractor/SCHEMA.md) |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, e2e | ✓ | `text_npz_v1` |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | **Мало чанков:** upstream **`transcript_chunk_embedder`**: **1** whisper-чанк; **`min_chunks_required=2`** → блок дисперсии не выполняется |
| **B** | ✗ | ≥2 чанка, заполненные `topvar_*` / `l2_variance`, `emit_extra_metrics=true` |
| **C** | ✗ | Нет матрицы чанков, `require_chunks=true`, topic missing / invalid |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи ↔ схема | ✓ | **39** имён, `allow_extra_keys: false` |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ожидаемые NaN на **A** | ✓ | При **`n_chunks < min_chunks_required`**: **`tp_embstats_l2_variance`**, **`tp_embstats_topvar_1…8`**, **`tp_embstats_n_chunks`**, **`tp_embstats_dim`** остаются **NaN** (ветка [`extract`](../../../../TextProcessor/src/extractors/embedding_stats_extractor/main.py) не присваивает `n_chunks`/`dim` до успешного пути дисперсии); **`tp_embstats_present=0`** |
| Тайминги | ✓ | **`emit_extra_metrics=false`** → **`tp_embstats_load_ms`**, **`tp_embstats_compute_ms`** — **NaN** (принудительно обнуляются после измерения) |

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
| Обучаемая модель в шаге | **Нет** — дисперсия по готовым чанк-эмбеддингам + энтропия по upstream `topic_probs` |

#### §6–§8 (кратко)

L3 DoD — **✗**.

---

## 1. Мета

Реализация: **1.2.0** ([`main.py`](../../../../TextProcessor/src/extractors/embedding_stats_extractor/main.py)).

---

## 2.1. L2: `result_store` и блокировка

Скрипт `TextProcessor/src/extractors/embedding_stats_extractor/scripts/audit_v4_npz_stats.py` (выход: `storage/audit_v4/embedding_stats_extractor_l2/embedding_stats_extractor_audit_v4_stats.json`) берёт 5 путей A+B и выделяет `tp_embstats_*`.

**Факт по `storage/result_store/youtube` (2026-04-14):** из 5 путей **2** файла `meta.status=ok` и содержат полный срез (**39** ключей); **3** файла `meta.status=error` и не содержат табличного слоя (пустой `feature_names`).

Причина блокировки — сбой всего `text_processor` на части mock-run (часто OOM в эмбеддерах до выполнения downstream шагов), а не логика `embedding_stats_extractor`.

---

## 2. Наблюдения на наборе **A**

### 2.1 Путь данных

| Поле | Значение |
|------|----------|
| `tp_embstats_enabled` | **1** |
| `tp_embstats_disabled_by_policy` | **0** |
| `tp_embstats_source_used_whisper` | **1** |
| `tp_embstats_source_used_youtube_auto` | **0** |
| `tp_embstats_used_legacy_key_flag` | **0** |
| `tp_embstats_unsafe_relpath_flag`, `tp_embstats_dim_mismatch_flag`, `tp_embstats_nan_inf_flag` | **0** |

Перекрёстно: `text_processor/_render/render_context.json` → **`transcript_chunk_embedder.total_chunks` = 1**, **`whisper_chunks` = 1**.

### 2.2 Дисперсия по чанкам

| Поле | Значение |
|------|----------|
| `tp_embstats_min_chunks_required` | **2** |
| `tp_embstats_require_chunks_enabled` | **0** |
| `tp_embstats_present` | **0** — **`l2_variance`** не вычислялась |
| `tp_embstats_n_chunks`, `tp_embstats_dim`, `tp_embstats_l2_variance`, `tp_embstats_topvar_1…8` | **NaN** (согласовано с отсутствием успешного блока дисперсии) |

### 2.3 Topic entropy

`tp_embstats_compute_topic_entropy_enabled` = **1** на артефакте; **`tp_embstats_topic_entropy_present` = 1**, **`tp_embstats_topic_probs_present` = 1**, **`topic_probs_invalid_flag` = 0**; **`tp_embstats_topic_entropy`**, **`_norm`**, **`_perplexity`** — конечны (отдельный путь от дисперсии).

### 2.4 HTML

`text_processor/_render/embedding_stats_extractor_report.html`.

---

## 3. Вердикт

**Плюсы**

- Полное совпадение **39** ключей с machine schema; на **A** видны **оба** пути: «нет дисперсии» и «есть topic entropy».
- Источник транскрипта (**whisper**) отражён флагами `source_used_*`.

**Минусы / внимание**

- При **`n_chunks < min_chunks_required`** **`n_chunks`/`dim`** остаются **NaN** (меньше телеметрии, чем у [`embedding_shift_indicator_extractor`](embedding_shift_indicator_extractor_audit_v4.md), где счётчик заполняется до early exit).
- Для L1 по дисперсии/topvar нужен run с **≥ `min_chunks_required`** чанками и по желанию **`emit_extra_metrics=true`**.

---

## 4. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Схема ↔ NPZ ↔ код | **9** | Множества имён совпали |
| Полнота эмпирики на **A** | **6** | Дисперсия не считалась; topic-path только частично компенсирует |
| Документированность ветвлений | **8** | README описывает `present`; код не заполняет `n_chunks` при коротком матриксе |
| Готовность к модели / продукту | **8** | Потребители: опираться на **`tp_embstats_present`** + явная обработка NaN |

**Итог L1: ~7.9 / 10** (условно: контракт **9/10**, «счастливый» путь дисперсии на **A** не показан — добирать **B**).
