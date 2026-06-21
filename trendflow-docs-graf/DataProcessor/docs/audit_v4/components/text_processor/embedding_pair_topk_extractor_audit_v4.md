# Audit v4 — `embedding_pair_topk_extractor` (TextProcessor)

**Дата отчёта:** 2026-04-14 (дополнение L2 + L1)  
**Уровень отчёта (план §3.1):** **L1 — draft** (исторический набор **A**) + попытка **L2** (A+B) — **заблокирована** на текущем `result_store`.  
**Артефакт (канон L1):** `storage/result_store/youtube/-Q6fnPIybEI/4c3bf25b-e300-47b3-915e-4699c72ab190/text_processor/text_features.npz`  
**Артефакт (воспроизводимый A для L2-скрипта):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/text_processor/text_features.npz`  
**Срез компонента:** **69** ключей (`tp_embpair_*`, legacy `tp_pairtopk_*`) — [`embedding_pair_topk_extractor_output_v1`](../../../../TextProcessor/schemas/embedding_pair_topk_extractor_output_v1.json). Вектора читаются из `doc.tp_artifacts` (title/description/transcript chunk matrix); отдельного «своего» NPZ нет.  
**Статистика L2 (инструмент):** `storage/audit_v4/embedding_pair_topk_extractor_l2/embedding_pair_topk_extractor_audit_v4_stats.json`  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт L2: `TextProcessor/src/extractors/embedding_pair_topk_extractor/scripts/audit_v4_npz_stats.py` (`--seed 0`)  
**Engineering log 4.2:** [`../audit_4_2/text_processor/embedding_pair_topk_extractor_engineering_log_v4_2.md`](../audit_4_2/text_processor/embedding_pair_topk_extractor_engineering_log_v4_2.md)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано · **◐** частично / только **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика + семантика + вердикт | ✓ | Этот документ |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты | ✓ | Machine JSON + [`SCHEMA.md`](../../../../TextProcessor/src/extractors/embedding_pair_topk_extractor/SCHEMA.md) |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, e2e | ✓ | `text_npz_v1` |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | Один чанк транскрипта, **`top_k_slots=5`**, **`top_k=10`** |
| **B** | ✗ | M чанков ≥ `top_k_slots`, FAISS/auto |
| **C** | ✗ | Пустые эмбеддинги, dim mismatch |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи ↔ схема | ✓ | **69** имён, `allow_extra_keys: false` |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ожидаемые NaN | ✓ | **`emit_extra_metrics=False`** → **NaN** в `tp_embpair_n_chunks`, one-hot источника транскрипта, `tp_embpair_use_faiss_mode`, `tp_embpair_require_faiss` ([`_apply_extra_metrics_block`](../../../../TextProcessor/src/extractors/embedding_pair_topk_extractor/main.py)) |
| Слоты top2–top8 / индексы | ✓ | При **одном** чанке и `export_topk_slots` до **5**: только **top1** конечен; слоты **2–5** (и **6–8**) остаются **NaN** по коду padding |

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
| Обучаемая модель в шаге | **Нет** (cosine / top-k по готовым эмбеддингам) |

#### §6–§8 (кратко)

L3 DoD — **✗**.

---

## 1. Мета

Реализация: **1.3.0** ([`main.py`](../../../../TextProcessor/src/extractors/embedding_pair_topk_extractor/main.py)). В схеме зафиксировано до **8** export-слотов; `top_k_slots` в конфиге на **A** = **5**.

---

## 2.1. L2: `result_store` и блокировка

Скрипт `TextProcessor/src/extractors/embedding_pair_topk_extractor/scripts/audit_v4_npz_stats.py` (выход: `storage/audit_v4/embedding_pair_topk_extractor_l2/embedding_pair_topk_extractor_audit_v4_stats.json`) берёт 5 путей A+B и выделяет `tp_embpair_*` + legacy `tp_pairtopk_*`.

**Факт по `storage/result_store/youtube` (2026-04-14):** из 5 путей **2** файла `meta.status=ok` и содержат полный срез (**56** canon + **13** legacy = **69** ключей); **3** файла `meta.status=error` и не содержат табличного слоя (пустой `feature_names`).

Причина блокировки — сбой всего `text_processor` на части mock-run (часто OOM в эмбеддерах до выполнения downstream шагов), а не логика `embedding_pair_topk_extractor`.

---

## 2. Наблюдения на наборе **A**

### 2.1 Флаги и конфиг

| Поле | Значение |
|------|----------|
| `tp_embpair_present` | **1** |
| `tp_embpair_title_present` / `desc` / `transcript_chunks` | **1** / **1** / **1** |
| `tp_embpair_used_legacy_key_flag` | **0** (канонический `transcripts[source].chunk_embeddings_relpath`) |
| Ошибки | `dim_mismatch`, `unsafe_relpath`, `nan_inf`, `zero_norm` — **0** |
| `tp_embpair_title_desc_cosine` | **0.8555** (совпадает с **`tp_cos_title_desc`** в [`cosine_metrics_extractor_audit_v4.md`](cosine_metrics_extractor_audit_v4.md) на том же run) |
| `tp_embpair_top_k` | **10** |
| `tp_embpair_top_k_slots` / `requested` | **5** / **5**; **`top_k_slots_clamped`** **0** |
| `tp_embpair_compute_title_desc` / `title_transcript_topk` | **1** / **1** |
| Export | `export_topk_slots` / `indices` / `summary` — **1** |

### 2.2 Top-k по title vs чанки

| Слот | cosine | idx |
|------|--------|-----|
| top1 | **0.8531** | **0** |
| top2 … top8 | **NaN** | **NaN** |

`tp_embpair_title_transcript_topk_mean` / `max` = **0.8531** (один конечный score в summary).

Legacy `tp_pairtopk_*` зеркалит канонические значения на заполненных слотах.

### 2.3 HTML

`text_processor/_render/embedding_pair_topk_extractor_report.html`.

---

## 3. Вердикт

**Плюсы**

- Полное совпадение **69** ключей с machine schema.
- Согласованность **`tp_embpair_title_desc_cosine`** с **`cosine_metrics_extractor`** (независимая перекрёстная проверка).
- Явная семантика **слотов**, **legacy** и **extra** блока.

**Минусы / внимание**

- Потребители должны понимать: **NaN в слотах** ≠ ошибка, если чанков меньше, чем слотов / чем число уникальных top scores.
- Без **`emit_extra_metrics`** нет **`tp_embpair_n_chunks`** в NPZ — для мониторинга плотности чанков включить флаг или читать upstream.

---

## 4. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Схема ↔ NPZ ↔ код | **9** | Полное совпадение множеств |
| Документированность NaN | **8.5** | Extra block + padding слотов |
| Перекрёстная согласованность | **9** | title–desc cosine |
| Edge coverage | **6** | Один чанк, без FAISS pressure |

**Итог L1: ~8.4 / 10** (до **B/C**, multi-chunk, §4.8).
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
