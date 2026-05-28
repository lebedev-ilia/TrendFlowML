# Audit v4 — `transcript_chunk_embedder` (TextProcessor)

**Дата:** 2026-04-14  
**Уровень отчёта (план §3.1):** **L2** (целевые наборы **A+B**, **5** путей из `result_store`; фактически данные компонента только на subset из-за ошибок `text_processor`, см. `dataset_quality`).  
**L2 stats (JSON):** `storage/audit_v4/transcript_chunk_embedder_l2/transcript_chunk_embedder_audit_v4_stats.json`  
**Engineering log 4.2:** `DataProcessor/docs/audit_v4/components/audit_4_2/text_processor/transcript_chunk_embedder_engineering_log_v4_2.md`  
**Артефакт (табличный срез):** `…/text_processor/text_features.npz`  
**Срез компонента:** **16** ключей `tp_tchunk_*` — [`transcript_chunk_embedder_output_v1`](../../../../TextProcessor/schemas/transcript_chunk_embedder_output_v1.json); **`allow_extra_keys: false`**. Матрица чанков: **`transcript_{source}_chunk_embeddings.npy`** ([`main.py`](../../../../TextProcessor/src/extractors/transcript_chunk_embedder/main.py)).  
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
| Ссылки на контракты | ✓ | Machine JSON + [`SCHEMA.md`](../../../../TextProcessor/src/extractors/transcript_chunk_embedder/SCHEMA.md) |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, e2e | ✓ | `text_npz_v1` + **`tp_artifacts.transcripts`**. |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | **`present=1`**, **1** источник (**whisper**), **1** чанк, **D=1024**; **youtube_auto** выключен в прогоне (**`present=0`**); **`emit_extra_metrics=false`** → **5× NaN** (batch/overlap/max chunks/cache) |
| **B** | ◐ | В L2 набор **B** формально включён (4 доп. видео), но `text_processor` падает на **3/5** путях → компонентный срез есть только на subset (см. §2.1 и JSON `dataset_quality`). Целевые вариации B: **youtube_auto**, много чанков, **`emit_extra_metrics=true`**, disk cache |
| **C** | ✗ | Нет ASR, **`require_asr`**, пустой транскрипт, **`emit_confidence_metrics=false`** |

### 2.1. L2: `result_store` и блокировка

L2 запущен на стандартных **5** путях A+B (см. `paths` в JSON). Результат:

- на **2/5** путях `meta.status=ok` и присутствует полный `tp_tchunk_*` (**16** ключей), а также `text_processor/_artifacts/transcript_whisper_chunk_embeddings.npy` (shape **(1, 1024)**, float32, значения конечны, \(L2\) нормы строк ~1);
- на **3/5** путях `meta.status=error`, `feature_names` пустой → компонентный срез отсутствует, артефакт не записан; причина — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

Итог: **L2 blocked** для `transcript_chunk_embedder` (нужны **5/5** OK `text_processor`).

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи ↔ схема | ✓ | **16** имён |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ожидаемые NaN на **A** | ✓ | **`tp_tchunk_batch_size`**, **`max_chunk_tokens_model`**, **`overlap_ratio`**, **`max_chunks_total`**, **`cache_enabled`** при **`emit_extra_metrics=false`** ([`main.py`](../../../../TextProcessor/src/extractors/transcript_chunk_embedder/main.py) ~551–562) |
| Confidence | ✓ | На **A** **`conf_present=1`**, mean/min/max **конечны** и совпадают (один чанк) |

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
| Модель | **`get_model`** + **`dp_models`** metadata; на **A** **D=1024** (e5-large-класс в профиле прогона) |
| Токенизатор | Строго **`shared_tokenizer_v1`** |

#### §6–§8 (кратко)

L3 DoD — **✗**.

---

## 1. Мета

Реализация: **1.3.0**.

---

## 2. Наблюдения на наборе **A** (исторические L1-заметки)

### 2.1 Табличный срез

| Поле | Значение |
|------|----------|
| **`tp_tchunk_present`** | **1** |
| Источники | **`sources_count`** **1**; **`whisper_present`** **1**; **`youtube_auto_present`** **0** |
| Чанки | **`whisper_chunks`** **1**; **`youtube_chunks`** **0** |
| Размерность | **`embedding_dim`** **1024** |
| Уверенность | **`conf_present`** **1**; mean/min/max **≈0.903** |
| Extra | **NaN** в пяти полях (см. §4.2) |

### 2.2 Артефакт

**`transcript_whisper_chunk_embeddings.npy`**: форма **(1, 1024)**, **float32**, L2 нормы строк **≈1**.

### 2.3 HTML

`text_processor/_render/transcript_chunk_embedder_report.html`.

---

## 3. Вердикт

**Плюсы**

- Полное совпадение **16** ключей с NPZ; upstream для **`transcript_aggregator`** на **A** согласован (**1** чанк).

**Минусы / внимание**

- Число чанков **1** ограничивает проверку overlap / multi-chunk путей на **A** — нужны **B/C**.
- **`emit_extra_metrics`** скрывает полезные для отладки пороги chunking в таблице.

---

## 4. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Схема ↔ NPZ ↔ код | **9** | Описание NaN в schema |
| Полнота эмпирики на **A** | **8** | Один чанк / один источник |
| Документированность ветвлений | **8** | README + SCHEMA |
| Готовность к модели / продукту | **8** | Confidence + dimension stable |

**Итог L1: ~8.2 / 10** (условно: **B/C**, **§4.8**).
