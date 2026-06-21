# Audit v4 — `transcript_aggregator` (TextProcessor)

**Дата:** 2026-04-15  
**Уровень отчёта (план §3.1):** **L2** (целевые наборы **A+B**, **5** путей из `result_store`; фактически данные компонента только на subset из-за ошибок `text_processor`, см. `dataset_quality`).  
**L2 stats (JSON):** `storage/audit_v4/transcript_aggregator_l2/transcript_aggregator_audit_v4_stats.json`  
**Engineering log 4.2:** `DataProcessor/docs/audit_v4/components/audit_4_2/text_processor/transcript_aggregator_engineering_log_v4_2.md`  
**Артефакт (табличный срез):** `…/text_processor/text_features.npz`  
**Срез компонента:** **19** ключей `tp_tragg_*` — [`transcript_aggregator_output_v1`](../../../../TextProcessor/schemas/transcript_aggregator_output_v1.json); **`allow_extra_keys: false`**. Чанки: **`transcript_whisper_chunk_embeddings.npy`** (и при наличии — **`youtube_auto`**); агрегаты: **`transcript_{source}_agg_{mean,max}.npy`**, **`transcript_combined_agg_{mean,max}.npy`** ([`main.py`](../../../../TextProcessor/src/extractors/transcript_aggregator/main.py)).  
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
| Ссылки на контракты | ✓ | Machine JSON + [`SCHEMA.md`](../../../../TextProcessor/src/extractors/transcript_aggregator/SCHEMA.md) |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, e2e | ✓ | `text_npz_v1` + `_artifacts/*.npy` |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | **Whisper** чанки есть → **`present_whisper=1`**, **`present_combined=1`**; **youtube_auto** на **A** нет → **`present_youtube=0`**; **`emit_extra_metrics=false`** → **9** extra-полей **NaN**; **`compute_std=0`** → even with extra on, std slots **NaN** |
| **B** | ◐ | В L2 набор **B** формально включён (4 доп. видео), но `text_processor` падает на **3/5** путях → компонентный срез есть только на subset (см. §2.0 и JSON `dataset_quality`). Целевые вариации B: оба источника, **`emit_extra_metrics=true`**, **`compute_std=true`**, **`require_chunks`** |
| **C** | ✗ | Нет **`tp_artifacts`**, нет файла чанков, **`compute_mean/max`**, **`write_artifacts=false`** |


### 2.0. L2: `result_store` и блокировка

L2 запущен на стандартных **5** путях A+B (см. `paths` в JSON). Результат:

- на **2/5** путях `meta.status=ok` и присутствует полный `tp_tragg_*` (**19** ключей); ожидаемые агрегаты по флагам записаны (**(1024,)**), \(L2\approx1\);
- на **3/5** путях `meta.status=error`, `feature_names` пустой → компонентный срез отсутствует; причина — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

Итог: **L2 blocked** для `transcript_aggregator` (нужны **5/5** OK `text_processor`).

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи ↔ схема | ✓ | **19** имён; имена **`youtube_auto`** в хвосте ключа (`tp_tragg_youtube_auto_*`) |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ожидаемые NaN на **A** | ✓ | **`tp_tragg_*_n_chunks`**, **`tp_tragg_*_mean_std`**, **`tp_tragg_*_max_std`** — **NaN** при **`emit_extra_metrics=false`** ([`main.py`](../../../../TextProcessor/src/extractors/transcript_aggregator/main.py) ~124–127) |

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
| Модель | **Metadata-only** resolve **`model_name`** через **`dp_models`** (без forward в агрегаторе) |
| Вычисления | PyTorch: L2-norm, decay weights, mean/max по чанкам |

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
| **`tp_tragg_present`** | **1** |
| **`tp_tragg_present_whisper` / `present_youtube` / `present_combined`** | **1** / **0** / **1** |
| Политика вычислений | **`compute_mean`/`max`/`combined`** **1**; **`compute_std`** **0**; **`write_artifacts`** **1** |
| **`tp_tragg_decay_rate`** | **≈0.01** |
| Extra block | все **NaN** (**`emit_extra_metrics=false`**) |

### 2.2 Артефакты на **A**

В `…/text_processor/_artifacts/`:

- **`transcript_whisper_chunk_embeddings.npy`** (upstream чанки),
- **`transcript_whisper_agg_mean.npy`**, **`transcript_whisper_agg_max.npy`**,
- **`transcript_combined_agg_mean.npy`**, **`transcript_combined_agg_max.npy`**

(**youtube_auto**-файлы отсутствуют, согласовано с **`present_youtube=0`**.)

### 2.3 Реестр

Обновляются **`doc.tp_artifacts.transcript_aggregates`** и **`transcripts[...].agg_*_relpath`** ([`main.py`](../../../../TextProcessor/src/extractors/transcript_aggregator/main.py) ~384–401).

### 2.4 HTML

`text_processor/_render/transcript_aggregator_report.html`.

---

## 3. Вердикт

**Плюсы**

- Все **19** полей согласованы с NPZ; источники (**whisper** / **youtube** / **combined**) отражены отдельными **`present_*`**.
- Фиксированные имена **`.npy`** и регистрация **relpath** для downstream (**`embedding_source_id`**, cosine и т.д.).

**Минусы / внимание**

- Без **`emit_extra_metrics=true`** в таблице нет **`n_chunks`** по источникам — для L2 полезен прогон с флагом.
- **`tp_tragg_present_youtube`** относится к **youtube_auto**, не к «любому» YouTube-тексту — имя уже закреплено в схеме.

---

## 4. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Схема ↔ NPZ ↔ код | **9** | В т.ч. блок extra / NaN |
| Полнота эмпирики на **A** | **8** | Whisper + combined, без youtube_auto; extra выключен |
| Документированность ветвлений | **8** | README, canonical vs legacy путей чанков |
| Готовность к модели / продукту | **8** | Mean/max/combined, decay |

**Итог L1: ~8.2 / 10** (условно: **B/C**, **§4.8**, **`emit_extra_metrics` + `compute_std`**).
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
