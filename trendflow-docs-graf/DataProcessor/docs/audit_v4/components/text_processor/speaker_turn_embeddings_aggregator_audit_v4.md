# Audit v4 — `speaker_turn_embeddings_aggregator` (TextProcessor)

**Дата отчёта:** 2026-04-14 (дополнение L2 + L1)  
**Уровень отчёта (план §3.1):** **L1 — draft** (исторический набор **A**) + попытка **L2** (A+B) — **заблокирована** на текущем `result_store`.  
**Артефакт (канон L1):** `storage/result_store/youtube/-Q6fnPIybEI/4c3bf25b-e300-47b3-915e-4699c72ab190/text_processor/text_features.npz`  
**Артефакт (воспроизводимый A для L2-скрипта):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/text_processor/text_features.npz`  
**Срез компонента:** **17** ключей `tp_spkemb_*` — [`speaker_turn_embeddings_aggregator_output_v1`](../../../../TextProcessor/schemas/speaker_turn_embeddings_aggregator_output_v1.json). Per-speaker вектора — **`speaker_{spkXXX}_mean.npy` / `_max.npy`** в `_artifacts/` (при успешном прогоне).  
**Статистика L2 (инструмент):** `storage/audit_v4/speaker_turn_embeddings_aggregator_l2/speaker_turn_embeddings_aggregator_audit_v4_stats.json`  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт L2: `TextProcessor/src/extractors/speaker_turn_embeddings_aggregator/scripts/audit_v4_npz_stats.py` (`--seed 0`)  
**Engineering log 4.2:** [`../audit_4_2/text_processor/speaker_turn_embeddings_aggregator_engineering_log_v4_2.md`](../audit_4_2/text_processor/speaker_turn_embeddings_aggregator_engineering_log_v4_2.md)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано · **◐** частично / только **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика + семантика + вердикт | ✓ | Этот документ |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты | ✓ | Machine JSON + [`SCHEMA.md`](../../../../TextProcessor/src/extractors/speaker_turn_embeddings_aggregator/SCHEMA.md) |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, e2e | ✓ | `text_npz_v1` |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | **Вход отсутствует:** нет **`speaker_diarization.speaker_segments` + ASR с `start_sec`/`end_sec`** и нет **`doc.speakers`** → **`tp_spkemb_present=0`** |
| **B** | ✗ | Diarization + таймкодированный ASR, **`emit_extra_metrics=true`** |
| **C** | ✗ | Legacy **`doc.speakers`**, **`require_input=true`** при пустоте |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи ↔ схема | ✓ | **17** имён, `allow_extra_keys: false` |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ожидаемые NaN на **A** | ✓ | **`emit_extra_metrics=false`** → **`tp_spkemb_batch_size`**, **`max_speakers`**, **`max_turns_per_speaker`**, **`min_chars_per_turn`**, **`max_chars_per_turn`** — **NaN** ([`_apply_extra_metrics_spkemb`](../../../../TextProcessor/src/extractors/speaker_turn_embeddings_aggregator/main.py)) |

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
| Модель | Sentence-transformers через **`get_model_with_meta`** — на **A** **`_encode_texts`** **не вызывался** (нет групп спикеров) |

#### §6–§8 (кратко)

L3 DoD — **✗**.

---

## 1. Мета

Реализация: **1.3.0**.

---

## 2.1. L2: `result_store` и блокировка

Скрипт `TextProcessor/src/extractors/speaker_turn_embeddings_aggregator/scripts/audit_v4_npz_stats.py` (выход: `storage/audit_v4/speaker_turn_embeddings_aggregator_l2/speaker_turn_embeddings_aggregator_audit_v4_stats.json`) берёт 5 путей A+B и проверяет:

- табличный срез `tp_spkemb_*` (**17** ключей),
- per-speaker артефакты `text_processor/_artifacts/speaker_spkXXX_{mean,max}.npy` (ожидаются только если `tp_spkemb_present=1` и `write_artifacts=1`).

**Факт по `storage/result_store/youtube` (2026-04-14):** из 5 путей **2** файла `meta.status=ok` и содержат полный `tp_spkemb_*`. На этих run `tp_spkemb_present=0` (валидный пустой исход), поэтому `speaker_spk*.npy` **ожидаемо отсутствуют**.

Ещё **3** пути имеют `meta.status=error` и не содержат табличного слоя (пустой `feature_names`).

Причина блокировки — сбой всего `text_processor` на части mock-run (часто OOM в эмбеддерах до выполнения downstream шагов), а также отсутствие «счастливого» входа diar+ASR/legacy speakers на текущих A/B.

---

## 2. Наблюдения на наборе **A**

### 2.1 Вход и флаги

| Поле | Значение |
|------|----------|
| **`tp_spkemb_present`** | **0** |
| **`tp_spkemb_input_present`** | **0** |
| **`tp_spkemb_input_mode_diar_asr`** / **`legacy_doc_speakers`** | **0** |
| **`tp_spkemb_asr_present`**, **`tp_spkemb_diar_present`** | **0** |
| Счётчики | **`speakers_total`/`embedded`/`turns_total`** — **0** |

Условие diar+ASR в коде жёсткое: оба списка должны быть **`list`**, и выравнивание ASR к diar требует **`start_sec`/`end_sec`** у ASR-сегментов; иначе сегменты пропускаются и может не остаться текстов по спикерам. Режим **legacy** требует непустой **`doc.speakers`**. На reference **A** ни один путь не активировался.

### 2.2 Конфигурация (видимая в NPZ)

**`tp_spkemb_write_artifacts` 1**, **`compute_mean`/`compute_max` 1**, **`require_input` 0** — пустой вход **не** даёт исключение.

### 2.3 Артефакты

Файлов **`speaker_spk*_*.npy`** в `…/text_processor/_artifacts/` **нет** — согласовано с **`present=0`**.

### 2.4 HTML

`text_processor/_render/speaker_turn_embeddings_aggregator_report.html`.

---

## 3. Вердикт

**Плюсы**

- Полное совпадение **17** ключей со схемой; «тихий» пустой исход при **`require_input=false`** предсказуем.

**Минусы / внимание**

- На reference **A** **не** проверяются эмбеддинги и запись **`.npy`** — для L1 по смыслу компонента нужен run с **diarization + временными ASR-сегментами** или **legacy `doc.speakers`**.
- Наличие «обычного» ASR в пайплайне **не** делает **`tp_spkemb_asr_present=1`**, пока не выполнен режим diar+ASR ([`_group_speakers`](../../../../TextProcessor/src/extractors/speaker_turn_embeddings_aggregator/main.py)).

---

## 4. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Схема ↔ NPZ ↔ код | **9** | Множества имён совпали |
| Полнота эмпирики на **A** | **5** | Только пустой путь |
| Документированность ветвлений | **8** | README описывает оба режима |
| Готовность к модели / продукту | **8** | **`tp_spkemb_present`** однозначен |

**Итог L1: ~7.7 / 10** (условно: **B** для «счастливого» пути).
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
