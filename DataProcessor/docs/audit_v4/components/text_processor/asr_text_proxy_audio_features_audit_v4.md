# Audit v4 — `asr_text_proxy_audio_features` (TextProcessor)

**Дата отчёта:** 2026-04-14 (дополнение L2 + L1)  
**Уровень отчёта (план §3.1):** **L1 — draft** (исторический набор **A**) + попытка **L2** (A+B) — **заблокирована** на текущем `result_store` (см. раздел **3** ниже).  
**Артефакт (канон L1):** `storage/result_store/youtube/-Q6fnPIybEI/4c3bf25b-e300-47b3-915e-4699c72ab190/text_processor/text_features.npz`  
**Артефакт (воспроизводимый A для L2-скрипта):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/text_processor/text_features.npz`  
**Статистика L2 (инструмент):** `storage/audit_v4/asr_text_proxy_audio_features_l2/asr_text_proxy_audio_features_audit_v4_stats.json`  
**Срез компонента:** все ключи с префиксом `tp_asrproxy_` в `feature_names` / `feature_values` (37 полей, логический контракт [`asr_text_proxy_audio_features_output_v1`](../../../../TextProcessor/schemas/asr_text_proxy_audio_features_output_v1.json)). Отдельного NPZ у экстрактора нет — см. [`SCHEMA.md`](../../../../TextProcessor/src/extractors/asr_text_proxy_audio_features/SCHEMA.md).  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт L2: `TextProcessor/src/extractors/asr_text_proxy_audio_features/scripts/audit_v4_npz_stats.py` (`--seed 0`)  
**Engineering log 4.2:** [`../audit_4_2/text_processor/asr_text_proxy_audio_features_engineering_log_v4_2.md`](../audit_4_2/text_processor/asr_text_proxy_audio_features_engineering_log_v4_2.md)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано · **◐** частично / только **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика + семантика + вердикт | ✓ | Этот документ |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты | ◐ | Machine JSON + SCHEMA; upstream ASR — [`asr_extractor` L1](../../audio_processor/asr_extractor_audit_v4.md) |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, фичи в e2e | ✓ | Слияние в `text_npz_v1` |
| Путь под `run_id` | ✓ | `text_processor/text_features.npz` |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | L1: исторический run; L2-скрипт: воспроизводимый `e2bc964f-…` |
| **B** | ✗ | Пять путей в скрипте; **2/5** `text_processor` ok, **3/5** error — см. раздел **3** ниже |
| **C** | ✗ | Пустой ASR, `require_asr_text`, token-only / decode fail |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |
| **L2** (A+B) | ✗ | Скрипт + JSON; полный L2 **blocked**, пока нет **5** успешных `text_processor` на B |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Список ключей, сверка со схемой | ✓ | 37 imён; `allow_extra_keys: false` — совпадение множеств |
| dtype | ✓ | `float32` в `feature_values` |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Доля NaN на **A** при `tp_asrproxy_present=1` | ✓ | **0** NaN среди `tp_asrproxy_*` на этом файле |

#### §4.3 — Распределения

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Перцентили | ◐ | L1: один ряд; L2: `aggregate` / `aggregate_ok_subset` в JSON (**2** OK строки) |

#### §4.7 — Трактовка

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Наблюдения → выводы | ✓ | §6–7 |

#### §4.8 — Golden на **A**

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Сигнатура | ✗ | TODO |

#### §4.10 — Empty (**C**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `present=0`, NaN в метриках | ✗ | На **A** транскрипт непустой |

#### §4.12 — Anti-leakage

| Вопрос | Ответ |
|--------|--------|
| Корпусные нормы? | Нет |
| Онлайн API в этом шаге? | Нет (CPU; token path тянет локальный tokenizer при token-only ASR) |

#### §5.3 — Сверка с Models

| Вопрос | Ответ | Комментарий |
|--------|-------|-------------|
| В Baseline v1.0 tabular? | **Нет** / **частично** | Прямых `tp_asrproxy_*` в [`BASELINE_MODEL.md`](../../../../../Models/docs/contracts/BASELINE_MODEL.md) нет; прокси по смыслу близки к речевому табличному слою |
| Зависимость от ASR NPZ | **Косвенная** | Источник текста — `doc.asr` (инжест из AudioProcessor), не чтение `asr_extractor` NPZ этим экстрактором |

#### §6–§8 (кратко)

Verdict + шкала — ниже; L3 DoD / скрипт со seed — **✗**.

---

## 1. Мета артефакта `text_features.npz`

| Поле | Значение |
|------|----------|
| `meta.schema_version` | `text_npz_v1` |
| `meta.producer` | `text_processor` |
| `meta.producer_version` | `1.3.0` |
| `meta.models_used` | **`[]`** (на уровне агрегатора; у ASR-proxy нет записи модели — ожидаемо для чисто эвристического CPU-шага) |
| Размер `feature_values` | **752** скаляра (в т.ч. 37 `tp_asrproxy_*`) |

Реализация экстрактора: **1.2.0** ([`main.py`](../../../../TextProcessor/src/extractors/asr_text_proxy_audio_features/main.py)).

---

## 2. Наблюдения на наборе **A**

| Величина | Значение |
|----------|----------|
| `tp_asrproxy_present` | **1** |
| `tp_asrproxy_audio_duration_sec` | **198** |
| `tp_asrproxy_duration_from_payload_flag` | **0** (длительность с документа) |
| `tp_asrproxy_segments_count` | **7** |
| `tp_asrproxy_text_chars` | **331** |
| `tp_asrproxy_word_count` | **55** |
| `tp_asrproxy_speech_rate_wpm` | **16.67** (согласуется с \(55 / (198/60)\)) |
| `tp_asrproxy_speech_rate_wpm_ratio_to_baseline` | **0.104** (baseline 160 WPM в конфиге) |
| `tp_asrproxy_has_confidence` | **1**; `tp_asrproxy_confidence_present_rate` **1** |
| `tp_asrproxy_confidence_mean` | **~0.903** |
| Флаги ошибок / усечения | все **0** |

**Сверка ключей со схемой:** множество имён из `asr_text_proxy_audio_features_output_v1.json` **совпадает** с множеством `tp_asrproxy_*` в NPZ; лишних/пропавших ключей нет.

---

## 3. Попытка L2 (A+B) и блокировка на `result_store`

Скрипт `TextProcessor/src/extractors/asr_text_proxy_audio_features/scripts/audit_v4_npz_stats.py` (выход: `storage/audit_v4/asr_text_proxy_audio_features_l2/asr_text_proxy_audio_features_audit_v4_stats.json`) сканирует **пять** путей `text_processor/text_features.npz` (тот же набор `video_id`/`run_id`, что у Visual L2).

**Факт по `storage/result_store/youtube` (2026-04-14):** только **два** файла с `meta.status=ok`; на **трёх** остальных `meta.status=error`, **`feature_names` пустой** — среза `tp_asrproxy_*` нет. Типичная причина в `meta.error`: сбой **TitleEmbedder** (**CUDA OOM** / `intfloat/multilingual-e5-large`) до выполнения downstream-экстракторов.

На **двух** OK-файлах: **37** ключей `tp_asrproxy_*`, **0** NaN внутри среза. Полный L2 (корреляции по пяти run, heatmap) **невозможен** до перепрогона `text_processor` на проблемных run. В JSON см. `dataset_quality`, `aggregate_ok_subset` (`n_rows=2`).

---

## 4. HTML / `_render`

Отчёт пайплайна: `text_processor/_render/asr_text_proxy_audio_features_report.html` (тот же run).

---

## 5. Сверка с кодом

1. Стабильный шаблон `_stable_template()` задаёт все ключи до ветвлений; пустой транскрипт даёт `present=0` и NaN в метриках (на **A** не проявилось).
2. Шумовой прокси: среднее по доступным из `rare_ratio` и `low_conf_rate` с cap 1.0; на **A** `noise_proxy_present=1`, значения умеренные.
3. Ритм и интонация включаются флагами `enable_*` — на файле все включены (1.0).

---

## 6. Вердикт

**Плюсы**

- Жёсткий **machine schema** с `allow_extra_keys: false` и фактическое совпадение с NPZ на **A**.
- Явные **маски** (`present`, `has_confidence`, флаги duration/token/degrade) — удобно для потребителей.
- Семантика «**не WER, не акустика**» зафиксирована в SCHEMA.

**Минусы / внимание**

- Потребители должны резать **`feature_names` по префиксу** или знать про merge в общий `text_features.npz`.
- Пустой **`meta.models_used`** на уровне `text_processor` не отражает опциональный **shared_tokenizer** на token-only пути — при аудите token-only стоит явно документировать вызов.
- Нужны **B/C** для пустого текста, `token_decode_failed`, короткого/длинного контента.
- Эмпирический **L2** на пяти run сейчас упирается не в сам экстрактор, а в **успешность всего `text_processor`** (на части mock-run — OOM эмбеддера до заполнения `text_features.npz`).

---

## 7. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Согласованность схема ↔ NPZ | **9** | Полное совпадение множества ключей на **A** |
| Документированность / маски | **8.5** | SCHEMA + флаги |
| Готовность к downstream без сюрпризов | **8** | Merge layout; явный prefix |
| Покрытие edge-case эмпирикой | **6** | Только непустой transcript на **A** |

**Итог L1: ~8.2 / 10** (округляя **8/10** до закрытия **C** и golden §4.8).
