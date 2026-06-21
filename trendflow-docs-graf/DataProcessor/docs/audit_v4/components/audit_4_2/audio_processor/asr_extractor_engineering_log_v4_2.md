# `asr_extractor` — инженерный журнал и связь с Audit v4 / **4.2**

**Назначение:** зафиксировать **изменения кода и метаданных** компонента относительно **эмпирического отчёта L2 (Audit 4.2)**, не дублируя полные таблицы статистики NPZ. Статистики выхода (наборы **A** и **B**), перцентили, корреляции и вердикт остаются **source-of-truth** в основном отчёте аудита.

**Дата журнала:** 2026-04-06  
**Версия компонента в коде (на момент журнала):** `2.3.2` (`DataProcessor/AudioProcessor/src/extractors/asr_extractor/main.py`)

---

## 1. Канонические документы (читать в этом порядке)

| Документ | Роль |
|----------|------|
| [Отчёт Audit 4.2 L2 — `asr_extractor`](../audio_processor/asr_extractor_audit_v4.md) | Таблицы по **A+B**, критерии плана, вердикт **L2 ~8.3/10**, пробелы до L3 |
| [Критерии и план v4](../../AUDIT_4_CRITERIA_AND_PLAN.md) | §3.1 уровни отчёта, §4.* метрики, §12.2 наблюдаемость оркестратора |
| [Компонентный README — `asr_extractor`](../../../../AudioProcessor/src/extractors/asr_extractor/docs/README.md) | Контракт полей, флаги CLI, оптимизации, env |
| [SCHEMA.md — `asr_extractor`](../../../../AudioProcessor/src/extractors/asr_extractor/docs/SCHEMA.md) | NPZ / `meta`, optional профилирование |
| [Телеметрия оркестратора — AudioProcessor](../../../../AudioProcessor/docs/ORCHESTRATOR_TELEMETRY.md) | `scheduler_runtime_report.json`, batch + single-file |
| [Журнал прогонов v4](../../RUN_LOG.md) | Воспроизводимость путей `result_store` / audit JSON |

**Артефакты L2 (статистика выхода), на которых строился отчёт:**

- JSON агрегатов: `storage/audit_v4/asr_extractor_l2/asr_extractor_audit_v4_stats.json` (`paths`, `aggregate`, `correlation_tabular`, …)
- Графики: `storage/audit_v4/asr_extractor_l2/figures/`
- Скрипт: `DataProcessor/AudioProcessor/src/extractors/asr_extractor/scripts/audit_v4_npz_stats.py` (`--seed 0`)

**Reference run A** в отчёте указывает `meta.producer_version`: **`2.2.0`** — см. §1 отчёта. Последующие версии **не меняют контракт tabular/массивов**, добавляют **optional** поля в `meta` и поведение рантайма (см. §3).

---

## 2. Сводка статистик из Audit 4.2 (коротко, без копии таблиц)

Ниже — **зафиксированные в отчёте факты**, чтобы журнал был самодостаточным для навигации; детали чисел — только в [основном отчёте](../audio_processor/asr_extractor_audit_v4.md).

- **Набор B (5 прогонов):** `token_total` 6…219, `segments_count` 1–2, языки **ms** и **en**, **0** NaN в tabular на всех пяти файлах.
- **Табличные поля и производные:** перцентили по фичам, корреляции (часть NaN у констант вроде `sample_rate` — ожидаемо), гистограммы в `figures/`.
- **Набор C** (empty / нет речи / короткое аудио): в отчёте **✗** — **TODO L3**.
- **§4.8 Golden на A:** **✗** — нет зафиксированной регрессионной сигнатуры JSON/хэш.
- **§6.1 Wall-time / RAM** в отчёте отмечены как **✗**; с **2.3.x** частично закрывается **внутренним профилем** и **оркестраторской телеметрией** (§3.2–3.3), а не одной только таблицей §4.

---

## 3. Изменения кода после отчёта L2 (хронология)

### 3.1 Версия **2.3.0** — этап 2: профилирование внутри экстрактора

- **`meta` (NPZ):** при успешном прогоне дополнительно (через савер): `asr_stage_timings_ms`, при `AP_ASR_RESOURCE_PROFILE=1` — `asr_resource_profile`.
- **Лог:** строка `ASR | profiling [run_segments|extract_batch_segments]: …`
- **Утилиты:** `src/extractors/asr_extractor/utils/resource_profile.py` (`AP_ASR_RESOURCE_PROFILE`).
- **Связь с аудитом:** даёт измеримые **wall / RSS / GPU** на уровне фаз; полезно для закрытия замечаний §6.1 при следующем обновлении отчёта.

### 3.2 Версия **2.3.1** — оптимизации без изменения качества decode по умолчанию

- Пропуск **`detect_language`** при **фиксированном** `--asr-language` (не `auto`).
- Кеш **`N_FRAMES`** / `n_audio_ctx`, модульный импорт `whisper`, меньше лишних копий numpy→mel, разумный перенос mel на device.
- **Batch:** распределение по файлам через **`indices_by_file`** (O(сегменты)).
- **Влияние на статистики §4:** **нет** (те же алгоритмы decode и tokenizer).

### 3.3 Версия **2.3.2** — память и опциональный «один detect» при `auto`

- **`run_segments`:** потоковая загрузка PCM (**одно окно в RAM** за раз); метаданные окон заранее.
- **`extract_batch_segments`:** нет хранения всех PCM; загрузка аудио **перед каждым инференс-батчем**; в `asr_stage_timings_ms` добавлены `load_meta_only_ms`, `load_audio_lazy_ms`.
- **`AP_ASR_LANG_DETECT_ONCE=1`:** один **`detect_language` на файл** (batch) или на весь **single-file** `run_segments`; последующие окна декодятся с закэшированным `lang_code` — **возможная деградация при смене языка внутри файла**.
- **Пустой сегмент** после ошибки чтения: ранний выход без Whisper (`audio_1d.size == 0`).
- **Влияние на статистики §4 при `AP_ASR_LANG_DETECT_ONCE`:** могут измениться **`lang_code_by_segment` / `lang_conf`** на поздних окнах относительно «полного detect на каждое окно»; tabular агрегаты по языку и downstream, чувствительные к языку, следует сравнивать в **контрольном** прогоне при включении флага.

### 3.4 Оркестратор (вне NPZ `asr_extractor`, но влияет на продуктовый прогон)

- **Телеметрия по экстракторам** в `scheduler_runtime_report.json` (`orchestrator_telemetry`): покрытие **single-file** и **batch** (`MainProcessor.run_batch` / `process_video`). См. `AudioProcessor/docs/ORCHESTRATOR_TELEMETRY.md`.
- **Связь с аудитом:** не входит в `audit_v4_npz_stats.py`, но даёт **сквозные** метрики для §6.1 и SRE.

---

## 4. Что не менялось относительно отчёта 4.2

- Контракт **token IDs → `shared_tokenizer_v1`**, no-fallback на whisper-tokens.
- Набор **основных** ключей NPZ v2 и логика **tabular** (плотность, WPM, quality-агрегаты при включённых флагах).
- Выводы **§5.3** отчёта (роль в Baseline v1.0, token path, analytics) остаются актуальны.

---

## 5. Рекомендации для следующего прогона аудита / L3

1. Зафиксировать в отчёте **`producer_version`** фактического прогона (≥ **2.3.2**).
2. При **включённых** `AP_ASR_LANG_DETECT_ONCE` / `AP_ASR_RESOURCE_PROFILE` — явно указать env в `RUN_LOG` и приложить при необходимости **доп. срез** meta (не смешивать с baseline L2 без пометки).
3. Закрыть **§4.8** (golden/signature на A) и **набор C** — как в отчёте.
4. При желании обновить §6.1: подтянуть **`asr_stage_timings_ms`** и **`orchestrator_telemetry.events`** в единую таблицу «wall / пик RSS / GPU» для типичного e2e.

---
---

## Навигация

[Audit v4 hub](../README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
