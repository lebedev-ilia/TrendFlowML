# Audit v4.2 — общий итог L2 по процессорам

**Дата сводки:** 2026-04-15  
**План и критерии:** [AUDIT_4_CRITERIA_AND_PLAN.md](AUDIT_4_CRITERIA_AND_PLAN.md)  
**Журнал прогонов и путей:** [RUN_LOG.md](RUN_LOG.md)  
**Engineering bridge (профилирование / оптимизации после L2):** [components/audit_4_2/README.md](components/audit_4_2/README.md)

## Назначение документа

**Audit v4.2** — слой **после** эмпирического **L2** (наборы **A+B**, типично **5** прогонов из `result_store`): инженерные журналы связывают отчёты с изменениями кода, не подменяя канонические `*_audit_v4.md`. Этот файл фиксирует **сквозной итог L2 + состояние bridge 4.2** по **AudioProcessor**, **VisualProcessor** и **TextProcessor**.

Ниже **L3** и **§8 DoD** нигде не закрыты; статус `passed` в смысле плана не применяется.

**Следующий крупный этап (прогон 60+ видео):** чек-лист подготовки — [PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md](PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md) (оптимизации, курация набора, вывод метрик).

---

## Сводная таблица по подсистемам (L2)

| Подсистема | Компонентов в контуре L2 | Статистика L2 (JSON) | Статус эмпирики на **A+B** | Bridge 4.2 (engineering logs) | Каноническая сводка |
|------------|--------------------------|----------------------|----------------------------|-------------------------------|---------------------|
| **AudioProcessor** | **21** | `storage/audit_v4/<extractor>_l2/*.json` (+ figures у ряда экстракторов) | **Закрыто:** product stats на **5** run; отчёты и `RUN_LOG` выровнены | **Есть** по большинству экстракторов ([`audit_4_2/audio_processor/`](components/audit_4_2/audio_processor/)) | [AUDIT_V4_L1_AUDIO_PROCESSOR_SUMMARY.md](AUDIT_V4_L1_AUDIO_PROCESSOR_SUMMARY.md) (заголовок L2 **A+B**) |
| **VisualProcessor** | **23** (**17** modules + **5** `core_*` + `ocr_extractor` — записи `RUN_LOG`) | `storage/audit_v4/<module>_l2/*.json` | **Почти закрыто:** **22** в `in_progress (v4 L2)`; **`micro_emotion`** — **blocked** (невалидный PCA на одном B-run, нет **5/5** OK NPZ) | **Есть** для core/modules/OCR ([`audit_4_2/visual_processor/`](components/audit_4_2/visual_processor/)) | [VISUAL_PROCESSOR_AUDIT_V4_SUMMARY.md](VISUAL_PROCESSOR_AUDIT_V4_SUMMARY.md) (оценки A/B/C и риски по компонентам) |
| **TextProcessor** | **22** | `storage/audit_v4/<extractor>_l2/*_audit_v4_stats.json` | **Tooling закрыт; эмпирика blocked:** на общем наборе **A+B** только **2/5** успешных `text_processor` → табличный срез `tp_*` только на subset (детали в JSON `dataset_quality` / `text_processor_error`) | **Есть** по всем 22 ([`audit_4_2/text_processor/`](components/audit_4_2/text_processor/)) | L1-эмпирика: [AUDIT_V4_L1_TEXT_PROCESSOR_SUMMARY.md](AUDIT_V4_L1_TEXT_PROCESSOR_SUMMARY.md); кросс-итог L1+L2: [AUDIT_V4_L1_CROSS_PROCESSORS_SUMMARY.md](AUDIT_V4_L1_CROSS_PROCESSORS_SUMMARY.md) |

**Итого по записям L2 в `RUN_LOG`:** **21 + 23 + 22 = 66** (Audio + Visual + TextProcessor), что согласуется с **66** L1-отчётами в [components/README.md](components/README.md).

---

## AudioProcessor

- **L2:** все **21** экстрактор закрыты на уровне **A+B** (продуктовая статистика, навигация в отчётах и `RUN_LOG`).
- **Субъективный итог волны:** ~**8 / 10** — сильная склейка контракт ↔ NPZ; отдельный класс дефектов (tabular/meta, строки в float) выявлен и частично исправлен; для «зелёного» продукта остаются **набор C**, **§4.8 golden**, **L3/§8**, повтор **A** после фиксов саверов.
- **Audit v4.2:** env-gated профилирование и `meta.stage_timings_ms` на большинстве компонентов — см. таблицу в [`components/audit_4_2/README.md`](components/audit_4_2/README.md).

---

## VisualProcessor

- **L2:** для **23** компонентов (записи `RUN_LOG`: **17** modules + **5** `core_*` + **`ocr_extractor`**) зафиксированы уровень **L2**, пути JSON и статусы.
- **Блокер:** **`micro_emotion`** — ошибка PCA на одном run набора **B**, нет полного **5/5** OK; остальные модули/core — **in_progress (v4 L2)** с типичными рисками (маски, NaN-оси, вырожденные оси времени).
- **Слабое место валидации смысла метрик на текущих 5 run:** **`action_recognition`** (`num_clips=1` на **B** — динамика не проверяется).
- Детализация по строкам (оценка **A/B/C**, риски): [VISUAL_PROCESSOR_AUDIT_V4_SUMMARY.md](VISUAL_PROCESSOR_AUDIT_V4_SUMMARY.md).

---

## TextProcessor

- **L2 tooling:** скрипты и JSON для **всех 22** экстракторов; записи в `RUN_LOG` и engineering logs в [`audit_4_2/text_processor/`](components/audit_4_2/text_processor/).
- **Блокер сквозной:** на стандартных **5** путях **A+B** пайплайн `text_processor` успешен только на **2/5** → полный L2 по данным **формально blocked** до **5/5** OK (пересборка `result_store` или исправление корневой ошибки e2e).
- L1-оценка волны на **A** остаётся **≈8.15/10**; детали по именам — в кросс-сводке и в [AUDIT_V4_L1_TEXT_PROCESSOR_SUMMARY.md](AUDIT_V4_L1_TEXT_PROCESSOR_SUMMARY.md).

---

## Сквозные выводы (все три процессора)

1. **Audio** — эталонный контур L2: статистика на **5** run, блокеры перенесены в **C / golden / L3** и гигиену артефактов после фиксов.
2. **Visual** — L2 собран почти везде; явные точечные блокеры (**`micro_emotion`**) и качество выборки (**`action_recognition`**).
3. **Text** — инструментарий L2 выровнен с Audio/Visual, но **общий e2e** `text_processor` ограничивает интерпретацию **любого** табличного среза на **B**.
4. **Audit v4.2** — единая точка входа для bridge-доков: [components/audit_4_2/README.md](components/audit_4_2/README.md); дальнейшие шаги плана: **§4.8**, набор **C**, оркестраторные отчёты и long-run профили (см. §12 плана и audio-сводку).
5. **Массовый прогон (60+ видео)** — план: [PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md](PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md); **чеклист до запуска:** [CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md](CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md).

---

## Дочерние сводки

- [AUDIT_V4_L1_AUDIO_PROCESSOR_SUMMARY.md](AUDIT_V4_L1_AUDIO_PROCESSOR_SUMMARY.md)  
- [VISUAL_PROCESSOR_AUDIT_V4_SUMMARY.md](VISUAL_PROCESSOR_AUDIT_V4_SUMMARY.md)  
- [AUDIT_V4_L1_TEXT_PROCESSOR_SUMMARY.md](AUDIT_V4_L1_TEXT_PROCESSOR_SUMMARY.md)  
- [AUDIT_V4_L1_CROSS_PROCESSORS_SUMMARY.md](AUDIT_V4_L1_CROSS_PROCESSORS_SUMMARY.md) (L1 + блок TextProcessor L1/L2)
---

## Навигация

[Audit v4 hub](components/audit_4_2/README.md) · [DataProcessor](../MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
