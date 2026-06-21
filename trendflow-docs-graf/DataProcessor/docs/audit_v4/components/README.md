# Audit v4 — индекс отчётов по компонентам

Отчёты **L1 (draft)** разложены по подсистемам:

| Папка | Содержимое |
|--------|------------|
| [`audio_processor/`](audio_processor/) | экстракторы `AudioProcessor` (`*_extractor_audit_v4.md`) |
| [`visual_processor/modules/`](visual_processor/modules/) | модули `VisualProcessor` (сценарные / продуктовые NPZ поверх core) |
| [`visual_processor/core/`](visual_processor/core/) | core: CLIP, depth, лица, детекции, optical flow, OCR |
| [`text_processor/`](text_processor/) | экстракторы / срезы `TextProcessor` (агрегированный `text_features.npz`) |

Сквозные документы уровня плана: [`../AUDIT_4_CRITERIA_AND_PLAN.md`](../AUDIT_4_CRITERIA_AND_PLAN.md), журнал: [`../RUN_LOG.md`](../RUN_LOG.md).

### Audit 4.2 — engineering bridge (после L2)

- [`audit_4_2/README.md`](audit_4_2/README.md) — инженерные журналы Audit 4.2 после L2: [`audio_processor/`](audit_4_2/audio_processor/), [`text_processor/`](audit_4_2/text_processor/), [`visual_processor/core/`](audit_4_2/visual_processor/core/), [`visual_processor/modules/`](audit_4_2/visual_processor/modules/).

Сводки:

- **Три процессора (кросс-сводка L1):** [`../AUDIT_V4_L1_CROSS_PROCESSORS_SUMMARY.md`](../AUDIT_V4_L1_CROSS_PROCESSORS_SUMMARY.md)
- **Audit v4.2 — общий итог L2 по процессорам:** [`../AUDIT_V4_2_L2_CROSS_PROCESSORS_SUMMARY.md`](../AUDIT_V4_2_L2_CROSS_PROCESSORS_SUMMARY.md)
- **Подготовка к большому прогону (60+ видео):** план [`../PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md`](../PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md) · чеклист [`../CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md`](../CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md)
- AudioProcessor: [`../AUDIT_V4_L1_AUDIO_PROCESSOR_SUMMARY.md`](../AUDIT_V4_L1_AUDIO_PROCESSOR_SUMMARY.md)
- VisualProcessor (L1): [`../AUDIT_V4_L1_VISUAL_PROCESSOR_SUMMARY.md`](../AUDIT_V4_L1_VISUAL_PROCESSOR_SUMMARY.md)
- VisualProcessor (Audit v4, L2+): [`../VISUAL_PROCESSOR_AUDIT_V4_SUMMARY.md`](../VISUAL_PROCESSOR_AUDIT_V4_SUMMARY.md)
- TextProcessor: [`../AUDIT_V4_L1_TEXT_PROCESSOR_SUMMARY.md`](../AUDIT_V4_L1_TEXT_PROCESSOR_SUMMARY.md)
---

## Навигация

[Audit v4 hub](audit_4_2/README.md) · [DataProcessor](../../MAIN_INDEX.md) · [Vault](../../../../docs/MAIN_INDEX.md)
