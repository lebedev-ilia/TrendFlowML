# Audit v3 — tempo_extractor: Вопросы 1-го раунда

**Дата**: 2026-03-13  
**Компонент**: `tempo_extractor` (Tier-0 baseline)  
**Контекст**: Аудит нацелен на изменения и улучшения логики алгоритмов, а не на оптимизацию.

---

## 1. Per-extractor schema_version

**Текущее состояние**: `tempo_extractor` не в маппинге `run_cli.py` → fallback `audio_npz_v1`.

**Вопрос**: Ввести ли per-extractor схему?

| Вариант | Описание |
|---------|----------|
| **A** | Добавить `tempo_extractor_npz_v1` в маппинг `run_cli.py` |
| **B** | Оставить fallback `audio_npz_v1` |

**Рекомендация**: **A** — добавить `tempo_extractor_npz_v1`. Tier-0 baseline компонент обязан иметь явную схему по Audit v3.

---

## 2. Canonical axis (segment_start_sec / segment_end_sec / segment_mask)

**Текущее состояние**: NPZ хранит `windowed_times_sec` (центры сегментов) и `windowed_bpm`. Нет `segment_start_sec`, `segment_end_sec`, `segment_mask`. Сегменты имеют `start_sec`, `end_sec` в Segmenter, но они не передаются в payload.

**Вопрос**: Добавлять ли canonical axis по аналогии с loudness/spectral/rhythmic?

| Вариант | Описание |
|---------|----------|
| **A** | Добавить `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`; переименовать `windowed_bpm` → `bpm_by_segment` (или оставить оба для совместимости) |
| **B** | Оставить `windowed_times_sec`/`windowed_bpm`; добавить только `segment_center_sec` как alias |
| **C** | Не менять — текущая структура достаточна |

**Рекомендация**: **A** — добавить canonical axis. Audit v3 требует согласованности с другими extractors. `rhythmic_extractor` (shared family `tempo`) уже использует canonical axis. Передача `start_sec`/`end_sec` из segments в payload — минимальное изменение.

---

## 3. Partial segment failures (segment_mask)

**Текущее состояние**: При ошибке обработки любого сегмента `_one()` выбрасывает исключение → весь run падает с error. Нет `segment_mask` для частичных сбоев.

**Вопрос**: Обрабатывать ли частичные сбои сегментов (как loudness/spectral)?

| Вариант | Описание |
|---------|----------|
| **A** | Добавить try/except в `_one()`: при сбое сегмента — `bpm=NaN`, `segment_mask[i]=False`; не прерывать run |
| **B** | Оставить fail-fast при первом сбое |
| **C** | Логировать сбой, подставлять NaN для сегмента, `segment_mask[i]=False`, продолжать |

**Рекомендация**: **A** — добавить graceful degradation. Audit v3: «ошибки сегментов не срывают весь экстрактор» (loudness). Tempo — signal processing, сбои редки, но при повреждённом аудио/артефактах лучше вернуть частичный результат.

---

## 4. Empty semantics

**Текущее состояние**: При `audio_present=false` tempo не запускается (pipeline-level). При пустых segments — `segments_loader` raises до вызова extractor. При сбое `_estimate_from_np` (пустой сигнал, librosa error) — raise → error.

**Вопрос**: Нужны ли явные empty semantics?

| Вариант | Описание |
|---------|----------|
| **A** | Добавить `status="empty"`, `empty_reason="tempo_all_segments_failed"` при `segment_mask` все False (если введём partial failures) |
| **B** | Добавить проверку `audio too short` (<1s) → empty с `audio_too_short` |
| **C** | Оставить как есть — empty обрабатывается на уровне pipeline (audio_present=false) |
| **D** | A + C: empty только при all segments failed (если введём mask) |

**Рекомендация**: **D** — при введении `segment_mask` (вопрос 3) возвращать empty, когда все сегменты failed. Проверку `audio_too_short` не добавлять — tempo даёт BPM даже для коротких клипов; quality gates (warnings) уже есть.

---

## 5. NaN policy для missing значений

**Текущее состояние**: NPZ saver использует `payload.get()` без `or 0`. Функция `add()` вызывает `as_float(value)` → при `None` возвращается NaN. Для `run()` без `windowed_bpm` поля `tempo_windowed_bpm_*`, `segments_count` могут быть None → уже NaN.

**Вопрос**: Нужны ли изменения?

| Вариант | Описание |
|---------|----------|
| **A** | Оставить как есть — as_float(None)=NaN уже применяется |
| **B** | Явно документировать NaN policy в SCHEMA.md |
| **C** | Проверить все add() вызовы на отсутствие `or 0` |

**Рекомендация**: **A** + **B** — политика уже корректна; зафиксировать в SCHEMA.md. В saver нет `or 0` — всё ок.

---

## 6. Render: offline-only, без CDN

**Текущее состояние**: `render.py` использует Chart.js CDN: `<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>`.

**Вопрос**: Перевести ли рендер на offline-only?

| Вариант | Описание |
|---------|----------|
| **A** | Переписать на vanilla canvas (как loudness/spectral) |
| **B** | Оставить Chart.js CDN |
| **C** | Встроить Chart.js локально в _render/assets |

**Рекомендация**: **A** — переписать на vanilla canvas. Audit v3: «без CDN-зависимостей», «интерактивность без внешнего интернета».

---

## 7. SCHEMA.md и machine schema

**Текущее состояние**: `SCHEMA.md` отсутствует. Machine schema не создана.

**Вопрос**: Создавать ли схемы?

| Вариант | Описание |
|---------|----------|
| **A** | Создать `SCHEMA.md` (human) и `schemas/tempo_extractor_npz_v1.json` (machine) |
| **B** | Только SCHEMA.md |
| **C** | Отложить |

**Рекомендация**: **A** — создать обе схемы. Audit v3 требует human + machine schema для каждого audited компонента.

---

## 8. Имена ключей: windowed_times_sec vs segment_center_sec

**Текущее состояние**: `windowed_times_sec`, `windowed_bpm` — legacy имена. Rhythmic использует `segment_center_sec`, `segment_mask`.

**Вопрос**: Переименовывать ли для унификации?

| Вариант | Описание |
|---------|----------|
| **A** | Добавить `segment_center_sec` (= windowed_times_sec), `bpm_by_segment` (= windowed_bpm); сохранить `windowed_*` для обратной совместимости (deprecated) |
| **B** | Полностью заменить: только `segment_center_sec`, `bpm_by_segment` (bump schema) |
| **C** | Оставить `windowed_times_sec`/`windowed_bpm` без изменений |

**Рекомендация**: **B** — при введении canonical axis использовать только новые имена. Bump `schema_version` до `tempo_extractor_npz_v1`. Downstream (onset, rhythmic, high_level_semantic) читают `windowed_bpm`/`tempo_bpm` — нужно проверить и обновить контракты.

---

## 9. Warnings: расширение empty_reason?

**Текущее состояние**: `warnings` — список (`tempo_out_of_range`, `low_confidence`, `signal_too_quiet`). Хранятся в NPZ как `warnings` (object array). Не влияют на status.

**Вопрос**: Связывать ли критические warnings с empty?

| Вариант | Описание |
|---------|----------|
| **A** | Оставить как есть — warnings только в meta, status=ok |
| **B** | При `low_confidence` + `tempo_out_of_range` возвращать `status="empty"`, `empty_reason="tempo_low_confidence"` |
| **C** | Добавить `empty_reason="tempo_signal_too_quiet"` при `signal_too_quiet` |

**Рекомендация**: **A** — оставить warnings как quality gates. Audit: «warnings не доминируют» — достаточно для sanity. Empty резервировать для «нет данных», а не «низкое качество».

---

## 10. run() vs run_segments(): поддержка run()

**Текущее состояние**: Есть два пути: `run()` (full audio, опционально `windowed_bpm` по фиксированным окнам) и `run_segments()` (Segmenter windows). Pipeline всегда использует `run_segments()`.

**Вопрос**: Нужно ли поддерживать `run()` в audited контракте?

| Вариант | Описание |
|---------|----------|
| **A** | Оставить `run()` для legacy/direct usage; документировать как «не primary path» |
| **B** | Убрать `run()`, оставить только `run_segments()` |
| **C** | Оставить оба; `run()` при отсутствии segments возвращает empty |

**Рекомендация**: **A** — оставить `run()` для обратной совместимости. Primary path — `run_segments()`. Документировать в SCHEMA.md.

---

## 11. README: раздел Render (dev-only)

**Текущее состояние**: README есть, но раздела «Render (dev-only)» по шаблону Audit v3 нет.

**Вопрос**: Дополнить ли README?

| Вариант | Описание |
|---------|----------|
| **A** | Добавить раздел «Render (dev-only)»: файлы, как читать, типовые распределения, аномалии, связь с NPZ, Top/Anti-top |
| **B** | Оставить как есть |
| **C** | Минимальные правки |

**Рекомендация**: **A** — добавить полный раздел по шаблону DECISIONS_AND_RULES (Key facts, Config highlights, How to QA, Top/Anti-top).

---

## 12. Downstream: onset_extractor, rhythmic_extractor, high_level_semantic

**Текущее состояние**:
- **onset_extractor**: опционально использует `tempo_payload` для `onset_tempo_consistency`
- **rhythmic_extractor**: shared family `tempo`, читает свои BPM (не из tempo_extractor)
- **high_level_semantic**: опционально интерполирует `tempo_bpm` на `times_s`

**Вопрос**: Нужны ли изменения контрактов при переименовании ключей (вопрос 8)?

| Вариант | Описание |
|---------|----------|
| **A** | При переименовании обновить onset/main_processor — читать `bpm_by_segment` или `windowed_bpm` по schema_version |
| **B** | Сохранить обратную совместимость: писать и `windowed_bpm`, и `bpm_by_segment` в NPZ |
| **C** | Оставить `windowed_times_sec`/`windowed_bpm` без переименования (вопрос 8C) |

**Рекомендация**: Зависит от ответа на вопрос 8. При **8B** — **A** (обновить downstream). При **8C** — изменений не требуется.

---

## Сводка рекомендаций

| # | Вопрос | Рекомендация |
|---|--------|--------------|
| 1 | schema_version | A — tempo_extractor_npz_v1 |
| 2 | Canonical axis | A — добавить segment_* |
| 3 | Partial failures | A — segment_mask, graceful degradation |
| 4 | Empty semantics | D — empty при all segments failed |
| 5 | NaN policy | A+B — оставить, документировать |
| 6 | Render | A — offline vanilla canvas |
| 7 | SCHEMA.md + machine | A — создать обе |
| 8 | Имена ключей | B — segment_center_sec, bpm_by_segment |
| 9 | Warnings → empty | A — не связывать |
| 10 | run() support | A — оставить, документировать |
| 11 | README Render | A — расширить раздел |
| 12 | Downstream | Зависит от 8 |

---

**Следующий шаг**: Ответьте на вопросы (можно по номерам, например «1A, 2A, 3A, …»). При необходимости будут заданы уточняющие вопросы второго раунда. После ваших ответов будут внесены изменения и подготовлен финальный отчёт.
---

## Навигация

[Audit v3 index](../README.md) · [Extractor README](../../../src/extractors/tempo_extractor/docs/README.md) · [AudioProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
