# Audit v3 — voice_quality_extractor: Вопросы 1-го раунда

**Дата**: 2026-03-13  
**Компонент**: `voice_quality_extractor` (Derivatives, optional)  
**Контекст**: Аудит нацелен на изменения и улучшения логики алгоритмов, а не на оптимизацию.

---

## 1. Per-extractor schema_version

**Текущее состояние**: `voice_quality_extractor` не в маппинге `run_cli.py` → fallback `audio_npz_v1`.

**Вопрос**: Ввести ли per-extractor схему?

| Вариант | Описание |
|---------|----------|
| **A** | Добавить `voice_quality_extractor_npz_v1` в маппинг `run_cli.py` |
| **B** | Оставить fallback `audio_npz_v1` |

**Рекомендация**: **A** — добавить `voice_quality_extractor_npz_v1`. Audit v3 требует явной схемы для audited компонентов.

---

## 2. Canonical axis (segment_start_sec / segment_end_sec / segment_mask)

**Текущее состояние**: NPZ хранит `segment_centers_sec`, `segment_durations_sec`. Нет `segment_start_sec`, `segment_end_sec`, `segment_mask`. Сегменты имеют `start_sec`, `end_sec` в Segmenter, но они не передаются в payload/NPZ.

**Вопрос**: Добавлять ли canonical axis по аналогии с loudness/spectral/tempo?

| Вариант | Описание |
|---------|----------|
| **A** | Добавить `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`; переименовать/дополнить `segment_centers_sec` → `segment_center_sec` |
| **B** | Оставить `segment_centers_sec`/`segment_durations_sec`; добавить только `segment_center_sec` как alias |
| **C** | Не менять — текущая структура достаточна |

**Рекомендация**: **A** — canonical axis обеспечивает согласованность с другими extractors и упрощает time-alignment.

---

## 3. Per-segment arrays (jitter_by_segment, shimmer_by_segment, hnr_by_segment)

**Текущее состояние**: Экстрактор агрегирует метрики по сегментам (mean, std, min, max). Per-segment значения (all_jitter, all_shimmer, all_hnr) не сохраняются в NPZ — только агрегаты.

**Вопрос**: Добавлять ли per-segment массивы для analytics?

| Вариант | Описание |
|---------|----------|
| **A** | Добавить `jitter_by_segment`, `shimmer_by_segment`, `hnr_by_segment` (float32[N]) — для timeline/распределений |
| **B** | Оставить только агрегаты; per-segment данные только при `enable_time_series` (как сейчас) |
| **C** | Добавить per-segment только для model-facing фичей (vq_voice_quality_score по сегментам) |

**Рекомендация**: **A** — per-segment массивы полезны для analytics и render; согласуются с loudness/spectral. Feature-gating можно сохранить (только при включённых jitter/shimmer/hnr).

---

## 4. Partial segment failures (segment_mask)

**Текущее состояние**: При ошибке `_estimate_f0` или `_compute_voice_quality_metrics` в сегменте — исключение пробрасывается, весь run падает. Нет `segment_mask` для частичных сбоев.

**Вопрос**: Обрабатывать ли частичные сбои сегментов?

| Вариант | Описание |
|---------|----------|
| **A** | Добавить try/except в `_process_segment`: при сбое — `segment_mask[i]=False`, `jitter/shimmer/hnr=NaN`; не прерывать run |
| **B** | Оставить fail-fast при первом сбое |
| **C** | Логировать сбой, подставлять NaN, `segment_mask[i]=False`, продолжать |

**Рекомендация**: **A** — graceful degradation. Сегменты без голоса (no f0) или с артефактами не должны срывать весь экстрактор.

---

## 5. Empty semantics

**Текущее состояние**: При пустых segments — `ValueError` до вызова extractor. При сбое f0 во всех сегментах — `RuntimeError`. Нет явных `status="empty"`, `empty_reason`.

**Вопрос**: Нужны ли явные empty semantics?

| Вариант | Описание |
|---------|----------|
| **A** | Добавить `status="empty"`, `empty_reason="voice_quality_all_segments_failed"` при `segment_mask` все False |
| **B** | Добавить проверку `audio too short` (<1s) → empty с `audio_too_short` |
| **C** | Оставить как есть — empty обрабатывается на уровне pipeline |
| **D** | A + C: empty только при all segments failed (если введём mask) |

**Рекомендация**: **D** — при введении `segment_mask` возвращать empty, когда все сегменты failed. Использовать каноничный `voice_quality_all_segments_failed` (расширение словаря empty_reason).

---

## 6. Feature gating: default preset

**Текущее состояние**: Все feature flags по умолчанию `False` (enable_jitter, enable_shimmer, enable_hnr, enable_f0_stats, enable_time_series). При включённом extractor без ни одной фичи — payload почти пустой (только metadata).

**Вопрос**: Нужен ли preset по умолчанию при включённом extractor?

| Вариант | Описание |
|---------|----------|
| **A** | Оставить все False; при включённом extractor без фичей — пустой payload (только meta) |
| **B** | Добавить preset `enable_basic_voice_quality=True`: jitter + shimmer + hnr + f0_stats (без time_series) |
| **C** | Добавить preset `enable_basic_voice_quality=True`: только jitter + shimmer + hnr (без f0_stats, без time_series) |
| **D** | Fail-fast при включённом extractor и 0 фичей |

**Рекомендация**: **C** — при включённом extractor иметь базовый preset (jitter, shimmer, hnr) для осмысленного выхода. f0_stats и time_series — opt-in. Audit v3: если extractor включён, он не должен падать с "no features enabled".

---

## 7. Render: offline-only, без CDN

**Текущее состояние**: `render.py` использует Plotly CDN: `<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>`.

**Вопрос**: Перевести ли рендер на offline-only?

| Вариант | Описание |
|---------|----------|
| **A** | Переписать на vanilla canvas (как loudness/spectral/tempo) |
| **B** | Оставить Plotly CDN |
| **C** | Встроить Plotly локально в _render/assets |

**Рекомендация**: **A** — vanilla canvas, без CDN. Audit v3 требует offline render.

---

## 8. NPZ: payload vs flat keys

**Текущее состояние**: NPZ saver сохраняет `feature_names`/`feature_values` + `f0`, `amps`, `hnr_vals`, `segment_centers_sec`, `segment_durations_sec`. Render читает из `payload` (если payload есть в NPZ). Текущий saver не пишет `payload` — данные в flat keys. Но render ожидает `payload` в npz_data.

**Вопрос**: Унифицировать ли структуру NPZ?

| Вариант | Описание |
|---------|----------|
| **A** | Убрать payload; render читает только flat keys (feature_names/values, segment_*, jitter_by_segment и т.д.) |
| **B** | Оставить payload для обратной совместимости; render поддерживает оба |
| **C** | Только flat keys; payload deprecated |

**Рекомендация**: **A** — Audit v3: NPZ = flat keys, no payload. Render должен читать из flat keys.

---

## 9. Pitch integration: per-segment alignment

**Текущее состояние**: `voice_quality` получает `pitch_payload` от `pitch_extractor`. Pitch даёт f0 для full audio или для своих сегментов. Voice_quality обрабатывает сегменты `families.voice_quality` — они могут не совпадать с `families.pitch`. При несовпадении pitch_payload не используется (f0 оценивается заново для каждого сегмента).

**Вопрос**: Как обрабатывать несовпадение сегментов pitch vs voice_quality?

| Вариант | Описание |
|---------|----------|
| **A** | Оставить как есть: pitch_payload используется только для run() (full audio); для run_segments() — своя оценка f0 |
| **B** | Требовать shared family: voice_quality и pitch должны использовать одни и те же сегменты (Segmenter contract) |
| **C** | Интерполяция: при разных сегментах — интерполировать f0 из pitch по времени center_sec |
| **D** | Документировать: pitch integration работает только при совпадении families; иначе — своя оценка |

**Рекомендация**: **D** — документировать текущее поведение. Shared family (B) — возможное улучшение на уровне Segmenter, но не в scope данного аудита.

---

## 10. NaN policy для missing значений

**Текущее состояние**: `_validate_output` отклоняет NaN/inf. При partial failures (если введём) — NaN в per-segment массивах. Агрегаты (mean, std) должны считаться только по валидным сегментам.

**Вопрос**: Нужны ли изменения?

| Вариант | Описание |
|---------|----------|
| **A** | При partial failures: NaN в jitter_by_segment и т.д.; агрегаты — только по finite; убрать reject NaN из _validate_output для per-segment |
| **B** | Оставить _validate_output как есть; при NaN в model-facing скалярах — error |
| **C** | Документировать NaN policy в SCHEMA.md |

**Рекомендация**: **A** + **C** — NaN для failed сегментов в per-segment; model-facing скаляры (агрегаты) — только по валидным; документировать в SCHEMA.md.

---

## 11. SCHEMA.md и machine schema

**Текущее состояние**: Нет `SCHEMA.md`, нет `schemas/voice_quality_extractor_npz_v1.json`.

**Вопрос**: Создавать ли?

| Вариант | Описание |
|---------|----------|
| **A** | Создать `SCHEMA.md` (human) и `schemas/voice_quality_extractor_npz_v1.json` (machine) |
| **B** | Только SCHEMA.md |
| **C** | Отложить до стабилизации контракта |

**Рекомендация**: **A** — обязательно для audited компонента.

---

## 12. Downstream: main_processor flat_payload

**Текущее состояние**: `main_processor` для voice_quality пишет в flat_payload: `vq_jitter`, `vq_shimmer`, `vq_hnr_like_db`. При feature-gating эти поля могут отсутствовать (0.0 fallback).

**Вопрос**: Нужны ли изменения при feature-gating?

| Вариант | Описание |
|---------|----------|
| **A** | При отсутствии фичи — NaN в flat_payload (не 0.0) |
| **B** | Оставить 0.0 fallback |
| **C** | Не писать ключ при отсутствии фичи |

**Рекомендация**: **A** — NaN для missing; согласуется с Audit v3 (no zero-stubs).

---

## 13. Итоговая таблица решений (реализовано по рекомендациям)

| # | Вопрос | Рекомендация | Реализовано |
|---|--------|--------------|-------------|
| 1 | schema_version | A — voice_quality_extractor_npz_v1 | ✅ |
| 2 | Canonical axis | A — segment_start/end/center, segment_mask | ✅ |
| 3 | Per-segment arrays | A — jitter/shimmer/hnr_by_segment | ✅ |
| 4 | Partial failures | A — segment_mask, graceful degradation | ✅ |
| 5 | Empty semantics | D — voice_quality_all_segments_failed | ✅ |
| 6 | Feature gating preset | C — enable_basic: jitter+shimmer+hnr | ✅ |
| 7 | Render offline | A — vanilla canvas | ✅ |
| 8 | NPZ payload | A — flat keys only | ✅ |
| 9 | Pitch alignment | D — документировать | ✅ |
| 10 | NaN policy | A + C | ✅ |
| 11 | SCHEMA.md + machine | A | ✅ |
| 12 | main_processor NaN | A | ✅ |
---

## Навигация

[Audit v3 index](../README.md) · [Extractor README](../../../src/extractors/voice_quality_extractor/docs/README.md) · [AudioProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
