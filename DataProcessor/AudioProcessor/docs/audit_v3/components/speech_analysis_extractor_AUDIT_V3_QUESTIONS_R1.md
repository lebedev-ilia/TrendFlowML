# Audit v3 — speech_analysis_extractor: Вопросы 1-го раунда

**Дата**: 2026-03-13  
**Компонент**: `speech_analysis_extractor`  
**Контекст**: Аудит нацелен на изменения и улучшения логики алгоритмов, а не на оптимизацию.

---

## 1. Feature gating: дефолты для ASR/diarization/pitch

**Текущее состояние**: Все три флага по умолчанию `False`:
- `enable_asr_metrics=False`
- `enable_diarization_metrics=False`
- `enable_pitch_metrics=False`

При включённом `speech_analysis` без флагов компонент возвращает минимальный payload (duration_sec, sample_rate, device_used, speech_analysis_contract_version). Сегменты asr/diarization всё равно обязательны (валидация), но результаты ASR/diarization не запрашиваются.

**Вопрос**: Какие feature flags включать по умолчанию?

| Вариант | Описание |
|---------|----------|
| **A** | Оставить все `False` — минимальная стоимость, пользователь явно включает нужное |
| **B** | `enable_asr_metrics=True`, `enable_diarization_metrics=True` по умолчанию — типичный use case «речь + спикеры» |
| **C** | Только `enable_asr_metrics=True` по умолчанию |
| **D** | Только `enable_diarization_metrics=True` по умолчанию |

**Рекомендация**: **A** — оставить все `False`. Компонент bundle/агрегатор, стоимость определяется зависимостями (ASR, diarization). Явное включение флагов даёт контроль над стоимостью. Audit v3 требует, чтобы «включённый extractor не падал» — при всех False он возвращает валидный minimal payload, что достаточно.

---

## 2. Per-extractor schema_version

**Текущее состояние**: `speech_analysis_extractor` не в маппинге `run_cli.py` → fallback `audio_npz_v1`.

**Вопрос**: Ввести ли per-extractor схему?

| Вариант | Описание |
|---------|----------|
| **A** | Добавить `speech_analysis_extractor_npz_v1` в маппинг `run_cli.py` |
| **B** | Оставить fallback `audio_npz_v1` |

**Рекомендация**: **A** — добавить `speech_analysis_extractor_npz_v1`. Это соответствует Audit v3 (per-extractor schema) и даёт чёткую версию контракта.

---

## 3. Empty semantics: audio too short (<5s)

**Текущее состояние**: При `dur_sec < 5.0` компонент выбрасывает `RuntimeError` (error, не empty).

**Вопрос**: Как обрабатывать аудио короче 5 секунд?

| Вариант | Описание |
|---------|----------|
| **A** | Возвращать `status="empty"`, `empty_reason="audio_too_short"` (как spectral_extractor при <1s) |
| **B** | Оставить `RuntimeError` (status=error) |
| **C** | Снизить порог до 1s и возвращать empty при <1s (как spectral) |
| **D** | Оставить 5s, но возвращать empty вместо error |

**Рекомендация**: **D** — оставить порог 5s (для речи он осмыслен), но возвращать `status="empty"`, `empty_reason="audio_too_short"` вместо RuntimeError. Это согласуется с Audit v3 (valid empty, не error).

---

## 4. Empty semantics: silence detection

**Текущее состояние**: При тихом аудио (peak < threshold, rms < threshold) возвращается `status="empty"`, `empty_reason="audio_missing_or_extract_failed"`.

**Вопрос**: Какой `empty_reason` использовать для тихого аудио?

| Вариант | Описание |
|---------|----------|
| **A** | Оставить `audio_missing_or_extract_failed` (каноничный из словаря) |
| **B** | Добавить/использовать `audio_silent` (более точная семантика) |
| **C** | Другой вариант |

**Рекомендация**: **A** — оставить `audio_missing_or_extract_failed`. Он уже в каноничном словаре и покрывает «нет полезного аудио». Добавление `audio_silent` потребует расширения словаря и не даёт существенной выгоды для downstream.

---

## 5. NaN policy для missing значений

**Текущее состояние**: NPZ saver использует `payload.get("...") or 0` для скаляров (asr_token_total, speaker_count и т.д.). При отсутствии значения записывается 0.

**Вопрос**: Как кодировать отсутствующие значения?

| Вариант | Описание |
|---------|----------|
| **A** | Использовать `np.nan` для missing (как в spectral_extractor) |
| **B** | Оставить `0` для обратной совместимости |
| **C** | NaN только для feature-gated полей, когда фича выключена |

**Рекомендация**: **A** — использовать NaN для missing. Audit v3: «никаких нулей-заглушек вместо missing». Feature-gated поля при выключенной фиче не должны попадать в NPZ (optional keys absent) или должны быть NaN, если ключ присутствует.

---

## 6. Canonical axis (segment_start_sec / segment_end_sec)

**Текущее состояние**: `speech_analysis_extractor` — bundle/агрегатор. Агрегирует ASR (per-segment) и diarization в run-level скаляры. Есть `asr_lang_id_by_segment` (по ASR-сегментам) и `speaker_ids` (по спикерам). Нет единой canonical time axis.

**Вопрос**: Добавлять ли canonical segment axis?

| Вариант | Описание |
|---------|----------|
| **A** | Не добавлять — компонент агрегирует в run-level, canonical axis не применим |
| **B** | Добавить `segment_start_sec`/`segment_end_sec` по ASR-сегментам (для asr_lang_id_by_segment) |
| **C** | Добавить axis по diarization turns |

**Рекомендация**: **A** — не добавлять. Компонент по сути run-level агрегатор. `asr_lang_id_by_segment` и `speaker_ids` — это индексы, не time series. Добавление axis усложнит контракт без явной пользы для model-facing path.

---

## 7. Render: offline-only, без CDN

**Текущее состояние**: `render.py` использует Plotly CDN: `<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>`.

**Вопрос**: Перевести ли рендер на offline-only?

| Вариант | Описание |
|---------|----------|
| **A** | Переписать на vanilla canvas/JS без CDN (как spectral_extractor) |
| **B** | Оставить Plotly CDN |
| **C** | Встроить Plotly локально (bundle в _render/assets) |

**Рекомендация**: **A** — переписать на vanilla canvas. Audit v3: «Интерактивность без внешнего интернета», «без CDN-зависимостей».

---

## 8. SCHEMA.md и machine schema

**Текущее состояние**: `SCHEMA.md` отсутствует. Machine schema (`speech_analysis_extractor_npz_v1.json`) не создана.

**Вопрос**: Создавать ли схемы?

| Вариант | Описание |
|---------|----------|
| **A** | Создать `SCHEMA.md` (human) и `schemas/speech_analysis_extractor_npz_v1.json` (machine) |
| **B** | Только SCHEMA.md |
| **C** | Отложить |

**Рекомендация**: **A** — создать обе схемы. Audit v3 требует human + machine schema для каждого audited компонента.

---

## 9. Минимальная длительность: 5s vs 1s

**Текущее состояние**: Порог 5 секунд.

**Вопрос**: Оставить 5s или снизить до 1s (как у spectral)?

| Вариант | Описание |
|---------|----------|
| **A** | Оставить 5s — для речи осмысленный минимум |
| **B** | Снизить до 1s — унификация с другими audio extractors |
| **C** | Сделать конфигурируемым (параметр min_duration_sec) |

**Рекомендация**: **A** — оставить 5s. Для speech analysis короткие клипы (<5s) дают мало полезной информации. Spectral работает с сырым сигналом, speech — с речью, требования различаются.

---

## 10. Privacy: asr_lang_id_by_segment

**Текущее состояние**: Сохраняются `asr_lang_id_by_segment` (int IDs) и `asr_lang_distribution`. Raw text не сохраняется.

**Вопрос**: Нужны ли изменения по privacy?

| Вариант | Описание |
|---------|----------|
| **A** | Оставить как есть — language IDs не PII |
| **B** | Убрать asr_lang_id_by_segment из model-facing (перевести в analytics) |
| **C** | Добавить feature flag для отключения lang_id_by_segment |

**Рекомендация**: **A** — оставить. Language IDs — не raw text, не PII. Audit v3 для speech: «raw ASR текст по умолчанию не сохраняем» — выполняется.

---

## 11. Empty при пустых сегментах (no-fallback)

**Текущее состояние**: При пустых `asr_segments` или `diar_segments` валидация падает с `ValueError` («segments is empty (no-fallback)») → status=error.

**Вопрос**: Менять ли поведение при пустых сегментах?

| Вариант | Описание |
|---------|----------|
| **A** | Оставить error — no-fallback, сегменты обязательны |
| **B** | Возвращать empty с `empty_reason="dependency_missing"` или `"audio_missing_or_extract_failed"` |
| **C** | Другой empty_reason |

**Рекомендация**: **A** — оставить error. Сегменты приходят из Segmenter; если families.asr/diarization пусты, это либо отсутствие аудио, либо конфигурационная ошибка. No-fallback согласован с preflight rules.

---

## 12. README: раздел Render (dev-only)

**Текущее состояние**: README есть, но раздел «Render (dev-only)» по шаблону Audit v3 может быть неполным.

**Вопрос**: Дополнить ли README по Audit v3?

| Вариант | Описание |
|---------|----------|
| **A** | Добавить/расширить раздел «Render (dev-only)»: файлы, как читать, типовые распределения, аномалии, связь с NPZ |
| **B** | Оставить как есть |
| **C** | Минимальные правки |

**Рекомендация**: **A** — добавить полный раздел по шаблону DECISIONS_AND_RULES (Key facts, Config highlights, How to QA, Top/Anti-top).

---

## Сводка рекомендаций

| # | Вопрос | Рекомендация |
|---|--------|--------------|
| 1 | Feature defaults | A — все False |
| 2 | schema_version | A — speech_analysis_extractor_npz_v1 |
| 3 | audio too short | D — empty, не error, порог 5s |
| 4 | silence empty_reason | A — audio_missing_or_extract_failed |
| 5 | NaN policy | A — NaN для missing |
| 6 | Canonical axis | A — не добавлять |
| 7 | Render | A — offline vanilla canvas |
| 8 | SCHEMA.md + machine | A — создать обе |
| 9 | min duration | A — оставить 5s |
| 10 | Privacy | A — оставить как есть |
| 11 | Empty segments | A — оставить error |
| 12 | README Render | A — расширить раздел |

---

**Следующий шаг**: Ответьте на вопросы (можно по номерам, например «1A, 2A, 3D, …»). При необходимости будут заданы уточняющие вопросы второго раунда. После ваших ответов будут внесены изменения и подготовлен финальный отчёт.
