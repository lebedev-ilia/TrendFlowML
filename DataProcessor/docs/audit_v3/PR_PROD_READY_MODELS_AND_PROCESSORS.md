# PR: Prod-ready модели + процессоры (финальные контракты + закрытие Visual + критерии аудита остальных)
Дата: 2026-02-19  
Статус: DRAFT (PR plan), но фиксирует “что считаем prod-ready по логике/фичам”

---

## 0) Цель PR

Сделать **DataProcessor + Models/Encoder** prod-ready по части:

- функциональности и логики алгоритмов,
- качества/полезности фич,
- воспроизводимости/версий,
- согласованности “процессоры ↔ модели”.

Оптимизация производительности, бенчи, и массовые прогоны/обучение — отдельные следующие этапы.

---

## 1) Scope PR

### 1.1 Модели (FINAL правила)

В этом PR фиксируем **финальные (v2) правила интерфейса** моделей со всеми процессорами:

- `Models/docs/contracts/MODEL_INTERFACE_V2.md` — source-of-truth интерфейса (FeatureSpec / TokenStreams / TokenSpec / SamplingPlan / privacy / versioning)
- обновление `Models/docs/contracts/ENCODER_CONTRACT.md` — v2 семантика Encoder (Tokenizer + Learned Pooling)

### 1.2 VisualProcessor (закрывающий блок)

Так как audit Visual уже сделан, в этом PR фиксируем:

- что именно нужно довести/заморозить, чтобы Visual был prod-ready по логике/фичам,
- какие компоненты/пресеты являются baseline/v1/vNext,
- acceptance criteria (качество + контрактные инварианты).

### 1.3 Остальные процессоры (критерии предстоящего audit v3)

Для `AudioProcessor`, `TextProcessor`, `Segmenter` в этом PR:

- **не переписываем логику сейчас**,  
- но фиксируем критерии/правила, по которым они будут приведены к новым контрактам в предстоящем audit v3.

---

## 2) Крупные этапы (stages)

### Stage A — “Финальные контракты моделей” (DoD = документы + версии)

**Выходы**:

- `MODEL_INTERFACE_V2.md` принят как FINAL.
- `ENCODER_CONTRACT.md` обновлён (v2: tokenizer + learned pooling).
- В индексах документации добавлены ссылки.

**Acceptance**:

- есть явные версии: `model_interface_version`, `token_stream_schema_version`, `feature_spec_version`, `token_spec_version`, `sampling_plan_version`
- описаны privacy ограничения (raw text)

---

### Stage B — “Visual prod-ready” (DoD = закрытие логики/фич)

**Выходы**:

- обновлён `DataProcessor/docs/audit_v3/VISUALPROCESSOR_ASSESSMENT_REPORT.md`:
  - отдельный блок “prod-ready checklist”
  - список конкретных подгонок компонентов под v2 интерфейс

**Acceptance**:

- baseline path: стабильный табличный subset (через `feature_names/feature_values`) и понятные presets
- token path: план TokenStreams/EventStream + readiness для tokenizer
- строгие схемы/tiers/empty semantics

---

### Stage C — “Чеклист аудита v3 для Audio/Text/Segmenter”

**Выходы**:

- единый чеклист аудита: что именно должны сделать процессоры, чтобы соответствовать v2 контрактам

**Acceptance**:

- для каждого процессора определены:
  - time-axis семантика
  - какие sequences/tokens они публикуют
  - privacy правила
  - required vs optional политики

---

### Stage D — “Интеграция и QA пак” (следующий PR/эпик после этого)

Не делаем в этом PR, но фиксируем как следующий этап:

- QA-пак видео 10–20 → массовые прогоны
- калибровка budgets/config’ов по распределениям
- обучение baseline/vNext

---

## 3) Что будет дробиться в отдельные планы разработки

Каждый stage выше дальше дробится на epics:

- Implement FeatureSpec pipeline (baseline)
- Implement TokenStreams artifacts
- Implement Tokenizer + Learned Pooling encoder
- Implement SamplingPlan + multi-pass Segmenter
- Implement Visual event stream / tracking
- Audit v3: Audio
- Audit v3: Text
- Audit v3: Segmenter


