# Audit v4 — `emotion_diarization_extractor`

**Дата:** 2026-04-06  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A+B**; **C** и §8 DoD — не закрыты).  
**Reference A (reproducible):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/emotion_diarization_extractor/emotion_diarization_extractor_features.npz`  
**Статистика L2 (A+B):** `storage/audit_v4/emotion_diarization_extractor_l2/emotion_diarization_extractor_audit_v4_stats.json` (+ `figures/`)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6), `--seed 0`

**Engineering log 4.2 (после L2):** `DataProcessor/docs/audit_v4/components/audit_4_2/audio_processor/emotion_diarization_extractor_engineering_log_v4_2.md`

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано на текущем уровне · **◐** частично / только набор **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика NPZ + семантика полей + вердикт | ✓ | Отчёт + `docs/README.md` / `SCHEMA.md` |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты DataProcessor / Models | ◐ | `emotion_diarization_extractor_npz_v1.json`, `SCHEMA.md`; Models — §5.3 |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Первая волна AudioProcessor | ✓ | `emotion_diarization_extractor` |
| Путь артефакта + `run_id` | ✓ | Шапка + [`RUN_LOG.md`](../../RUN_LOG.md) |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | `youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759` |
| **B** ≥5 видео | ✓ | `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`, `-Q6fnPIybEI/e2bc964f-…` |
| **C** edge | ✗ | тишина, `<5s`, все окна masked, срыв модели |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L2 product stats**, не L3 | ✓ | `RUN_LOG`: L2 закрыт по **A+B** |
| Нет заявления полного §8 | ✓ | DoD не закрыт |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `dtype` / `shape` | ✓ | N=7, C=4; `emotion_probs` (7,4); оси времени (7,) |
| Сверка с JSON-схемой | ◐ | Ключи совпадают; CI/runtime validation не приложена |
| Согласованность dominant | ✓ | `dominant_emotion_id` в tabular = **2** = `argmax(emotion_mean_probs)`; `dominant_emotion_prob` ≈ max mean prob |

#### §4.1a — Семантика типов, строки, NaN

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Tabular через `as_float` | ✓ | На **A** NaN в `feature_values` нет; `dominant_emotion_id` как float **2.0** — ожидаемо |
| `emotion_probs` строки | ✓ | Суммы по классу после нормализации ~1 на **A** |
| Object-поля с dict | ✓ | `emotion_distribution` и др. — **0-dim `object`**, значение `arr.item()` — `dict` |

#### §4.2 — NaN, Inf, нули

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Маски и NaN | N/A **A** | Все сегменты валидны; **C** — `NaN` в `emotion_confidence`, `-1` в `emotion_id` |

#### §4.3 — Распределения

| Критерий | Статус | Заметка |
|----------|--------|---------|
| p01…p99 по tabular на **A+B** | ✓ | `storage/audit_v4/emotion_diarization_extractor_l2/emotion_diarization_extractor_audit_v4_stats.json` |
| `segments_count` / `emotion_entropy` на **A+B** | ✓ | `segments_count`: min=7, max=20, mean=14.2; `emotion_entropy`: min≈0.598, max≈1.221, mean≈0.906 |

#### §4.4 — Категориальные / object

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `emotion_labels`, dict-ключи int | ✓ | `['a','n','h','s']` — 4 класса |
| `meta.models_used` | ✓ | `emotion_diarization_large_inprocess` + digest |
| `meta.model_name` / `weights_digest` | ◐ | на проанализированном **A** были **`None`** — савер брал из payload, экстрактор не прокидывал; **исправлено в коде** для следующих прогонов |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `segment_*_sec` строго N | ✓ | Выравнивание с Segmenter |
| `segments_total` vs `segments_count` | ✓ | meta `segments_total`=7; tabular `segments_count`=7 при полной маске |

#### §4.6 — Корреляции (**B**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Внутренние (между tabular на A+B) | ✓ | `storage/audit_v4/emotion_diarization_extractor_l2/emotion_diarization_extractor_audit_v4_stats.json` + `figures/tabular_corr_heatmap.png` |
| С ASR / speech_analysis | ✗ | L3 / C / далее |

#### §4.7 — Трактовка

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Наблюдения → §4.7 | ◐ | §5–6 |

#### §4.8 — Golden на **A**

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Хеш probs / tabular | ✗ | TODO |

#### §4.9 — Sampling, N, перекрытие

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `families.emotion` | ✓ | N=7 на **A** |
| Мин. длительность 5s | N/A **A** | Политика на уровне экстрактора; e2e прошёл |

#### §4.10 — `empty` / `empty_reason` (**C**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Артефакты silent / too_short | ✗ | **C** |

#### §4.11 — >24 scalar

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Порог | N/A | F=7 tabular + плотные `[N,C]` |

#### §4.12 — Anti-leakage

| Вопрос (план) | Ответ | Комментарий |
|-----------------|-------|-------------|
| Только текущее аудио? | Да | |
| Глобальная нормализация по датасету? | Нет | |
| Онлайн загрузка весов? | Нет | `dp_models` offline |

#### §5 — Документация полей

| Подпункт | Статус | Заметка |
|----------|--------|---------|
| §5.1–§5.2 README | ◐ | Audit v4 таблица NPZ; уточнено: **rms/peak не в NPZ** |
| §5.3 | ◐ | Ниже |

##### §5.3 — Сверка с Models (`emotion_diarization_extractor`)

| Вопрос (план §5.3) | Ответ | Комментарий |
|--------------------|-------|-------------|
| В минимальном Baseline v1.0 (3 аудио)? | **Нет** | [`BASELINE_MODEL.md`](../../../../../Models/docs/contracts/BASELINE_MODEL.md): `clap_extractor`, `loudness_extractor`, `tempo_extractor`. |
| Tabular / FeatureSpec | **Частично** | 7 скаляров + богатые последовательности |
| Dense / encoder | **Частично** | `emotion_probs` + time axis при включённых фичах |
| Segmenter | **Да** | `families.emotion`, mask |
| Downstream | **Да** | Возможен `speech_analysis` и др. |

#### §6 — Verdict

| Критерий | Статус |
|----------|--------|
| Блок §6 плана | ✓ |

#### §6.1 — Оценка, операционно

| Критерий | Статус |
|----------|--------|
| Баллы 0–10 | ✓ |
| Wall-time / RAM | ✗ |

#### §7 — Порядок работ

| Шаг | Статус |
|-----|--------|
| Идентификация + NPZ | ✓ |
| Общий скрипт, seed | ✗ |
| §4 на L3 | ✗ |

#### §8 — Definition of Done

**Не закрыт:** **C**, golden, межкомпонентные корреляции (§4.6), full §8 DoD.

#### §9–§11

| Раздел | Статус | Заметка |
|--------|--------|---------|
| §10 журнал | ◐ | git hash TODO |
| §11 | N/A | |

---

## 1. Мета (фрагмент, набор **A** до фикса payload)

| Поле | Значение |
|------|----------|
| `schema_version` | `emotion_diarization_extractor_npz_v1` |
| `producer_version` | `3.1.0` |
| `status` | `ok` |
| `emotion_contract_version` | `emotion_contract_v1` |
| `features_enabled` | `probs`, `mean_probs`, `ids`, `confidence`, `entropy`, `dominant`, `quality_metrics` |
| `segments_total` | 7 |
| `models_used` | large inprocess, digest присутствует |
| `model_name` (meta) | **null** (до фикса) |
| `weights_digest` (meta) | **null** (до фикса) |

---

## 2. Tabular (набор **A**)

Порядок имён (**`npz_savers/emotion_diarization.py`**):

1. `segments_count` — 7  
2. `emotion_entropy` — ≈ 0.598  
3. `dominant_emotion_id` — 2.0  
4. `dominant_emotion_prob` — ≈ 0.715  
5. `emotion_transitions_count` — 1  
6. `emotion_stability_score` — ≈ 0.957  
7. `emotion_diversity_score` — ≈ 0.431  

NaN в `feature_values`: **0**.

---

## 3. Сверка с кодом

1. Inference по батчам, нормализация строк `probs`, strict scatter по `valid_indices`.
2. Агрегаты и `emotion_transitions_count` только по `segment_mask`.
3. Dict-метрики сохраняются как **scalar object** массивы с `.item()` = `dict`.
4. **`rms` / `peak`**: есть в **payload** экстрактора, **не** сериализуются в NPZ савером.

---

## 4. Вердикт

**Плюсы:** строгая ось N, mask, обогащённые метрики (transitions, stability, diversity); опциональные probs/mean/quality; offline модель через ModelManager.

**Минусы:** вне minimal audio baseline; L1 без B/C; до фикса **пробел meta.model_name/weights_digest** при наличии `models_used`.

---

## 5. Оценка 0–10

| Критерий | Балл |
|----------|------|
| Стабильность / контракт | **8** |
| Полезность tabular | **8** |
| Полезность для encoder | **8** (при `emotion_probs` и оси времени) |
| Аналитика / UI | **9** |

**Итог: ~8/10** при условии L2/L3.
