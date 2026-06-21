# Audit v4 / **4.2** — `asr_extractor`

**Дата:** 2026-04-07 (**закрытие L2** для Audit 4.2; L1 от 2026-04-06 сохранён ниже как reference **A**).  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A + B**, **5** прогонов `youtube/*` из `result_store`).  
**Набор B (diversity):** см. JSON `storage/audit_v4/asr_extractor_l2/asr_extractor_audit_v4_stats.json` (`paths`).  
**Инструмент:** `DataProcessor/AudioProcessor/src/extractors/asr_extractor/scripts/audit_v4_npz_stats.py` (seed по умолчанию **0**).  
**Reference A (исторический один run):** `storage/result_store/youtube/-Q6fnPIybEI/4c3bf25b-e300-47b3-915e-4699c72ab190/asr_extractor/asr_extractor_features.npz`  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано · **◐** частично / только **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика + семантика + вердикт | ✓ | Этот документ + `docs/README.md` |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты Models/DataProcessor | ◐ | §5.3 и README; полный перечень §1 плана не копировался |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| AudioProcessor, NPZ из e2e | ✓ | `asr_extractor` |
| Путь под `run_id` | ✓ | Шапка + `RUN_LOG.md` |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | Один reference run (§1–4 ниже, старый `run_id`) |
| **B** | ✓ | **5** видео, разные `token_total` (6…219), `segments_count` 1–2; языки встречались **ms**, **en** |
| **C** | ✗ | empty / нет речи / короткое аудио — **TODO** (L3) |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L2**, не `passed` | ✓ | Журнал + JSON + графики в `storage/audit_v4/asr_extractor_l2/figures/` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Список ключей NPZ, длины | ✓ | §2–4 отчёта |
| Machine schema check | ◐ | Не автоматизировано в отчёте |

#### §4.1a — Типы и tabular

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Разделение float / счётчик / строка | ◐ | README + этот отчёт; полный чеклист всех имён — TODO |
| Строки vs `as_float` | ✓ | На компоненте нет строкового поля в `feature_values` |
| `lang_id` vs `lang_code` | ✓ | Документировано: id — legacy/Whisper, код — продуктовый |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Доля NaN в tabular на **A** | ✓ | 0 на проанализированном файле |

#### §4.3 — Распределения

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Перцентили по всем tabular полям | ✓ | Скрипт: `aggregate.tabular_per_feature` (p01/p50/p99 по 5 run); гистограммы `figures/hist_tabular_*` |
| Вырождение при N=1 | ✓ | На части B всё ещё бывает **N=1** сегмент → квантили quality совпадают (ожидаемо) |

#### §4.4 — Object / строки

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Частоты языков, длины строк | ◐ | На B: **ms** и **en** (`lang_codes` в JSON per_file); полный частотный отчёт — опционально расширить скрипт |
| Топы token id / гистограмма | ◐ | В JSON есть `derived.token_len_stats`; детальная гистограмма token-id — TODO |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Монотонность центров сегментов | ✓ | N=1 |
| Union video timestamps | N/A | Аудио окна ASR |

#### §4.6 — Корреляции (**B**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| ρ по tabular и длинам | ✓ | `correlation_tabular.matrix` в JSON; heatmap `figures/tabular_corr_heatmap.png` (5×5 runs — много NaN у констант **`sample_rate`** / rate=1.0; ожидаемо) |

#### §4.7 — Трактовка

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Таблица наблюдений → выводы | ◐ | §6–7 |

#### §4.8 — Golden на **A**

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Сигнатура для регрессии | ✗ | TODO JSON/хэш |

#### §4.9 — Sampling, N

| Критерий | Статус | Заметка |
|----------|--------|---------|
| duration × N | ◐ | N=1, ~12 s; `asr_window_sec`/`stride` в NPZ |
| Два прогона policy | ✗ |
| Перекрытие ASR-окон | ◐ | Проверить при N>1 на **B** |

#### §4.10 — Empty (**C**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Кейсы `empty` / `empty_reason` | ✗ |

#### §4.11 — >24 scalar

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Триггер | N/A/◐ | На run порядка ~20 tabular скаляров |
| Корр. на **B** при росте фич | ✗ |

#### §4.12 — Anti-leakage

| Вопрос | Ответ | Комментарий |
|--------|-------|-------------|
| Только локальное аудио? | Да | |
| Глобальная норма по корпусу внутри шага? | Нет | |
| Онлайн API? | Нет | inprocess + ModelManager |

#### §5 — Документация

| Подпункт | Статус | Заметка |
|----------|--------|---------|
| §5.1–§5.2 README | ◐ | Каталог; можно добавить полные колонки шаблона §5.1 |
| §5.3 | ◐ | Ниже |

##### §5.3 — Сверка с Models (`asr_extractor`)

| Вопрос | Ответ | Комментарий |
|--------|-------|---------------|
| В Baseline v1.0 (3 аудио)? | **Нет** | [`BASELINE_MODEL.md`](../../../../../Models/docs/contracts/BASELINE_MODEL.md): `clap`, `loudness`, `tempo` только. |
| Tabular для прогноза | **Частично** | Плотность, WPM, quality-агрегаты — кандидаты в расширенный FeatureSpec. |
| Token path | **Да** | `token_ids` + `shared_tokenizer_v1`; стыкуется с идеей токенов в [`MODEL_INTERFACE_V2.md`](../../../../../Models/docs/contracts/MODEL_INTERFACE_V2.md) / text pipeline. |
| Как mel dense для AudioEncoder | **Нет** | Дискретные токены по сегментам, не time-series mel. |
| Analytics / QA | **Да** | `segment_quality_*`, язык, confidence. |

#### §6 — Verdict

| Критерий | Статус |
|----------|--------|
| Блок §6 плана | ✓ |

#### §6.1

| Критерий | Статус |
|----------|--------|
| Баллы 0–10 | ✓ |
| Wall-time / RAM | ✗ |

#### §7–§8

| Критерий | Статус |
|----------|--------|
| Скрипт + seed | ✓ | `audit_v4_npz_stats.py`, `--seed 0` |
| L3 DoD | ✗ | Нужен набор **C** + §4.9/§4.10 полнее |

#### §9–§11

| Раздел | Статус |
|--------|--------|
| §10 полнота журнала | ◐ |
| Остальное | N/A / по плану |

---

## 0. Набор **B** (L2) — сводка по 5 прогонам

Источник: автоматический отчёт (см. шапку). Характеристики **tabular** по пяти NPZ (разные `video_id` / `run_id`):

| Поле | min | max | p50 (медиана по run) | Комментарий |
|------|-----|-----|----------------------|-------------|
| `token_total` | 6 | 219 | 152 | Сильный разброс длины распознанной речи по клипам |
| `token_density_per_sec` | ~0.40 | ~5.71 | ~3.63 | Согласуется с разной длительностью и наполненностью |
| `speech_rate_wpm` | ~18.6 | ~263 | ~167 | Хвосты ожидаемы при короткой речи / другом языке |
| `segments_count` | 1 | 2 | 2 | Практически все прогоны — 1–2 ASR-окна |
| `token_variance` | 0 | 81 | — | Ненулевая дисперсия появляется при **N≥2** сегментах |

**NaN в `feature_values`:** на всех пяти файлах **0** (доля NaN по run = 0).  

**Корреляции:** см. `figures/tabular_corr_heatmap.png` и `correlation_tabular` в JSON; высокие |ρ| между группами `asr_quality__*_mean/p50/p90` и с `token_*` частично отражают общую «информативность» клипа, а не обязательно избыточность одной фичи — для §4.11 на большем **B** имеет смысл кластеризовать группы качества.

**Этап 2** (профилирование ресурсов, оркестратор, ускорение без деградации логики): по плану Audit 4.2 — [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md) §12.2; рекомендуется общий слой в оркестраторе, затем подключение экстракторов. Журнал последующих правок кода и ссылка на артефакты статистики L2: [`../audit_4_2/audio_processor/asr_extractor_engineering_log_v4_2.md`](../audit_4_2/audio_processor/asr_extractor_engineering_log_v4_2.md).

---

## 1. Мета и статус **(набор A, reference single run)**

Ниже — детальный разбор **одного** reference run (L1); контракт и код не изменились, мета актуальна для B.

| Поле | Значение |
|------|----------|
| `schema_version` | `asr_extractor_npz_v2` |
| `producer_version` | `2.2.0` |
| `status` | `ok` |
| `asr_text_contract_version` | `asr_text_contract_v1` |
| `device_used` | `cuda` |
| `features_enabled` | `token_sequences`, `token_counts`, `token_total`, `token_density`, `speech_rate`, `lang_distribution`, `segments_with_speech`, `avg_segment_duration`, `token_variance` |

### `models_used` (воспроизводимость)

- **Whisper:** `whisper_small_inprocess`, `weights_digest` `a4a7b738709c3a0221aea182a25ce72964ce1c45b8b25f933bd0859385e6792d`, runtime `inprocess`, `fp16`, `cuda`.
- **Токенизатор:** `shared_tokenizer_v1`, digest `c0cb7277b7f6efc61e33bc5daf6f17142babb0bb68b2d5dd600c96471a90c62e`, `tokenizers` / CPU.

---

## 2. Объём данных на reference run (**A**)

| Метрика | Значение |
|---------|----------|
| Число ASR-сегментов `N` | **1** (одно длинное окно family `asr`) |
| `audio_duration_sec` (scalar) | **12.028** s |
| `segment_*` | start **0**, end **12.028** s, center **6.014** s |
| `token_ids_by_segment` | 1 список, длина **31** токен |
| ID токенов (shared tokenizer) | min **0**, max **145074** (окладывается на живой vocab `shared_tokenizer_v1`, не на 51k Whisper) |
| `token_counts` | `[31]` |
| `token_total` (tabular) | **31** |
| `lang_code_by_segment` | **`ms`** |
| `lang_conf_by_segment` | **~0.998** |
| `lang_id_by_segment` | **50282** |
| `lang_distribution` | `{"ms": 1}` |

---

## 3. Tabular: `feature_names` / `feature_values`

Порядок — как в `npz_savers/asr.py` (`add(...)`).

| Имя | Значение | Комментарий |
|-----|----------|-------------|
| `segments_count` | 1 | |
| `sample_rate` | 16000 | Гц |
| `token_total` | 31 | |
| `token_density_per_sec` | 2.577 | `31 / 12.028…` ✓ |
| `speech_rate_wpm` | 118.95 | `(31/1.3) / (12.028/60)` ✓ |
| `segments_with_speech` | 1 | |
| `avg_segment_duration_sec` | 12.028 | один сегмент |
| `token_variance` | **0** | при **одном** сегменте по коду всегда 0 — ожидаемо |
| `asr_quality__avg_logprob_mean/p50/p90` | −0.425 | один сегмент → все квантилы совпадают |
| `asr_quality__compression_ratio_*` | 1.189 | |
| `asr_quality__no_speech_prob_*` | 0.00130 | низкий, согласуется с речью |

**NaN в tabular:** на этом файле **нет** (все `feature_values` конечны).

---

## 4. Массивы времени и сэмплинг

| Массив | NaN | Примечание |
|--------|-----|------------|
| `segment_start_sec` / `end` / `center` | 0 | длина 1, монотонность тривиальна |
| `lang_conf_by_segment` | 0 | |
| `asr_window_sec` | 30 | из `asr_segments_meta` |
| `asr_stride_sec` | 25 | |
| `asr_max_windows` | **−1** | «неизвестно» по контракту савера |
| `asr_sampling_profile` | `"semantic"` | object scalar |

На run с **N > 1** имеет смысл дополнительно проверить монотонность `segment_center_sec` и перекрытие окон (как у pitch).

---

## 5. `segment_quality_by_segment`

Один элемент:

```text
avg_logprob ≈ -0.425, compression_ratio ≈ 1.189, no_speech_prob ≈ 0.00130, temperature = 0.0
```

Совпадает с полями Whisper `DecodingResult` (числовые, без сырого текста). Язык в этом dict не дублируется (он в отдельных массивах).

---

## 6. Сверка с кодом (`main.py`)

1. **Пайплайн:** mel 80-канальный → `whisper_model.decode` (beam при `temperature=0`) → текст → **строго** `shared_tokenizer_v1.encode` → `token_ids` (без fallback на whisper token ids).
2. **Язык:** `detect_language` → предпочтительно **код** и **confidence** из dict вероятностей; `lang_id` — best-effort (часто **токен/индекс из Whisper**, см. `lang_tok[0]`). Для продуктовой логики опираться на **`lang_code_by_segment`**, а не на числовой `lang_id`, если нужна стабильная семантика.
3. **Агрегаты:** `token_density_per_sec`, `speech_rate_wpm`, `token_variance` — формулы в `run_segments` совпадают с табличными числами на этом примере.
4. **`segment_quality_by_segment`:** не feature-gated — всегда пишется (privacy-safe числа).

---

## 7. Вердикт

**Плюсы**

- Контракт token IDs ↔ shared tokenizer соблюдён; `models_used` полон для кэша/аудита.
- Нет обязательного хранения сырого текста в NPZ (хорошо для privacy + TextProcessor).
- Табличные и по-сегментные качественные метрики пригодны для **tabular baseline**, гейтинга «качество ASR» и аналитики.
- Один длинный ASR-контур на коротком клипе — нормальный режим Segmenter.

**Минусы / внимание**

- При **N=1** `token_variance` и квантили `asr_quality__*` не информативны — нужен diversity set с несколькими окнами.
- **`lang_id_by_segment`** легко ошибочно трактовать как enum языка — это устаревший/вспомогательный сигнал рядом с Whisper.
- Для **моделей по токенам** нужен decoder baseline (TextProcessor / embedding pipeline); сам NPZ несёт ids, не текст.

---

## 8. Оценка 0–10

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Масштабирование пайплайна (окна, GPU, sequential) | **9** | Длинные окна снижают число вызовов; узкое место — latency Whisper на длинном контенте |
| Готовность фич для **моделей** (tabular) | **8** | Плотность речи, WPM, quality means/p90 при N≫1 |
| Готовность для **token / text** downstream | **9** | Сильный контракт ids + digests |
| Демонстрация параметров видео / QA | **8** | Язык, confidence, no_speech_prob, compression_ratio |

**Итог L1 (A): 8.7 / 10** (округляя продуктово **9/10** для Tier‑0 ASR при условии документированного использования `lang_code` и тестов на многосегментных роликах).

### Оценка **L2 (Audit 4.2, A+B)**

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Стабильность контракта на diversity | **9** | Одинаковый набор ключей NPZ, нет неожиданных NaN в tabular |
| Полнота эмпирики §4 | **8** | Перцентили и корр на 5 run; нет **C**, нет golden JSON §4.8 |
| Готовность tooling | **9** | Воспроизводимый скрипт + JSON + графики |
| Риски вырождения фич | **7** | При малом числе сегментов quality-квантили малоинформативны |

**Итог L2: ~8.3 / 10** — уровень **product stats** достигнут; до **L3** не хватает набора **C**, edge `empty` и зафиксированной регрессионной сигнатуры на **A**.
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
