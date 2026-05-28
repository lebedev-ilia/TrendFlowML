# Audit v4 — `color_light` (VisualProcessor)

**Дата:** 2026-04-06 (обновление: 2026-04-13)  
**Уровень отчёта (план §3.1):** **L2 — product stats** (**A + B**, 5 run).  
**Артефакт (набор A, фактический в `storage/result_store`):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/color_light/color_light_features.npz`  
**Код / контракт:** `DataProcessor/VisualProcessor/modules/color_light/` · machine schema: [`VisualProcessor/schemas/color_light_npz_v2.json`](../../../../../VisualProcessor/schemas/color_light_npz_v2.json) · [`docs/SCHEMA.md`](../../../../../VisualProcessor/modules/color_light/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано на текущем уровне · **◐** частично / только набор **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика NPZ + семантика + вердикт | ✓ | Ниже + `docs/SCHEMA.md` |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Контракты DP / Models | ◐ | §5.3 |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| VisualProcessor | ✓ | Hard dep: **`scene_classification`**, ось Segmenter |
| Путь + run | ✓ | [`RUN_LOG.md`](../../../RUN_LOG.md) |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | e2e reference |
| **B** | ✓ (JSON) | `storage/audit_v4/color_light_l2/color_light_audit_v4_stats.json` (A+B, 5 run) |
| **C** | ✗ | `after_filt_empty`, нет `scene_classification`, нет timestamps |

#### §3.1 — Уровень

| Критерий | Статус |
|----------|--------|
| L2, не L3 | ✓ |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи NPZ vs `color_light_npz_v2.json` | ✓ | **13** ключей, **без лишних** при `allow_extra_keys=false` |
| `manifest.notes` | ✓ | **`null`** на **A** |
| `frame_compact_features` | ✓ | **`(M, 16)`** float32, имена **16** строк стабильны (см. снимок §1) |
| Ось **N** vs **M** | ◐ | На **A**: **`frame_indices`** и **`sequence_frame_indices`** **идентичны** (оба **36**); **`times_s`** = **`sequence_times_s`**. В `SCHEMA.md` заложено **N** и **M** как разные длины — здесь совпали после пересечения с сценами |
| `meta.processed_frames` vs длина оси | ◐ | `processed_frames=250`, **`len(frame_indices)=36`** — разные сущности (segmenter vs пересечение с индексами сцен); не путать при логировании |

#### §4.1a — Семантика типов / NaN в dict

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `video_features` | ◐ | **543** float-ключа-скаляра на **A**; **7** значений **NaN**: `color_distribution_gini`, `nima_mean`, `nima_std`, `laion_mean`, `laion_std`, `cinematic_lighting_score`, `professional_look_score` — по смыслу **опциональные/внешние оценки**, не прогнанные или недоступны; нужна явная политика в docs / `empty_reason` vs «тихий NaN» |
| `aggregated.frame_compact` | ✓ | Сводка по компактным фичам: `mean`, `std`, `p25/p50/p75`, **`valid_rows`**, **без NaN** в агрегатах на **A** |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `frame_compact_features` | ✓ | **0%** NaN, **0%** Inf |
| `video_features` | ◐ | **7** NaN, **0** Inf |

#### §4.3 — Распределения (**A**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Компактные фичи | ✓ (JSON) | На **A+B**: `M_total=142`, `M_min=18`, `M_max=36`; по всем значениям compact min/max **0 … ~2.90**, mean **~0.505**, p95 **~1.93** (см. JSON `compact_all.summary`) |
| Пример `hue_mean_norm` | ✓ (JSON) | См. `compact_all.per_dim["0"]` (p01/p50/p99) |

#### §4.4 — Object / debug

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `scenes` | ✓ | **4** сцены на **A** (`store_debug_objects=true`) |
| `frames` | ✓ | Per-scene вложенные фичи |
| `sequence_inputs` | ✓ | Dict `frames` / `scenes` / `global` — compat, см. SCHEMA |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Монотонность | ✓ | `frame_indices`, `times_s` монотонны |

#### §4.6 — Корреляции

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Compact 16×16 | ✓ (JSON) | Корреляции по компактным фичам доступны как `compact_all.corr` (summary) |
| `video_features` (543) | ◐ (JSON, N=5) | В JSON есть `video_features_corr_across_runs` (top‑pairs + summary). **Важно:** при **N=5** интерпретировать осторожно; для стабильных выводов нужен более широкий B |

#### §4.7 — Трактовка

| Наблюдение | Вывод |
|------------|--------|
| Компакт **16** — стабильный вход модели | Хороший контракт для encoder |
| Разрежённая ось (**36** кадров) | Нормально при пересечении сцен; на **B** проверить чувствительность к числу сцен |
| **7** NaN в большом tabular `video_features` | Риск для pipeline, который flatten’ит dict без маски — документировать ключи и причину |

#### §4.8 — Golden **A**

| Критерий | Статус | Заметка |
|----------|--------|---------|
| | ✗ | TODO |

#### §4.9 — Sampling

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Параметры в `meta` | ✓ | `stride=5`, `max_frames_per_scene=350`, `hue_hist_bins=36`, palette-kmeans knobs, `module_sampling_policy_version` |

#### §4.10 — empty / error

| Критерий | Статус | Заметка |
|----------|--------|---------|
| | ✗ | Нужен **C** |

#### §4.11 — Много scalar

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `video_features` **543** | ◐ | На L1 зафиксирован размер; **L2** — кластеризация/корреляции |

#### §4.12 — Anti-leakage

| Вопрос | Ответ |
|--------|--------|
| Локально по видео? | Да |
| Глобальный mean по датасету в модуле? | Нет явного на **A** |
| Онлайн? | `models_used=[]`; часть **NaN**-ключей намекает на опциональные внешние скореры (нужна фиксация) |

#### §5.3 — Models

| Вопрос | Ответ |
|--------|--------|
| Encoder-friendly | **`frame_compact_features (M,16)`** + опционально pooling по `aggregated.frame_compact` |
| Тяжёлый tabular | **`video_features`** — только при явном feature selection |

#### §6 — Verdict

**Итог L2 (A+B, 5 run):** схема и NPZ **совпадают**, строки `manifest.json` для `color_light` имеют `status=ok`, `schema_version=color_light_npz_v2`, `producer_version=2.0.2`. Компактный блок **качественный** и без NaN на всех 5 run; `video_features` стабильно **543** ключа, NaN-ключи стабильны (**7** опциональных скореров).

**Оценка:** **~8.5 / 10** до закрытия **C** и **golden (§4.8)**.

#### §8 — DoD

**Не закрыт:** **C**, golden **§4.8** и полный DoD (§8).

---

## 2. L2 stats (A+B, 5 run) — артефакт

- JSON: `storage/audit_v4/color_light_l2/color_light_audit_v4_stats.json`
- Итог по этим 5 run: **M_total=142**, диапазон **M=18…36**; `video_features` — **543** ключа, NaN-ключи стабильны (**7**).

---

## 1. Снимок **A**

| Величина | Значение |
|----------|----------|
| `M` (кадры на выходе) | 36 |
| `frame_compact_features` | (36, 16), float32 |
| NaN в compact | 0 |
| Сцены в `scenes` | 4 |
| Ключей в `video_features` | 543 |
| NaN в `video_features` | 7 (имена в §4.1a) |
| Имена compact | `hue_mean_norm`, `hue_std_norm`, `hue_entropy_weighted`, `sat_mean_norm`, `val_mean_norm`, `L_mean_norm`, `global_contrast_norm`, `local_contrast_mean_norm`, `colorfulness_norm`, `skin_tone_ratio`, `overexposed_ratio`, `underexposed_ratio`, `vignetting_score_norm`, `soft_light_prob`, `dominant_lab_a_norm`, `dominant_lab_b_norm` |
