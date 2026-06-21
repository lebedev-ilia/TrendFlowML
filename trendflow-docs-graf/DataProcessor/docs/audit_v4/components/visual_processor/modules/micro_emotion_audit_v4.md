# Audit v4 — `micro_emotion` (VisualProcessor)

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (A+B; фактически **4 OK** NPZ + **1 error** в B).  
**Артефакт (A):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/micro_emotion/micro_emotion.npz`  
**JSON stats (OK артефакты):** `storage/audit_v4/micro_emotion_l2/micro_emotion_audit_v4_stats.json`  
**Контракт:** [`VisualProcessor/schemas/micro_emotion_npz_v3.json`](../../../../../VisualProcessor/schemas/micro_emotion_npz_v3.json) · [`modules/micro_emotion/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/micro_emotion/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Один NPZ | ✓ | `micro_emotion.npz` |
| Инфраструктура | ◐ | OpenFace в Docker ([`SCHEMA.md`](../../../../../VisualProcessor/modules/micro_emotion/docs/SCHEMA.md)); на **A** в meta указаны `docker_image`, `device=cuda` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи vs `micro_emotion_npz_v3` | ✓ | Совпадение; **`allow_extra_keys: false`** |
| `manifest.notes` | ✓ | **`null`** |
| Размеры | ✓ | **N=250**, **F=21**, **`compact22` (250,22)**, **V=75**, **K=0** событий |

#### §4.1a — Маска лица и NaN

| Критерий | Статус | Заметка |
|----------|--------|---------|
| SCHEMA | ✓ | Вне лица → **NaN** в per-frame числах, **`face_present_any=false`** |
| На **A** | ✓ | **`face_present_any` true: 7.6%** (**19** кадров) |
| `frame_features` при лице | ✓ | NaN в ячейках **~4.8%** при **`face_present_any`** — **1** кадр с частичным NaN |
| Сводка | ✓ | `summary`: **`frames_with_face=19`**, **`frames_processed_openface=18`** — один face-кадр без успешного OpenFace |
| `compact22` | ✓ | **~92.8%** NaN глобально (ожидаемо); при лице почти плотно, **1** проблемный ряд |

#### §4.2 — Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `feature_values`, `frame_features`, `compact22` | ✓ | **0** Inf на **A** |

#### §4.3 — Video-level `feature_values` (**V=75**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| NaN | ◐ | **4** из **75** полей **NaN**: `landmark_visibility_reliable`, `occlusion_flag`, `au_pca_var_explained_4`, `au_pca_var_explained_5` — прозрачно для downstream §4.1a |
| Доля | ✓ | **~5.3%** элементов вектора |

#### §4.4 — События

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **K=0** | ◐ | Массивы `event_*` пустые; на **A** нет зарегистрированных micro-expression peaks (пороги/контент) — валидно при **пустой** оси событий |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `frame_indices`, `times_s` | ✓ | Монотонны |

#### §4.11 — Много scalar

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **V=75** | ◐ | **>24** — на **L2** добавлены корреляции video-vector по доступным OK NPZ (4 run); полный L2 по ≥5 требует закрыть `-Ga4edhrfog` |

#### §4.12 — Anti-leakage / детерминизм

| Вопрос | Ответ |
|--------|--------|
| Внешний сервис? | **Docker OpenFace** — детерминизм и версия image критичны; зафиксировать в run metadata |
| `models_used` | **0** на **A** при наличии внешнего движка — не путать с «нет модели» |

#### §5.3 — Models

| Вопрос | Ответ |
|--------|--------|
| Encoder | **`compact22 (N,22)`** + маска **`face_present_any`** и finite-check по строкам |
| Tabular | **`feature_values (V,)`** + `microexpr_features` dict |

#### §6 — Verdict

**Итог L2:** по **4 OK** NPZ (A + 3/4 из B) схема и артефакты **совпадают**, агрегаты устойчивы; корреляции video-vector (V=75) собраны в JSON. Один run набора B (`-Ga4edhrfog/e2dc8851-…`) завершился **`status=error`** (нет NPZ) из-за PCA `n_components` при малом числе samples — требуется добить B (≥5 OK) для полного L2.

**Оценка:** **~8.6 / 10** на L2 (частично blocked до 5 OK run).

#### §8 — DoD

**Не закрыт:** B (≥5 OK), C, §4.8.

---

## 1. Снимок **A**

| Величина | Значение |
|----------|----------|
| N | 250 |
| F (wide) | 21 |
| OpenFace обработано | 18 / 19 face-кадров |
| `frame_features` NaN (все ячейки) | ~84% |
| `compact22` NaN (все ячейки) | ~92.8% |
| K (micro-events) | 0 |
| V (video scalars) | 75 |

---

## 4.3b — L2 stats (A+B)

- **JSON**: `storage/audit_v4/micro_emotion_l2/micro_emotion_audit_v4_stats.json`
- **Итоги по OK артефактам**: `n_runs=4`, **N_total=1000**, `face_present_any` True **70** (**7%**), `K_total=2`, `video_feature_values_nan_total=16`.
- **B-run error** (нет NPZ): `youtube / -Ga4edhrfog / e2dc8851-6c51-43c0-9757-3c0fed803348` → PCA `n_components=3` при `min(n_samples,n_features)=2` (см. `manifest.json`).
---

## Навигация

[Audit v4 hub](../../audit_4_2/README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
