# Audit v4 — `high_level_semantic` (VisualProcessor)

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A+B**, **5** run).  
**Артефакт (A):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/high_level_semantic/high_level_semantic.npz`  
**JSON stats (A+B):** `storage/audit_v4/high_level_semantic_l2/high_level_semantic_audit_v4_stats.json`  
**Контракт:** [`VisualProcessor/schemas/high_level_semantic_npz_v2.json`](../../../../../VisualProcessor/schemas/high_level_semantic_npz_v2.json) · [`modules/high_level_semantic/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/high_level_semantic/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Один NPZ | ✓ | `high_level_semantic.npz` |
| Upstream | ◐ | `core_clip` embeddings, сцены из cut_detection; опционально audio/text флаги в `meta` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи vs `high_level_semantic_npz_v2` | ✓ | Совпадение множеств; **`allow_extra_keys: false`** |
| `manifest.notes` | ✓ | **`null`** на **A** |
| Размеры | ✓ | **N=48**, **S=4**, **D=512**, **F=8**, **E=4**, **T=752** |

#### §4.1a — Семантика `scene_embedding_mean_norm`

| Критерий | Статус | Заметка |
|----------|--------|---------|
| SCHEMA | ✓ | По [`SCHEMA.md`](../../../../../VisualProcessor/modules/high_level_semantic/docs/SCHEMA.md): **норма mean-вектора до L2-нормализации** (quality proxy) |
| На **A** | ✓ | Значения **~0.96–0.98**; при этом строки **`scene_embeddings`** имеют **‖·‖₂ = 1** — согласовано с пайплайном «mean → затем normalize» |

#### §4.2 — NaN / Inf

| Массив | NaN (**A**) | Заметка |
|--------|-------------|---------|
| `scene_embeddings` | 0% | |
| `frame_features` | **~25.5%** элементов | Доминируют колонки **`loudness_dbfs`**, **`tempo_bpm`** — **100%** NaN |
| `text_feature_values` | **~20.7%** | Часть ключей TextProcessor отсутствует / не агрегирована |
| Inf | ✓ | **0%** в проверенных float-массивах |

#### §4.2b — NaN / present_ratio (A+B, 5 run)

На **A+B** (5 run) часть модальностей опциональна по доступности upstream:

- **Аудио**: `loudness_dbfs`, `tempo_bpm` — **100% NaN** на всех 5 run (нет audio‑артефактов / require‑флаги не включены).
- **Эмоции**: `emo_valence/arousal/intensity` могут быть **100% NaN** на части видео (нет лиц / `emotion_face` sparse).
- **Текст**: `text_feature_*` на части run **пустые** (T=0), если `text_processor/text_features.npz` отсутствует.

См. `high_level_semantic_audit_v4_stats.json`: `all_nan_frame_feature_names_union` и `T_set`.

#### §4.1a — Опциональные модальности в `frame_features`

| Колонка | `present_ratio` | NaN по кадрам (**A**) |
|---------|-----------------|-------------------------|
| `clip_sim_prev`, `clip_novelty_prev` | **~0.979** | **~2.1%** |
| `scene_pos_norm` | **1.0** | 0% |
| `loudness_dbfs`, `tempo_bpm` | **0.0** | **100%** |
| `emo_valence`, `emo_arousal`, `emo_intensity` | **1.0** | 0% |

На **A** в `meta`: **`require_audio_loudness`** / **`require_audio_tempo`** / **`require_audio_clap`** не включены (как минимум loudness/tempo не подтянуты) — NaN выглядят **ожидаемо**, не как «тихий баг».

#### §4.3 — Сцены и события (**A**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `scene_id` | ✓ | **4** уникальных id (**0…3**) на **48** кадрах |
| События **E=4** | ◐ | `event_type_id`: **1, 1, 200, 1**; см. `ui.event_type_map` |
| Ось | ✓ | `frame_indices`, `times_s` монотонны |

#### §4.4 — `features` / `ui`

| Ключ | На **A** |
|------|----------|
| `features` | 6 ключей: `n_frames`, `n_scenes`, `clip_sim_prev_mean`, `clip_novelty_prev_mean`, `hard_cuts_count`, `semantic_jump_events_count` |
| `ui` | `event_type_map`, `feature_groups`, `upstream` |

#### §4.11 — Много scalar

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **T=752** | ◐ | **>24** — на **L2** зафиксировано, что на части run `T=0` (нет TextProcessor артефакта), поэтому cross-run корреляции по `text_feature_*` пропущены; для L3 нужен B/C с `T=752` на всех run |

#### §4.12 — Anti-leakage

| Вопрос | Ответ |
|--------|--------|
| Локально по run? | Да (агрегация upstream-артефактов) |
| `models_used` | **3** записи на **A** (не пусто — зафиксировать роли в §5 код/README при необходимости) |

#### §5.3 — Models

| Вопрос | Ответ |
|--------|--------|
| Encoder | **`scene_embeddings (S,D)`** + **`frame_features (N,F)`** с маской/NaN policy; per-frame `scene_id` для alignment |
| Tabular | **`text_feature_values (T,)`** + summary `features` |

#### §6 — Verdict

**Итог L2:** схема и NPZ **совпадают** на **5 run** (A+B), `axis_ok_all=true`. Опциональные модальности дают ожидаемые NaN‑паттерны:

- аудио‑колонки (`loudness_dbfs`, `tempo_bpm`) — **полностью NaN** на всех 5 run (нет upstream / require‑флаги не включены);
- эмо‑колонки (`emo_*`) — могут быть полностью NaN на части видео (face‑sparse);
- текстовый блок на части run отсутствует (T=0), что делает cross-run корреляции по `text_feature_*` нерепрезентативными без целевого B/C.

**Оценка:** **~8.6 / 10** на L2 (не `passed` до L3/§8).

#### §8 — DoD

**Не закрыт:** C, §4.8 (golden), корреляции/отбор по **T** (требует набора с `T>0` на всех run).

---

## 1. Снимок **A**

| Величина | Значение |
|----------|----------|
| N / S / D | 48 / 4 / 512 |
| F / E / T | 8 / 4 / 752 |
| `scene_embeddings` ‖row‖₂ | 1.0 |
| `scene_embedding_mean_norm` | ~0.96–0.98 |
| `frame_features` NaN (global) | ~25.5% |
| `text_feature_values` NaN | ~20.7% |
| `meta.processed_frames` | 48 |
| `meta.require_text_processor` | false (на **A**) |

---

## 4.3b — L2 stats (A+B, 5 run)

- **JSON**: `storage/audit_v4/high_level_semantic_l2/high_level_semantic_audit_v4_stats.json`
- **Итоги**:
  - **N_total**: **543**
  - **S**: **2…8** (зависит от видео)
  - **D**: **512** (стабильно)
  - **F**: **8** (стабильно)
  - **T**: `0 | 752` (зависит от наличия `text_processor/text_features.npz`)
