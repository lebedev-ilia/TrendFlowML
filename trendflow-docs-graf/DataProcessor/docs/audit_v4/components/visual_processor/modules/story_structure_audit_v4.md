# Audit v4 — `story_structure` (VisualProcessor)

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A+B**, **5** прогонов из `result_store`).  
**Статистика (A+B):** `storage/audit_v4/story_structure_l2/story_structure_audit_v4_stats.json`  
**Артефакты (5 run):** см. `RUN_LOG.md` запись `story_structure` (A+B)  
**Контракт:** [`VisualProcessor/schemas/story_structure_npz_v3.json`](../../../../../VisualProcessor/schemas/story_structure_npz_v3.json) · [`modules/story_structure/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/story_structure/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ось и deps | ✓ | Segmenter **`frame_indices`**, `union_timestamps_sec` ([`SCHEMA.md`](../../../../../VisualProcessor/modules/story_structure/docs/SCHEMA.md)) |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи vs `story_structure_npz_v3` | ✓ | Полное совпадение; **`allow_extra_keys: false`** |
| `manifest.notes` | ✓ | **`null`** на **A** |
| **N**, пары **N−1** | ✓ | **N=48**; **`embedding_sim_next`**, **`embedding_diff_next`**: **(47,)** |
| Пики энергии | ✓ | **P=2**; индексы пиков — **позиции в выборке** `[0, N)` (**13**, **26**) |
| Topic peaks | ✓ | **`topic_shift_peaks_idx`**: **T=0** (пустой массив) — допустимо при отсутствии кривой |
| Табличные **F** | ✓ | **`feature_names`** / **`feature_values`**: **F=22** |

#### §4.1a — Семантика индексов (tabular vs ось)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **`climax_frame_index`** | ◐ | На **A** = **185** при **`frame_indices.max()=337`**, **`N=48`** — это **union/исходный** номер кадра, **не** индекс в `(0…N−1)`; согласуется с **`frame_indices[story_energy_peaks_idx]`** (второй пик **→ 185**) |
| Downstream | ◐ | Энкодеры не должны смешивать **индекс в последовательности** и **union frame index** без явного контракта |

#### §4.2 — NaN / Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Основные кривые | ✓ | **`story_energy_curve`**, **motion**, **`embedding_change_rate_per_sec`**, **`embedding_sim_next` / `embedding_diff_next`**, **`times_s`**: **0%** NaN на **A** |
| **`topic_shift_curve`** | ✓ | **100%** NaN на **A** при **`topic_shift_curve_present=False`** ([`SCHEMA.md`](../../../../../VisualProcessor/modules/story_structure/docs/SCHEMA.md): текст может отсутствовать) |
| **`feature_values`** | ◐ | **0%** NaN на **A**, но **`hook_to_avg_energy_ratio` ≈ −4.2e5** — экстремум, вероятно деление на почти ноль; стоит **клиппить/проверять** в потребителях |
| **`frame_feature_present_ratio`** | ✓ | На **A** константа **0.75** — согласуется с отсутствием finite-ветки **topic_shift** в числителе (доля finite среди объединённых model-facing кривых) |
| Inf | ✓ | **0** в float-массивах на **A** |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `frame_indices` | ✓ | Строго возрастают (**0…337**, **48** точек) |

#### §5.3 — Models

| Вопрос | Ответ |
|--------|-------|
| `meta.models_used` | На **A**: **4** записи |
| `meta.text_mode` | **`ocr_clip_text`** при пустой topic-кривой — полезно трактовать как «путь текста не дал usable curve», а не как silent failure |

#### §6 — Verdict

**Итог L2:** по 5 run (A+B) схема и NPZ **совпадают**, `manifest.status=ok`, `schema_version=story_structure_npz_v3`, `producer_version=3.0.2`.  
Визуальные/CLIP кривые плотные; `topic_shift_curve_present=False` на всех 5 run ⇒ текстовая ветка не проверена (topic peaks всегда пустые).  
`hook_to_avg_energy_ratio` в tabular может быть экстремальным по модулю (на A+B min≈**−8.6e5**, max≈**6.9e5**) — downstream должен быть робастным к таким значениям.

**Оценка:** **~8.1 / 10** на L2.

#### §8 — DoD

**Не закрыт:** C, §4.6, §4.8; сценарий с **`topic_shift_curve_present=True`**.

---

## 1. L2 summary (A+B, 5 run)

По агрегатам JSON:

- **N_total**: **467**
- **F**: **22** на всех
- **P_set**: `[2,4,5]` (число story-energy peaks)
- **topic_shift_curve_present**: `False` на всех 5 (T_set: `[0]`)
- **hook_to_avg_energy_ratio**: min≈**−862862**, max≈**686092**

## 2. Снимок **A** (исторический, L1)

| Величина | Значение |
|----------|----------|
| N | 48 |
| P (energy peaks) | 2 |
| T (topic peak idx) | 0 |
| F (tabular) | 22 |
| `story_energy_peaks_idx` | 13, 26 |
| `frame_indices` @ peaks | 94, 185 |
| `topic_shift_curve_present` | False |
| `any_face_present` true ratio | ~0.625 |
| `hook_to_avg_energy_ratio` | ~−4.2×10⁵ |
---

## Навигация

[Audit v4 hub](../../audit_4_2/README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
