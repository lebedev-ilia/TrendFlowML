# Audit v4 — `shot_quality` (VisualProcessor)

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A+B**, **5** прогонов из `result_store`).  
**Статистика (A+B):** `storage/audit_v4/shot_quality_l2/shot_quality_audit_v4_stats.json`  
**Артефакты (5 run):** см. `RUN_LOG.md` запись `shot_quality` (A+B)  
**Контракт:** [`VisualProcessor/schemas/shot_quality_npz_v3.json`](../../../../../VisualProcessor/schemas/shot_quality_npz_v3.json) · [`modules/shot_quality/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/shot_quality/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Hard deps | ✓ | `core_clip`, `core_depth_midas`, `core_object_detections`, `core_face_landmarks`, `cut_detection`; ось Segmenter (`frame_indices`) ([`SCHEMA.md`](../../../../../VisualProcessor/modules/shot_quality/docs/SCHEMA.md)) |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи vs `shot_quality_npz_v3` | ✓ | Полное совпадение; **`allow_extra_keys: false`** |
| `manifest.notes` | ✓ | **`null`** на **A** |
| **N, S, F, P, K** | ✓ | **N=48**, **S=4**, **F=48**, **P=10**, **K=3** |
| `shot_ids` vs `shot_frame_count` | ✓ | Согласованы для всех шотов |

#### §4.1a — `quality_probs` и shot top-k

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `quality_probs` сумма по строке | ✓ | На **A**: **≈1.0** (float16; min/max в пределах **~1±1e-4**) — согласуется с zero-shot softmax по **P** классам |
| `shot_quality_topk_probs` сумма | ◐ | На **A**: **~0.31** на строку — это **не** полное распределение, а **среднее по кадрам** для top-**K** меток; энкодеру не стоит ожидать **∑=1** без явного контракта |
| NaN | ✓ | **0%** в `quality_probs` и в shot-level вероятностях/ids на **A** |

#### §4.2 — NaN / Inf (`frame_features`)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Доля NaN (агрегат по ячейкам) | ◐ | **~9.9%** — концентрируется в **6** из **48** признаков |
| Полностью NaN (100% кадров) | ◐ | **4** столбца: **`vignetting_level`**, **`chromatic_aberration_level`**, **`lens_sharpness_drop_off`**, **`rolling_shutter_artifacts_score`** — вероятно недоступны/заглушены в текущем пайплайне |
| Частично NaN (face-ROI) | ✓ | **`face_sharpness_tenengrad`**, **`face_noise_level_luma`** — **37.5%** NaN на **A**, согласуется с допустимой пустотой лиц в [`SCHEMA.md`](../../../../../VisualProcessor/modules/shot_quality/docs/SCHEMA.md) |
| Inf | ✓ | **0** в float-массивах на **A** |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `frame_indices` | ✓ | Строго возрастают (sorted+unique) |
| `times_s` | ✓ | **0%** NaN, выровнены по **N** |

#### §4.12 — Anti-leakage

| Вопрос | Ответ |
|--------|-------|
| Зависимости только из `core_*` / cut | Да; без «лейков» таргета вне пайплайна |

#### §5.3 — Models

| Вопрос | Ответ |
|--------|-------|
| `meta.models_used` | ✓ (после фикса) | В NPZ добавляется запись для **`impl_meta.clip_model_name`** (upstream `core_clip`); `model_signature` пересчитывается в `save_results` |
| `impl_meta` | Есть **`shot_quality_prompts_sha256`**, **`shot_boundaries_source`**, **`faces_available`** / **`faces_empty_reason`** |

#### §6 — Verdict

**Итог L2:** по 5 run (A+B) схема и NPZ **совпадают**, `manifest.status=ok`, `schema_version=shot_quality_npz_v3`, `producer_version=2.0.2`.  
`quality_probs` ведут себя как softmax по **P=10** (сумма по строке стабильно **≈1**, min≈**0.9998169**, max≈**1.000122**).  
`shot_quality_topk_probs` **не** обязаны суммироваться в 1 (это per-shot mean по кадрам top‑K), на A+B сумма по строке стабильно **~0.304…0.308**.  
Fully‑NaN «линзовые» признаки стабильны на A+B (**4**): `vignetting_level`, `chromatic_aberration_level`, `lens_sharpness_drop_off`, `rolling_shutter_artifacts_score` (см. JSON stats).

**Оценка:** **~8.2 / 10** на L2.

#### §8 — DoD

**Не закрыт:** C, §4.6, §4.8.

---

## 1. L2 summary (A+B, 5 run)

По агрегатам JSON:

- **N_total**: **543**
- **S_set**: `[2,4,6,8]`
- **F**: **48** на всех
- **P**: **10** на всех
- **K**: **3** на всех
- **`quality_probs` ∑ по строке**: min≈**0.9998169**, max≈**1.000122**
- **`shot_quality_topk_probs` ∑ по строке**: min≈**0.3037**, max≈**0.3084**
- **fully‑NaN фичи (union)**: `vignetting_level`, `chromatic_aberration_level`, `lens_sharpness_drop_off`, `rolling_shutter_artifacts_score`

## 2. Снимок **A** (исторический, L1)

| Величина | Значение |
|----------|----------|
| N | 48 |
| S | 4 |
| F | 48 |
| P | 10 |
| K | 3 |
| `quality_probs` ∑ (min, max) по строке | ~0.9999, ~1.0001 |
| `shot_quality_topk_probs` ∑ (min, max) по строке | ~0.307, ~0.308 |
| Признаки с любым NaN в `frame_features` | 6 |
