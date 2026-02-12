# ⏳ Baseline Audit — `uniqueness`

Компонент: `DataProcessor/VisualProcessor/modules/uniqueness/`  
Тип: Visual module (Tier‑0 baseline)  
Статус: **✅ CLOSED (baseline)** (2026‑01‑28)  

---

## Резюме

`uniqueness` считает **intra‑video** метрики повторяемости/разнообразия по sampled кадрам, используя **только `core_clip` embeddings**.

Hard deps (no‑fallback):
- `core_clip` (`embeddings.npz`) — эмбеддинги `frame_embeddings` (и `frame_indices`), полностью покрывающие sampling модуля.

---

## Соответствие `BASELINE_COMPONENT_AUDIT_CRITERIA.md`

### 1) Наследование / интерфейсы

- `UniquenessBaselineModule` наследуется от `BaseModule`
- реализует `process(frame_manager, frame_indices, config)`
- `required_dependencies()` → `["core_clip"]`
- `get_models_used()` включает upstream `core_clip.models_used` (best‑effort) для воспроизводимости

### 2) Контракты входа/выхода

- `frame_indices` строго из `frames_dir/metadata.json["uniqueness"]["frame_indices"]` (Segmenter‑owned)
- `times_s` строго из `union_timestamps_sec[frame_indices]` (no‑fallback)
- отсутствие `union_timestamps_sec` / не монотонность / непокрытие `frame_indices` → error
- отсутствие `core_clip` или непокрытие `frame_indices` → error
- safety‑лимит: `N > max_frames` → error (матрица \(N\times N\))

### 3) Per‑run storage + atomic save + validation

- Артефакт: `result_store/<platform>/<video>/<run_id>/uniqueness/uniqueness.npz` (**фиксированное имя**)
- Сохранение атомарное + `validate_npz()` в `BaseModule.save_results()` (fail‑fast)

---

## Артефакт (NPZ)

Путь: `.../uniqueness/uniqueness.npz`

Ключи:
- `frame_indices (N,) int32`
- `times_s (N,) float32`
- `max_sim_to_other (N,) float32`
- `cos_dist_next (N-1,) float32`
- `features` (dict, object-array)
- `meta` (dict, object-array)

Schema:
- `uniqueness_npz_v2`

---

## 4) Progress reporting & stage timings

- Progress events (stage-based): `start`, `load_deps`, `compute`, `save`, `done`
- Stage timings: `summary.stage_timings_ms`

---

## 5) Threshold policy (auto)

Baseline default:
- `repeat_threshold_mode="otsu"` (auto) по распределению `max_sim_to_other`
- safety clamp: `[repeat_threshold_min=0.90 .. repeat_threshold_max=0.99]`
- `repeat_threshold=0.97` используется только если `mode="fixed"`

Evidence (реальный прогон):
- `storage/reports/out/uniqueness_real/result_store/youtube/NSumhkOwSg/38a469df4909/uniqueness/uniqueness_features.npz`
  - `validate_npz`: OK
  - `times_s` monotonic: OK
  - `N=120` (в пределах max_frames=200)

Human-friendly demo (HTML):
- `storage/reports/out/uniqueness_real/demo_uniqueness_quality_20260116-040116-874336.html`

---

## Производительность (resource costs)

Источник правды:
- `docs/models_docs/resource_costs/uniqueness_costs_v1.json`

Evidence:
- `storage/reports/out/checklist-uniqueness/checklist_components_micro_results.json`

Unit:
- `frame` (per sampled frame, CPU module; note: \(O(N^2)\) по N)

---

## Проверка качества (human‑friendly)

Скрипт:
- `scripts/baseline/demo_uniqueness_quality.py`

Evidence (пример HTML):
- `storage/reports/out/uniqueness_real/demo_uniqueness_quality_20260116-040116-874336.html`

---

## Известные ограничения / next steps

- Сложность \(O(N^2)\) ⇒ строго лимитировать sampling (`max_frames<=200`).
- Для production‑use (межвидео “похожесть на топ‑референсы”) нужен отдельный слой с индексом/каталогом референсов; baseline‑модуль про intra‑video.


