# Критерии приёмки: quality_extractor

**Согласованы:** 2026-07-16  
**Компонент:** quality_extractor (AudioProcessor, CPU-only numpy)  
**Схема:** quality_extractor_npz_v2

---

## Универсальные гейты (U1–U6)

| ID | Критерий | Метод |
|----|----------|-------|
| U1 | validate_quality.py → rc=0 на всех NPZ | `python validate_quality.py <npz> --struct` |
| U2 | Ось времени: segment_start/end/center_sec одинаковой длины N, segment_mask bool | `--struct` |
| U3 | feature_values finite на ok-NPZ (при enable_basic_metrics=True: dc_offset/clipping_ratio/crest_factor_db конечны) | numpy isfinite |
| U4 | Expected-empty: segments=[] → status=empty, rc=0, empty_reason="quality_all_segments_empty" | синтетический тест |
| U5 | Golden-детерминизм: CPU-only numpy, OMP_NUM_THREADS=1 → max\|Δ\|=0.0 | 10 прогонов одного NPZ |
| U6 | Разные длины видео → все rc=0 (storage содержит разные видео) | валидатор по всем NPZ |

## Компонент-специфичные критерии (C1–C4)

> Применяются при **enable_basic_metrics=True** (дефолт).
> enable_dynamic_metrics и enable_frame_analysis — опциональные, в обязательные не входят.

| ID | Критерий | Порог |
|----|----------|-------|
| C1 | clipping_ratio ∈ [0,1] на всех ok-NPZ | 0 ≤ val ≤ 1.0 |
| C2 | quality_score ∈ [0,1] на всех ok-NPZ | 0 ≤ val ≤ 1.0 |
| C3 | std(crest_factor_db) по корпусу > 0 (различимость) | std > 0 |
| C4 | Empty-path возвращает все ключи схемы (feature_names/feature_values/segment_*/meta) | ключи присутствуют |
