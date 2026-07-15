# Прогресс валидации — color_light

## Состояние (2026-07-12)
- Компонент: color_light (CPU-only: numpy/opencv/sklearn/scipy). Hard dep: scene_classification (ось Segmenter).
- CRITERIA.md согласован (2026-07-12): U1–U6 + C1–C4. REPORT ещё нет.
- Под: 81ermplh4hysmt RUNNING, RTX 2000 Ada, ssh 213.173.99.24:25543 (порт из pod_control, НЕ 24135).
  venv /workspace/venv ok, видео /workspace/scene_videos ok.

## Сделано
- Прочитан онбординг, DECISIONS_AND_LESSONS, CRITERIA.md, валидатор validate_color_light.py.
- Профиль: configs/audit_v3/visual/visual_color_light_only.yaml (color_light=true, deps: core_clip,
  core_optical_flow, cut_detection, scene_classification).

## Результаты (2026-07-12, продолжение)
- НАЙДЕН+ПОДТВЕРЖДЁН БАГ (C4): color_distribution_gini=NaN, entropy≈0 в cl_run1/cl_gold — старый код
  `frame.get(key)`→default 0 → sum(hue)=0 → gini NaN, вырожд.гистограмма → entropy≈0.
  Фикс `getf` (читает frame["features"][hue_mean]) уже в processor.py (локально git = M, uncommitted).
  Переигровка color_light на deps cl_run1 с фиксом: gini=0.0726, entropy=2.489, NaN keys=6 (только by-design).
- U1 валидатор rc=0 на всех NPZ (cl_run1, cl_gold, fixed, short, empty). U2 оси неубыв. U3 health ок.
- U4 expected-empty (синтетика: сдвиг индексов сцен +1e6): status=empty, after_filt_empty, compact (0,16)
  float32, len=0, все ключи, validator rc=0. PASS.
- U5/C3 golden: multi-thread BLAS → 2/2128 элем на 1 ULP (1.19e-7); OMP_NUM_THREADS=1 → БИТ-ИДЕНТИЧНО, max|Δ|=0.
- vshort (23f): rc=0, NaN/Inf=0, 13/16 dims std>0, gini=0.00044, 6 NaN by-design.
- C2 intervideo: v1short(133f, gini .073) vs vshort(23f, gini .00044) — вариация есть.

## Финал (2026-07-12)
- vlong (250f, 92.6с): rc=0, status=ok, NaN/Inf=0, 16/16 std>0, gini=0.106, entropy=1.997, 6 NaN by-design.
- U6 PASS (23/133/250 кадров). ВСЕ гейты U1–U6 и критерии C1–C4 PASS.
- REPORT_2026-07-12.md написан. DECISIONS дополнен (3 урока). 
- ИТОГ отправлен владельцу: вердикт «штамповать при условии коммита getf-фикса в git».
- ✅ ЗАВЕРШЕНО. Владелец одобрил штамп. Фикс закоммичен: ветка fix/color_light-hue-extraction, commit f99742e.
- ✅ Штамп поставлен в COMPONENT_VALIDATION_CHECKLIST.md (v2, 07-12).
- Пост-коммит validate rc=0 на всех NPZ. Под погашен.
- ОСТАТОК для прода (не блокер): (1) смёржить ветку fix/... в main; (2) в k8s OMP_NUM_THREADS=1.

## Грабли/заметки
- Порт SSH берём из pod_control status (25543), выданный владельцем 24135 — закрыт.
