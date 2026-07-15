# Критерии приёмки: core_optical_flow (согласовано с владельцем 2026-07-07)

## Решения владельца
1. Инфра — **Вариант A**: inprocess-обход через torchvision `raft_small` (Triton-free, идентичный v3-артефакт),
   добавить `--runtime inprocess` в main.py по аналогии с core_depth_midas.
2. golden cam_* — **фиксировать RANSAC-seed** (cv2.setRNGSeed) → детерминизм, diff=0.
3. Пресеты — **попробовать все** (raft_256 / raft_384 / raft_512), сравнить качество и скорость.

## Универсальные хард-гейты (pass/fail)
- H1. Валидатор выхода rc=0: `validate_core_optical_flow_npz.py --struct --ranges` (и --qa, если конфиг есть).
- H2. Ось времени: `times_s ⊆ union_timestamps_sec`, неубывающая; `dt_seconds[0]=NaN`, `dt[1:]>0` где finite.
- H3. finite/health: `motion_norm_per_sec_mean` finite, не константа на корпусе; корректные shape/dtype (N,).
- H4. Expected-empty: контракт — empty недопустим; при `<2` кадров компонент даёт корректный **error** (не падение мусором).
- H5. golden-детерминизм (см. C4).
- H6. Разные длины видео (матрица §1 протокола) отрабатывают без ошибок.

## Критерии под компонент
- **C1 (различимость motion):** по корпусу CV per-video среднего `motion_norm_per_sec_mean` > 0.3;
  внутри динамичных роликов `p95/median` motion ≥ 2.0.
- **C2 (разделяющая способность):** медиана motion на статике (talking-head/слайды) заметно ниже, чем
  на экшн/динамике — соотношение медиан ≥ 2×.
- **C3 (диапазоны/согласованность), на 100% finite-значений:** `flow_dir_dispersion`,`bg_ratio`,`flow_consistency`∈[0,1];
  `flow_dir_sin/cos_mean`∈[−1,1]; `flow_consistency ≈ 1/(1+flow_div_abs_mean)` (допуск 5e-3); `cam_affine_scale`≥0.
- **C4 (golden):** между двумя прогонами одного ролика diff=0 для детерминированных массивов
  (`frame_indices`, `times_s`, `motion_norm_per_sec_mean`, `flow_mag_std/p95`, `flow_dx/dy_mean`, `flow_dir_*`,
  `flow_div_abs_mean`, `flow_consistency`). Для `cam_*` — после фиксации RANSAC-seed diff=0 тоже.

## Замечания
- ДЕФЕКТ (в REPORT): batch-путь `core_optical_flow_batch.py` не пишет audit-v3 per-frame фичи →
  batch-NPZ провалит структурный валидатор. Для валидации используется per-video main.py; batch надо
  синхронизировать перед прод-масштабом.
- Пресеты сравниваются по качеству (motion-кривая, разделяющая способность) и скорости (ms/pair).
