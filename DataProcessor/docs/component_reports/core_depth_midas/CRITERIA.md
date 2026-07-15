# Критерии приёмки: core_depth_midas (согласовано с владельцем 2026-07-06)

Компонент: VisualProcessor core provider, Tier-0 карты глубины MiDaS.
Рантайм валидации: **inprocess** torch.hub MiDaS_small, out 256×256 (Triton-free).
Golden: **на GPU** (по решению владельца).

## Универсальные хард-гейты (pass/fail)
- U1. Валидатор `validate_core_depth_midas_npz.py --struct --ranges --qa` → rc=0 на всех роликах.
- U2. Ось времени: `times_s` = `union_timestamps_sec[frame_indices]`, неубывающая, `len(times_s)=len(frame_indices)=N`.
- U3. Health/finite: `depth_maps` finite (nan_rate=0), не константа; выходы finite по документированной политике.
- U4. Expected-empty: **by-design нет** — глубина существует для любого кадра (владелец подтвердил). No-fallback: при пустых `frame_indices` компонент падает штатно — это корректное поведение, не valid-empty.
- U5. Golden-детерминизм: 2 прогона на GPU, зафиксировать (см. C4).
- U6. Разные длины видео (~10с…8мин+) отрабатывают без ошибок.

## Критерии под компонент
- C1. `depth_maps_norm` и `preview_depth_maps_norm` ∈ [0,1] (finite); nan_rate raw `depth_maps` = 0 на всех кадрах всех роликов.
- C2. `depth_complexity_score` ∈ [0,1] и НЕ константа по корпусу. **Порог переопределён 2026-07-06 (владелец ОК):**
  масштабо-инвариантный `CV(complexity между роликами) ≥ 0.10` (факт 0.188). Исходный абсолютный `std > 0.01`
  был мискалиброван (метрика = средний \|градиент\| норм.[0,1] карты 256², естественная шкала ~0.004, факт std=0.00076).
- C3. `foreground_background_separation_proxy` и `depth_range_robust` — finite и различимы между роликами (не константа).
- C4. Golden на GPU: побайтовая идентичность массивов depth желательна; при недетерминизме bicubic-интерполяции допускается max|Δ| ≤ 1e-3 по `depth_maps` (фиксируем фактическое значение как «GPU-стохастичность»).
- C5. `preview_frame_indices` ⊆ `frame_indices`; `meta.preview_k` = `len(preview_frame_indices)`; preview карты H,W = depth_maps H,W.

## Решения по открытым вопросам (владелец: «реши сам»)
- Seq для Encoder: валидируем ОБЕ ветки — per-frame агрегаты как dense time-series (mean/std/p05/p95/range_robust/complexity/fg_bg_proxy) И `depth_maps_norm` (model_facing тензор карт). Полные карты (N,256,256) тяжелы для трансформер-Encoder — в отчёте отметим, что практическая seq для модели = per-frame агрегаты + опц. downsampled карты; карты также идут в backend/shot_quality.
