# Критерии приёмки: title_to_hashtag_cosine_extractor

**Версия кода:** 1.2.0  
**Дата согласования:** 2026-07-17

## Универсальные хард-гейты

| Гейт | Описание | Статус |
|------|----------|--------|
| U1 | validate rc=0: batch-validate 28/28 NPZ OK | ✅ PASS |
| U2 | Ось времени согласована | N/A — скалярный компонент, нет временных массивов |
| U3 | health_score=1.0, NaN rate=0.0 на ok-NPZ | ✅ PASS |
| U4 | Expected-empty path: нет tp_artifacts → present=0, cosine=NaN (graceful) | ✅ PASS |
| U5 | Golden-детерминизм: max\|Δ\|=0.0 (numpy, CPU-only) | ✅ PASS |
| U6 | Разные длины видео | N/A — компонент работает с фиксированными 1D-эмбеддингами |

## Компонент-специфические критерии

| Критерий | Описание | Статус |
|----------|----------|--------|
| C1 | cosine ∈ [-1,1] при present=1; NaN by design при present=0 | ✅ PASS |
| C2 | NaN rate=0.0 на ok-NPZ (нет неожиданных NaN) | ✅ PASS |
| C3 | Все 5 флагов ∈ {0,1}, finite | ✅ PASS |
| C4 | Консистентность: present=1 ↔ cosine finite | ✅ PASS |

## Примечания

- cosine NaN by design при present=0 (нет tp_artifacts/relpath/файл/dim_mismatch/zero_norm) — не дефект.
- Тестовый корпус — одно видео (-Q6fnPIybEI), cosine ~const 0.85 — проблема датасета, не компонента.
- GPU не нужен (CPU-only numpy).
