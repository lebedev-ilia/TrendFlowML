# band_energy_extractor — критерии приёмки (согласованы 2026-07-16)

## Универсальные хард-гейты (U1–U6)

| Гейт | Критерий | Ожидание |
|------|----------|----------|
| U1 | Валидатор rc=0 (validate_band_energy.py --struct) | rc=0 на всех NPZ (ok И empty) |
| U2 | Ось времени / duration | duration присутствует в meta для ok-NPZ; при time_series=False — N/A для segment_centers |
| U3 | Finite/health | 0 NaN/Inf в band_energy_shares для ok-NPZ; не константа на корпусе |
| U4 | Expected-empty путь | NPZ с status=empty валиден структурно (rc=0), keys присутствуют с корректными shape |
| U5 | Golden-детерминизм | max\|Δ\| ≤ 1e-10 при повторном прогоне (librosa детерминирован) |
| U6 | Разные длины | Видео 3с–30с+ отрабатывают без падений |

## Специфичные критерии (C1–C4)

| Критерий | Порог | Обоснование |
|----------|-------|-------------|
| C1 | sum(band_energy_shares) ∈ [0.99, 1.01] для всех ok-NPZ | Внутренняя валидация в main.py; нарушение = баг нормализации |
| C2 | 0 NaN/Inf в band_energy_shares для ok-NPZ | Shares finite — основной контракт; NaN допустимы ТОЛЬКО при status=empty |
| C3 | std(band_share_mid) > 0.05 на корпусе ≥5 ok-NPZ | Различимость сигнала (факт: std=0.23 на 15 видео) |
| C4 | empty-path: band_energy_shares shape (3,) с np.nan; band_edges_hz shape (3,2) с np.nan | Фиксированный shape — downstream ожидает [3] всегда |

## Примечания

- **NaN by design**: в band_energy_shares NaN допустимы ТОЛЬКО при status=empty (C4 выше). Для ok — никогда.
- **balance_metrics**: опциональны (feature-gated), в текущих NPZ не включены. При включении проверять finite.
- **time_series**: опционально, в текущих NPZ не включены. При включении проверять alignment segment_mask.
- **Golden-метод**: сравнение на 10 прогонах video_id=-Q6fnPIybEI, max\|Δ\|≤3.4e-13 (фактически 0).
