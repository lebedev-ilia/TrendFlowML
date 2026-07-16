# Критерии приёмки: source_separation_extractor

Согласованы с владельцем 2026-07-17.

## Универсальные хард-гейты (U1–U6)

| # | Критерий | Порог | Примечание |
|---|---|---|---|
| U1 | validate_source_separation.py rc | =0 (schema + struct) | Для всех NPZ |
| U2 | Ось времени согласована | segment_start/end/center_sec, mask — длины идентичны N | Нет рассинхрона |
| U3 | Различимость | std(share_drums_mean) > 0.01 по ≥3 видео | После фикса |
| U4 | expected-empty path | audio_too_short (<5s) → valid empty NPZ, rc=0 | share_mean=[NaN,NaN,NaN,NaN] by design |
| U5 | Golden-детерминизм | max\|Δshare_mean\| ≤ 0.02 (2 прогона, один файл) | Demucs CPU stochastic by design; порог учитывает это |
| U6 | Разные длины | Короткое / среднее / длинное видео — без падений | |

## Критерии компонента (C1–C4)

| # | Критерий | Порог | Примечание |
|---|---|---|---|
| C1 | share_mean диапазон | ∈[0,1] каждый из 4 источников; сумма ≈ 1 ± 0.001 | Для ok-пути |
| C2 | dominant_source_id | ∈{0,1,2,3} | 0=vocals, 1=drums, 2=bass, 3=other |
| C3 | source_balance_score | ∈[0,1] | Entropy-based нормализованный |
| C4 | audio_too_short NPZ | share_mean=[NaN,NaN,NaN,NaN] by design — ОК | Не считать дефектом |

## Особые исключения (NaN by design)

- `share_mean=[NaN,NaN,NaN,NaN]` при status=empty (audio_too_short / audio_silent / audio_missing) — не дефект
- Golden max|Δ| > 0 (до 2%) — by design (Demucs не полностью детерминирован на CPU при multi-thread)

## Архитектурный фикс (2026-07-17)

Убран `_logmel_to_waveform` (псевдо-wav → нестабильные результаты, max|Δ|=0.32).
Теперь: реальный wav → resample 44100Hz → stereo → apply_model(htdemucs).
