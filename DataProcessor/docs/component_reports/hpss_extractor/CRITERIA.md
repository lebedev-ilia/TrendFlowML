# Критерии приёмки: hpss_extractor

**Дата:** 2026-07-16  
**Компонент:** hpss_extractor (AudioProcessor, librosa HPSS, CPU-only)

---

## Универсальные хард-гейты (U1–U6)

| # | Критерий | Порог / ожидание |
|---|---|---|
| U1 | validate_hpss.py rc=0 (schema + --struct) | rc=0 для ok-NPZ; rc=0 для empty-NPZ (N=0 — ожидаемо) |
| U2 | Ось времени: segment_start/end/center_sec | Не убывает, конечны, согласованы (все 6 полей единой длины N) |
| U3 | Finite/health: feature_values | nan_rate=0.0 для ok-NPZ; F=0 при empty (no NaN) |
| U4 | Expected-empty | status=empty при audio_missing_or_extract_failed: F=0, N=0, rc=0 |
| U5 | Golden (повторяемость) | При одинаковом config_hash: max\|Δ(harmonic_share_by_segment)\|=0.0 |
| U6 | Разные длины видео | N растёт с длиной (видео 5..30 сегментов) |

---

## Компонентно-специфичные критерии (hpss)

| # | Критерий | Порог / политика |
|---|---|---|
| C1 | hpss_harmonic_share ∈ [0,1] | Всегда, все ok-NPZ |
| C2 | hpss_percussive_share ∈ [0,1] | Всегда, все ok-NPZ |
| C3 | harmonic_share + percussive_share > 1 при margin≥2 | **by design** (librosa масштабирует маски при margin>1) — NOT FAIL |
| C4 | hpss_separation_quality ∈ [0,1] | После фикса clip(0,1): не должна уходить в отрицат. |
| C5 | hpss_dominance ∈ {harmonic, percussive, mixed} | Строковое поле в meta — проверять при наличии |
| C6 | segment_mask — все True | mask_ok=True на корпусе (нет failed сегментов) |

---

## Исключения и NaN-политика

- `hpss_harmonic_share_by_segment`, `hpss_percussive_share_by_segment`: NaN **допустим** для failed-сегментов (by design, задокументировано в main.py), на текущем корпусе mask_ok=True у всех.
- `hpss_separation_quality`: после фикса clip(0,1) — в диапазоне. Значение 0.0 возможно при margin≥2 (не ошибка, а артефакт перекрывающихся масок).
- **sum(h+p) > 1** по сегментам при margin=2.0 — **ожидаемое поведение** librosa (маски масштабируются), не дефект.

---

## Seq-фичи для Encoder (модельная пригодность)

| Поле | Тип | Token stream | В NPZ? | Ось времени |
|---|---|---|---|---|
| hpss_harmonic_share_by_segment | seq | dense per-segment | ✅ | segment_center_sec |
| hpss_percussive_share_by_segment | seq | dense per-segment | ✅ | segment_center_sec |
| feature_values (агрегаты) | agg | — | ✅ (model_facing) | — |

---

## Тайминги (CPU-only, без пода)

- Стадия process_segments_ms: 671–2971 мс/видео при N=5 (зависит от версии librosa/OS)
- GPU не нужен.
