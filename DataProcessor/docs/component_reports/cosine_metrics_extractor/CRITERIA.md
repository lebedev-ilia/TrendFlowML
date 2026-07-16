# cosine_metrics_extractor — Критерии приёмки

**Согласовано:** 2026-07-17  
**Компонент:** CosineMetricsExtractor v1.3.0  
**Схема:** cosine_metrics_extractor_output_v1 (39 ключей tp_cos_*)  
**Тип:** CPU-only, numpy/math, нет нейросети, нет GPU

---

## Универсальные хард-гейты (U1–U6)

| ID | Критерий | Pass |
|----|----------|------|
| U1 | Валидатор `validate_cosine_metrics_extractor_text_npz.py --struct --ranges` → rc=0 на ≥3 NPZ разных видео | rc=0 |
| U2 | Ось времени — НЕ ПРИМЕНИМО (скалярный компонент, нет seq/time) | N/A (ok by design) |
| U3 | Все 5 косинусов finite и ∈[-1,1] при `*_present=1`; NaN только при `*_present=0` | finite при present=1, NaN при absent=1 |
| U4 | Expected-empty: нет title → `tp_cos_empty_no_title=1`, `tp_cos_title_desc=NaN`; нет всех → все 5 косинусов NaN, валидатор rc=0 | rc=0, флаги корректны |
| U5 | Golden-детерминизм: 2 прогона одного входа → diff=0 (numpy/math, нет GPU/random) | max\|Δ\|=0 |
| U6 | Разные наборы входов (только title+desc, только transcript, все четыре, matrix-mode) — без исключений | без exceptions |

---

## Критерии компонента (C1–C4)

| ID | Критерий | Порог |
|----|----------|-------|
| C1 | Валидатор batch на 28 NPZ из `storage/result_store` → rc=0 (при status=error — допустим пустой срез) | rc=0 или пустой срез при status!=ok |
| C2 | Косинусные пары на ok-записях ∈[0.6, 1.0] (подтверждено эмпирически на ≥3 видео; при < 0.6 пересмотреть порог) | ≥0.6 на ok-видео |
| C3 | One-hot `transcript_agg_source_*`: сумма=0 (нет транскрипта) или =1 (есть) — никогда >1 | сумма ∈{0.0, 1.0} |
| C4 | NaN by design = ровно 5 полей при `emit_extra_metrics=False`: `tp_cos_load_ms`, `tp_cos_compute_ms`, `tp_cos_tc_n_comments_used`, `tp_cos_tc_sims_std`, `tp_cos_tc_sims_p95` | ровно 5 NaN, остальные finite/0/1 |

---

## NaN by design (явные исключения, не дефекты)

- `tp_cos_load_ms`, `tp_cos_compute_ms` при `emit_extra_metrics=False` → NaN ✓
- `tp_cos_tc_n_comments_used`, `tp_cos_tc_sims_std`, `tp_cos_tc_sims_p95` при `emit_extra_metrics=False` → NaN ✓
- Любой косинус когда один или оба вектора absent (`*_present=0`) → NaN ✓
- `tp_cos_tc_sims_std/p95/n_comments_used` при `comments_mode=aggregates` и `emit_extra_metrics=True` → NaN (matrix-only extras) ✓
