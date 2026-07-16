# CRITERIA.md — comments_aggregator

Версия компонента: CommentsAggregationExtractor v1.3.0  
Дата согласования: 2026-07-17 (авто-штамп при 100% PASS)

## Универсальные хард-гейты (U1–U6)

| Гейт | Описание | Статус |
|------|----------|--------|
| U1 | Валидатор rc=0 (validate_comments_aggregator_text_npz.py --struct --ranges) | PASS |
| U2 | Ось времени n/a — текстовый компонент, нет временной последовательности | PASS |
| U3 | Числа корректны: present∈{0,1}, count≥0, dim>0 при present=1, std≥0; зеркала согласованы | PASS |
| U4 | Expected-empty: при отсутствии embeddings → present=0.0, dim=NaN, rc=0, нет exception | PASS |
| U5 | Golden: семантические поля max\|Δ\|=0.0; timing-поля (tp_commentsagg_agg_mean_ms/agg_median_ms) excluded by design (измеряют CPU-время) | PASS |
| U6 | Разные видео: 5 разных video_id + 22 ok-прогона отработали без падений | PASS |

## Компонентные критерии

| Критерий | Порог / условие | Статус |
|----------|----------------|--------|
| C1 | dim=1024.0 для всех ok-NPZ (соответствует модели multilingual-e5-large) | PASS |
| C2 | mean_std и median_std ∈ [0, ∞) при compute_std=True; NaN при compute_std=False — by design | PASS |
| C3 | Зеркала согласованы: tp_commentsagg_* = tp_comments_agg_* = tp_cagg_* (present/count/dim/std) | PASS |
| C4 | agg_mean_ms и agg_median_ms excluded from golden (timing, не семантика) | PASS |

## NaN by design

- `tp_commentsagg_dim=NaN` при expected-empty (нет embeddings) — норма, не дефект.
- `tp_commentsagg_mean_std=NaN` и `tp_commentsagg_median_std=NaN` при `compute_std=False` — норма.
- `tp_commentsagg_agg_mean_ms=NaN` при `emit_extra_metrics=False` или `compute_mean=False` — норма.
- error-NPZ (status=error): пустой срез comments_aggregator — компонент не запускался при ошибке TextProcessor, это expected (rc=0 для валидатора).
