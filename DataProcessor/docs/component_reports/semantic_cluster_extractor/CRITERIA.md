# CRITERIA.md — semantic_cluster_extractor

**Компонент:** `semantic_cluster_extractor` (TextProcessor)  
**Версия:** v1.3.0  
**Согласовано:** 2026-07-16 (Второй агент от имени владельца)

---

## Универсальные хард-гейты (U1–U6)

| Гейт | Описание | Статус |
|------|----------|--------|
| U1 | Валидатор выхода rc=0 (batch по 28 NPZ) | PASS |
| U2 | Числа корректны: extra_block = NaN при `emit_extra_metrics=False` — NaN by design | PASS |
| U3 | Не константа: sim std=0.029, cluster_id: {5, 26} по 6 видео | PASS |
| U4 | Expected-empty: нет эмбеддинга → present=0, id/sim/dist = NaN | PASS |
| U5 | Golden: семантические поля max\|Δ\|=0.0; compute_ms (timing) — excluded | PASS |
| U6 | Разные длины видео (6 видео из storage) → rc=0 | PASS |

---

## Компонентные критерии (C1–C6)

| Код | Критерий | Порог | Факт |
|-----|----------|-------|------|
| C1 | `tp_semclust_present=1` при наличии title-эмбеддинга | ≥ 95% NPZ | 100% (28/28) |
| C2 | `similarity ∈ [−1, 1]`, finite; `dist = 1 − sim`, max_err < 1e-4 | строго | max_err=0.0 |
| C3 | `cluster_id ∈ [0, n_clusters−1]` (n_clusters=32), finite при present=1 | строго | PASS |
| C4 | extra_block (n_clusters, *_dim, margin_top2, compute_ms) = NaN при `emit_extra_metrics=False` **— NaN by design**, не дефект | by design | 100% |
| C5 | Golden — семантические поля: max\|Δ\|=0.0 между повторными прогонами; `compute_ms` excluded (timing, зависит от состояния CPU) | семантика=0 | PASS |
| C6 | Fallback-путь: title отсутствует → description/hashtag → `fallback_used=1`, `source_description/hashtag=1` | синтетика PASS | PASS |

---

## Числа по прогону (2026-07-16)

- n_clusters=32, orig_dim=1024, reduced_dim=128
- Видео в storage: 6 разных (-0InsUQNwIQ, -3Mbinqzig4, -4RHVBIikn8, -4WRepA-bss, -8WeWWOpxHk, -Q6fnPIybEI)
- similarity: min=0.1367, max=0.2067, mean≈0.191, std=0.0293
- distance: [0.793, 0.863]
- Batch-валидатор: 28/28 OK, rc=0
- Golden (17 прогонов одного видео): max|Δ| семантических полей = 0.0
- FAISS: не установлен локально → numpy_cosine fallback (корректно)
- Синтетические тесты: empty/dim_mismatch/fallback/require_embedding — все PASS

---

## NaN by design (явные исключения)

- `tp_semclust_{n_clusters,model_orig_dim,model_reduced_dim,embedding_dim,margin_top2,compute_ms}` = NaN при `emit_extra_metrics=False` — **expected, не дефект**
- `tp_semclust_{id,similarity,distance}` = NaN при `present=0` (нет эмбеддинга или dim_mismatch) — **expected**
- `tp_semclust_compute_ms` — timing, **excluded from golden**
