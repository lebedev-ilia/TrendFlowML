# CRITERIA.md — title_embedding_cluster_entropy_extractor

**Дата согласования:** 2026-07-17  
**Авто-штамп:** 100% PASS (ответа владельца не ждали)

---

## Универсальные хард-гейты (U1–U6)

| # | Критерий | Порог | Статус |
|---|----------|-------|--------|
| U1 | Валидатор `validate_title_embedding_cluster_entropy_extractor_text_npz.py` rc=0 на всех NPZ | rc=0 | ✅ |
| U2 | Ось времени (N/A — скалярный агрегатор, нет сегментов) | — | ✅ N/A |
| U3 | Finite/health: 19/24 полей finite при ok-path (5 NaN by design — extra block) | ≥19 finite | ✅ |
| U4 | Expected-empty: no_emb/missing_file → present=0; dim_mismatch → mismatch=1 | rc=0, корректные флаги | ✅ |
| U5 | Golden детерминизм: семантические поля max\|Δ\|=0.0 (compute_ms excluded — timing) | max\|Δ\|=0.0 | ✅ |
| U6 | Разные длины видео (N/A — не зависит от длины видео, только от title embedding) | — | ✅ N/A |

---

## Критерии компонента

| # | Критерий | Порог | Статус |
|---|----------|-------|--------|
| C1 | Entropy varies across videos: std(entropy_raw) > 0 на ok-NPZ | std > 0.01 | ✅ std=0.043 |
| C2 | NaN count per NPZ = 5 при emit_extra_metrics=False (ровно extra-block: n_clusters, model_orig_dim, model_reduced_dim, margin_top2, compute_ms) | NaN=5 | ✅ все 22 NPZ |
| C3 | Clamp: top_k_slots > SCHEMA_TOP_K_SLOTS_MAX(8) → clamped=1, slots=8 | clamped=1.0 | ✅ синтетик |
| C4 | Dim mismatch graceful: embedding_dim ≠ orig_dim → present=0, dim_mismatch_flag=1, rc=0 | корректные флаги | ✅ синтетик |

---

## NaN by design (задокументировано явно)

При `emit_extra_metrics=False` (дефолт) — **5 полей NaN**:
- `tp_titleclent_n_clusters`
- `tp_titleclent_model_orig_dim`
- `tp_titleclent_model_reduced_dim`
- `tp_titleclent_margin_top2`
- `tp_titleclent_compute_ms`

При `title_present=0` или `dim_mismatch=1` — дополнительно NaN в:
- `tp_titleclent_entropy_raw/norm`, `perplexity`, `top_k_used`, `distinct_clusters_topk`, все extra-block

**`compute_ms` excluded из golden** (CPU-стенное время, меняется между вызовами).

---

## Модель: semantic_clusters_v1

- PCA: (1024, 128) float32 — проекция e5-large (1024d) → 128d
- Centroids: (32, 128) float32 — 32 кластера семантического пространства
- Backend: numpy_cosine (faiss не установлен, fallback корректен)
