# Критерии приёмки: comments_embedder

**Согласовано:** 2026-07-16  
**Компонент:** comments_embedder (TextProcessor)  
**Версия:** 1.3.0  

## Универсальные гейты (U1–U6)

| Гейт | Описание | Порог |
|------|----------|-------|
| U1 | Валидатор rc=0 (`validate_comments_embedder_text_npz.py`) | 100% OK |
| U2 | Консистентность n_input/n_deduped/n_selected/count | 0 расхождений |
| U3 | Finite/health core-ключей, L2-нормы, различимость | см. C1–C2 |
| U4 | Expected-empty (нет комментариев) → present=0, error=None | PASS |
| U5 | Golden-детерминизм (CPU sentence-transformers) | max\|Δ\|=0.0 |
| U6 | Разные длины (n=1,5,15+ комментариев) без ошибок | PASS |

## Компонентные критерии (C1–C4)

**C1 — Health score core-ключей = 1.0**  
8 core-полей (present, count, dim, n_input, n_deduped, n_selected, total_chars_used, truncated_flag): NaN rate = 0.0.

**C2 — L2-нормы эмбеддингов ∈ [0.999, 1.001]**  
Матрица `comments_embeddings.npy`: все строки L2-нормализованы (результат: [1.0000, 1.0000]).

**C3 — Shape консистентность**  
`emb.shape[0] == tp_commentsemb_count == tp_commentsemb_n_selected` для каждого ok-NPZ.

**C4 — NaN by design (extra-ключи при emit_extra_metrics=False)**  
10 extra-ключей (cache_enabled, cache_hit, fp16, device_cuda, model_digest_u24, compute_enabled, write_artifact_enabled, artifact_written, select_ms, encode_ms) = NaN при дефолтном `emit_extra_metrics=False`. Это штатное поведение, НЕ ошибка. Downstream (comments_aggregator) должен проверять наличие `.npy`-артефакта, а не флаг artifact_written.
