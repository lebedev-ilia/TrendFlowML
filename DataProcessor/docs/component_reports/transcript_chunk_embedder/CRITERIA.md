# Критерии приёмки: transcript_chunk_embedder

Согласованы 2026-07-16 (авто-штамп по итогам брифинга).

## Универсальные хард-гейты (U1–U6)

| Гейт | Описание |
|------|----------|
| U1 | Валидатор `validate_transcript_chunk_embedder_text_npz.py` → rc=0 |
| U2 | Ось чанков согласована: feature_names/values 16 ключей на всех ветках |
| U3 | Finite/health: NaN только by-design (5 extra metrics + conf stats при conf_present=0); эмбеддинги finite, L2=1 |
| U4 | Expected-empty (нет ASR): present=0, 16 ключей, error=None |
| U5 | Golden детерминизм CPU: max|Δ|=0.00 |
| U6 | Разные длины видео (1–20 сегментов, 1–3 чанка) отрабатывают корректно |

## Критерии компонента

| ID | Критерий | Порог |
|----|----------|-------|
| C1 | embedding_dim | = 1024 (multilingual-e5-large) |
| C2 | L2 нормы каждого чанка в артефакте | ∈ [0.9999, 1.0001] |
| C3 | conf_mean/min/max = NaN by design при conf_present=0 | Допустимо. Реальный Whisper ASR не выдаёт confidence в текущем payload; NaN ожидается. При conf_present=1 — finite. |
| C4 | При present=1: whisper_chunks ≥ 1, artifact shape=(N, 1024), все элементы finite | PASS |

## Заметки
- Компонент CPU-only (sentence-transformers in-process). Под не нужен.
- emit_extra_metrics=False (по умолчанию) → 5 полей (batch_size, max_chunk_tokens_model, overlap_ratio, max_chunks_total, cache_enabled) = NaN — by design.
- Различимость между видео: cos-similarity mean-векторов 0.70–0.88 на корпусе storage.
