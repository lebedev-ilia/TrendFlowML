# Критерии приёмки: key_extractor

**Утверждено**: 2026-07-16  
**Компонент**: key_extractor (AudioProcessor)  
**Версия**: 2.1.1  
**Схема**: key_extractor_npz_v1  

---

## Универсальные хард-гейты (U1–U6)

| Гейт | Описание | Порог |
|------|----------|-------|
| U1 | validate_key.py rc=0 | rc=0 |
| U2 | Ось времени: ss < se, center ≈ (ss+se)/2 | True |
| U3 | Finite/health: nan_rate в feature_values | = 0.0 |
| U4 | Expected-empty (audio_missing → valid empty NPZ, все ключи схемы есть) | rc=0, no crash |
| U5 | Golden-детерминизм: librosa побайтово детерминирован | max\|Δ\|=0.0 |
| U6 | Разные длины видео (N=5..30 сегментов) отрабатывают | no crash |

---

## Компонентные критерии (C1–C4)

| Критерий | Описание | Порог |
|----------|----------|-------|
| C1 | nan_rate в feature_values при status=ok | = 0.0 |
| C2 | key_id_by_segment: при segment_mask=True → 0..23; при mask=False → -1 | строго |
| C3 | Golden: два прогона одного аудио с одинаковым конфигом | max\|Δ\|=0.0 |
| C4 | chroma_reused=True by design — все сегменты получают одинаковый ключ при shared chroma. unique_conf=1 per video — НЕ баг | зафиксировано |

---

## Известные исключения (by design)

- `key_scores = zeros(24)` при `enable_detailed_scores=False` — by design, всегда присутствует в NPZ
- `chroma_reused=True` → все сегменты идентичны по ключу/confidence — by design (global chroma из chroma_extractor)
- `sample_rate` и `hop_length` в feature_values — конфиг-константы, std=0 по видео — нормально
