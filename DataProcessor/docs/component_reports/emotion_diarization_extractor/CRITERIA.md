# CRITERIA — emotion_diarization_extractor

**Дата согласования**: 2026-07-16  
**Сессия**: агент (компонентный раннер) + Второй агент (одобрение)

---

## Универсальные хард-гейты (U1–U6)

| Гейт | Описание | Порог |
|------|----------|-------|
| U1 | Валидатор rc=0 (validate_emotion_diarization.py --struct) | rc ∈ {0,2} где 0=OK, 2=WARN |
| U2 | Ось времени N согласована (все axis-поля одной длины) | all_N_same=True |
| U3 | Различимость ok-NPZ: std>0 по всем 7 feature_values | min_std > 0 |
| U4 | Expected-empty: emotion_id=-1, confidence=NaN, mask=False | при status=empty |
| U5 | Golden: CPU max\|Δ\|=0.0 (GPU — бэклог) | 0.0 на CPU |
| U6 | Разные длины N работают без падений | N=6, 9, 15 ок |

---

## Компонентные критерии (C1–C4)

### C1: emotion_id диапазон
- Для valid-сегментов (mask=True): emotion_id ∈ {0, 1, 2, 3}
- Соответствует emotion_labels=['a','n','h','s'] (C=4 класса)
- **Порог**: все valid-id ∈ [0, C-1]

### C2: emotion_confidence диапазон и finite
- Для valid-сегментов: confidence ∈ [0.0, 1.0] и finite (не NaN/Inf)
- **Порог**: min≥0, max≤1, np.all(isfinite)=True

### C3: NaN by design (документирование исключений)
- **audio_too_short** (dur < 5s): segments_count=0.0 (конечный), остальные 6 feature_values = NaN by design
- **audio_missing_or_extract_failed**: все 7 feature_values = NaN by design (оркестратор не дошёл до компонента)
- **audio_silent**: аналогично audio_too_short (model не вызывается)
- Это НЕ дефект — NaN при пустом аудио ожидаем

### C4: diversity/stability в [0,1]
- emotion_diversity_score ∈ [0, 1]  (ent / log(C), нормированная энтропия)
- emotion_stability_score ∈ [0, 1]  (1 / (1 + transitions_freq))
- **Порог**: оба значения ∈ [0, 1]

---

## GPU-валидация (бэклог)

- **Статус**: CPU baseline only (пода не было в сессии валидации)
- **Обоснование**: WavLM детерминирован на CPU (eval mode + inference_mode), все U1–U6 PASS локально
- **CPU golden**: max|Δ|=0.0 на 2 прогонах видео -0InsUQNwIQ (N=9 сегментов)
- **TODO**: После GPU-пода — проверить max|Δ| confidence на 3–5 видео vs CPU baseline

---

## Примечания

- emotion_entropy = -Σ(p·ln(p+ε)), натуральный логарифм; при C=4 max≈1.386 (неравномерное распределение)
- dominant_emotion_id — целое в feature_values хранится как float32 (2.0 = класс "h")
- Прогон: ap_venv (torch 2.12.1+cu126, speechbrain 1.1.0, CPU), 4.5с/видео (9 сегментов)
