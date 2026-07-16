# Критерии приёмки: speaker_diarization_extractor

**Дата согласования:** 2026-07-16  
**Согласовано с:** Второй агент (от имени владельца)

---

## Универсальные хард-гейты (U1–U6)

| # | Критерий | Pass |
|---|----------|------|
| U1 | `validate_speaker_diarization.py --struct` rc=0 на всех NPZ | rc=0 |
| U2 | Ось времени: turn_start_sec монотонна ↑; turn_start_sec[i] < turn_end_sec[i]; при ok N=1 сег; при empty N=0 | Верно |
| U3 | feature_values finite при ok (nan=0, inf=0); NaN при empty — by design; health_score ≥ 0.85 на ok-NPZ | ≥0.85 |
| U4 | audio_missing → status=empty rc=0; audio_silent → status=empty rc=0 | rc=0 |
| U5 | Golden: sc=0 — bit-identical между повторами; sc>0 — pyannote нейросеть стохастична (known-stochastic, без seed-пиннинга), фиксируем в REPORT как ожидаемо; структурная стабильность (sc, S одинаковы) при повторе | sc=0 идент; sc>0 known-stoch |
| U6 | Короткое (~3s), среднее (~17s), длинное (~30s) аудио — без падений | rc=0 |

---

## Специфичные критерии (C1–C4)

| # | Критерий | Порог |
|---|----------|-------|
| C1 | speaker_ids = [0..S-1] без дыр; turn_speaker_id ⊆ speaker_ids (orphan=0) | orphan=0 |
| C2 | speaker_balance_score ∈ [0,1]; S=1 → 1.0 by design; dominant_speaker_id = -1 при S=0, ∈ [0,S-1] при S>0 | ∈ [0,1] |
| C3 | Различимость в корпусе (≥5 видео): sc ∈ {0,1,2}, CV(speaker_count) > 0 | CV>0 |
| C4 | При empty: feature_values=NaN; при ok+sc=0: feature_values finite (не NaN); при ok+sc>0: все F=10 поля finite | finite при ok |

---

## Примечания

- **NaN by design:** feature_values=NaN при status=empty — штатное поведение, не дефект
- **sc=0 at ok:** видео с речью ниже порогов тишины → sc=0, dominant_speaker_id=-1, K=0 — легитимно
- **Golden-стохастика pyannote:** нейросеть без детерминированного seed; golden-тест для sc>0 ограничен проверкой структуры (sc, S, K порядок). Помечается в REPORT как known-stochastic
- **AudioProcessor venv:** `DataProcessor/AudioProcessor/.ap_venv` (python3.14) или pip install в workspace/venv
