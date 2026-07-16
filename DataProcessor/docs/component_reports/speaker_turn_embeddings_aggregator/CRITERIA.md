# Критерии приёмки — speaker_turn_embeddings_aggregator

**Дата согласования:** 2026-07-17  
**Согласовал:** Второй агент (от имени владельца)  
**Версия компонента:** 1.3.0  
**Схема:** speaker_turn_embeddings_aggregator_output_v1 (17 ключей tp_spkemb_*)

---

## Универсальные хард-гейты

| Гейт | Критерий | Результат |
|------|----------|-----------|
| U1 | batch-валидатор 28/28 NPZ → OK (rc=0) | PASS: 28/28 OK |
| U2 | ось времени согласована | N/A — TextProcessor, нет frame_indices |
| U3 | различимость: cos_sim(Alice,Bob) < 0.99 | PASS: 0.9236 |
| U4 | expected-empty: require_input=False + пустой вход → present=0, no crash | PASS |
| U5 | golden: CPU ST детерминирован, max\|Δ\|=0.0 | PASS: 0.0 (legacy + diar+ASR) |
| U6 | разные длины (1/2/3 тёрна) отрабатывают без падений | PASS (синтетика) |

## Компонентные критерии

| Код | Критерий | Результат |
|-----|----------|-----------|
| C1 | 5 extra-ключей (batch_size/max_speakers/max_turns_per_speaker/min_chars/max_chars) = NaN при emit_extra_metrics=False; finite при True — by design | PASS |
| C2 | speakers_embedded ≤ speakers_total; при present=1 → embedded ≥ 1 | PASS |
| C3 | mode-флаги взаимоисключающие: diar_asr XOR legacy (не могут быть оба 1.0) | PASS |
| C4 | require_input=True + пустой вход → RuntimeError (не тихий empty) | PASS |

## Известные исключения (NaN by design)

- 5 extra-ключей NaN при emit_extra_metrics=False — ожидаемо, ключи всегда присутствуют (контракт 17 ключей выполнен)
- В storage 28/28 NPZ имеют present=0 — ожидаемо для MVP-датасета без diar+ASR pipeline
