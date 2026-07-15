# Критерии приёмки: behavioral

**Согласовано с владельцем:** 2026-07-11 («Сделай всё по своим рекомендациям»).
**Решения владельца:**
- (а) body_lean_angle — **ЧИНИТЬ** (убрать насыщающий множитель `*5.0`, дать вариативность) + выставить `producer_version=2.0.1`.
- (б) вторичные NaN mouth/pose при наличии лица — принять как **by design** (иерархия опор landmarks: лицо → pose/руки → рот) + явная запись в SCHEMA.md.

## Универсальные хард-гейты (pass/fail)
- **U1 — валидатор rc=0:** `utils/validate_behavioral.py` → schema VALID, structure без ошибок, rc=0.
- **U2 — ось времени:** `times_s == union_timestamps_sec[frame_indices]`; `frame_indices` строго возрастают; 0% NaN в `times_s` и `seq_timestamp_norm`.
- **U3 — finite/health:** 0 Inf в `seq_*`; dtype/shape по схеме `behavioral_npz_v1`; `seq_timestamp_norm ∈ [0,1]`; `seq_gesture_prob_*` ∈ [0,1] и сумма probs по классам ≈ 1.0 (на кадрах с landmarks).
- **U4 — expected-empty:** видео без лиц/landmarks → все `landmarks_present=False`, весь основной `seq_*` = NaN (by design), `aggregated` валиден (не падение), rc=0.
- **U5 — golden-детерминизм:** чистый numpy → повтор того же видео → diff=0 по `seq_*` и сигнатуре `aggregated`.
- **U6 — разные длины:** набор B (≥5 видео, ~10с … 5мин+) отрабатывает без падений.

## Критерии под компонент
- **C1 — NaN↔маска (строго):** для основного seq-блока (`seq_num_hands` и др.) 100% соответствие маске: 0 NaN при `landmarks_present=True`, 0 finite при `landmarks_present=False`.
- **C2 — иерархия вторичных опор (by design):** вторичные NaN `seq_mouth_*` (нужны лицевые точки) и pose-полей (нужны pose-точки) при `landmarks_present=True` — ДОПУСТИМЫ и задокументированы в SCHEMA.md как иерархия опор. Проверка: NaN только там, где отсутствуют соответствующие под-landmarks (не случайные).
- **C3 — различимость body_lean (ключевой):** после фикса `seq_body_lean_angle` НЕ константа на наборе B — `std > 1e-3` по конечным кадрам (было =1.0 константа = FAIL). Дополнительно спот-проверка: ≥1 другой seq-признак не константа (arm_openness/speech_activity_proxy).
- **C4 — aggregated:** ≥24 скалярных поля (факт 33); доля NaN-полей ≤ 20%, и NaN только в полях, зависящих от малого числа опорных кадров (early_*/ratios), что документировано; между видами B значения агрегатов варьируются (не константа).

## Примечания к назначению выхода (Encoder-fit)
- `seq_*` [N, F] + `landmarks_present` mask — dense-seq с осью SoT; для Encoder нужен pooling/маска (высокая разрежённость на контенте без лица — ожидаемо).
- `aggregated` — video-level скаляры для аналитиков/tabular head.
