# Критерии приёмки: ocr_extractor

**Согласовано с владельцем:** 2026-07-11 («Все ок, по вопросам реши сам»).
**Движок (решение агента):** `ppocr_rec_onnx` (recommended, offline via dp_models ModelManager).
**retain_raw_ocr_text=true (решение агента):** проверяется лёгким dev-smoke; прод-штамп — на `retain=false`.

## Универсальные хард-гейты (pass/fail)
- **U1 — валидатор rc=0:** `utils/validate_ocr.py --struct` (+ `--qa`) → schema VALID, structure без ошибок, rc=0.
- **U2 — ось времени:** `times_s == union_timestamps_sec[frame_indices]`; `frame_indices` строго возрастают; 0% NaN в `times_s`.
- **U3 — finite/health:** `times_s` finite; `frame_indices` int32 в диапазоне [0, len(uts)); структуры `ocr_raw` валидны.
- **U4 — expected-empty:** ролик без `text_region`/без текста → `status=empty`, `empty_reason` задан, `ocr_raw` пуст, rc=0 (не падение).
- **U5 — golden-детерминизм:** повтор того же ролика тем же движком → воспроизводимость (см. C4).
- **U6 — разные длины:** матрица длин (~10с … 8мин+) отрабатывает без падений.

## Критерии под компонент
- **C1 — привязка к оси:** `frame` каждой строки `ocr_raw` ∈ `frame_indices` = **100%**.
- **C2 — privacy:** при `retain=false` ни в одной строке нет `text_raw`/`text_norm` (**100%**); присутствуют `text_sha256` + `text_len`. `meta.retain_raw_ocr_text=false`.
- **C3 — различимость:** на роликах С текстом `R>0` и `rec_confidence ∈ [0,1]`; на роликах без `text_region` — valid empty (не падение). Корпус не константа (R варьируется между роликами).
- **C4 — golden-набор:** повтор ролика (ppocr_rec_onnx) → идентичный набор строк по `(frame, bbox, text_sha256)`; порог по `rec_confidence` **max|Δ| ≤ 1e-3** (фактический замерить; ONNX может быть побайтово детерминирован → 0).

## Примечания к назначению выхода (Encoder-fit)
- Ось `frame_indices`/`times_s` — seq/time-axis, согласована с SoT (union_timestamps_sec).
- `ocr_raw` — **sparse-события** текста (analytics-tier): для аналитиков и downstream
  (`franchise_recognition`, `text_scoring`). Плотных числовых per-frame фич для Encoder компонент не отдаёт — это ожидаемо (analyst/sparse-компонент, не dense-seq).
