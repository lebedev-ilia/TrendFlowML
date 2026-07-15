# CRITERIA — emotion_face (согласовано 2026-07-13)

Компонент: EmoNet (n_expression=8) → valence/arousal/intensity + 8 эмоций (Neutral,Happy,Sad,Surprise,Fear,Disgust,Anger,Contempt).
Схема: `emotion_face_npz_v3`. Валидатор: `modules/emotion_face/utils/validate_emotion_face.py`.

## Универсальные хард-гейты (pass/fail)
- **U1** Валидатор выхода rc=0 (`--struct`, `--qa`). --qa может печатать «пропуск» (QA-конфига нет) — это rc=0, ок.
- **U2** Ось времени: `len(times_s)==N`, `times_s` неубывает; все per-frame ключи длины N; `emotion_probs (N,8)`.
- **U3** Health на processed-кадрах: valence/arousal/probs finite; не константа; верные dtype/shape/range.
- **U4** Expected-empty: видео без лиц → `status=empty`, `empty_reason=no_faces_in_video`, все массивы N=0, rc=0.
- **U5** Golden-детерминизм: EmoNet fp32 (AMP отключён) → повтор прогона побайтово, `max|Δvalence|=max|Δprobs|=0`.
- **U6** Разные длины видео отрабатывают (матрица роликов разной длительности).

## Критерии под компонент (числовые)
- **C1** valence,arousal ∈ [-1,1]; intensity=sqrt(v²+a²) ∈ [0, √2] на processed-кадрах (processed_mask=True).
- **C2** emotion_probs по строке ≈1 (softmax): |Σ_c probs − 1| ≤ 1e-3 на processed-кадрах; на непроцессир. — NaN by design.
- **C3** Различимость: CV(valence_mean по роликам) > ~15% ИЛИ std между роликами заметен; эмоции не глобальная константа.
- **C4** processed_mask ⊆ face_present (обрабатываются только кадры с лицом); valence/arousal=NaN на непроцессир. by design.

## Решения владельца (2026-07-13)
- Keyframes-баг (NameError seq/times_s в run → keyframes всегда []) — **фиксить сейчас** (seq_proc/times_s_proc).
- Golden — **fp32 (use_amp=False)** для детерминизма приёмки. AMP оставить конфигурируемым (прод-скорость).
- nanargmax all-NaN → защита (dominant=-1 на пустых строках).
