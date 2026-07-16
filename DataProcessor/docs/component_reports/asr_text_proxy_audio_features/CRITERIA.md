# Критерии приёмки: asr_text_proxy_audio_features

> Одобрено: Второй агент (от имени владельца), 2026-07-17.
> Компонент: CPU-only TextProcessor агрегатор (proxy аудиофич через ASR текст).

## Универсальные хард-гейты (U1–U6)

| Гейт | Критерий |
|------|----------|
| U1 | validate_asr_text_proxy_text_npz.py --struct --ranges → rc=0 на всех NPZ |
| U2 | tp_asrproxy_audio_duration_sec == VideoDocument.audio_duration_sec (Δ=0) |
| U3 | std(tp_asrproxy_speech_rate_wpm) > 5 на корпусе present=1 видео |
| U4 | Пустой ASR (no segments) → present=0, n_keys=37, нет исключений |
| U5 | Golden: max\|Δ\|=0.0 (CPU-only Python+numpy, детерминирован без условий) |
| U6 | Видео длиной 5/60/600с отрабатывают без падений |

## Критерии компонента (C1–C4)

**C1 — NaN confidence by design:**
`tp_asrproxy_confidence_mean/std/chunked_min/low_conf_rate = NaN` при `tp_asrproxy_confidence_present_rate=0`
— норма: Whisper (продакшн ASR) не передаёт поле confidence. Не дефект.

**C2 — NaN rhythm/intonation by design:**
`tp_asrproxy_speech_rate_wpm/speech_char_density/pause_density/filler_ratio/sentence_intonation/text_noise_*= NaN`
при `tp_asrproxy_present=0` (нет транскрипта) — by design, downstream обязан проверять present перед чтением.

**C3 — wpm в разумном диапазоне:**
При `present=1`: `tp_asrproxy_speech_rate_wpm ∈ [1, 500]` wpm.
(Реальный датасет: 5.6–217.6 wpm; быстрая речь ~250 wpm max.)

**C4 — различимость на корпусе:**
При ≥5 видео с present=1: `std(tp_asrproxy_speech_rate_wpm) > 5`.
(Датасет: std=86.9 — высокая различимость.)
