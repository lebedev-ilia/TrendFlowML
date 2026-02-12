## `asr_text_proxy_audio_features` (ASR Text-based Audio Features)

### Назначение

Извлекает **audio-like proxy** признаки из текста **ASR транскрипции**, чтобы оценивать качество распознавания/“шумность”/ритм речи **без прямого анализа звукового сигнала**.

Production-интенция:
- **ASR приходит из AudioProcessor** (это источник истины для транскрипции в проде).
- Если транскрипта нет — это **валидный empty** (признаки = NaN + masks), без эвристических fallback.
- `audio_duration_sec` должен быть всегда (Segmenter→AudioProcessor contract). Если отсутствует → **error** (fail-fast).

**Версия**: 1.1.0  
**Категория**: text-based audio proxy  
**GPU**: не требуется

### Входы

- **`VideoDocument`**:
  - `audio_duration_sec: float` (**required**)
  - `asr: dict` (**optional**, preferred) — payload от AudioProcessor:
    - `segments: List[dict]`:
      - `text: str`
      - `confidence: float | null`
      - `start_sec/end_sec: float | null`
    - `total_audio_duration_sec: float | null` *(optional)*
  - `transcripts_meta: dict` (**legacy**, optional) — временный alias, будет удалён после аудита AudioProcessor.

### Выходы (stable flat features)

Экстрактор пишет признаки в `result.features_flat` (плоский dict числовых скаляров). Имена стабильны и короткие (префикс `tp_asrproxy_`).

#### Presence / masks

- `tp_asrproxy_present` (0/1): есть ли транскрипт (текст не пустой)
- `tp_asrproxy_has_confidence` (0/1): есть ли confidence хотя бы у одного сегмента
- `tp_asrproxy_segments_count` (float): число сегментов ASR (dict) в payload
- `tp_asrproxy_text_chars`, `tp_asrproxy_word_count` (float): размеры текста (после лимитов)
- `tp_asrproxy_confidence_present_rate` (float \([0..1]\) или NaN)

#### Audio meta

- `tp_asrproxy_audio_duration_sec` (float): длительность аудио (sec), всегда присутствует (иначе error)
- `tp_asrproxy_duration_from_payload_flag` (0/1)
- `tp_asrproxy_duration_invalid_flag` (0/1)

#### Confidence (ASR quality proxies)

- `tp_asrproxy_confidence_mean` (float \([0..1]\) или NaN)
- `tp_asrproxy_confidence_std` (float или NaN)
- `tp_asrproxy_confidence_chunked_min` (float или NaN)
- `tp_asrproxy_low_conf_rate` (float \([0..1]\) или NaN)

#### Noise proxies (не утверждаем WER; это proxy “шумности/грязности” текста)

- `tp_asrproxy_text_noise_rare_ratio` (float \([0..1]\) или NaN)
- `tp_asrproxy_text_noise_oov_ratio` (float \([0..1]\) или NaN)
- `tp_asrproxy_noise_proxy` (float \([0..1]\) или NaN)
- `tp_asrproxy_noise_proxy_present` (0/1)

#### Rhythm (requires `audio_duration_sec`)

- `tp_asrproxy_speech_rate_wpm` (float или NaN)
- `tp_asrproxy_speech_char_density` (float или NaN)
- `tp_asrproxy_pause_density` (float ≥ 0 или NaN)
- `tp_asrproxy_filler_ratio` (float \([0..1]\) или NaN)

#### Intonation

- `tp_asrproxy_sentence_intonation` (float \([0..1]\) или NaN)

> Отдельные “качественные” компоненты (NER/langdetect/speaker_count) можно сделать позже: отдельный extractor с ML‑моделью через `dp_models` (ModelManager), без runtime downloads.

### Алгоритм (кратко)

1. Load: читаем `audio_duration_sec` (required) и `asr.segments` (optional)
2. Normalize: `normalize_whitespace` + join segments
3. Compute: confidence/noise/rhythm/intonation (можно выключать через `enable_*`)
4. Export: `features_flat` + masks

### Конфигурация (feature-gating)

Параметры конструктора (передаются через `TextProcessor/run_cli.py --extractor-params-json`):

- `enable_basic` (bool, default true): confidence метрики
- `enable_noise` (bool, default true): noise proxies
- `enable_rhythm` (bool, default true): rhythm features
- `enable_intonation` (bool, default true): intonation proxy
- `low_conf_threshold` (float, default 0.5): порог для `tp_asrproxy_low_conf_rate`
- `enabled` (bool, default true): компонентный feature-gating
- `max_text_chars` (int, default 200000): лимит на длину транскрипта, `tp_asrproxy_text_truncated_flag=1` если сработал

### Обработка ошибок и empty semantics

- **Если нет `audio_duration_sec`** → `RuntimeError` (fail-fast).
- **Если транскрипта нет** → валидный empty на уровне extractor’а:
  - `tp_asrproxy_present=false`
  - остальные признаки = `NaN` (модель/аналитика отличает “нет данных” от “0”)

### Performance characteristics

- CPU: низкие (текстовые эвристики, numpy)
- GPU: не используется
- Типичное время: ~0.05–0.3s на средний transcript

### Зависимости

- `numpy`
- `unicodedata`

### Связанные компоненты / зависимости

- `TextProcessor/run_cli.py`: сбор `features_flat` в итоговый NPZ
- `AudioProcessor`: источник ASR и `audio_duration_sec` (контракт фиксируем здесь; AudioProcessor адаптируем позже)


