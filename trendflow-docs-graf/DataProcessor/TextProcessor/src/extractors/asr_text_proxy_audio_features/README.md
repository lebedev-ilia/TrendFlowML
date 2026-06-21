## `asr_text_proxy_audio_features` (ASR Text-based Audio Features)

### Назначение

Извлекает **audio-like proxy** признаки из текста **ASR транскрипции**, чтобы оценивать качество распознавания/“шумность”/ритм речи **без прямого анализа звукового сигнала**.

Production-интенция:
- **ASR приходит из AudioProcessor** (это источник истины для транскрипции в проде): **`doc.asr`** (и временно **`transcripts_meta`**).
- Экстрактор **не** читает **`doc.transcripts`** (лексика может; здесь — только структурированный ASR payload).
- Если транскрипта нет — **валидный empty** по умолчанию (`require_asr_text=false`): признаки = NaN + masks.
- Длительность: предпочтительно **`audio_duration_sec`** на документе; иначе — из payload ASR с **`tp_asrproxy_duration_from_payload_flag=1`** (деградация). Режим **`strict_document_duration=true`** запрещает отсутствие **`audio_duration_sec`**.
- Порядок экстракторов в DAG / теги **не** меняют входной ASR для этого компонента.

**Версия**: 1.2.0  
**Категория**: text-based audio proxy  
**GPU**: не требуется

**Описание фич и нормальные диапазоны (док+валидатор):** [`docs/FEATURE_DESCRIPTION.md`](docs/FEATURE_DESCRIPTION.md) · `utils/validate_asr_text_proxy_text_npz.py`

**Контракт Audit v3**: [`SCHEMA.md`](SCHEMA.md) · machine: [`../../../schemas/asr_text_proxy_audio_features_output_v1.json`](../../../schemas/asr_text_proxy_audio_features_output_v1.json)  
**Audit v4:** [`../../../../docs/audit_v4/components/text_processor/asr_text_proxy_audio_features_audit_v4.md`](../../../../docs/audit_v4/components/text_processor/asr_text_proxy_audio_features_audit_v4.md) · L2 stats: `scripts/audit_v4_npz_stats.py` → `storage/audit_v4/asr_text_proxy_audio_features_l2/`

### Входы

- **`VideoDocument`**:
  - `audio_duration_sec: float | null` (**желательно**; иначе при наличии duration в payload — fallback + флаг)
  - `asr: dict` (**optional**, preferred) — payload от AudioProcessor:
    - `segments: List[dict]`:
      - `text: str`
      - `confidence: float | null`
      - `start_sec/end_sec: float | null`
    - `total_audio_duration_sec: float | null` *(optional)*
  - `transcripts_meta: dict` (**legacy**, optional) — временный alias, будет удалён после аудита AudioProcessor.
  - Опционально **token-only** ветка Audit v3: `token_ids_by_segment` (+ старт/конец сегментов); декодирование через `shared_tokenizer_v1`, без сохранения raw текста в артефактах. Сбой декода → `tp_asrproxy_token_decode_failed_flag=1`.

### Выходы (stable flat features)

Экстрактор пишет признаки в `result.features_flat` (плоский dict числовых скаляров). Имена стабильны и короткие (префикс `tp_asrproxy_`).

#### Presence / masks

- `tp_asrproxy_present` (0/1): есть ли транскрипт (текст не пустой)
- `tp_asrproxy_has_confidence` (0/1): есть ли confidence хотя бы у одного сегмента
- `tp_asrproxy_segments_count` (float): число сегментов ASR (dict) в payload
- `tp_asrproxy_text_chars`, `tp_asrproxy_word_count` (float): размеры текста (после лимитов)
- `tp_asrproxy_confidence_present_rate` (float \([0..1]\) или NaN)

#### Configuration flags (для отладки и аудита)

- `tp_asrproxy_enabled` (0/1): был ли экстрактор включен
- `tp_asrproxy_basic_enabled` (0/1): были ли включены базовые признаки
- `tp_asrproxy_noise_enabled` (0/1): были ли включены признаки шума
- `tp_asrproxy_rhythm_enabled` (0/1): были ли включены признаки ритма
- `tp_asrproxy_intonation_enabled` (0/1): были ли включены признаки интонации
- `tp_asrproxy_require_asr_text_enabled` (0/1): `require_asr_text`
- `tp_asrproxy_strict_document_duration_enabled` (0/1): `strict_document_duration`
- `tp_asrproxy_low_conf_threshold` (float): использованный порог для низкой confidence
- `tp_asrproxy_words_per_minute_baseline` (float): baseline WPM (в коде clamped > 0)
- `tp_asrproxy_max_text_chars` (float): использованный лимит символов

#### Audio meta

- `tp_asrproxy_audio_duration_sec` (float): итоговая длительность (sec)
- `tp_asrproxy_duration_from_payload_flag` (0/1): длительность взята из payload (документ не содержал `audio_duration_sec`)
- `tp_asrproxy_duration_invalid_flag` (0/1): была ли обнаружена невалидная длительность (≤ 0)

#### Validation flags

- `tp_asrproxy_text_truncated_flag` (0/1): был ли текст обрезан из-за превышения `max_text_chars`
- `tp_asrproxy_asr_schema_invalid_flag` (0/1): была ли обнаружена невалидная схема ASR payload
- `tp_asrproxy_conf_invalid_flag` (0/1): были ли обнаружены невалидные значения confidence
- `tp_asrproxy_token_decode_failed_flag` (0/1): сбой token-id decode path

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

#### Rhythm (requires валидной длительности и текста при включённом rhythm)

- `tp_asrproxy_speech_rate_wpm` (float или NaN)
- `tp_asrproxy_speech_rate_wpm_ratio_to_baseline` (float или NaN): отношение WPM к baseline
- `tp_asrproxy_speech_char_density` (float или NaN)
- `tp_asrproxy_pause_density` (float ≥ 0 или NaN)
- `tp_asrproxy_filler_ratio` (float \([0..1]\) или NaN)

#### Intonation

- `tp_asrproxy_sentence_intonation` (float \([0..1]\) или NaN)

> Отдельные “качественные” компоненты (NER/langdetect/speaker_count) можно сделать позже: отдельный extractor с ML‑моделью через `dp_models` (ModelManager), без runtime downloads.

### Алгоритм (кратко)

1. Load: длительность (`audio_duration_sec` или из payload) и сегменты ASR / token-id path
2. Normalize: `normalize_whitespace` + join segments
3. Compute: confidence/noise/rhythm/intonation (можно выключать через `enable_*`)
4. Export: `features_flat` + masks

### Конфигурация (feature-gating)

Параметры конструктора (передаются через `TextProcessor/run_cli.py --extractor-params-json`):

- `enabled` (bool, default true): компонентный feature-gating
- `enable_basic` (bool, default true): confidence метрики
- `enable_noise` (bool, default true): noise proxies
- `enable_rhythm` (bool, default true): rhythm features
- `enable_intonation` (bool, default true): intonation proxy
- `require_asr_text` (bool, default false): fail-fast, если joined transcript пуст
- `strict_document_duration` (bool, default false): fail-fast, если нет `audio_duration_sec` на документе
- `low_conf_threshold` (float, default 0.5): порог для `tp_asrproxy_low_conf_rate`
- `words_per_minute_baseline` (float, default 160.0): baseline для `tp_asrproxy_speech_rate_wpm_ratio_to_baseline`
- `max_text_chars` (int, default 200000): лимит на длину транскрипта, `tp_asrproxy_text_truncated_flag=1` если сработал

### Обработка ошибок и empty semantics

- **Если нет валидной длительности ни на документе, ни в payload** → `RuntimeError` (fail-fast).
- **Если `strict_document_duration` и нет `audio_duration_sec`** → `RuntimeError`.
- **Если транскрипта нет** и **`require_asr_text=false`** → валидный empty:
  - `tp_asrproxy_present=false`
  - метрики = `NaN` где уместно
- **`require_asr_text=true`** и пустой текст → `RuntimeError`

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
---

## Навигация

[SCHEMA](SCHEMA.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
