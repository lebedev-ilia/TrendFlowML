# Главный индекс документации AudioProcessor

Этот документ служит единой точкой входа для навигации по документации всех extractors AudioProcessor. Каждый раздел содержит краткое описание extractor'а и ссылку на его README.

---

## Тестирование

- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** — руководство по smoke- и full-тестам
- **[TESTING_STRUCTURE.md](TESTING_STRUCTURE.md)** — структура папок и скриптов

---

## AudioProcessor Extractors

### asr_extractor
**Краткое описание**: Извлекает транскрипцию речи через Whisper ASR (inprocess, ModelManager-only, offline). Выход — token IDs через `shared_tokenizer_v1` (privacy-first) для downstream компонентов (TextProcessor), privacy-safe quality metrics, нормализованные language codes. Строгий token contract (no fallback), feature-gated aggregates, per-segment quality. GPU preferred, schema `asr_extractor_npz_v2`, версия 2.2.0 (Audit v3).

**Полный документ**: [src/extractors/asr_extractor/README.md](../src/extractors/asr_extractor/docs/README.md)

### band_energy_extractor
**Краткое описание**: Извлекает доли энергии по 3 фиксированным полосам (low/mid/high) из Segmenter-окон (shared family `spectral`). Audit v3: только librosa (no fallback), нормализация включена по умолчанию, model_facing — только shares; optional analytics sequences с `segment_mask` (строгое выравнивание). CPU-only, schema `band_energy_extractor_npz_v1`, версия 2.1.0 (Audit v3).

**Полный документ**: [src/extractors/band_energy_extractor/README.md](../src/extractors/band_energy_extractor/docs/README.md)

### chroma_extractor
**Краткое описание**: Извлекает chroma (pitch class profile) в каноничном контракте `n_chroma=12` + L1 frame-normalization. Audit v3: tuning оценивается один раз на полном аудио (при неудаче `tuning=0.0`), `run_segments()` даёт strict-aligned per-segment sequences через `segment_mask` (если включено), а `key_extractor` может переиспользовать chroma in-memory как `shared_features`. CPU-only, schema `chroma_extractor_npz_v1`, версия 2.1.0 (Audit v3).

**Полный документ**: [src/extractors/chroma_extractor/README.md](../src/extractors/chroma_extractor/docs/README.md)

### clap_extractor
**Краткое описание**: Считает семантический аудио-эмбеддинг CLAP по Segmenter-окнам (`families.clap`). Audit v3: ModelManager-only (offline, no-network), strict-aligned `embedding_sequence` с `segment_mask` и time-axis (`start/end/center`), robust aggregation по валидным сегментам, репорт trim (если окно > 10s). Tier-0 baseline, required. GPU preferred, schema `clap_extractor_npz_v1`, версия 1.1.0 (Audit v3).

**Полный документ**: [src/extractors/clap_extractor/README.md](../src/extractors/clap_extractor/docs/README.md)

### emotion_diarization_extractor
**Краткое описание**: Извлекает эмоциональную диаризацию через SpeechBrain Speech_Emotion_Diarization. Audit v3: ModelManager-only (offline, no-network), строго по Segmenter окнам (`families.emotion`), strict-aligned `emotion_id`/`emotion_confidence` sequences с `segment_mask` + time-axis (`start/end/center`), агрегаты (entropy/dominant/transitions/stability/diversity), empty для `audio_silent` и `audio_too_short`. GPU preferred, schema `emotion_diarization_extractor_npz_v1`, версия 3.1.0 (Audit v3).

**Полный документ**: [src/extractors/emotion_diarization_extractor/README.md](../src/extractors/emotion_diarization_extractor/docs/README.md)

### hpss_extractor
**Краткое описание**: Извлекает Harmonic-Percussive Source Separation (HPSS) признаки — разложение аудио на гармоническую и перкуссионную компоненты. Audit v3: только `run_segments()` по `families.hpss`, strict-aligned `segment_start_sec`/`segment_end_sec`/`segment_center_sec`/`segment_mask`, `hpss_harmonic_share_by_segment`/`hpss_percussive_share_by_segment` (NaN для failed), waveform paths в meta.extra (без массивов в NPZ), offline HTML renderer. `enable_energy_metrics=True` по умолчанию. CPU-only, schema `hpss_extractor_npz_v1`, версия 2.1.0 (Audit v3).

**Полный документ**: [src/extractors/hpss_extractor/docs/README.md](../src/extractors/hpss_extractor/docs/README.md)

### key_extractor
**Краткое описание**: Определяет тональность (ключ) аудио — основной тональный центр и лад (мажор/минор). Audit v3: только `run_segments()` (run отключён), strict alignment через `segment_start_sec`/`segment_end_sec`/`segment_center_sec`/`segment_mask`, `key_id_by_segment`/`key_confidence_by_segment`, `key_id` (0–23) model-facing. Default `key_method=librosa`. Offline HTML render (vanilla canvas). CPU-only, schema `key_extractor_npz_v1`, версия 2.1.0 (Audit v3).

**Полный документ**: [src/extractors/key_extractor/README.md](../src/extractors/key_extractor/docs/README.md)

### loudness_extractor
**Краткое описание**: Считает громкость/динамику: RMS, peak, dBFS, опционально LUFS (если установлен pyloudnorm), статистики по short-term RMS. Audit v3: `run_segments()` по `families.primary`, strict alignment через `segment_start_sec`/`segment_end_sec`/`segment_center_sec`/`segment_mask` + `NaN` padding для failed сегментов, offline HTML render (без CDN). Tier‑0 baseline, required. CPU-only, schema `loudness_extractor_npz_v2`, версия 2.1.0 (Audit v3).

**Полный документ**: [src/extractors/loudness_extractor/README.md](../src/extractors/loudness_extractor/docs/README.md)

### mel_extractor
**Краткое описание**: Извлекает Mel-спектрограмму и производные признаки. Audit v3: `mel_extractor_npz_v2`, каноническая ось сегментов `segment_start_sec`/`segment_end_sec`/`segment_center_sec` + `segment_mask` (strict alignment), segment-aligned sequences (`mel_mean_by_segment` и др.) при `--mel-enable-time-series`, offline HTML render (vanilla canvas, без CDN), float32 путь без CUDA autocast. Версия 2.1.0 (Audit v3).

**Полный документ**: [src/extractors/mel_extractor/README.md](../src/extractors/mel_extractor/docs/README.md)

### mfcc_extractor
**Краткое описание**: Извлекает MFCC (Mel-frequency cepstral coefficients). Audit v3: `mfcc_extractor_npz_v2`, каноническая ось сегментов `segment_start_sec`/`segment_end_sec`/`segment_center_sec` + `segment_mask` (strict alignment), segment-aligned sequences (`mfcc_mean_by_segment`, `mfcc_energy_by_segment`) при `--mfcc-enable-time-series`, offline HTML render (vanilla canvas, без CDN), float32 без autocast, full MFCC только в debug .npy. Версия 2.1.0 (Audit v3).

**Полный документ**: [src/extractors/mfcc_extractor/docs/README.md](../src/extractors/mfcc_extractor/docs/README.md)

### onset_extractor
**Краткое описание**: Определяет онсеты (атаки звука). Audit v3: `onset_extractor_npz_v2`, каноническая ось сегментов (aggregation semantics), onset_times только в debug .npy, offline HTML render (vanilla canvas), basic_features=True по умолчанию, units="frames" исправлен. Версия 2.0.0, CPU-only.

**Полный документ**: [src/extractors/onset_extractor/README.md](../src/extractors/onset_extractor/docs/README.md)

### pitch_extractor
**Краткое описание**: Извлекает основную частоту (f0) из аудио сигнала с использованием PYIN, YIN и опционально torchcrepe. Audit v3: `pitch_extractor_npz_v2`, каноническая ось сегментов, empty при `pitch_all_segments_empty` (не ошибка), `enable_basic_stats=True` по умолчанию, f0_series только в debug .npy, offline HTML render (vanilla canvas), voice_quality загружает f0 из .npy. Версия 2.0.0, GPU optional.

**Полный документ**: [src/extractors/pitch_extractor/README.md](../src/extractors/pitch_extractor/docs/README.md)

### quality_extractor
**Краткое описание**: Извлекает базовые метрики качества аудио для оценки технического состояния записи. Audit v3: `quality_extractor_npz_v2`, каноническая ось сегментов, `enable_basic_metrics=True` по умолчанию, empty при `quality_all_segments_empty`, snr_db/dc_offset_abs удалены, time series только в .npy, offline HTML render (vanilla canvas). Версия 2.0.0, CPU-only.

**Полный документ**: [src/extractors/quality_extractor/docs/README.md](../src/extractors/quality_extractor/docs/README.md)

### rhythmic_extractor
**Краткое описание**: Ритмические метрики (beat tracking/regularity). Audit v3: `rhythmic_extractor_npz_v2`, required family `tempo`, canonical axis (`segment_*_sec` + `segment_mask`), no-beats = `status=ok` (NaN policy), beat events token-ready (`beat_times_sec` + `beat_segment_index`) с `.npy` fallback, offline HTML render (vanilla canvas). Версия 2.0.0, CPU-only.

**Полный документ**: [src/extractors/rhythmic_extractor/README.md](../src/extractors/rhythmic_extractor/docs/README.md)

### source_separation_extractor
**Краткое описание**: Source separation shares (vocals/drums/bass/other) via ModelManager-only inprocess PyTorch. Audit v3: `source_separation_extractor_npz_v2`, canonical axis + `segment_mask` (silent windows masked), structured per-source stats arrays (no dict object), fail-fast NaN/inf, short audio → `empty(audio_too_short)`, offline HTML render (vanilla canvas). Версия 3.0.0, GPU preferred.

**Полный документ**: [src/extractors/source_separation_extractor/README.md](../src/extractors/source_separation_extractor/docs/README.md)

### speaker_diarization_extractor
**Краткое описание**: Диаризация спикеров (кто говорит когда) на полном аудио. Audit v3: `speaker_diarization_extractor_npz_v2`, ModelManager-only (no-network), Segmenter-owned family `diarization` (single full-audio window), token-ready turn arrays (`turn_*`), structured per-speaker arrays (no object dict), offline HTML render (vanilla canvas). Версия 3.1.0, GPU preferred.

**Полный документ**: [src/extractors/speaker_diarization_extractor/README.md](../src/extractors/speaker_diarization_extractor/docs/README.md)

### spectral_entropy_extractor
**Краткое описание**: Спектральная энтропия (опционально flatness/spread). Audit v3: `spectral_entropy_extractor_npz_v2`, shared family `spectral`, per-segment агрегаты (`entropy_*_by_segment`) + `segment_mask`, NaN policy (без нулевых заглушек), short audio → `empty(audio_too_short)`, all segments failed → `empty(spectral_entropy_all_segments_failed)`, offline HTML render (vanilla canvas). Версия 2.0.0, CPU-only.

**Полный документ**: [src/extractors/spectral_entropy_extractor/README.md](../src/extractors/spectral_entropy_extractor/docs/README.md)

### spectral_extractor
**Краткое описание**: Извлекает базовые спектральные признаки: спектральный центроид, ширина полосы, плоскостность, rolloff, скорость пересечения нуля (ZCR), контраст. Версия 2.0.0, CPU-only.

**Полный документ**: [src/extractors/spectral_extractor/README.md](../src/extractors/spectral_extractor/docs/README.md)

### speech_analysis_extractor
**Краткое описание**: Предоставляет компактный "обзор речи" путем комбинирования результатов ASR (Whisper), Speaker Diarization (pyannote.audio) и опционально Pitch. Использует существующие результаты зависимых компонентов. Версия 2.1.0, GPU optional.

**Полный документ**: [src/extractors/speech_analysis_extractor/README.md](../src/extractors/speech_analysis_extractor/docs/README.md)

### tempo_extractor
**Краткое описание**: Оценивает темп (BPM) и простые ритмические признаки на базе librosa. Tier-0 baseline, required. Версия 1.1.0, CPU-only.

**Полный документ**: [src/extractors/tempo_extractor/README.md](../src/extractors/tempo_extractor/docs/README.md)

### voice_quality_extractor
**Краткое описание**: Извлекает метрики качества голоса для оценки стабильности и гармоничности. Использует прокси-метрики jitter, shimmer и HNR-подобную метрику на основе оценки f0. Версия 2.0.0, GPU optional.

**Полный документ**: [src/extractors/voice_quality_extractor/README.md](../src/extractors/voice_quality_extractor/docs/README.md)

---

## Статистика

Всего extractors: **21**

**Tier-0 baseline (required)**:
- `clap_extractor`
- `tempo_extractor`
- `loudness_extractor`
- `asr_extractor`

**Tier-1 (optional)**:
- Остальные 17 extractors

