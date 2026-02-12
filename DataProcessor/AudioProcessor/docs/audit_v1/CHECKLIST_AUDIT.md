# AudioProcessor audit checklist index (audit_v1)

**Цель**: единый индекс всех audit‑файлов по AudioProcessor и его extractors.  
**Source-of-truth критерии**: `AP_AUDIT_CRITERIA.md`.

---

## Как пользоваться

- Каждый extractor аудируем отдельно и создаём файл в:  
  `AudioProcessor/docs/audit_v1/components/<extractor_name>_AUDIT.md`
- После создания/обновления audit‑файла добавляем строку в таблицу ниже.
- Статусы:
  - `planned` — запланирован, но аудит не начат
  - `in_review` — аудит идёт (вопросы/дизайн обсуждение)
  - `in_progress` — внедряем изменения в код/README
  - `done` — аудит закрыт, компонент соответствует критериям
  - `blocked` — заблокирован (нет данных/модели/апстрима/контракта)

---

## Индекс audit‑файлов

### Основные документы

- `AP_AUDIT_CRITERIA.md` — критерии аудита (этот каталог)

### Orchestrator/Writer (core components)

| Component | Audit file | Status | Notes |
|-----------|------------|--------|-------|
| `orchestrator_writer` (`run_cli.py`) | `components/orchestrator_writer_AUDIT.md` | `done` | Progress reporting (JSON-lines), stage timings, resource metrics, error handling (fail-fast, retry, OOM fallback), UI render |

### Tier-0 Baseline Extractors (required)

| Extractor | Audit file | Status | Notes |
|-----------|------------|--------|-------|
| `clap_extractor` | `components/clap_extractor_AUDIT.md` | `done` | Full audit by AP_AUDIT_CRITERIA.md; Segmenter contract, ModelManager, NPZ schema, UI render |
| `tempo_extractor` | `components/tempo_extractor_AUDIT.md` | `done` | Full audit by AP_AUDIT_CRITERIA.md; Segmenter contract, signal processing, NPZ schema, UI render |
| `loudness_extractor` | `components/loudness_extractor_AUDIT.md` | `done` | Full audit by AP_AUDIT_CRITERIA.md; Segmenter contract, signal processing, NPZ schema, UI render |

### Non-Baseline Extractors (optional)

| Extractor | Audit file | Status | Notes |
|-----------|------------|--------|-------|
| `asr_extractor` | `components/asr_extractor_AUDIT.md` | `done` | Full audit by `AP_AUDIT_CRITERIA.md` completed. Feature gating, token validation, error codes, progress reporting, HTML renderer. |
| `speaker_diarization_extractor` | `components/speaker_diarization_extractor_AUDIT.md` | `done` | Full audit by AP_AUDIT_CRITERIA.md; Triton-backed; speaker embeddings; clustering methods (agglomerative/kmeans/auto); speaker count estimation (heuristic/silhouette/fixed); feature gating; detailed error codes; progress reporting; clustering metrics; contract versioning |
| `emotion_diarization_extractor` | `components/emotion_diarization_extractor_AUDIT.md` | `done` | Full audit by AP_AUDIT_CRITERIA.md; Triton-backed; emotion probs; feature gating; detailed error codes; progress reporting; probability validation; emotion_labels validation; additional aggregates (transitions, distribution, stability, diversity); quality metrics; contract versioning |
| `source_separation_extractor` | `components/source_separation_extractor_AUDIT.md` | `done` | Full audit by AP_AUDIT_CRITERIA.md; Triton-backed; source shares; feature gating; detailed error codes; progress reporting; shares/energies validation; source_order validation; additional aggregates (transitions, distribution, stability, balance); quality metrics; contract versioning; preprocessing params validation |
| `speech_analysis_extractor` | `components/speech_analysis_extractor_AUDIT.md` | `done` | Full audit by AP_AUDIT_CRITERIA.md; bundle extractor; feature gating; detailed error codes; progress reporting; segments validation; payload validation; additional metrics (ASR: speech_rate_wpm, lang_distribution; diarization: balance_score, transitions_count; pitch: range, stability, distribution); contract versioning; silence detection |
| `video_audio_extractor` | `components/video_audio_extractor_AUDIT.md` | `blocked` | **REMOVED**: AudioProcessor не должен извлекать аудио из видео (это делает Segmenter). Компонент удален из кодовой базы. |
| `mfcc_extractor` | `components/mfcc_extractor_AUDIT.md` | `done` | Full audit by AP_AUDIT_CRITERIA.md; segment-based processing; feature gating; detailed error codes; progress reporting; output validation; parameter validation; additional metrics; contract versioning; UI render; optional normalization; improved GPU heuristic |
| `mel_extractor` | `components/mel_extractor_AUDIT.md` | `done` | Full audit by AP_AUDIT_CRITERIA.md; segment-based processing; feature gating; detailed error codes; progress reporting; output validation; parameter validation; additional metrics; contract versioning; UI render; optional normalization; GPU logic |
| `onset_extractor` | `components/onset_extractor_AUDIT.md` | `done` | Full audit by AP_AUDIT_CRITERIA.md; segment-based processing; feature gating; detailed error codes; progress reporting; output validation; parameter validation; additional metrics; contract versioning; UI render; optional normalization; no-fallback policy; optional tempo integration |
| `chroma_extractor` | `components/chroma_extractor_AUDIT.md` | `done` | Full audit by AP_AUDIT_CRITERIA.md; segment-based processing; feature gating; detailed error codes; progress reporting; output validation; parameter validation; additional metrics; contract versioning; UI render; optional normalization; no-fallback policy; explicit backend selection |
| `spectral_extractor` | `components/spectral_extractor_AUDIT.md` | `done` | Full audit by AP_AUDIT_CRITERIA.md; segment-based processing; feature gating; detailed error codes; progress reporting; output validation; parameter validation; additional metrics; contract versioning; UI render; optional normalization |
| `quality_extractor` | `components/quality_extractor_AUDIT.md` | `done` | Full audit by AP_AUDIT_CRITERIA.md; segment-based processing; feature gating; detailed error codes; progress reporting; output validation; parameter validation; additional metrics; contract versioning; UI render; optional normalization |
| `rhythmic_extractor` | `components/rhythmic_extractor_AUDIT.md` | `done` | Full audit by AP_AUDIT_CRITERIA.md; segment-based processing; feature gating; detailed error codes; progress reporting; output validation; parameter validation; additional metrics; contract versioning; UI render; optional normalization; no-fallback policy; explicit backend selection; additional librosa/essentia parameters |
| `voice_quality_extractor` | `components/voice_quality_extractor_AUDIT.md` | `done` | Full audit by AP_AUDIT_CRITERIA.md; segment-based processing; feature gating; detailed error codes; progress reporting; output validation; parameter validation; additional metrics; contract versioning; UI render; optional normalization; no-fallback policy; f0 method selection (YIN/PYIN/torchcrepe); optional integration with pitch_extractor |
| `hpss_extractor` | `components/hpss_extractor_AUDIT.md` | `done` | Full audit by AP_AUDIT_CRITERIA.md; segment-based processing; explicit HPSS parameter selection (kernel_size, margin, power); feature gating; detailed error codes; progress reporting; output/parameter validation; additional metrics (harmonic_stability, percussive_stability, separation_quality, balance_score, dominance); spectral features from separated components; contract versioning; UI render; optional audio normalization; per-run storage for .npy files |
| `key_extractor` | `components/key_extractor_AUDIT.md` | `done` | Full audit by AP_AUDIT_CRITERIA.md; Segmenter contract, feature gating, detailed error codes, progress reporting, confidence categorization, additional metrics (stability, distribution, quality), key change detection, contract versioning, UI render, optional audio normalization, integration with chroma_extractor |
| `band_energy_extractor` | `components/band_energy_extractor_AUDIT.md` | `done` | Full audit by AP_AUDIT_CRITERIA.md; Segmenter contract, feature gating, detailed error codes, progress reporting, additional metrics (balance, dynamics, distribution), contract versioning, UI render, optional audio normalization, integration with spectral_extractor |
| `spectral_entropy_extractor` | `components/spectral_entropy_extractor_AUDIT.md` | `done` | Full audit by AP_AUDIT_CRITERIA.md; Segmenter contract, feature gating, detailed error codes, progress reporting, additional metrics (dynamics, distribution, diversity), contract versioning, UI render, optional audio normalization, integration with spectral_extractor |
| `pitch_extractor` | `components/pitch_extractor_AUDIT.md` | `done` | Full audit by AP_AUDIT_CRITERIA.md; signal processing; run_segments support; feature gating; detailed error codes; progress reporting; parameter/output validation; additional metrics (contour_smoothness, jump_count, octave_distribution, centroid, skewness, kurtosis); contract versioning; per-run storage for .npy files |

> Примечание: каталог `components/` может быть пустым до первого завершённого аудита. Это нормально.

---

## Приоритизация аудита

### Критично (baseline Tier-0)
1. ✅ `clap_extractor` — **DONE** (full audit by AP_AUDIT_CRITERIA.md)
2. ✅ `tempo_extractor` — **DONE** (full audit by AP_AUDIT_CRITERIA.md)
3. ✅ `loudness_extractor` — **DONE** (full audit by AP_AUDIT_CRITERIA.md)
4. ✅ `orchestrator_writer` — **DONE** (production-ready)

### Важно (production-ready)
5. ✅ `asr_extractor` — **DONE** (Full audit by `AP_AUDIT_CRITERIA.md` completed)
6. ✅ `speaker_diarization_extractor` — **DONE** (Full audit by `AP_AUDIT_CRITERIA.md` completed)
7. ✅ `emotion_diarization_extractor` — **DONE** (Full audit by `AP_AUDIT_CRITERIA.md` completed)
8. ✅ `source_separation_extractor` — **DONE** (Full audit by `AP_AUDIT_CRITERIA.md` completed)
9. ✅ `speech_analysis_extractor` — **DONE** (Full audit by `AP_AUDIT_CRITERIA.md` completed)

### Опционально (nice-to-have)
8. Остальные signal processing extractors (mfcc, mel, spectral, etc.)
9. `emotion_diarization_extractor`
10. `source_separation_extractor`

---

## Связь с baseline аудитами

Tier-0 baseline extractors имеют два уровня аудитов:
1. **Baseline аудиты** (фокус на baseline контрактах):
   - `docs/baseline/components/audio/CLAP_EXTRACTOR_BASELINE_AUDIT.md`
   - `docs/baseline/components/audio/TEMPO_EXTRACTOR_BASELINE_AUDIT.md`
   - `docs/baseline/components/audio/LOUDNESS_EXTRACTOR_BASELINE_AUDIT.md`

2. **Полные аудиты** (по `AP_AUDIT_CRITERIA.md`):
   - `AudioProcessor/docs/audit_v1/components/clap_extractor_AUDIT.md`
   - `AudioProcessor/docs/audit_v1/components/tempo_extractor_AUDIT.md`
   - `AudioProcessor/docs/audit_v1/components/loudness_extractor_AUDIT.md`

Полные аудиты включают:
- Полный чек-лист по `AP_AUDIT_CRITERIA.md`
- Progress reporting и stage timings
- Feature gating (если применимо)
- UI render скрипты
- Детальная документация параллелизма
- Compliance summary по всем критериям

