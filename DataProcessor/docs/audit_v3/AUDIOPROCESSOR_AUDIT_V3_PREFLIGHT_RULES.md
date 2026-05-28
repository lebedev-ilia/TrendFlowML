# Audit v3 — AudioProcessor preflight rules (source-of-truth)
Дата: 2026-02-22  
Статус: FINAL (для текущего audit v3)

Этот документ фиксирует **рабочие правила старта аудита AudioProcessor**: audit pack, порядок, sampling policy, схемы, ModelManager/no-network, privacy и run-log дисциплина.

Связанные документы:
- Global audit rules (source-of-truth): `DataProcessor/docs/audit_v3/DECISIONS_AND_RULES.md`
- Главный чеклист аудита Audio/Text/Segmenter: `DataProcessor/docs/audit_v3/AUDIT_V3_CRITERIA_AUDIO_TEXT_SEGMENTER.md`
- Артефакты/meta/empty_reason/NPZ: `DataProcessor/docs/contracts/ARTIFACTS_AND_SCHEMAS.md`
- Система схем (human+machine+validation): `DataProcessor/docs/contracts/SCHEMAS_SYSTEM.md`
- Segmenter contract (audio/segments.json): `DataProcessor/docs/contracts/SEGMENTER_CONTRACT.md`
- Privacy/retention: `DataProcessor/docs/contracts/PRIVACY_AND_RETENTION.md`
- Model interface v2: `Models/docs/contracts/MODEL_INTERFACE_V2.md`

---

## 0) Audit pack (smoke set) — FINAL

**FINAL**: в качестве стартового audit pack (smoke) используем 3 видео:
- `example/example_videos/video1_fixed.mp4`
- `example/example_videos/video2_fixed.mp4`
- `example/example_videos/video3_fixed.mp4`

Важно (реальность repo, зафиксировано ffprobe):
- эти `*_fixed.mp4` **не содержат audio stream** (только video track),
- поэтому они отлично проверяют **empty semantics** и “не падать без аудио”,
- но **не покрывают** проверку реальной аудио‑экстракции (ASR/CLAP/tempo/loudness/…).

**Дополнение (AudioProcessor validation set, audio-present)**:
- `example/example_videos/video1.mp4`
- `example/example_videos/video2.mp4`
- `example/example_videos/video3.mp4`

Это дополнение используется только для QA аудио‑экстракции и не отменяет fixed‑smoke set.

Правила использования:
- smoke прогоняем **перед** любыми существенными правками контракта/алгоритмов;
- после существенных правок прогоняем снова и фиксируем запись в `RUN_LOG.md`;
- расширенный validation pack (10–20 видео) добавим после стабилизации Tier‑0 (см. критерии аудита).

---

## 1) Порядок аудита AudioProcessor — FINAL

**FINAL**: аудитируем **все** аудио-компоненты (все extractors AudioProcessor). Порядок выбираем так, чтобы:
- сначала стабилизировать **базовые signal-processing** контракты и time-axis semantics,
- затем переходить к **model-based** и **privacy-sensitive** экстракторам,
- bundle/агрегаторы аудитировать после их deps.

Рекомендуемый порядок (working order, можно корректировать по факту зависимостей, но фиксируем изменения в run-log):
1. Tier‑0 baseline: `loudness_extractor` ✅, `tempo_extractor`, `clap_extractor`, `asr_extractor` ✅
2. Signal / вспомогательные: `quality_extractor`, `spectral_extractor`, `mfcc_extractor`, `mel_extractor`, `chroma_extractor`, `pitch_extractor`, `rhythmic_extractor`, `onset_extractor`
3. Derivatives (опциональные переиспользования): `key_extractor`, `voice_quality_extractor`, `band_energy_extractor`, `spectral_entropy_extractor`, `hpss_extractor`
4. Speech/diarization (privacy-sensitive): `speaker_diarization_extractor`, `emotion_diarization_extractor`
5. Separation: `source_separation_extractor`
6. Bundle/aggregator: `speech_analysis_extractor` (после `asr` + `speaker_diarization` + (опц.) `pitch`)

**Статус аудита (2026-02-22)**:
- ✅ `loudness_extractor` — Audit v3 complete (`loudness_extractor_npz_v1`)
- ✅ `asr_extractor` — Audit v3 complete (`asr_extractor_npz_v1`, строгий token contract, privacy-safe outputs, sampling policy knobs)

---

## 2) Unified sampling policy rules (Audio) — FINAL

Контекст: конечная цель моделей — **предсказание популярности**; нам нужна **детерминированность, воспроизводимость и качество** сигналов без “тихих” отклонений.

**FINAL (unified)**:
- **Segmenter — единственный владелец sampling**. AudioProcessor не “придумывает” окна сам.
- Каждый аудио-extractor должен иметь **явно определённый required family** в `frames_dir/audio/segments.json` (schema `audio_segments_v1`).
- **No-fallback по sampling**:
  - если extractor включён и required family отсутствует/пустой → **fail-fast** (error), кроме случаев, которые явно признаны “valid empty” (см. ниже);
  - запрещены неявные подмены сегментов “возьмём primary/spectral вместо X” для audited режима.
- **Допускаемые отхождения (exceptions)**: если для качества популярности реально нужно “shared family” или fallback,
  - это должно быть **явно задокументировано** в README/SCHEMA.md конкретного extractor’а как **exception**,
  - иметь чёткую мотивацию (“почему так лучше для качества/стабильности”),
  - и приводить к bump `schema_version` (контракт меняется).

Empty semantics (audio) — правило:
- если аудио отсутствует как модальность (например, видео без аудио/с нулевой дорожкой) → допустим `status="empty"` с каноничным `empty_reason` (см. `ARTIFACTS_AND_SCHEMAS.md`);
- если sampling family отсутствует из-за некорректной работы Segmenter → это **error** (так как нарушен контракт upstream).

---

## 3) ModelManager-only + no-network enforcement — FINAL

**FINAL**:
- Любая модель/веса/токенизатор должны резолвиться через `dp_models` (ModelManager).
- Любые сетевые загрузки в runtime запрещены для audited компонентов.
- Если компонент технически требует внешних моделей (например HF/whisperx/pyannote) и пока не заведён в `dp_models`,
  - компонент **не может** считаться `audited v3 passed`, пока не будет полностью offline через ModelManager,
  - временно допускается режим rollout (dev-only) с явной маркировкой в README как “NOT AUDITED / NETWORK-DEPENDENT”.

Reproducibility:
- `meta.models_used[]` и `meta.model_signature` должны отражать **фактически использованные** модели (runtime/engine/precision/device/weights_digest).
- Запрещено писать в meta “*_triton”, если реально используется “*_inprocess” (и наоборот).

---

## 4) Schemas system (Audio) — FINAL

**FINAL**: каждый audited extractor должен быть “known schema” и валидироваться fail-fast:
- Human schema: `DataProcessor/AudioProcessor/src/extractors/<name>/SCHEMA.md`
- Machine schema (vp_schema_v1): `DataProcessor/AudioProcessor/schemas/<schema_version>.json`

Дополнение:
- рекомендуется перейти от общего `schema_version="audio_npz_v1"` к **per-extractor schema_version**, чтобы:
  - фиксировать keys/dtype/shape без расплывчатости,
  - включить `allow_extra_keys=false` для audited состояния,
  - безопасно развивать контракты через bump версий.

---

## 5) Privacy rules (Audio) — FINAL

**FINAL**:
- raw text/транскрипты/слова по умолчанию **не сохраняем** в model_facing артефактах;
- любые raw поля допустимы только как **debug-only** под явным retain/enable флагом, с предупреждением в render/README.

---

## 6) Run-log discipline — FINAL

**FINAL**: после любых существенных изменений (контракт/версии/алгоритм/политики sampling/модели) обязателен факт-лог в:
- `DataProcessor/docs/audit_v3/RUN_LOG.md`

Минимум для аудио в run-log:
- `audio_duration_sec`, `sample_rate`
- `N_segments` по каждой required family
- stats длительностей сегментов: min/p50/p90/max
- если есть sequences/tokens: `N_tokens`, доля masked/empty


