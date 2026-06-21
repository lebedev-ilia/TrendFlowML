# AudioProcessor — Audit v4: общий итог (L2, набор **A+B**)

**Дата сводки:** 2026-04-13  
**Опорный run (набор A, воспроизводимый):** `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`  
**План и критерии:** [AUDIT_4_CRITERIA_AND_PLAN.md](AUDIT_4_CRITERIA_AND_PLAN.md)  
**Журнал прогонов:** [RUN_LOG.md](RUN_LOG.md)

## Статус волны

**Итог по AudioProcessor:** **все 21 компонент закрыты на уровне L2 (A+B)** (product stats собраны; отчёты и `RUN_LOG.md` обновлены; есть ссылки на JSON+figures для каждого компонента).

**Итог по программе (субъективно):** **~8 / 10** — контракты и NPZ в целом согласованы; выявлен и частично устранён класс дефектов «категориальная строка / несогласованный флаг → NaN или вводящие в заблуждение meta». Для «продуктово зелёного» уровня всё ещё нужны **golden §4.8**, **набор C**, и **L3/§8** (см. ниже).

## Что закрыто / что осталось

### Закрыто (на уровне L2)

- **A+B статистика** (JSON + figures) для всех компонентов.
- **Сквозная навигация**: отчёты L2 + ссылки в `RUN_LOG.md`.
- **Audit v4.2 (engineering bridge, после L2)**: добавлены `meta.stage_timings_ms` и env-gated resource profiles по большинству компонентов (см. `components/audit_4_2/README.md`).

### Осталось (для L3 / DoD и полной 4.2)

- **Set C (edge cases)**: silence/too short/invalid inputs/пограничные семейства сегментов.
- **Golden / hash (§4.8)**: фиксация сигнатуры по свежим NPZ.
- **Повторный прогон A** там, где были фиксы (для чистого golden на актуальном коде).
- **Audit v4.2 long-run**: серия/батчи для измерения wall-time/RSS/GPU “before/after” + оркестраторные отчёты (`scheduler_runtime_report.json`) — шаги §12.1–12.2.

## Сквозные темы

1. **Tabular = только float.** Строки (`device_used`, `backend`, `f0_method`, и т.д.) не должны проходить через `add()` → `as_float` → **NaN**. Исправления в соответствующих `npz_savers/*.py` + перенос в **meta** / optional keys схемы.
2. **Согласованность `meta.features_enabled` с фактическим содержимым.** Пример: `speech_analysis_extractor` — `pitch_metrics` в meta без мержей pitch → лишние NaN; исправлено в `main.py` (`_features_enabled`).
3. **Полный payload в `run_segments`.** Пример: `spectral_extractor` — отсутствие `hop_length` / `n_fft` / `duration` в payload при сегментном режиме → NaN; плюс `device_used` в tabular.
4. **Документация vs савер.** Примеры: `speaker_diarization_extractor` (F=10 vs «минимум 7» в старых docs); уточнения по семантике `duration` / охвату окон (`rhythmic_extractor`, `tempo_extractor`).
5. **Тяжёлые старые артефакты.** Многие записи в `RUN_LOG` помечены: после фиксов нужен **повторный прогон A** для §4.8 и чистой статистики NaN.

## Таблица по компонентам

Оценка в колонке «вердикт отчёта» — из соответствующего `components/audio_processor/*_audit_v4.md` (округлённо, условно «после доводки L2/L3» там, где явно указано).

| Компонент | Отчёт | Вердикт (из отчёта) | Ключевое на **A** / фикс |
|-----------|--------|---------------------|---------------------------|
| `asr_extractor` | [asr_extractor_audit_v4.md](components/audio_processor/asr_extractor_audit_v4.md) | ~**9**/10 | N=1 окно на run; эм. диаризация и др. |
| `band_energy_extractor` | [band_energy_extractor_audit_v4.md](components/audio_processor/band_energy_extractor_audit_v4.md) | ~**8**/10 | **L2** (**A+B**); `meta.duration` сегментного e2e; **C** — TODO |
| `chroma_extractor` | [chroma_extractor_audit_v4.md](components/audio_processor/chroma_extractor_audit_v4.md) | ~**8**/10 | **L2** (**A+B**); time_series / сегментный режим; **C** — TODO |
| `clap_extractor` | [clap_extractor_audit_v4.md](components/audio_processor/clap_extractor_audit_v4.md) | ~**8.5–9**/10 | **L2** (**A+B**); `device_used` vs `models_used` |
| `emotion_diarization_extractor` | [emotion_diarization_extractor_audit_v4.md](components/audio_processor/emotion_diarization_extractor_audit_v4.md) | ~**8**/10 | **L2** (**A+B**); dict в NPZ; meta model fields |
| `key_extractor` | [key_extractor_audit_v4.md](components/audio_processor/key_extractor_audit_v4.md) | ~**8**/10 | **L2** (**A+B**); строки в tabular → NaN; фикс савера |
| `loudness_extractor` | [loudness_extractor_audit_v4.md](components/audio_processor/loudness_extractor_audit_v4.md) | ~**9**/10 | **L2** (**A+B**); NaN 0 на **A** |
| `hpss_extractor` | [hpss_extractor_audit_v4.md](components/audio_processor/hpss_extractor_audit_v4.md) | ~**7.5→8.5**/10 | **L2** (**A+B**); много NaN + time_series (фикс в коде) |
| `mel_extractor` | [mel_extractor_audit_v4.md](components/audio_processor/mel_extractor_audit_v4.md) | ~**8**/10 | **L2** (**A+B**); `device_used` в tabular → NaN (фикс савера) |
| `mfcc_extractor` | [mfcc_extractor_audit_v4.md](components/audio_processor/mfcc_extractor_audit_v4.md) | ~**8**/10 | **L2** (**A+B**); `device_used` → NaN (фикс савера) |
| `onset_extractor` | [onset_extractor_audit_v4.md](components/audio_processor/onset_extractor_audit_v4.md) | ~**8**/10 | **L2** (**A+B**); `backend` → NaN (фикс савера) |
| `pitch_extractor` | [pitch_extractor_audit_v4.md](components/audio_processor/pitch_extractor_audit_v4.md) | **8**/10 | **L2** (**A+B**); `backend` строкой в `meta` (не в tabular) |
| `quality_extractor` | [quality_extractor_audit_v4.md](components/audio_processor/quality_extractor_audit_v4.md) | ~**8**/10 | **L2** (**A+B**); `device_used` строкой в `meta` (не в tabular) |
| `rhythmic_extractor` | [rhythmic_extractor_audit_v4.md](components/audio_processor/rhythmic_extractor_audit_v4.md) | ~**8.5**/10 | **L2** (**A+B**); чистый tabular; семантика `duration_sec` |
| `source_separation_extractor` | [source_separation_extractor_audit_v4.md](components/audio_processor/source_separation_extractor_audit_v4.md) | ~**8.5**/10 | **L2** (**A+B**); N=1 на **A**; строки в meta |
| `speaker_diarization_extractor` | [speaker_diarization_extractor_audit_v4.md](components/audio_processor/speaker_diarization_extractor_audit_v4.md) | ~**9**/10 | **L2** (**A+B**); выравнивание SCHEMA (F=10) |
| `spectral_entropy_extractor` | [spectral_entropy_extractor_audit_v4.md](components/audio_processor/spectral_entropy_extractor_audit_v4.md) | ~**8.5**/10 | **L2** (**A+B**); F=2 tabular; чисто на **A** |
| `spectral_extractor` | [spectral_extractor_audit_v4.md](components/audio_processor/spectral_extractor_audit_v4.md) | ~**8**/10 | **L2** (**A+B**); исторически 4 NaN на старом A (device + пропуски run_segments) |
| `speech_analysis_extractor` | [speech_analysis_extractor_audit_v4.md](components/audio_processor/speech_analysis_extractor_audit_v4.md) | ~**8**/10 | **L2** (**A+B**); `pitch_metrics` vs отсутствие pitch |
| `tempo_extractor` | [tempo_extractor_audit_v4.md](components/audio_processor/tempo_extractor_audit_v4.md) | ~**8.5**/10 | **L2** (**A+B**); чистый tabular; meta без contract |
| `voice_quality_extractor` | [voice_quality_extractor_audit_v4.md](components/audio_processor/voice_quality_extractor_audit_v4.md) | ~**8**/10 | **L2** (**A+B**); `f0_method` → NaN |

**Всего компонентов в таблице:** **21**; **все 21 — L2 (A+B)** (L1 = 0).

## Следующие шаги (общие)

1. Зафиксировать **git commit** в `RUN_LOG.md` после стабилизации фиксов.  
2. **Повторный прогон A** для компонентов с исправленными саверами / payload.  
3. **Golden / hash (§4.8)** по свежим NPZ.  
4. **Набор B** (≥5 видео): распределения, корреляции плана §4.6.  
5. **Набор C**: empty, короткое аудио, edge по семействам сегментов.

---

*Файл сгенерирован для навигации по волне Audit v4; детали и таблицы критериев — в индивидуальных `components/audio_processor/*_audit_v4.md`.*
---

## Навигация

[Audit v4 hub](components/audit_4_2/README.md) · [DataProcessor](../MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
