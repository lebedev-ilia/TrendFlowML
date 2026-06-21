# AudioProcessor — Audit v3 (документация прогона)

Единые критерии Audit v3 для Audio/Text/Segmenter:  
[`DataProcessor/docs/audit_v3/AUDIT_V3_CRITERIA_AUDIO_TEXT_SEGMENTER.md`](../../../docs/audit_v3/AUDIT_V3_CRITERIA_AUDIO_TEXT_SEGMENTER.md) — раздел **§5 AudioProcessor**.

Карта экстракторов и ссылки на README:  
[`AudioProcessor/docs/MAIN_INDEX.md`](../MAIN_INDEX.md)

Dev run-log репозитория:  
[`DataProcessor/docs/audit_v3/RUN_LOG.md`](../../../docs/audit_v3/RUN_LOG.md)

Декларативный DAG (cross-processor):  
[`DataProcessor/docs/reference/component_graph.yaml`](../../../docs/reference/component_graph.yaml)

---

## Отчёты по компонентам

| Extractor | Report | Questions |
|-----------|--------|-----------|
| asr_extractor | [report](components/asr_extractor_AUDIT_V3_REPORT.md) | — |
| band_energy_extractor | [report](components/band_energy_extractor_AUDIT_V3_REPORT.md) | — |
| chroma_extractor | [report](components/chroma_extractor_AUDIT_V3_REPORT.md) | — |
| clap_extractor | [report](components/clap_extractor_AUDIT_V3_REPORT.md) | — |
| emotion_diarization_extractor | [report](components/emotion_diarization_extractor_AUDIT_V3_REPORT.md) | — |
| hpss_extractor | [report](components/hpss_extractor_AUDIT_V3_REPORT.md) | — |
| key_extractor | [report](components/key_extractor_AUDIT_V3_REPORT.md) | — |
| loudness_extractor | [report](components/loudness_extractor_AUDIT_V3_REPORT.md) | — |
| mel_extractor | [report](components/mel_extractor_AUDIT_V3_REPORT.md) | — |
| mfcc_extractor | [report](components/mfcc_extractor_AUDIT_V3_REPORT.md) | — |
| onset_extractor | [report](components/onset_extractor_AUDIT_V3_REPORT.md) | — |
| pitch_extractor | [report](components/pitch_extractor_AUDIT_V3_REPORT.md) | — |
| quality_extractor | [report](components/quality_extractor_AUDIT_V3_REPORT.md) | — |
| rhythmic_extractor | [report](components/rhythmic_extractor_AUDIT_V3_REPORT.md) | — |
| source_separation_extractor | [report](components/source_separation_extractor_AUDIT_V3_REPORT.md) | — |
| speaker_diarization_extractor | [report](components/speaker_diarization_extractor_AUDIT_V3_REPORT.md) | — |
| spectral_entropy_extractor | [report](components/spectral_entropy_extractor_AUDIT_V3_REPORT.md) | — |
| spectral_extractor | [report](components/spectral_extractor_AUDIT_V3_REPORT.md) | — |
| speech_analysis_extractor | [report](components/speech_analysis_extractor_AUDIT_V3_REPORT.md) | [questions](components/speech_analysis_extractor_AUDIT_V3_QUESTIONS_R1.md) |
| tempo_extractor | [report](components/tempo_extractor_AUDIT_V3_REPORT.md) | [questions](components/tempo_extractor_AUDIT_V3_QUESTIONS_R1.md) |
| voice_quality_extractor | [report](components/voice_quality_extractor_AUDIT_V3_REPORT.md) | [questions](components/voice_quality_extractor_AUDIT_V3_QUESTIONS_R1.md) |

---

## Связанные индексы

- [TextProcessor Audit v3](../../TextProcessor/docs/audit_v3/README.md)
- [DataProcessor docs MAIN_INDEX](../../../docs/MAIN_INDEX.md)
- [Vault root INDEX](../../../../docs/MAIN_INDEX.md)
---

## Навигация

[AudioProcessor](../MAIN_INDEX.md) · [DataProcessor](../../../docs/MAIN_INDEX.md) · [Vault](../../../../docs/MAIN_INDEX.md)
