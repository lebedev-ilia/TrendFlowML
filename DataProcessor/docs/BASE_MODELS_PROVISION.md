# Offline-провижен публичных базовых моделей (A2)

Базовые публичные модели **не** хранятся в едином HF-репо `trendflow_models`
(раздел `public_base_models` манифеста), но нужны компонентам. На мульти-ноде без
них ноды пойдут в сеть (нарушение no-network). Скрипт кладёт их в канонические
пути под `DP_MODELS_ROOT` (= `DataProcessor/dp_models`).

Скрипт: `DataProcessor/scripts/provision_base_models.py`

## Использование (на машине с `.data_venv` и сетью)

```bash
# что нужно и что уже есть
python DataProcessor/scripts/provision_base_models.py --list

# показать команды без выполнения
python DataProcessor/scripts/provision_base_models.py --dry-run

# всё доступное
python DataProcessor/scripts/provision_base_models.py

# подмножество
python DataProcessor/scripts/provision_base_models.py --only e5 source_separation

# gated pyannote (нужен HF_TOKEN + принятая лицензия на hf.co/pyannote/*)
HF_TOKEN=hf_xxx python DataProcessor/scripts/provision_base_models.py --only pyannote
```

## Реестр и стратегии

| id | что | стратегия | канонический путь (под DP_MODELS_ROOT) | примечание |
|---|---|---|---|---|
| `e5` | intfloat/multilingual-e5-large (Text) | script | `text/embeddings/intfloat_multilingual-e5-large` | **критично** для TextProcessor offline |
| `source_separation` | source separation large (Audio) | script | `audio/source_separation/large.pt` | — |
| `pyannote` | speaker-diarization | script (**gated**) | `audio/pyannote_speaker_diarization/` | нужен HF_TOKEN + лицензия |
| `wavlm_large` | microsoft/wavlm-large | hf_snapshot | `hf_cache/hub/` | база speechbrain emotion |
| `wav2vec2_base` | facebook/wav2vec2-base | hf_snapshot | `hf_cache/hub/` | опционально |
| `clap_630k` | LAION CLAP 630k | manual | `audio/laion_clap/clap_ckpt.pt` | скачать из релизов LAION-AI/CLAP |
| `places365_resnet50` | Places365 ResNet50 | manual | `visual/places365/resnet50_places365.pth.tar` | уже есть в Triton (ONNX) |

Стратегии: `script` — вызывает существующий `save_*/download_*`; `hf_snapshot` —
`huggingface_hub.snapshot_download` в `hf_cache`; `manual` — печатает URL/инструкцию
(fragile-источники намеренно не хардкодятся).

## Интеграция

- **Локально/staging:** запускать после `bootstrap.sh` (или добавить шаг
  `--with-base-models`, вызывающий этот скрипт после Phase 4/models).
- **k8s (мульти-нода):** запускать провижен на машине, которая наполняет
  `models-pvc` (можно добавить в `model-download` Job шаг вызова этого скрипта в
  образе DataProcessor — там есть torch/sentence-transformers). Тогда базовые
  модели попадут на общий PVC вместе с весами из `trendflow_models`.

## Проверено
- `--list` и `--dry-run` отрабатывают; gated pyannote корректно пропускается без
  токена; manual-модели печатают инструкцию. Реальная загрузка — на машине с сетью
  и `.data_venv` (torch/sentence-transformers/huggingface_hub).
