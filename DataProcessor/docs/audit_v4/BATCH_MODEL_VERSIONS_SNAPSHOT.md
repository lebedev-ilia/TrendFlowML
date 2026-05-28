# Снимок версий моделей и сервисов (батч, чеклист п. 1.2)

**Дата фиксации:** 2026-04-15  
**Источник конфигурации моделей:** [`DataProcessor/configs/global_config.yaml`](../../configs/global_config.yaml) (см. также [BATCH_FULL_PROFILE_REFERENCE.md](BATCH_FULL_PROFILE_REFERENCE.md)).

## Конфиг `global_config.yaml` (ключевые имена / размеры)

| Подсистема | Параметр | Значение в шаблоне |
|------------|----------|---------------------|
| Схема конфига | `version` | `1.0.0` |
| ASR (Whisper) | `processors.audio.extractors.asr.model_size` | `small` |
| ASR alt | `emotion_diarization` / др. (см. YAML) | `model_size: small` / `large` по секциям |
| Text embedders | `model_name` (title/description/… ) | `intfloat/multilingual-e5-large` |
| CLIP (in-process) | `core_clip.model_name` | `ViT-B/32` |
| Triton specs | `core_clip` | `clip_image_224_triton`, `clip_text_triton` |
| Triton specs | `core_depth_midas` | `midas_256_triton` |
| Triton specs | `core_optical_flow` | `raft_256_triton` |
| Детекции | `core_object_detections.model` | `visual/yolo/yolo11x_41_best.pt` (от `DP_MODELS_ROOT`) |
| Endpoint | `inline_config.global.triton_http_url` | `http://localhost:8000` (в бою заменить на реальный Triton) |

Точные версии **развёрнутых** моделей в Triton задаются образом/репозиторием ModelManager — их нужно дописать под ваш стенд (имя контейнера, тег image, `triton_model_spec` → ревизия).

## Зафиксированные зависимости Python (TextProcessor)

Файл: [`DataProcessor/TextProcessor/requirements.txt`](../../TextProcessor/requirements.txt) (жёсткие пины, актуальны на дату снимка).

| Пакет | Версия |
|-------|--------|
| numpy | 1.26.4 |
| sentence-transformers | 5.2.2 |
| transformers | 5.0.0 |
| huggingface_hub | 1.3.7 |
| torch (рекомендация в комментарии файла) | `2.9.1+cu126` (+ `torchaudio`, `torchvision` с тем же индексом PyTorch) |
| triton (Python, не сервер) | 3.5.1 |

Полный список — в самом `requirements.txt`. Окружения **Audio** / **Visual** могут ставить `torch` и пакеты отдельно; для воспроизводимости батча зафиксируйте **`pip freeze`** или единый lockfile в п. 1.2 при первом успешном пилоте.

## Прочие сервисы

| Сервис | Примечание |
|--------|------------|
| **Triton Inference Server** | Версия = образ/деплой; URL из env / `global_config` |
| **OpenFace** (micro_emotion) | Docker-образ по документации модуля |
| **Fetcher / ingestion** | Версия стека — по `backend` / docker-compose при батче через API |

## Text — политика батча (чеклист п. 1.5–1.6), 2026-04-15

| Параметр | Решение |
|----------|---------|
| **`emit_extra_metrics` / `compute_std`** | **Везде `true`** в замороженном YAML (шаблон `global_config.yaml` часто `false` — переопределить). |
| **FAISS vs NumPy** | **Только FAISS**; окружение с `faiss-*`; не полагаться на numpy fallback для батча. |

Подробности: [BATCH_FULL_PROFILE_REFERENCE.md](BATCH_FULL_PROFILE_REFERENCE.md) § Text.

## Config hash (чеклист п. 1.3), 2026-04-15

Вычислено по текущему [`DataProcessor/configs/global_config.yaml`](../../configs/global_config.yaml) (без profile / без CLI — **не** полный run-hash из `main.py`):

| Метрика | Значение |
|---------|----------|
| **16-hex** (`sha256( yaml.safe_dump({"global_config": …}, sort_keys=True, allow_unicode=True).encode() )[:16]`) | `be0de63d921ffaf1` |
| **SHA256** сырого файла конфига | `958075440cbc45a3e79f6220103a94ad09b9d9a66c67c4585c4001c8cb7f07b9` |

Перед Go пересчитайте hash на **итоговом** замороженном YAML, если он отличается от шаблона или в прогоне участвуют нетривиальный `profile` / `visual_cfg` / аргументы CLI (см. `DataProcessor/main.py`, сборка `cfg_for_hash`).

## Обновление

После смены `global_config.yaml`, образа Triton или `requirements.txt` обновите этот файл и строки **1.2–1.3** в [CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md](CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md).
