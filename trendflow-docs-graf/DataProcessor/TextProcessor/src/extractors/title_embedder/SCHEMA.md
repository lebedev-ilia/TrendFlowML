# `title_embedder` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `title_embedder` |
| Класс | `TitleEmbedder` |
| Machine schema | `DataProcessor/TextProcessor/schemas/title_embedder_output_v1.json` |
| `schema_version` (логический контракт `features_flat`) | `title_embedder_output_v1` |
| Версия реализации | `1.2.0` (см. `TitleEmbedder.VERSION`) |

## Назначение

Считать **плотный эмбеддинг** заголовка (`doc.title`) через `dp_models` / sentence-transformers (offline), сохранить вектор в per-run **`title_embedding.npy`** (опционально) и выставить **скалярные метрики** в `result.features_flat` (`tp_titleemb_*`) для NPZ и мониторинга.

## Audit v3 preflight (модель)

В полном Audit v3 TextProcessor зафиксирована единая модель эмбеддингов: **`intfloat/multilingual-e5-large`** через ModelManager ([preflight §0.5](../../../../docs/audit_v3/TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md)). Конструктор по умолчанию в коде может указывать другую модель для обратной совместимости — **профиль прогона** обязан задать `model_name` согласно политике аудита.

## Upstream

- **`TagsExtractor`** (если включён раньше в `MainProcessor`): заголовок может быть **очищен от inline `#тегов`**; этот экстрактор **не** мутирует `doc`.
- Пустой title: **валидный empty** (`tp_titleemb_title_present=0`, эмбеддинг не считается), если только не **`require_title=true`** → **fail-fast**.

## Артефакты vs `features_flat`

| Что | Где |
|-----|-----|
| Вектор dim D, float32, L2-normalized | `title_embedding.npy` (если `write_artifact=true`), плюс `doc.tp_artifacts["embeddings"]["title"]` с `relpath`, `model_name`, `weights_digest`, `model_version`, `dim` |
| Скаляры для агрегата NPZ | `result.features_flat` |

В **`result`** нет сырого вектора (размер/конфиденциальность); только метаданные (`model_name`, `model_version`, `weights_digest`) и `features_flat`.

## Полный перечень `features_flat`

Source of truth: `main.py` → `_stable_features_template()` и `features_flat.update(...)` после кодирования.

| Ключ | Смысл |
|------|--------|
| `tp_titleemb_present` | 1.0 если эмбеддинг реально посчитан |
| `tp_titleemb_dim` | Размерность вектора или NaN |
| `tp_titleemb_norm_raw` | L2 норма **до** финальной нормализации (если `compute_raw_norm`), иначе NaN |
| `tp_titleemb_l2_norm` | L2 норма итогового вектора (ожидается ≈1) |
| `tp_titleemb_title_present` | 1.0 если title непустой после нормализации |
| `tp_titleemb_require_title_enabled` | Параметр `require_title` |
| `tp_titleemb_compute_enabled` | Параметр `compute_embedding` |
| `tp_titleemb_write_artifact_enabled` | Параметр `write_artifact` |
| `tp_titleemb_artifact_written` | Файл артефакта успешно записан |
| `tp_titleemb_cache_enabled` | Дисковый кеш включён |
| `tp_titleemb_cache_hit` | 0/1 при включённом кеше; иначе по веткам 0.0 или NaN |
| `tp_titleemb_fp16` | fp16 на GPU |
| `tp_titleemb_device_cuda` | CUDA device |
| `tp_titleemb_model_digest_u24` | Первые 6 hex символов `weights_digest` как float |
| `tp_titleemb_encode_ms` | Время encode (мс) |
| `tp_titleemb_compute_raw_norm` | Параметр `compute_raw_norm` |

Параметр **`emit_extra_metrics`** в текущей версии **не добавляет** ключей в `features_flat` (зарезервирован).

## Версионирование

Изменение набора или смысла ключей → bump **`title_embedder_output_v2`** + `RUN_LOG.md` + отчёт компонента.
---

## Навигация

[README](README.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
