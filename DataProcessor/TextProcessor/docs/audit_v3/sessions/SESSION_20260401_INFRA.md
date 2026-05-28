# Session — TextProcessor Audit v3 infrastructure (2026-04-01)

## Scope

Закрытие инфраструктуры до первого компонентного отчёта: **граф зависимостей в коде**, **machine schema** для `text_npz_v1`, **CLI**, **декларативный граф**, **плейсхолдеры corpus packs**, обновление **preflight / SCHEMAS_SYSTEM / индексов**.

## Code

- `src/core/main_processor.py` — `DEPENDENCIES`: `TagsExtractor` — корень Tier-0; `LexicalStatsExtractor`, `ASRTextProxyExtractor`, `DescriptionEmbedder` зависят от `TagsExtractor` (раньше `DescriptionEmbedder` был без этой связи).
- `run_cli.py` — дефолтный `devices_config.cpu`: `TagsExtractor` первым; флаг `--require-known-schema` → `validate_npz(..., require_known_schema=...)`; пример в help для `--devices-config-json`.
- Инициализация `v_ok` перед записью NPZ (корректный `error_code` при исключении до валидации).

## Schemas

- `schemas/text_npz_v1.json` — контракт `text_features.npz` (`feature_names`, `feature_values`, `payload`, `meta`, опционально `primary_embedding*`).
- `schemas/README.md` — краткий реестр.

## Config / docs

- `config/corpus_packs.placeholder.yaml` — заблаговременные плейсхолдеры pack’ов (similar titles / cluster entropy / taxonomy).
- `DataProcessor/docs/contracts/SCHEMAS_SYSTEM.md` — явная секция TextProcessor.
- `DataProcessor/docs/audit_v3/TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md` — ссылка на placeholder YAML (§7).
- `DataProcessor/docs/reference/component_graph.yaml` — стадия `text_processor_tier0` (tags → lexical / asr_proxy / title_embedder / description_embedder).

## Next

Первый компонентный аудит по preflight (Tier-0), например `langs_extractor` / `tags_extractor` — после подтверждения владельцем.
