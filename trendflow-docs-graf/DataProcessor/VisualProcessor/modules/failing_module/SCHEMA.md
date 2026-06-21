# `failing_module` — schema

`failing_module` — **utility / test-only** компонент для PR‑4 evidence (демонстрация того, что *optional* module может упасть, не останавливая весь пайплайн).

## Artifact contract

- **NPZ артефакт не создаётся** (компонент завершает работу с exit code `2` до сохранения результатов).
- Соответственно, **JSON schema отсутствует** и `schema_version` не применимы.

## CLI compatibility

Компонент принимает стандартные аргументы VisualProcessor:

- `--frames-dir`
- `--rs-path`

и завершается с ошибкой намеренно.
---

## Навигация

[README](README.md) · [VisualProcessor](../../docs/MAIN_INDEX.md) · [DataProcessor](../../../docs/MAIN_INDEX.md) · [Vault](../../../../docs/MAIN_INDEX.md)
