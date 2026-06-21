# VisualProcessor Schemas (production)

This directory contains **machine-readable** schemas for VisualProcessor artifacts (primarily NPZ).

## Goals

- **Strict typing**: validate keys, dtypes, and shapes.
- **Versioning**: schemas are indexed by `meta.schema_version` (NPZ is the source-of-truth).
- **Tiering**: each field is labeled as one of:
  - `model_facing` — intended for downstream models / encoders,
  - `analytics` — for analysis/interpretation,
  - `debug` — render/QA/diagnostics, not part of production contract.
- **Fail-fast** (for known schemas): if an artifact claims a `schema_version` that has a schema here,
  the validator will raise errors on violations.

## Contract spec

See the formal schema-system contract (Audit v3 / production):

- `DataProcessor/docs/contracts/SCHEMAS_SYSTEM.md`

## Files

- `*.json`: schema documents keyed by `schema_version` (e.g. `core_clip_npz_v2.json`).
- Python implementation:
  - `registry.py`: loads schema JSON documents.
  - `npz_validator.py`: validates `.npz` artifacts against a schema.
  - `cli.py`: validate one or more `.npz` paths (dev/CI tooling).

## Schema format (vp_schema_v1)

Each schema JSON has the following top-level keys:

- `schema_system_version`: must be `"vp_schema_v1"`.
- `schema_version`: the NPZ `meta.schema_version` this schema validates.
- `producer`: component name (matches NPZ `meta.producer`).
- `artifact_kind`: `"npz"`.
- `allow_extra_keys`: if `false`, unknown NPZ keys are errors.
- `meta`:
  - `required_keys`: list of required keys inside `meta` dict.
  - `optional_keys`: list of optional keys inside `meta` dict.
- `fields`: mapping `npz_key -> FieldSpec`.

`FieldSpec`:

- `required`: bool
- `tier`: `"model_facing" | "analytics" | "debug"`
- `dtype`: `"float32" | "int32" | "bool" | "str" | "object" | ...` or list of such tokens
- `shape`: list of dims or `null`
  - dims can be integers or strings (symbolic, can include `+/- int`, e.g. `"N-1"`).
  - empty list `[]` means scalar (ndim=0).
- `description`: optional string (human docs)
---

## Навигация

[VisualProcessor](../docs/MAIN_INDEX.md) · [DataProcessor](../../docs/MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
