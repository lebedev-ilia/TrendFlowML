# TextProcessor — Wave 3 Normalization

Этап нормализации `TextProcessor` (portfolio + production).  
Управляющий план: [../../docs/PORTFOLIO_NORMALIZATION_PLAN.md](../../docs/PORTFOLIO_NORMALIZATION_PLAN.md)  
Журнал: [../../docs/PORTFOLIO_PROGRESS_LOG.md](../../docs/PORTFOLIO_PROGRESS_LOG.md)  
Шаблон (AudioProcessor): [../../AudioProcessor/docs/NORMALIZATION_WAVE2.md](../../AudioProcessor/docs/NORMALIZATION_WAVE2.md)

---

## Статус: `done` (документация и навигация; 2026-05-28)

---

## 1. Структура модуля (as-is)

| Путь | Роль |
|------|------|
| `run_cli.py`, `rlp.py` | CLI entrypoints |
| `src/` | Core + 22 extractors |
| `schemas/` | Machine NPZ schemas (`text_npz_v1`, per-extractor) |
| `config/` | Corpus packs, placeholders |
| `docs/` | MAIN_INDEX, audit_v3, FAISS notes |
| `scripts/` | Smoke audit v3 (`smoke_each_extractor_audit_v3.py`) |

**Extractors:** 22 (см. [MAIN_INDEX.md](MAIN_INDEX.md))

**Ключевые upstream:**
- `AudioProcessor` / `asr_extractor` — token IDs → текст
- `VideoDocument` — title, description, comments, tags
- `dp_models` — `intfloat/multilingual-e5-large` (единая embedding модель)
- FAISS corpus packs — для top-k / cluster extractors

---

## 2. Уже есть (хорошая база)

- [docs/MAIN_INDEX.md](MAIN_INDEX.md) — индекс extractors
- [docs/audit_v3/README.md](audit_v3/README.md) — preflight + 22 audit reports
- [docs/audit_v3/TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md](../../docs/audit_v3/TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md)
- Per-extractor `README.md` в `src/extractors/*/`

---

## 3. DoD Wave 3

- [x] Inventory: 22 extractors, tier/DAG, deps на ASR и embeddings
- [x] [EXTRACTOR_DEPENDENCIES.md](EXTRACTOR_DEPENDENCIES.md)
- [x] Единый layout: `README.md` + `SCHEMA.md` + `docs/FEATURE_DESCRIPTION.md` (22/22)
- [x] Prod smoke checklist (§6 в EXTRACTOR_DEPENDENCIES)
- [x] Ссылки в `docs/MAIN_INDEX.md`, `docs/audit_v3/README.md`

---

## 4. Каноничный doc layout (отличие от AudioProcessor)

| Файл | Путь |
|------|------|
| README | `src/extractors/<name>/README.md` |
| SCHEMA | `src/extractors/<name>/SCHEMA.md` |
| FEATURE_DESCRIPTION | `src/extractors/<name>/docs/FEATURE_DESCRIPTION.md` |

Унификация с Audio (`docs/README.md`) — отложена (массовый перенос ссылок); текущий layout **зафиксирован как canonical для Text**.

---

## 5. Следующий шаг

- `Wave 4`: VisualProcessor (core vs modules, schemas, docs)
---

## Навигация

[TextProcessor](MAIN_INDEX.md) · [DataProcessor](../../docs/MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
