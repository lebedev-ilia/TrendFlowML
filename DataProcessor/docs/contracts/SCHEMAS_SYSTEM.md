# Schema System (production) — human + machine contracts

Этот документ фиксирует **общую систему схем** артефактов DataProcessor (Audit v3 / production-ready).
Он дополняет `ARTIFACTS_AND_SCHEMAS.md` и формализует **машиночитаемые схемы** + правила валидатора.

---

## 1) Зачем это нужно

Цель — сделать контракты компонентов **строгими и проверяемыми**:

- NPZ остаётся **source-of-truth**, но контракт NPZ должен быть:
  - понятен человеку (`SCHEMA.md` рядом с кодом),
  - проверяем машиной (schema JSON + runtime/CI validation).
- Любое “тихое” изменение ключей/типов/форм должно ловиться **fail-fast** (для audited компонентов).

---

## 2) Термины

- **Human schema**: `SCHEMA.md` рядом с компонентом (human-friendly).
- **Machine schema**: JSON‑документ в общем реестре схем (keys/dtype/shape/tiers/required).
- **Tier**:
  - `model_facing` — идёт в downstream модели/энкодеры,
  - `analytics` — для аналитики/интерпретации,
  - `debug` — для QA/рендера/диагностики, не часть прод контракта.
- **Known schema**: `meta.schema_version` присутствует в реестре схем и валидируется строго.

---

## 3) Layout: где лежат схемы

### VisualProcessor (v1: реализовано)

- Machine schemas:
  - `DataProcessor/VisualProcessor/schemas/<schema_version>.json`
- Human schemas:
  - `DataProcessor/VisualProcessor/**/<component>/SCHEMA.md`

> Рекомендация: генерировать `SCHEMA.md` из JSON схемы, чтобы не было расхождений.

### AudioProcessor (реализовано)

- Machine schemas: `DataProcessor/AudioProcessor/schemas/<schema_version>.json`

### TextProcessor (реализовано — агрегатный NPZ)

- Machine schemas: `DataProcessor/TextProcessor/schemas/<schema_version>.json`
- На старте Audit v3 в реестре зафиксирован **`text_npz_v1`** (`run_cli.py` → `text_features.npz`).
- Per-extractor схемы добавляются по мере аудита компонентов: **`tags_extractor_output_v1`**, **`lexico_static_features_output_v1`**, **`asr_text_proxy_audio_features_output_v1`**, **`title_embedder_output_v1`**, **`description_embedder_output_v1`**, **`hashtag_embedder_output_v1`**, **`transcript_chunk_embedder_output_v1`**, **`comments_embedder_output_v1`**, **`speaker_turn_embeddings_aggregator_output_v1`**, **`transcript_aggregator_output_v1`**, **`comments_aggregator_output_v1`**, **`qa_embedding_pairs_extractor_output_v1`**, **`embedding_pair_topk_extractor_output_v1`**, **`semantics_topics_keyphrases_output_v1`**, **`embedding_stats_extractor_output_v1`**, **`cosine_metrics_extractor_output_v1`**, **`title_embedding_cluster_entropy_extractor_output_v1`**, **`title_to_hashtag_cosine_extractor_output_v1`**, **`semantic_cluster_extractor_output_v1`**, **`topk_similar_titles_extractor_output_v1`**, **`embedding_shift_indicator_extractor_output_v1`**, **`embedding_source_id_extractor_output_v1`** (вклады `tp_tags_*` / `tp_lex_*` / `tp_asrproxy_*` / `tp_titleemb_*` / `tp_descemb_*` / `tp_hashemb_*` / `tp_tchunk_*` / `tp_commentsemb_*` / `tp_spkemb_*` / `tp_tragg_*` / `tp_commentsagg_*` (+ legacy `tp_comments_agg_*`, `tp_cagg_*`) / `tp_qa_*` / `tp_embpair_*` (+ legacy `tp_pairtopk_*`) / `tp_topics_*` / `tp_embstats_*` / `tp_cos_*` / `tp_titleclent_*` / `tp_titlehashcos_*` / `tp_semclust_*` / `tp_topktitles_*` / `tp_embshift_*` / `tp_embid_*`; `artifact_kind`: `extractor_features_flat`).

Правило Audit v3: у каждого audited компонента — **human `SCHEMA.md` + machine JSON + версия** (`schema_version` / bump при контрактных изменениях).

---

## 4) Machine schema format: `vp_schema_v1`

Machine schema — JSON с верхнеуровневыми ключами:

- `schema_system_version`: `"vp_schema_v1"`
- `schema_version`: строка, совпадает с `meta.schema_version` внутри NPZ
- `producer`: имя компонента (совпадает с `meta.producer`)
- `artifact_kind`: `"npz"`
- `allow_extra_keys`: `true|false`
  - `false` рекомендуется для audited компонентов (контракт не меняется “тихо”)
- `meta`:
  - `required_keys`: список ключей, которые должны присутствовать в `meta` dict
  - `optional_keys`: список допустимых дополнительных meta‑ключей
- `fields`: map `npz_key -> FieldSpec`

### FieldSpec

- `required`: bool
- `tier`: `"model_facing" | "analytics" | "debug"`
- `dtype`: строка или список строк:
  - `"float32" | "int32" | "int16" | "bool" | "str" | "object" | "any"`
  - `"str"` означает numpy dtype kind `U/S`
- `shape`: `null` или список dim‑ов
  - `[]` означает scalar shape `()`
  - dim может быть:
    - integer (фиксированный размер),
    - string:
      - символическое имя (`"N"`, `"K"`, `"D"`),
      - выражение `"N-1"` / `"K+2"` (только `+/- int`),
      - `"any"`/`"*"` (не проверять dim).
- `description`: опционально

---

## 5) Правила required/optional (критично для production)

### 5.1 NPZ keys

- Если `FieldSpec.required=true` → ключ **обязан существовать** в NPZ.
- Если `FieldSpec.required=false` → ключ:
  - либо **отсутствует**,
  - либо существует и соответствует dtype/shape.

**Запрещено** кодировать “optional отсутствует” как `None` записанный в NPZ:
numpy сохранит это как `object` scalar, что ломает строгую типизацию.

Правило: если фича/блок выключен — **не писать ключ вообще**.

### 5.2 Meta keys

`meta` — dict, сохраненный как scalar object array (boxing).

- required meta keys: проверяется наличие ключа (значение может быть `None`, если это разрешено контрактом)
- optional meta keys: допускаются

---

## 6) Валидатор: runtime и CI

### Runtime policy (Audit v3)

В рантайме валидатор делает:

- baseline meta validation (`producer/schema_version/models_used/...`)
- schema validation keys/dtype/shape:
  - если `schema_version` известен → **error** на любое расхождение,
  - если `schema_version` неизвестен → **warning** (временный режим rollout).

Для audited компонентов правило жёсткое: schema должна быть **known** (и валидироваться строго).

### CI policy (рекомендуемая)

Для audited набора включаем gate:

- `require_known_schema=true`
- валидируем sample/golden артефакты (например из `dp_results/` или отдельного fixtures набора)

Пример (VisualProcessor):

```bash
PYTHONPATH="DataProcessor/VisualProcessor" \
  python -m schemas.cli --require-known-schema DataProcessor/dp_results/<...>/<component>/*.npz
```

---

## 7) Versioning: когда bump `schema_version`

Bump `schema_version` обязателен, если меняется **контракт**, в частности:

- добавление/удаление ключей (при `allow_extra_keys=false`)
- изменение dtype
- изменение shape
- изменение семантики missing/empty (NaN/-1 policy), если это влияет на downstream

Не требует bump (обычно):

- изменение внутренних алгоритмов при сохранении контрактных инвариантов (но bump `producer_version` обязателен)
- добавление debug‑полей, если:
  - они явно `tier=debug`,
  - и контракт разрешает extra keys (либо schema обновлена синхронно).

---

## 8) Практические рекомендации (стиль)

- Символьные размеры:
  - `N` — кадры (primary sampling),
  - `K` — top‑K,
  - `D` — embedding dim,
  - `H/W` — spatial dims.
- Строки:
  - для JSON‑совместимости и безопасности между venv можно сохранять `meta_json` как `dtype="U"` (string).
- Object arrays:
  - допускаются только там, где это неизбежно (например списки dict для OCR).
  - для `meta` всегда используем boxing (scalar object array).


