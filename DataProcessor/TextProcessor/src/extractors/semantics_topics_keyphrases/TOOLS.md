# `semantics_topics_keyphrases` — Tools / how to extend topics DB

## Where the topics DB lives

Bundled (offline) asset:

- `DataProcessor/dp_models/bundled_models/text/topics_v1/topics.jsonl`

This file is **not** a runtime download and is safe for offline workers.

dp_models spec (preferred resolver):
- `DataProcessor/dp_models/spec_catalog/text/topics_taxonomy_v1.yaml`

## How the extractor uses it

- At runtime, `SemanticTopicExtractor` loads `topics.jsonl`.
- Prompt embeddings are built **once** (per DB+model signature) into **cache**:
  - `default_cache_dir()/tp_topics_db/topics_topics_v1_<sha>.npy`
- Cache is **not** source-of-truth; it is only a performance accelerator.

## How to expand to 200–500 topics (recommended)

1. Add more lines to `topics.jsonl` (one JSON per line):
   - include both `prompts_ru` and `prompts_en` (multi-language policy: mixed prompts)
2. Keep `id` stable (never reuse ids).
3. Prefer ~3–10 prompts per topic for recall.
4. Run a smoke-run; the cache file will be rebuilt automatically on first use for the chosen embedding model.

## Higher-quality but more complex upgrades (future)

- Use a dedicated keyphrase algorithm (YAKE) instead of the lightweight scorer.
- Replace prompt-level retrieval with topic centroid embeddings + hard negative mining (better quality).
- Add a dedicated LanguageDetector component (via `dp_models`) and select prompt language per video.


