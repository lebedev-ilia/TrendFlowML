-- Benchmark Registry (Postgres) — MVP DDL
-- Source-of-truth: DynamicBatch/docs/BENCHMARK_REGISTRY_CONTRACT.md
-- Policy: append-only + active selection via valid_to IS NULL.

BEGIN;

CREATE TABLE IF NOT EXISTS benchmark_costs_v1 (
  id UUID PRIMARY KEY,

  -- identity keys
  component_id TEXT NOT NULL,
  component_part TEXT NOT NULL DEFAULT 'whole', -- whole | substep:<name>
  owner TEXT NOT NULL,                -- dataprocessor|fetcher|models
  stage TEXT NULL,                    -- baseline|v1|v2 (nullable for global)
  unit TEXT NOT NULL,                 -- frame|segment|prompt|url|...
  runtime TEXT NOT NULL,              -- triton|inprocess|...
  model_signature TEXT NULL,          -- nullable for pure CPU/no-model components
  model_branch TEXT NULL,             -- e.g. 224/336/448 or model variant (optional)

  -- bucketing / knobs / device
  input_bucket JSONB NOT NULL DEFAULT '{}'::jsonb,
  knobs JSONB NOT NULL DEFAULT '{}'::jsonb,
  device_profile JSONB NOT NULL DEFAULT '{}'::jsonb,

  -- provenance
  producer_version TEXT NOT NULL,
  git_commit TEXT NOT NULL,
  git_dirty BOOLEAN NOT NULL DEFAULT FALSE,
  schema_version TEXT NOT NULL,

  -- metrics payload: must include keys required by scheduler
  metrics JSONB NOT NULL DEFAULT '{}'::jsonb,

  -- raw artifact pointer (S3/MinIO URI)
  artifact_uri TEXT NOT NULL,

  -- validity window (append-only active selection)
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  valid_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  valid_to TIMESTAMPTZ NULL
);

-- Fast lookups for scheduler
CREATE INDEX IF NOT EXISTS idx_benchmark_costs_component_runtime_sig
  ON benchmark_costs_v1 (component_id, runtime, model_signature);

CREATE INDEX IF NOT EXISTS idx_benchmark_costs_component_part
  ON benchmark_costs_v1 (component_id, component_part);

CREATE INDEX IF NOT EXISTS idx_benchmark_costs_owner_stage
  ON benchmark_costs_v1 (owner, stage);

-- JSON query indexes
CREATE INDEX IF NOT EXISTS idx_benchmark_costs_input_bucket_gin
  ON benchmark_costs_v1 USING GIN (input_bucket);

CREATE INDEX IF NOT EXISTS idx_benchmark_costs_device_profile_gin
  ON benchmark_costs_v1 USING GIN (device_profile);

CREATE INDEX IF NOT EXISTS idx_benchmark_costs_knobs_gin
  ON benchmark_costs_v1 USING GIN (knobs);

-- Active-only helper index
CREATE INDEX IF NOT EXISTS idx_benchmark_costs_active
  ON benchmark_costs_v1 (component_id, runtime)
  WHERE valid_to IS NULL;

COMMIT;


