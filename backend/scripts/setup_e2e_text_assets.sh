#!/usr/bin/env bash
# Minimal offline text assets for full E2E (topics, semantic clusters, similar titles).
# Usage: ./backend/scripts/setup_e2e_text_assets.sh
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUNDLE="$REPO_ROOT/DataProcessor/dp_models/bundled_models/text"

python3 <<PY
import json
from pathlib import Path
import numpy as np

bundle = Path("$BUNDLE")
bundle.mkdir(parents=True, exist_ok=True)

# similar_titles_v1
st = bundle / "similar_titles_v1"
st.mkdir(parents=True, exist_ok=True)
emb_path = st / "embeddings.npy"
if not emb_path.is_file():
    np.save(emb_path, np.zeros((8, 1024), dtype=np.float32))
    (st / "ids.json").write_text(json.dumps([f"stub_{i}" for i in range(8)]), encoding="utf-8")
    print("created similar_titles_v1 stub")

# topics_taxonomy_v1
topics_dir = bundle / "topics_v1"
topics_dir.mkdir(parents=True, exist_ok=True)
topics_path = topics_dir / "topics.jsonl"
if not topics_path.is_file():
    rows = [
        {"id": i, "name": f"topic_{i}", "group": "e2e_stub", "prompts_en": [f"topic {i} content"]}
        for i in range(8)
    ]
    topics_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print("created topics_v1 stub")
elif topics_path.is_file():
    lines = []
    changed = False
    for line in topics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not (obj.get("prompts_en") or obj.get("prompts_ru")):
            obj["prompts_en"] = [str(obj.get("name") or f"topic_{obj.get('id', 0)}")]
            changed = True
        lines.append(json.dumps(obj, ensure_ascii=False))
    if changed:
        topics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("migrated topics_v1 (prompts_en)")

# semantic_clusters_v1
sc = bundle / "semantic_clusters_v1"
sc.mkdir(parents=True, exist_ok=True)
pca_p = sc / "pca.npy"
cent_p = sc / "centroids.npy"
cl_p = sc / "clusters.jsonl"
if not pca_p.is_file():
    np.save(pca_p, np.random.randn(1024, 128).astype(np.float32) * 0.01)
    np.save(cent_p, np.random.randn(32, 128).astype(np.float32) * 0.01)
    clusters = [{"cluster_id": i, "name": f"cluster_{i}", "group": "e2e_stub"} for i in range(32)]
    cl_p.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in clusters) + "\n", encoding="utf-8")
    print("created semantic_clusters_v1 stub")
elif cl_p.is_file():
    # Migrate legacy stub field "id" -> "cluster_id"
    lines = []
    changed = False
    for line in cl_p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if "cluster_id" not in obj and "id" in obj:
            obj["cluster_id"] = int(obj.pop("id"))
            changed = True
        lines.append(json.dumps(obj, ensure_ascii=False))
    if changed:
        cl_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("migrated semantic_clusters_v1 clusters.jsonl (cluster_id)")

print("text assets OK")
PY
