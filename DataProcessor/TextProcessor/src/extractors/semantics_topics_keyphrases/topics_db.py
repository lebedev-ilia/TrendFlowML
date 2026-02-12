from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.path_utils import default_cache_dir, textprocessor_root


@dataclass(frozen=True)
class TopicItem:
    id: int
    name: str
    prompts_ru: List[str]
    prompts_en: List[str]
    group: Optional[str] = None


@dataclass
class TopicsDB:
    version: str
    items: List[TopicItem]
    prompts: List[str]
    prompt_topic_ids: List[int]

    @property
    def n_topics(self) -> int:
        return len({x.id for x in self.items})

    @property
    def n_prompts(self) -> int:
        return len(self.prompts)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _topics_bundle_path(version: str = "v1") -> Path:
    """
    Resolve bundled topics db path.

    Prefer `dp_models/bundled_models/text/topics_v1/` (DP_MODELS_ROOT),
    but fall back to repo-local bundle path for dev envs.
    """
    # dp_models bundle path (repo layout)
    repo_root = textprocessor_root().parents[2]  # .../DataProcessor
    p = repo_root / "dp_models" / "bundled_models" / "text" / f"topics_{version}" / "topics.jsonl"
    if p.exists():
        return p

    # fallback: in-component assets (optional)
    p2 = textprocessor_root() / "src" / "extractors" / "semantics_topics_keyphrases" / "assets" / f"topics_{version}.jsonl"
    return p2


def resolve_topics_db(spec_name: str = "topics_taxonomy_v1") -> Tuple[Path, str, str]:
    """
    Resolve topics DB via dp_models (preferred) and return:
      (topics_path, db_weights_digest, topics_db_version)
    """
    try:
        from dp_models import get_global_model_manager  # type: ignore
    except Exception:
        # dev fallback: direct file path
        p = _topics_bundle_path("v1")
        if not p.exists():
            raise RuntimeError(f"SemanticTopicExtractor: topics DB missing (dev fallback): {p}")
        return p, "unknown", "v1"

    mm = get_global_model_manager()
    spec = mm.get_spec(model_name=str(spec_name))
    _d, _p, _runtime, _engine, weights_digest, resolved = mm.resolve(spec)
    rp = spec.runtime_params if isinstance(spec.runtime_params, dict) else {}
    rel = rp.get("topics_relpath")
    if not isinstance(rel, str) or not rel:
        raise RuntimeError(f"SemanticTopicExtractor: topics DB spec missing runtime_params.topics_relpath: {spec_name}")
    path = resolved.get(rel) or resolved.get(str(rel))
    if not path:
        raise RuntimeError(f"SemanticTopicExtractor: topics DB artifact not resolved: {rel} (spec={spec_name})")
    db_ver = str(rp.get("topics_db_version") or getattr(spec, "model_version", "unknown") or "unknown")
    return Path(path), str(weights_digest or "unknown"), db_ver


def load_topics_db(path: Path, version: str) -> TopicsDB:
    if not path.exists():
        raise RuntimeError(f"SemanticTopicExtractor: topics DB missing: {path}")

    items: List[TopicItem] = []
    prompts: List[str] = []
    prompt_topic_ids: List[int] = []

    raw = path.read_text(encoding="utf-8")
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        tid = int(obj["id"])
        name = str(obj.get("name") or "")
        ru = obj.get("prompts_ru") or []
        en = obj.get("prompts_en") or []
        group = obj.get("group")
        item = TopicItem(
            id=tid,
            name=name,
            prompts_ru=[str(x) for x in ru if str(x).strip()],
            prompts_en=[str(x) for x in en if str(x).strip()],
            group=(str(group) if group is not None else None),
        )
        items.append(item)
        for p in item.prompts_ru + item.prompts_en:
            prompts.append(p)
            prompt_topic_ids.append(tid)

    if not items or not prompts:
        raise RuntimeError(f"SemanticTopicExtractor: topics DB is empty or has no prompts: {path}")

    return TopicsDB(version=version, items=items, prompts=prompts, prompt_topic_ids=prompt_topic_ids)


def _cache_paths(db_path: Path, *, db_digest: str, model_name: str, model_weights_digest: str) -> Tuple[Path, Path]:
    cache_root = default_cache_dir() / "tp_topics_db"
    cache_root.mkdir(parents=True, exist_ok=True)
    sig = _sha256_bytes((db_path.read_bytes() + ("\n" + model_name).encode("utf-8") + ("\n" + str(db_digest)).encode("utf-8") + ("\n" + str(model_weights_digest)).encode("utf-8")))
    emb_path = cache_root / f"topics_{db_path.parent.name}_{sig}.npy"
    meta_path = cache_root / f"topics_{db_path.parent.name}_{sig}.meta.json"
    return emb_path, meta_path


def _prune_cache(cache_root: Path, *, max_total_mb: int) -> None:
    if int(max_total_mb) <= 0:
        return
    files = sorted(cache_root.glob("topics_*.npy"), key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
    total = 0
    sizes: List[Tuple[Path, int]] = []
    for p in files:
        try:
            sz = int(p.stat().st_size)
        except Exception:
            continue
        total += sz
        sizes.append((p, sz))
    limit = int(max_total_mb) * 1024 * 1024
    while total > limit and sizes:
        p, sz = sizes.pop(0)
        try:
            mp = p.with_suffix(".meta.json")
            if mp.exists():
                mp.unlink()
        except Exception:
            pass
        try:
            p.unlink()
        except Exception:
            pass
        total -= sz


def load_or_build_prompt_embeddings(
    db: TopicsDB,
    *,
    db_path: Path,
    db_digest: str,
    model_name: str,
    model_weights_digest: str,
    encode_fn: Any,
    cache_enabled: bool = True,
    cache_ttl_s: float = 7 * 24 * 3600.0,
    cache_max_total_mb: int = 512,
) -> np.ndarray:
    """
    Returns matrix (n_prompts, dim) float32, L2-normalized.
    Stored in cache_dir (NOT result_store) for performance; cache is not source-of-truth.
    """
    cache_root = default_cache_dir() / "tp_topics_db"
    cache_root.mkdir(parents=True, exist_ok=True)
    _prune_cache(cache_root, max_total_mb=int(cache_max_total_mb))

    emb_path, meta_path = _cache_paths(db_path, db_digest=str(db_digest), model_name=model_name, model_weights_digest=str(model_weights_digest))
    if cache_enabled and emb_path.exists():
        try:
            arr = np.asarray(np.load(emb_path), dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] == len(db.prompts):
                if cache_ttl_s > 0 and meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        created_at = float(meta.get("created_at_s") or 0.0)
                        if created_at > 0 and (time.time() - created_at) > float(cache_ttl_s):
                            raise RuntimeError("expired")
                    except Exception:
                        pass
                return arr
        except Exception:
            pass

    # Build embeddings (offline: model resolved via dp_models)
    vecs = encode_fn(db.prompts)
    arr = np.asarray(vecs, dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError("SemanticTopicExtractor: prompt embeddings must be a 2D matrix")

    # Save cache atomically
    tmp = emb_path.with_suffix(".tmp.npy")
    np.save(tmp, arr.astype(np.float32))
    tmp.replace(emb_path)
    try:
        meta_path.write_text(
            json.dumps(
                {
                    "schema_version": "tp_topics_db_cache_v1",
                    "model_name": model_name,
                    "model_weights_digest": str(model_weights_digest or "unknown"),
                    "db_digest": str(db_digest or "unknown"),
                    "n_prompts": int(arr.shape[0]),
                    "dim": int(arr.shape[1]),
                    "db_rel": str(db_path),
                    "created_at_s": float(time.time()),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass
    return arr


