#!/usr/bin/env python3
"""Синтетическая валидация embedding_source_id_extractor.

Запуск:
    python3 automation/runner/synth_embsrcid.py

Проверяет:
  U1 — validate_schema/structure/ranges на синтетических NPZ
  U3 — all-finite в ok-path; policy one-hot сумма=1; primary one-hot ∈{0,1}
  U4 — absent artifacts + strict=False → valid NPZ без краша
  U5 — golden=0 (bit-identical на одном input)
  C1 — все 5 политик
  C2 — все 6 error-кодов при strict=False
  C3 — vector_id == SHA256(float32 bytes)[:24] для 3 векторов
  C4 — 0 NaN в ok-path
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Пути
REPO_ROOT = Path(__file__).resolve().parents[2]
TP_ROOT = REPO_ROOT / "DataProcessor" / "TextProcessor"
sys.path.insert(0, str(TP_ROOT))

# Импорт компонента
from src.extractors.embedding_source_id_extractor.main import EmbeddingSourceIdExtractor
from src.extractors.embedding_source_id_extractor.utils.validate_embedding_source_id_extractor_text_npz import (
    validate_schema,
    validate_structure,
    validate_ranges,
    load_npz,
    extract_meta,
    _build_slice,
)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

results: List[Tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    icon = PASS if ok else FAIL
    print(f"  [{icon}] {name}" + (f": {detail}" if detail else ""))


# ──────────────────────────────────────────────────────────
# Вспомогательные инструменты
# ──────────────────────────────────────────────────────────

class MockDoc:
    """Минимальный mock VideoDocument для extract()."""
    def __init__(self, tp_artifacts: Optional[Dict[str, Any]] = None):
        self.tp_artifacts = tp_artifacts or {}


def _make_emb_file(tmpdir: Path, name: str, vec: Optional[np.ndarray] = None) -> str:
    """Сохраняет .npy в tmpdir и возвращает relpath."""
    if vec is None:
        vec = np.random.default_rng(42).random(1024).astype(np.float32)
    p = tmpdir / name
    np.save(str(p), vec)
    return name


def _synth_doc_transcript(tmpdir: Path, vec: Optional[np.ndarray] = None) -> MockDoc:
    relpath = _make_emb_file(tmpdir, "transcript_combined_agg_mean.npy", vec)
    return MockDoc({
        "transcripts": {
            "combined": {"agg_mean_relpath": relpath}
        }
    })


def _synth_doc_title(tmpdir: Path, vec: Optional[np.ndarray] = None) -> MockDoc:
    relpath = _make_emb_file(tmpdir, "title_emb.npy", vec)
    return MockDoc({
        "embeddings": {
            "title": {
                "relpath": relpath,
                "model_name": "e5-large",
                "model_version": "1.0",
                "weights_digest": "abc123",
            }
        }
    })


def _synth_doc_description(tmpdir: Path, vec: Optional[np.ndarray] = None) -> MockDoc:
    relpath = _make_emb_file(tmpdir, "desc_emb.npy", vec)
    return MockDoc({
        "embeddings": {
            "description": {
                "relpath": relpath,
            }
        }
    })


def _make_npz(features_flat: Dict[str, float], payload: Dict[str, Any], status: str = "ok") -> bytes:
    """Сериализует результат в NPZ-like формат для валидатора."""
    keys = list(features_flat.keys())
    vals = [features_flat[k] for k in keys]
    meta = {"status": status, "schema_version": "embedding_source_id_extractor_output_v1"}
    buf = io.BytesIO()
    np.savez(
        buf,
        feature_names=np.array(keys, dtype=object),
        feature_values=np.array(vals, dtype=np.float32),
        meta=np.array(meta, dtype=object),
        payload=np.array(payload, dtype=object),
    )
    return buf.getvalue()


def _extract_and_pack(extractor: EmbeddingSourceIdExtractor, doc: MockDoc) -> Tuple[Dict[str, float], Dict[str, Any], bytes]:
    result = extractor.extract(doc)
    ff = result["result"]["features_flat"]
    esid = result["result"]["embedding_source_id"]
    payload = {"embedding_source_id": esid}
    status = "ok" if ff.get("tp_embid_present", 0.0) > 0.5 else "ok"
    npz_bytes = _make_npz(ff, payload, status)
    return ff, esid, npz_bytes


def _validate_npz_bytes(npz_bytes: bytes) -> Tuple[bool, List[str]]:
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        f.write(npz_bytes)
        fname = f.name
    try:
        ok_schema = validate_schema(fname)
        struct_errs = validate_structure(fname)
        range_errs = validate_ranges(fname)
        return ok_schema, struct_errs + range_errs
    finally:
        os.unlink(fname)


def _manual_vector_id(vec: np.ndarray) -> str:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.dtype.byteorder not in ("<", "="):
        v = v.byteswap().newbyteorder("<")
    h = hashlib.sha256(v.tobytes(order="C")).hexdigest()
    return h[:24]


# ──────────────────────────────────────────────────────────
# Тесты
# ──────────────────────────────────────────────────────────

def test_u1_u3_u4_u5_c1_c2_c3_c4():
    rng = np.random.default_rng(0)

    # ── C1: все 5 политик ──────────────────────────────────
    print("\n=== C1: 5 политик ===")
    policies = [
        "transcript_first",
        "title_first",
        "description_first",
        "title_only",
        "transcript_only",
    ]
    policy_flags = {
        "transcript_first": "tp_embid_policy_transcript_first",
        "title_first": "tp_embid_policy_title_first",
        "description_first": "tp_embid_policy_description_first",
        "title_only": "tp_embid_policy_title_only",
        "transcript_only": "tp_embid_policy_transcript_only",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        tdir = Path(tmpdir)

        for pol in policies:
            ext = EmbeddingSourceIdExtractor(
                primary_source_policy=pol,
                strict_missing_primary=False,
                artifacts_dir=str(tdir),
            )
            # Предоставим все источники, чтобы политика могла выбрать
            _make_emb_file(tdir, "transcript_combined_agg_mean.npy", rng.random(512).astype(np.float32))
            _make_emb_file(tdir, "title_emb.npy", rng.random(512).astype(np.float32))
            _make_emb_file(tdir, "desc_emb.npy", rng.random(512).astype(np.float32))

            if pol in ("transcript_first", "transcript_only"):
                doc = MockDoc({
                    "transcripts": {"combined": {"agg_mean_relpath": "transcript_combined_agg_mean.npy"}},
                    "embeddings": {
                        "title": {"relpath": "title_emb.npy"},
                        "description": {"relpath": "desc_emb.npy"},
                    },
                })
            elif pol == "title_first":
                doc = MockDoc({
                    "embeddings": {"title": {"relpath": "title_emb.npy"}},
                    "transcripts": {"combined": {"agg_mean_relpath": "transcript_combined_agg_mean.npy"}},
                })
            elif pol == "description_first":
                doc = MockDoc({
                    "embeddings": {
                        "description": {"relpath": "desc_emb.npy"},
                        "title": {"relpath": "title_emb.npy"},
                    },
                    "transcripts": {"combined": {"agg_mean_relpath": "transcript_combined_agg_mean.npy"}},
                })
            else:  # title_only
                doc = MockDoc({
                    "embeddings": {"title": {"relpath": "title_emb.npy"}},
                })

            ff, esid, npz_bytes = _extract_and_pack(ext, doc)

            # Проверяем нужный policy flag = 1
            flag_key = policy_flags[pol]
            flag_ok = abs(ff.get(flag_key, 0.0) - 1.0) < 1e-6
            # Проверяем сумму policy one-hot = 1
            psum = sum(ff.get(k, 0.0) for k in policy_flags.values())
            sum_ok = abs(psum - 1.0) < 1e-3
            # Проверяем что эмбеддинг найден (present=1)
            present_ok = ff.get("tp_embid_present", 0.0) > 0.5
            record(f"C1/{pol}", flag_ok and sum_ok and present_ok,
                   f"flag={ff.get(flag_key, '?'):.0f} psum={psum:.1f} present={ff.get('tp_embid_present'):.0f}")

    # ── U3/C4: ok-path finite + primary one-hot ────────────
    print("\n=== U3+C4: ok-path finite, primary one-hot ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tdir = Path(tmpdir)
        vec = rng.random(768).astype(np.float32)
        ext = EmbeddingSourceIdExtractor(
            primary_source_policy="transcript_first",
            strict_missing_primary=False,
            artifacts_dir=str(tdir),
        )
        doc = _synth_doc_transcript(tdir, vec)
        ff, esid, npz_bytes = _extract_and_pack(ext, doc)

        # C4: 0 NaN
        nan_count = sum(1 for v in ff.values() if not np.isfinite(float(v)))
        record("C4 0-NaN ok-path", nan_count == 0, f"nan_count={nan_count}")

        # U3: policy one-hot sum
        psum = sum(ff.get(k, 0.0) for k in (
            "tp_embid_policy_transcript_first", "tp_embid_policy_title_first",
            "tp_embid_policy_description_first", "tp_embid_policy_title_only",
            "tp_embid_policy_transcript_only",
        ))
        record("U3 policy one-hot sum=1", abs(psum - 1.0) < 1e-3, f"sum={psum:.3f}")

        # U3: primary one-hot ∈{0,1}
        prim_sum = sum(ff.get(k, 0.0) for k in (
            "tp_embid_primary_is_transcript", "tp_embid_primary_is_title", "tp_embid_primary_is_description"
        ))
        record("U3 primary one-hot ∈{0,1}", 0.0 <= prim_sum <= 1.001, f"sum={prim_sum:.3f}")
        record("U3 primary_is_transcript=1", abs(ff.get("tp_embid_primary_is_transcript", 0.0) - 1.0) < 1e-6)

    # ── U1: validate_schema/structure/ranges на ok-path ───
    print("\n=== U1: валидатор на ok-path NPZ ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tdir = Path(tmpdir)
        ext = EmbeddingSourceIdExtractor(
            primary_source_policy="title_first",
            strict_missing_primary=False,
            artifacts_dir=str(tdir),
        )
        doc = _synth_doc_title(tdir, rng.random(384).astype(np.float32))
        ff, esid, npz_bytes = _extract_and_pack(ext, doc)
        ok_schema, errs = _validate_npz_bytes(npz_bytes)
        record("U1 validate_schema ok-path", ok_schema)
        record("U1 validate_struct/ranges ok-path", len(errs) == 0, f"ошибок={len(errs)}: {errs[:2]}")

    # ── U4: error-path (absent artifacts, strict=False) ───
    print("\n=== U4: expected-empty (no artifacts, strict=False) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tdir = Path(tmpdir)
        ext = EmbeddingSourceIdExtractor(
            primary_source_policy="transcript_first",
            strict_missing_primary=False,
            artifacts_dir=str(tdir),
        )
        doc_empty = MockDoc({})  # нет tp_artifacts
        try:
            ff, esid, npz_bytes = _extract_and_pack(ext, doc_empty)
            no_crash = True
            present_zero = ff.get("tp_embid_present", 1.0) < 0.5
            err_code = esid.get("error") == "no_embedding_found"
            ok_schema, errs = _validate_npz_bytes(npz_bytes)
            record("U4 no crash при absent artifacts", no_crash)
            record("U4 present=0 при absent artifacts", present_zero, f"present={ff.get('tp_embid_present')}")
            record("U4 error=no_embedding_found", err_code, f"got={esid.get('error')}")
            record("U4 validate_schema error-path", ok_schema)
        except Exception as e:
            record("U4 no crash при absent artifacts", False, str(e))

    # ── C2: все 6 error-кодов ─────────────────────────────
    print("\n=== C2: все 6 error-кодов при strict=False ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tdir = Path(tmpdir)

        ext = EmbeddingSourceIdExtractor(
            primary_source_policy="title_first",
            strict_missing_primary=False,
            artifacts_dir=str(tdir),
        )

        # 1. no_embedding_found — нет tp_artifacts
        doc = MockDoc({})
        try:
            r = ext.extract(doc)
            code = r["result"]["embedding_source_id"].get("error")
            record("C2 no_embedding_found", code == "no_embedding_found", f"got={code}")
        except Exception as e:
            record("C2 no_embedding_found", False, str(e))

        # 2. unsafe_relpath — путь ../evil
        doc = MockDoc({"embeddings": {"title": {"relpath": "../evil.npy"}}})
        try:
            r = ext.extract(doc)
            code = r["result"]["embedding_source_id"].get("error")
            record("C2 unsafe_relpath", code == "unsafe_relpath", f"got={code}")
        except Exception as e:
            record("C2 unsafe_relpath", False, str(e))

        # 3. embedding_file_missing — relpath ведёт в несуществующий файл
        doc = MockDoc({"embeddings": {"title": {"relpath": "nonexistent.npy"}}})
        try:
            r = ext.extract(doc)
            code = r["result"]["embedding_source_id"].get("error")
            record("C2 embedding_file_missing", code == "embedding_file_missing", f"got={code}")
        except Exception as e:
            record("C2 embedding_file_missing", False, str(e))

        # 4. embedding_load_failed — файл есть, но не NPY (битый)
        bad_file = tdir / "bad.npy"
        bad_file.write_bytes(b"NOT_A_NUMPY_FILE")
        doc = MockDoc({"embeddings": {"title": {"relpath": "bad.npy"}}})
        try:
            r = ext.extract(doc)
            code = r["result"]["embedding_source_id"].get("error")
            record("C2 embedding_load_failed", code == "embedding_load_failed", f"got={code}")
        except Exception as e:
            record("C2 embedding_load_failed", False, str(e))

        # 5. embedding_empty — пустой массив float32
        empty_file = tdir / "empty.npy"
        np.save(str(empty_file), np.array([], dtype=np.float32))
        doc = MockDoc({"embeddings": {"title": {"relpath": "empty.npy"}}})
        try:
            r = ext.extract(doc)
            code = r["result"]["embedding_source_id"].get("error")
            record("C2 embedding_empty", code == "embedding_empty", f"got={code}")
        except Exception as e:
            record("C2 embedding_empty", False, str(e))

        # 6. embedding_non_finite — вектор с NaN
        nan_file = tdir / "nan_vec.npy"
        vec_nan = np.full(128, np.nan, dtype=np.float32)
        np.save(str(nan_file), vec_nan)
        doc = MockDoc({"embeddings": {"title": {"relpath": "nan_vec.npy"}}})
        try:
            r = ext.extract(doc)
            code = r["result"]["embedding_source_id"].get("error")
            record("C2 embedding_non_finite", code == "embedding_non_finite", f"got={code}")
        except Exception as e:
            record("C2 embedding_non_finite", False, str(e))

    # ── C3: vector_id == SHA256 ───────────────────────────
    print("\n=== C3: vector_id совпадает с ручным SHA256 ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tdir = Path(tmpdir)
        ext = EmbeddingSourceIdExtractor(
            primary_source_policy="title_first",
            strict_missing_primary=False,
            artifacts_dir=str(tdir),
        )
        for i, seed in enumerate([0, 1, 2]):
            vec = rng.random(1024).astype(np.float32)
            _make_emb_file(tdir, f"title_{i}.npy", vec)
            doc = MockDoc({"embeddings": {"title": {"relpath": f"title_{i}.npy"}}})
            r = ext.extract(doc)
            got_id = r["result"]["embedding_source_id"].get("vector_id", "")
            expected_id = _manual_vector_id(vec)
            record(f"C3 vector_id[{i}]", got_id == expected_id,
                   f"got={got_id} exp={expected_id}")

    # ── U5: golden = 0 ────────────────────────────────────
    print("\n=== U5: golden (bit-identical на 2 прогонах) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tdir = Path(tmpdir)
        vec = rng.random(512).astype(np.float32)
        _make_emb_file(tdir, "gold.npy", vec)
        ext = EmbeddingSourceIdExtractor(
            primary_source_policy="title_first",
            strict_missing_primary=False,
            artifacts_dir=str(tdir),
        )
        doc = MockDoc({"embeddings": {"title": {"relpath": "gold.npy"}}})
        r1 = ext.extract(doc)
        r2 = ext.extract(doc)
        vid1 = r1["result"]["embedding_source_id"]["vector_id"]
        vid2 = r2["result"]["embedding_source_id"]["vector_id"]
        ff1 = r1["result"]["features_flat"]
        ff2 = r2["result"]["features_flat"]
        id_ok = vid1 == vid2
        feat_ok = all(abs(ff1.get(k, 0.0) - ff2.get(k, 0.0)) < 1e-9 for k in ff1)
        record("U5 vector_id bit-identical", id_ok, f"run1={vid1} run2={vid2}")
        record("U5 features_flat bit-identical", feat_ok,
               "max_delta=" + str(max(abs(ff1.get(k, 0.0) - ff2.get(k, 0.0)) for k in ff1)))

    # ── U4/C2: strict=True бросает RuntimeError ───────────
    print("\n=== U4/extra: strict=True бросает RuntimeError ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tdir = Path(tmpdir)
        ext_strict = EmbeddingSourceIdExtractor(
            primary_source_policy="transcript_first",
            strict_missing_primary=True,
            artifacts_dir=str(tdir),
        )
        doc = MockDoc({})
        try:
            ext_strict.extract(doc)
            record("strict=True кидает RuntimeError при absent", False, "нет исключения")
        except RuntimeError:
            record("strict=True кидает RuntimeError при absent", True)


def main():
    print("\n" + "=" * 60)
    print("  Синтетическая валидация: embedding_source_id_extractor")
    print("=" * 60)
    test_u1_u3_u4_u5_c1_c2_c3_c4()
    print("\n" + "=" * 60)
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"  Итого: {n_pass} PASS, {n_fail} FAIL из {len(results)}")
    print("=" * 60)
    if n_fail > 0:
        print("\nFAIL-тесты:")
        for name, ok, detail in results:
            if not ok:
                print(f"  ✗ {name}: {detail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
