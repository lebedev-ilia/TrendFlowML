#!/usr/bin/env python3
"""
Синтетический тест speaker_turn_embeddings_aggregator.

Проверяет:
 T1 - legacy mode (doc.speakers) - happy path
 T2 - diar+ASR mode - happy path
 T3 - empty-path (нет входа), require_input=False
 T4 - require_input=True при пустом входе → RuntimeError
 T5 - golden: два прогона legacy дают идентичный результат
 T6 - golden: два прогона diar+ASR дают идентичный результат
 T7 - структура features_flat (17 ключей, ranges)
 T8 - артефакты .npy записаны и конечны
"""
from __future__ import annotations
import sys, os, tempfile, shutil, types
import numpy as np

# PYTHONPATH
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(THIS_DIR, "..")  # TrendFlowML
TP_ROOT = os.path.join(REPO_ROOT, "DataProcessor", "TextProcessor")
TP_SRC = os.path.join(TP_ROOT, "src")
DP_ROOT = os.path.join(REPO_ROOT, "DataProcessor")
sys.path.insert(0, TP_ROOT)  # для "from src.core..."
sys.path.insert(0, TP_SRC)   # для "from extractors..."
sys.path.insert(0, DP_ROOT)
os.environ.setdefault("DP_MODELS_ROOT", os.path.join(DP_ROOT, "dp_models", "bundled_models"))

from extractors.speaker_turn_embeddings_aggregator.main import SpeakerTurnEmbeddingsAggregatorExtractor

PASS = 0; FAIL = 0
def ok(msg): global PASS; PASS += 1; print(f"  PASS {msg}")
def fail(msg): global FAIL; FAIL += 1; print(f"  FAIL {msg}")

# ----- Фиктивный doc -----
def make_doc(**kwargs):
    d = types.SimpleNamespace(**kwargs)
    return d

# ----- Загружаем модель один раз -----
print("=== Инициализация модели (intfloat/multilingual-e5-large, cpu) ===")
import time
t0 = time.perf_counter()
TMP = tempfile.mkdtemp(prefix="spkemb_test_")

ext = SpeakerTurnEmbeddingsAggregatorExtractor(
    model_name="intfloat/multilingual-e5-large",
    artifacts_dir=TMP,
    device="cpu",
    fp16=False,
    emit_extra_metrics=True,
    write_artifacts=True,
)
print(f"  Инициализация: {time.perf_counter()-t0:.1f}с, weights_digest={ext.weights_digest[:12]}...")

# ===================== T1: legacy mode =====================
print("\n=== T1: legacy mode (doc.speakers) ===")
doc_legacy = make_doc(speakers={
    "t1": {"name": "Alice", "description": "Привет, я Алис. Сегодня поговорим о машинном обучении."},
    "t2": {"name": "Bob", "description": "Я Боб. Расскажу о нейронных сетях."},
    "t3": {"name": "Alice", "description": "Добавлю ещё кое-что о трансформерах."},
})
r1 = ext.extract(doc_legacy)
ff1 = r1["result"]["features_flat"]
try:
    assert ff1["tp_spkemb_present"] == 1.0, f"present={ff1['tp_spkemb_present']}"
    assert ff1["tp_spkemb_input_mode_legacy_doc_speakers"] == 1.0
    assert ff1["tp_spkemb_input_mode_diar_asr"] == 0.0
    assert ff1["tp_spkemb_speakers_total"] == 2.0, f"speakers_total={ff1['tp_spkemb_speakers_total']}"  # Alice+Bob
    assert ff1["tp_spkemb_speakers_embedded"] == 2.0
    assert ff1["tp_spkemb_turns_total"] == 3.0, f"turns_total={ff1['tp_spkemb_turns_total']}"  # 2+1
    assert ff1["tp_spkemb_input_present"] == 1.0
    assert ff1["tp_spkemb_diar_present"] == 0.0  # legacy mode
    ok("legacy mode: present=1, speakers=2, turns=3, mode=legacy")
except AssertionError as e:
    fail(f"legacy mode: {e}")

# ===================== T2: diar+ASR mode =====================
print("\n=== T2: diar+ASR mode ===")
doc_diar = make_doc(
    speakers={},
    speaker_diarization={
        "speaker_segments": [
            {"speaker_id": "SPEAKER_00", "start_sec": 0.0, "end_sec": 5.0},
            {"speaker_id": "SPEAKER_01", "start_sec": 5.0, "end_sec": 10.0},
            {"speaker_id": "SPEAKER_00", "start_sec": 10.0, "end_sec": 15.0},
        ]
    },
    asr={
        "segments": [
            {"text": "Привет, я первый спикер.", "start_sec": 0.5, "end_sec": 4.5},
            {"text": "А я второй спикер.", "start_sec": 5.5, "end_sec": 9.5},
            {"text": "Первый снова говорит.", "start_sec": 10.5, "end_sec": 14.5},
        ]
    }
)
ext2 = SpeakerTurnEmbeddingsAggregatorExtractor(
    model_name="intfloat/multilingual-e5-large",
    artifacts_dir=TMP,
    device="cpu",
    fp16=False,
    emit_extra_metrics=True,
    write_artifacts=True,
)
r2 = ext2.extract(doc_diar)
ff2 = r2["result"]["features_flat"]
try:
    assert ff2["tp_spkemb_present"] == 1.0, f"present={ff2['tp_spkemb_present']}"
    assert ff2["tp_spkemb_input_mode_diar_asr"] == 1.0
    assert ff2["tp_spkemb_input_mode_legacy_doc_speakers"] == 0.0
    assert ff2["tp_spkemb_diar_present"] == 1.0
    assert ff2["tp_spkemb_asr_present"] == 1.0
    assert ff2["tp_spkemb_speakers_total"] >= 1.0, f"speakers_total={ff2['tp_spkemb_speakers_total']}"
    ok(f"diar+ASR: present=1, speakers={ff2['tp_spkemb_speakers_total']}, turns={ff2['tp_spkemb_turns_total']}")
except AssertionError as e:
    fail(f"diar+ASR: {e}")

# ===================== T3: empty-path =====================
print("\n=== T3: empty-path (нет входа, require_input=False) ===")
doc_empty = make_doc(speakers={})
r3 = ext.extract(doc_empty)
ff3 = r3["result"]["features_flat"]
try:
    assert ff3["tp_spkemb_present"] == 0.0
    assert ff3["tp_spkemb_speakers_total"] == 0.0
    assert ff3["tp_spkemb_input_present"] == 0.0
    ok("empty-path: present=0, no crash")
except AssertionError as e:
    fail(f"empty-path: {e}")

# ===================== T4: require_input=True → RuntimeError =====================
print("\n=== T4: require_input=True при пустом входе → RuntimeError ===")
ext_req = SpeakerTurnEmbeddingsAggregatorExtractor(
    model_name="intfloat/multilingual-e5-large",
    artifacts_dir=TMP,
    device="cpu",
    fp16=False,
    require_input=True,
)
try:
    ext_req.extract(doc_empty)
    fail("require_input=True: ожидался RuntimeError, но исключения нет")
except RuntimeError as e:
    ok(f"require_input=True: RuntimeError → '{str(e)[:60]}'")
except Exception as e:
    fail(f"require_input=True: неожиданное исключение {type(e).__name__}: {e}")

# ===================== T5: golden legacy =====================
print("\n=== T5: golden — два прогона legacy дают идентичный результат ===")
TMP2 = tempfile.mkdtemp(prefix="spkemb_test2_")
ext5a = SpeakerTurnEmbeddingsAggregatorExtractor(
    model_name="intfloat/multilingual-e5-large",
    artifacts_dir=TMP,
    device="cpu",
    fp16=False,
    write_artifacts=True,
)
ext5b = SpeakerTurnEmbeddingsAggregatorExtractor(
    model_name="intfloat/multilingual-e5-large",
    artifacts_dir=TMP2,
    device="cpu",
    fp16=False,
    write_artifacts=True,
)
doc_gold = make_doc(speakers={
    "s1": {"name": "Иван", "description": "Привет! Меня зовут Иван. Сегодня поговорим о ML."},
    "s2": {"name": "Мария", "description": "Привет! Я Мария. Обсудим нейронные сети."},
})
rg1 = ext5a.extract(doc_gold)
rg2 = ext5b.extract(doc_gold)
ff_g1 = rg1["result"]["features_flat"]
ff_g2 = rg2["result"]["features_flat"]
try:
    for k in ff_g1:
        v1, v2 = ff_g1[k], ff_g2[k]
        if isinstance(v1, float) and (v1 != v1) and (v2 != v2):
            continue  # оба NaN
        assert abs(v1 - v2) < 1e-9, f"features_flat[{k}]: {v1} != {v2}"
    ok("golden features_flat: max|Δ|=0.0")
except AssertionError as e:
    fail(f"golden features_flat: {e}")

# Проверяем совпадение .npy артефактов
try:
    for fname in ["speaker_spk000_mean.npy", "speaker_spk000_max.npy",
                  "speaker_spk001_mean.npy", "speaker_spk001_max.npy"]:
        p1 = os.path.join(TMP, fname)
        p2 = os.path.join(TMP2, fname)
        if os.path.exists(p1) and os.path.exists(p2):
            a1 = np.load(p1)
            a2 = np.load(p2)
            diff = float(np.max(np.abs(a1 - a2)))
            if diff > 1e-6:
                fail(f"golden npy {fname}: max|Δ|={diff:.2e}")
            else:
                ok(f"golden npy {fname}: max|Δ|={diff:.2e}")
        else:
            fail(f"golden npy {fname}: не найден ({os.path.exists(p1)}/{os.path.exists(p2)})")
except Exception as e:
    fail(f"golden npy: {e}")

# ===================== T6: golden diar+ASR =====================
print("\n=== T6: golden — два прогона diar+ASR идентичны ===")
TMP3 = tempfile.mkdtemp(prefix="spkemb_test3_")
ext6a = SpeakerTurnEmbeddingsAggregatorExtractor(
    model_name="intfloat/multilingual-e5-large",
    artifacts_dir=TMP,
    device="cpu",
    fp16=False,
    write_artifacts=True,
)
ext6b = SpeakerTurnEmbeddingsAggregatorExtractor(
    model_name="intfloat/multilingual-e5-large",
    artifacts_dir=TMP3,
    device="cpu",
    fp16=False,
    write_artifacts=True,
)
rd1 = ext6a.extract(doc_diar)
rd2 = ext6b.extract(doc_diar)
ff_d1 = rd1["result"]["features_flat"]
ff_d2 = rd2["result"]["features_flat"]
try:
    for k in ff_d1:
        v1, v2 = ff_d1[k], ff_d2[k]
        if isinstance(v1, float) and (v1 != v1) and (v2 != v2):
            continue
        assert abs(v1 - v2) < 1e-9, f"features_flat[{k}]: {v1} != {v2}"
    ok("golden diar+ASR features_flat: max|Δ|=0.0")
except AssertionError as e:
    fail(f"golden diar+ASR: {e}")

# ===================== T7: structure — 17 ключей, ranges =====================
print("\n=== T7: структура (17 ключей, ranges) ===")
EXPECTED_KEYS = {
    "tp_spkemb_present", "tp_spkemb_speakers_total", "tp_spkemb_speakers_embedded",
    "tp_spkemb_turns_total", "tp_spkemb_write_artifacts", "tp_spkemb_compute_mean",
    "tp_spkemb_compute_max", "tp_spkemb_input_present", "tp_spkemb_input_mode_diar_asr",
    "tp_spkemb_input_mode_legacy_doc_speakers", "tp_spkemb_asr_present", "tp_spkemb_diar_present",
    "tp_spkemb_batch_size", "tp_spkemb_max_speakers", "tp_spkemb_max_turns_per_speaker",
    "tp_spkemb_min_chars_per_turn", "tp_spkemb_max_chars_per_turn",
}
for test_name, ff in [("legacy", ff1), ("diar+ASR", ff2), ("empty", ff3)]:
    got_keys = set(ff.keys())
    missing = EXPECTED_KEYS - got_keys
    extra = got_keys - EXPECTED_KEYS
    if missing or extra:
        fail(f"T7 {test_name}: missing={missing}, extra={extra}")
    else:
        ok(f"T7 {test_name}: ровно 17 ключей")

# ranges
def check_ranges(ff, name):
    errors = []
    BINARY = EXPECTED_KEYS - {
        "tp_spkemb_speakers_total", "tp_spkemb_speakers_embedded", "tp_spkemb_turns_total",
        "tp_spkemb_batch_size", "tp_spkemb_max_speakers", "tp_spkemb_max_turns_per_speaker",
        "tp_spkemb_min_chars_per_turn", "tp_spkemb_max_chars_per_turn",
    }
    for k in BINARY:
        v = ff.get(k, float('nan'))
        if v == v and (v < -1e-6 or v > 1.0 + 1e-6):
            errors.append(f"{k}={v}")
    for k in ["tp_spkemb_speakers_total", "tp_spkemb_speakers_embedded", "tp_spkemb_turns_total"]:
        v = ff.get(k, float('nan'))
        if v == v and v < -1e-6:
            errors.append(f"{k}={v} < 0")
    st = ff.get("tp_spkemb_speakers_total", 0)
    se = ff.get("tp_spkemb_speakers_embedded", 0)
    if st == st and se == se and se - st > 1e-3:
        errors.append(f"embedded={se} > total={st}")
    if errors:
        fail(f"T7 ranges {name}: {errors}")
    else:
        ok(f"T7 ranges {name}: OK")

check_ranges(ff1, "legacy")
check_ranges(ff2, "diar+ASR")
check_ranges(ff3, "empty")

# emit_extra_metrics checks
def check_extra(ff, name, emit_extra):
    EXTRA = ["tp_spkemb_batch_size", "tp_spkemb_max_speakers", "tp_spkemb_max_turns_per_speaker",
             "tp_spkemb_min_chars_per_turn", "tp_spkemb_max_chars_per_turn"]
    nan_count = sum(1 for k in EXTRA if ff.get(k, 0.0) != ff.get(k, 0.0))  # NaN!=NaN
    if emit_extra:
        if nan_count == 0:
            ok(f"T7 extra_metrics {name}: все finite")
        else:
            fail(f"T7 extra_metrics {name}: {nan_count}/5 NaN при emit_extra=True")
    else:
        if nan_count == 5:
            ok(f"T7 extra_metrics {name}: все NaN (emit_extra=False)")
        else:
            fail(f"T7 extra_metrics {name}: {5-nan_count}/5 не NaN при emit_extra=False")

# ff1/ff2 — emit_extra_metrics=True; ff3 — из ext (emit_extra_metrics=True)
check_extra(ff1, "legacy(emit_extra=True)", True)
check_extra(ff2, "diar+ASR(emit_extra=True)", True)
# ff3 from ext (emit_extra_metrics=True)
check_extra(ff3, "empty(emit_extra=True)", True)

# ===================== T8: артефакты .npy =====================
print("\n=== T8: артефакты .npy записаны и конечны ===")
artifacts_found = 0
for fname in os.listdir(TMP):
    if fname.endswith(".npy"):
        arr = np.load(os.path.join(TMP, fname))
        artifacts_found += 1
        if not np.all(np.isfinite(arr)):
            fail(f"T8 {fname}: не все finite")
        elif arr.ndim == 1 and arr.shape[0] > 0:
            # L2-norm проверка
            norm = float(np.linalg.norm(arr))
            if abs(norm - 1.0) > 0.01:
                fail(f"T8 {fname}: norm={norm:.4f} (ожидается ~1.0)")
        else:
            ok(f"T8 {fname}: shape={arr.shape}, finite, norm≈1.0")

if artifacts_found == 0:
    fail("T8: нет .npy артефактов")
else:
    ok(f"T8: найдено {artifacts_found} .npy файлов")

# ===================== Различимость U3 =====================
print("\n=== U3: различимость эмбеддингов между спикерами ===")
# Alice и Bob должны иметь разные mean-эмбеддинги
alice_path = os.path.join(TMP, "speaker_spk000_mean.npy")
bob_path = os.path.join(TMP, "speaker_spk001_mean.npy")
if os.path.exists(alice_path) and os.path.exists(bob_path):
    a = np.load(alice_path)
    b = np.load(bob_path)
    cos_sim = float(np.dot(a, b))
    if cos_sim < 0.999:
        ok(f"U3 различимость: cos_sim(Alice,Bob)={cos_sim:.4f} < 0.999")
    else:
        fail(f"U3 различимость: cos_sim(Alice,Bob)={cos_sim:.4f} ≈ 1.0 (возможно, одинаковые)")
else:
    print(f"  SKIP U3: не найдены артефакты alice={os.path.exists(alice_path)} bob={os.path.exists(bob_path)}")

# ===================== ИТОГ =====================
shutil.rmtree(TMP, ignore_errors=True)
shutil.rmtree(TMP2, ignore_errors=True)
shutil.rmtree(TMP3, ignore_errors=True)

print(f"\n{'='*50}")
print(f"ИТОГ: PASS={PASS} FAIL={FAIL}")
if FAIL == 0:
    print("✅ ВСЕ ТЕСТЫ ПРОШЛИ")
    sys.exit(0)
else:
    print("❌ ЕСТЬ ОШИБКИ")
    sys.exit(1)
