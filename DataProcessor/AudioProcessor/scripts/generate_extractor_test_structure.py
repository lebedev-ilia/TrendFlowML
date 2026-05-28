#!/usr/bin/env python3
"""Генерирует docs/scripts/utils для всех extractors (кроме voice_quality — уже есть)."""

import os
from pathlib import Path

EXTRACTORS = [
    ("asr_extractor", "asr_extractor_npz_v2", "asr"),
    ("band_energy_extractor", "band_energy_extractor_npz_v1", "band_energy"),
    ("chroma_extractor", "chroma_extractor_npz_v1", "chroma"),
    ("clap_extractor", "clap_extractor_npz_v1", "clap"),
    ("emotion_diarization_extractor", "emotion_diarization_extractor_npz_v1", "emotion_diarization"),
    ("hpss_extractor", "hpss_extractor_npz_v1", "hpss"),
    ("key_extractor", "key_extractor_npz_v1", "key"),
    ("loudness_extractor", "loudness_extractor_npz_v2", "loudness"),
    ("mel_extractor", "mel_extractor_npz_v2", "mel"),
    ("mfcc_extractor", "mfcc_extractor_npz_v2", "mfcc"),
    ("onset_extractor", "onset_extractor_npz_v2", "onset"),
    ("pitch_extractor", "pitch_extractor_npz_v2", "pitch"),
    ("quality_extractor", "quality_extractor_npz_v2", "quality"),
    ("rhythmic_extractor", "rhythmic_extractor_npz_v2", "rhythmic"),
    ("source_separation_extractor", "source_separation_extractor_npz_v2", "source_separation"),
    ("speaker_diarization_extractor", "speaker_diarization_extractor_npz_v2", "speaker_diarization"),
    ("spectral_extractor", "spectral_extractor_npz_v2", "spectral"),
    ("spectral_entropy_extractor", "spectral_entropy_extractor_npz_v2", "spectral_entropy"),
    ("speech_analysis_extractor", "speech_analysis_extractor_npz_v1", "speech_analysis"),
    ("tempo_extractor", "tempo_extractor_npz_v1", "tempo"),
]

BASE = Path(__file__).resolve().parent.parent / "src" / "extractors"


def write_testing_report(extractor: str):
    p = BASE / extractor / "docs" / "TESTING_REPORT.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    name = extractor.replace("_extractor", "")
    content = f"""# Отчёт о тестировании {extractor} компонента

**Дата**: (обновить после прогона)
**Компонент**: `{extractor}`
**Schema**: `{extractor.replace("_extractor", "")}_extractor_npz_*`

---

## Резюме

- **Протестировано видео**: 20
- **Успешных прогонов**: X/20
- **Валидных артефактов**: X/20

---

## Файлы

- **Валидатор**: `DataProcessor/AudioProcessor/src/extractors/{extractor}/utils/validate_{name}.py`
- **Скрипт тестирования**: `DataProcessor/AudioProcessor/src/extractors/{extractor}/scripts/run_tests.sh`
- **Скрипт анализа**: `DataProcessor/AudioProcessor/src/extractors/{extractor}/utils/analyze_all_results.py`
"""
    p.write_text(content, encoding="utf-8")
    print(f"  Wrote {p}")


def write_run_tests(extractor: str, key: str):
    p = BASE / extractor / "scripts" / "run_tests.sh"
    p.parent.mkdir(parents=True, exist_ok=True)
    content = f'''#!/bin/bash
# Тесты {extractor} на 20 видео

set -e
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
BASE_DIR="${{BASE_DIR:-"$(cd "${{SCRIPT_DIR}}/../../../../../.." && pwd)"}}"
VIDEOS_DIR="${{VIDEOS_DIR:-"${{BASE_DIR}}/example/example_videos"}}"
RESULTS_DIR="${{RESULTS_DIR:-"${{BASE_DIR}}/DataProcessor/dp_results"}}"
GLOBAL_CONFIG="${{GLOBAL_CONFIG:-"${{BASE_DIR}}/DataProcessor/configs/audit_v3/audio/profile_{key}.yaml"}}"
PROFILE_PATH="${{PROFILE_PATH:-"${{BASE_DIR}}/DataProcessor/configs/audit_v3/audio/profile_{key}.yaml"}}"
PYTHON="${{PYTHON:-"${{BASE_DIR}}/DataProcessor/.data_venv/bin/python"}}"
MAIN_SCRIPT="${{MAIN_SCRIPT:-"${{BASE_DIR}}/DataProcessor/main.py"}}"

SHORTEST_VIDEO="-Q6fnPIybEI.mp4"
VIDEOS=("-7Ei8e05x30.mp4" "-5EYUqIlyJU.mp4" "-U5ipG4hohY.mp4" "-15jH8mtfJw.mp4" "-BXwIsW0t9w.mp4" "-Ga4edhrfog.mp4" "-7pz_DGQPos.mp4" "-FOB4jpQIg8.mp4" "-Q_Ch-vrvvM.mp4" "-OBC82ymkcs.mp4" "-2b9IMP1ih0.mp4" "-VX009hQoDA.mp4" "-ZLHxCNCpdA.mp4" "-3GDPu4XLZY.mp4" "-T4Rvscu7b4.mp4" "-FyF-rDXAOU.mp4" "-1eKh7CJbhM.mp4" "-BBSE2F58ik.mp4" "-Cnn3Nq_Lpk.mp4")

cd "${{BASE_DIR}}"
echo "Тесты {extractor} (20 видео)"
echo "[0/20] test_{key}_shortest"
"${{PYTHON}}" "${{MAIN_SCRIPT}}" --video-path "${{VIDEOS_DIR}}/${{SHORTEST_VIDEO}}" --global-config "${{GLOBAL_CONFIG}}" --profile-path "${{PROFILE_PATH}}" --platform-id youtube --video-id "test_{key}_shortest" --run-id "test_{key}_shortest" --output "${{RESULTS_DIR}}" --rs-base "${{RESULTS_DIR}}" --no-run-visual > "/tmp/{key}_test_shortest.log" 2>&1 && echo "  OK" || echo "  FAIL"
for i in "${{!VIDEOS[@]}}"; do
  vid="${{VIDEOS[$i]}}"
  rid="test_{key}_$((i+2))"
  vp="${{VIDEOS_DIR}}/$vid"
  [ ! -f "$vp" ] && echo "Skip $vid" && continue
  echo "[$((i+1))/20] $rid"
  "${{PYTHON}}" "${{MAIN_SCRIPT}}" --video-path "$vp" --global-config "${{GLOBAL_CONFIG}}" --profile-path "${{PROFILE_PATH}}" --platform-id youtube --video-id "$rid" --run-id "$rid" --output "${{RESULTS_DIR}}" --rs-base "${{RESULTS_DIR}}" --no-run-visual > "/tmp/{key}_test_$rid.log" 2>&1 && echo "  OK" || echo "  FAIL"
done
echo "Done"
'''
    p.write_text(content, encoding="utf-8")
    os.chmod(p, 0o755)
    print(f"  Wrote {p}")


def write_run_analyze(extractor: str, key: str):
    p = BASE / extractor / "scripts" / "run_analyze.sh"
    content = f'''#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
BASE_DIR="${{BASE_DIR:-"$(cd "${{SCRIPT_DIR}}/../../../../../.." && pwd)"}}"
RESULTS_DIR="${{RESULTS_DIR:-"${{BASE_DIR}}/DataProcessor/dp_results"}}"
PYTHON="${{PYTHON:-"${{BASE_DIR}}/DataProcessor/.data_venv/bin/python"}}"
ANALYZER="${{SCRIPT_DIR}}/../utils/analyze_all_results.py"
cd "${{BASE_DIR}}/DataProcessor"
PYTHONPATH="${{BASE_DIR}}/DataProcessor/AudioProcessor/src:${{PYTHONPATH}}" "${{PYTHON}}" "${{ANALYZER}}" --rs-base "${{RESULTS_DIR}}/youtube" --run-id-prefix "test_{key}_" --component-name "{extractor}"
'''
    p.write_text(content, encoding="utf-8")
    os.chmod(p, 0o755)
    print(f"  Wrote {p}")


def write_validate(extractor: str, schema: str):
    p = BASE / extractor / "utils" / f"validate_{extractor.replace('_extractor', '')}.py"
    name = extractor.replace("_extractor", "")
    content = f'''#!/usr/bin/env python3
"""Валидатор для {extractor}."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

def load_npz(npz_path: str) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    return {{k: (data[k].item() if data[k].dtype == object and data[k].size == 1 else data[k]) for k in data.files}}

def extract_meta(d: Dict) -> Dict:
    m = d.get("meta")
    return m.item() if hasattr(m, "item") else (m or {{}})

def validate(npz_path: str) -> bool:
    try:
        d = load_npz(npz_path)
        meta = extract_meta(d)
        if "meta" not in d:
            return False
        sv = str(meta.get("schema_version", ""))
        return "{schema}" in sv or "{extractor}" in sv
    except Exception:
        return False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz_path")
    args = p.parse_args()
    print("✅ VALID" if validate(args.npz_path) else "❌ INVALID")
    return 0 if validate(args.npz_path) else 1

if __name__ == "__main__":
    sys.exit(main())
'''
    p.write_text(content, encoding="utf-8")
    print(f"  Wrote {p}")


def write_analyze(extractor: str, key: str):
    p = BASE / extractor / "utils" / "analyze_all_results.py"
    name = extractor.replace("_extractor", "")
    content = f'''#!/usr/bin/env python3
"""Анализ результатов {extractor}."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Dict, Any

def analyze(rs_base: str = "dp_results/youtube", run_id_prefix: str = "test_{key}_", component_name: str = "{extractor}", npz_name: str = "{extractor}_features.npz") -> Dict[str, Any]:
    rs = Path(rs_base) / "youtube"
    if not rs.exists():
        return {{"total_videos": 0, "per_video": [], "summary": {{}}}}
    stats = []
    for run_dir in sorted(rs.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith(run_id_prefix):
            continue
        npz = run_dir / run_dir.name / component_name / npz_name
        if not npz.exists():
            continue
        stats.append({{"video_id": run_dir.name, "valid": True}})
    return {{"total_videos": len(stats), "per_video": stats, "summary": {{"valid_count": len(stats), "total_count": len(stats)}}}}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rs-base", default="dp_results/youtube")
    p.add_argument("--run-id-prefix", default="test_{key}_")
    p.add_argument("--component-name", default="{extractor}")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    r = analyze(args.rs_base, args.run_id_prefix, args.component_name)
    print(json.dumps(r, indent=2, ensure_ascii=False) if args.json else f"Всего: {{r['total_videos']}}, валидных: {{r['summary'].get('valid_count', 0)}}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
    p.write_text(content, encoding="utf-8")
    print(f"  Wrote {p}")


ALL_EXTRACTOR_KEYS = [
    "asr", "band_energy", "chroma", "clap", "emotion_diarization", "hpss", "key",
    "loudness", "mel", "mfcc", "onset", "pitch", "quality", "rhythmic",
    "source_separation", "speaker_diarization", "spectral", "spectral_entropy",
    "speech_analysis", "tempo", "voice_quality",
]


def write_profile(extractor: str, key: str):
    cfg_dir = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "audit_v3" / "audio"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    p = cfg_dir / f"profile_{key}.yaml"
    # Minimal: only this extractor enabled. Exclude key from false-list to avoid YAML duplicate overwrite.
    false_lines = "\n      ".join(
        f"{k}:\n        enabled: false" for k in ALL_EXTRACTOR_KEYS if k != key
    )
    content = f'''version: "1.0.0"
global:
  platform_id: "youtube"
  sampling_policy_version: "v1"
  dataprocessor_version: "audit3_test"
processors:
  audio:
    enabled: true
    required: false
    device: "auto"
    extractors:
      {key}:
        enabled: true
      {false_lines}
  text:
    enabled: false
  visual:
    enabled: false
'''
    p.write_text(content, encoding="utf-8")
    print(f"  Wrote {p}")


def main():
    for extractor, schema, key in EXTRACTORS:
        if extractor == "voice_quality_extractor":
            continue
        print(f"\n{extractor}:")
        write_testing_report(extractor)
        write_run_tests(extractor, key)
        write_run_analyze(extractor, key)
        write_validate(extractor, schema)
        write_analyze(extractor, key)
        write_profile(extractor, key)
    print("\nDone.")


if __name__ == "__main__":
    main()
