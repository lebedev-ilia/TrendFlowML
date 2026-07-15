#!/usr/bin/env python3
"""
Offline-провижен публичных БАЗОВЫХ моделей (задача A2 / масштаб-мультинода).

Эти модели НЕ хранятся в едином HF-репо `trendflow_models` (раздел
`public_base_models` манифеста), но нужны компонентам. На мульти-ноде без них
ноды пойдут в сеть (нарушение no-network). Скрипт кладёт их в канонические пути
под DP_MODELS_ROOT (= DataProcessor/dp_models), переиспользуя существующие
`save_*/download_*` скрипты, а для остального — печатает точную инструкцию/URL.

Запуск (на машине с DataProcessor/.data_venv и сетью):
  python DataProcessor/scripts/provision_base_models.py --list
  python DataProcessor/scripts/provision_base_models.py                 # все доступные
  python DataProcessor/scripts/provision_base_models.py --only e5 source_separation
  HF_TOKEN=... python DataProcessor/scripts/provision_base_models.py --only pyannote

Стратегии:
  script     — вызвать существующий скрипт репозитория;
  hf_snapshot— huggingface_hub.snapshot_download в hf_cache (для speechbrain-баз);
  manual     — напечатать инструкцию/URL (fragile-источники не хардкодим слепо).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]       # .../TrendFlowML
DP = REPO_ROOT / "DataProcessor"
DP_MODELS_ROOT = Path(os.environ.get("DP_MODELS_ROOT", str(DP / "dp_models")))
SCRIPTS = DP / "scripts"

# id -> спецификация базовой модели
REGISTRY = {
    "e5": {
        "desc": "intfloat/multilingual-e5-large (TextProcessor embeddings)",
        "target": DP_MODELS_ROOT / "text" / "embeddings" / "intfloat_multilingual-e5-large",
        "strategy": "script",
        "cmd": [sys.executable, str(SCRIPTS / "save_sentence_transformer_model.py"),
                "--model-name", "intfloat/multilingual-e5-large",
                "--output-dir", "{target}"],
        "gated": False,
    },
    "source_separation": {
        "desc": "source separation (AudioProcessor) large",
        "target": DP_MODELS_ROOT / "audio" / "source_separation" / "large.pt",
        "strategy": "script",
        "cmd": [sys.executable, str(SCRIPTS / "download_source_separation_models.py"),
                "--models-root", str(DP_MODELS_ROOT), "--sizes", "large"],
        "gated": False,
    },
    "pyannote": {
        "desc": "pyannote speaker-diarization (GATED: нужен HF_TOKEN + принятие лицензии)",
        "target": DP_MODELS_ROOT / "audio" / "pyannote_speaker_diarization",
        "strategy": "script",
        "cmd": [sys.executable, str(SCRIPTS / "save_pyannote_diarization_model.py"),
                "--output-dir", "{target}"],
        "gated": True,
    },
    "wavlm_large": {
        "desc": "microsoft/wavlm-large (база speechbrain emotion_diarization)",
        "target": DP_MODELS_ROOT / "hf_cache" / "hub",
        "strategy": "hf_snapshot",
        "repo": "microsoft/wavlm-large",
        "gated": False,
    },
    "wav2vec2_base": {
        "desc": "facebook/wav2vec2-base (опц. база)",
        "target": DP_MODELS_ROOT / "hf_cache" / "hub",
        "strategy": "hf_snapshot",
        "repo": "facebook/wav2vec2-base",
        "gated": False,
    },
    "clap_630k": {
        "desc": "LAION CLAP 630k-audioset-best.pt (audio embedding)",
        "target": DP_MODELS_ROOT / "audio" / "laion_clap" / "clap_ckpt.pt",
        "strategy": "manual",
        "hint": "Скачать 630k-audioset-best.pt из релизов LAION-AI/CLAP (github) и положить в target.",
        "gated": False,
    },
    "places365_resnet50": {
        "desc": "Places365 ResNet50 (scene_classification inprocess) — веса CSAIL, авто-скачивание",
        "target": DP_MODELS_ROOT / "visual" / "places365" / "resnet50_places365.pth.tar",
        "strategy": "manual",
        "hint": ("Авто (проверено): mkdir -p {target%/*} && cd {target%/*} && "
                 "wget http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar && "
                 "wget https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt . "
                 "Загрузка torchvision resnet50(num_classes=365); state_dict под 'module.'-префиксом. "
                 "Также доступны resnet18/resnet152_places365 (те же URL-шаблоны). Places365 есть и в Triton (ONNX)."),
        "gated": False,
    },
    # --- action_recognition: альтернативные backbone (выбор при анализе --backbone) ---
    "action_videomae": {
        "desc": "VideoMAE base finetuned Kinetics-400 (action_recognition --backbone videomae)",
        "target": DP_MODELS_ROOT / "visual" / "action_recognition" / "videomae_base_finetuned_kinetics",
        "strategy": "hf_snapshot",
        "repo": "MCG-NJU/videomae-base-finetuned-kinetics",
        "gated": False,
    },
    "action_videomaev2": {
        "desc": "VideoMAEv2 base (action_recognition --backbone videomaev2; может требовать trust_remote_code)",
        "target": DP_MODELS_ROOT / "visual" / "action_recognition" / "videomaev2_base",
        "strategy": "hf_snapshot",
        "repo": "OpenGVLab/VideoMAEv2-Base",
        "gated": False,
    },
    "action_hiera": {
        "desc": "Hiera (action_recognition --backbone hiera; нужен video-K400 чекпоинт)",
        "target": DP_MODELS_ROOT / "visual" / "action_recognition" / "hiera",
        "strategy": "hf_snapshot",
        "repo": "facebook/hiera-base-224-in1k-hf",
        "gated": False,
    },
    "osnet_reid": {
        "desc": "OSNet x1_0 market1501 ReID (трекер --track-embedder osnet). torchreid скачивает "
                "веса автоматически при первом использовании (нужен pip install torchreid).",
        "target": DP_MODELS_ROOT / "visual" / "reid" / "osnet_x1_0_market1501.pth",
        "strategy": "manual",
        "hint": ("Автозагрузка: `OSNetBoxEmbedder(weights_path=None)` (torchreid, нужна сеть). Offline: "
                 "положи osnet_x1_0_market1501.pth в target и запускай с --track-osnet-weights <target>."),
        "gated": False,
    },
}

# группы для bootstrap --with-action-backbones (провижен всех alt-backbone action_recognition)
ACTION_BACKBONE_IDS = ["action_videomae", "action_videomaev2", "action_hiera", "osnet_reid"]


def _present(target: Path) -> bool:
    if target.suffix:  # файл
        return target.is_file()
    return target.is_dir() and any(target.iterdir()) if target.exists() else False


def do_list() -> int:
    print(f"DP_MODELS_ROOT = {DP_MODELS_ROOT}\n")
    for mid, spec in REGISTRY.items():
        st = "PRESENT" if _present(spec["target"]) else "missing"
        gated = " [GATED]" if spec.get("gated") else ""
        print(f"  {mid:<20} {st:<8}{gated}  {spec['desc']}")
        print(f"      -> {spec['target']}")
    return 0


def run_one(mid: str, spec: dict, dry: bool) -> str:
    target = spec["target"]
    if _present(target):
        return "skip"
    strat = spec["strategy"]
    if strat == "manual":
        print(f"[manual] {mid}: {spec['hint']}\n         target: {target}")
        return "manual"
    if strat == "hf_snapshot":
        if dry:
            print(f"[dry] hf snapshot {spec['repo']} -> {target}")
            return "dry"
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print(f"[skip] {mid}: нужен huggingface_hub (pip install huggingface_hub)")
            return "fail"
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=spec["repo"], cache_dir=str(target))
        return "ok"
    if strat == "script":
        cmd = [c.replace("{target}", str(target)) for c in spec["cmd"]]
        if spec.get("gated") and not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
            print(f"[skip] {mid}: GATED — задай HF_TOKEN и прими лицензию модели.")
            return "skip-gated"
        if dry:
            print(f"[dry] {mid}: {' '.join(cmd)}")
            return "dry"
        target.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(cmd, cwd=str(REPO_ROOT))
        return "ok" if r.returncode == 0 else "fail"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser("provision_base_models")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="*", help="подмножество id (см. --list)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.list:
        return do_list()

    ids = args.only or list(REGISTRY)
    unknown = [i for i in ids if i not in REGISTRY]
    if unknown:
        print(f"неизвестные id: {unknown}. Доступно: {list(REGISTRY)}")
        return 2

    results = {}
    for mid in ids:
        print(f"\n=== {mid} ===")
        results[mid] = run_one(mid, REGISTRY[mid], args.dry_run)
    print("\n=== ИТОГ ===")
    for mid, r in results.items():
        print(f"  {mid:<20} {r}")
    fails = [m for m, r in results.items() if r == "fail"]
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
