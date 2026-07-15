#!/usr/bin/env python3
"""
Провижен меток Kinetics-400 в порядке выходных индексов модели SlowFast (pytorchvideo).

Зачем: `action_recognition` v3 отдаёт `clip_topk_action_ids` (индексы логитов головы SlowFast).
Чтобы владелец мог семантически сверить действия, нужна карта id→name в ТОЧНОМ порядке обучения
модели. pytorchvideo обучает slowfast_r50 на Kinetics-400 с фиксированным порядком классов,
заданным официальным `kinetics_classnames.json` (name→id). Этот скрипт скачивает его и пишет
`kinetics400_labels.txt` (400 строк, строка i = имя класса с индексом i).

Почему скрипт, а не хардкод: любой ручной список из 400 имён рискует перепутать ПОРЯДОК индексов
(модель отдаёт по своему обучению, а не по алфавиту). Скачивание канонического источника
гарантирует совпадение порядка с логитами → классы будут корректны.

Запуск (среда с сетью, напр. Cursor):
  DataProcessor/.data_venv/bin/python DataProcessor/scripts/provision_kinetics_labels.py
Пишет в: $DP_MODELS_ROOT/visual/action_recognition/kinetics400_labels.txt
(action_recognition сам подхватит его через _load_kinetics_class_names()).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

# Канонический источник pytorchvideo (тот же, что в их официальном туториале по классификации).
CLASSNAMES_URL = "https://dl.fbaipublicfiles.com/pytorchvideo/kinetics/kinetics_classnames.json"
NUM_CLASSES = 400


def _default_out_path() -> str:
    root = os.environ.get("DP_MODELS_ROOT")
    if not root:
        # относительно репозитория: DataProcessor/dp_models
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.join(here, os.pardir, "dp_models")
    return os.path.abspath(os.path.join(root, "visual", "action_recognition", "kinetics400_labels.txt"))


def fetch_id_to_name(url: str = CLASSNAMES_URL) -> list[str]:
    """Скачивает name->id JSON и возвращает список имён в порядке индекса (0..399)."""
    with urllib.request.urlopen(url, timeout=60) as r:
        raw = json.loads(r.read().decode("utf-8"))
    # формат pytorchvideo: {'"abseiling"': 0, ...} — имена в двойных кавычках, значения = id
    id_to_name: dict[int, str] = {}
    for name, idx in raw.items():
        clean = str(name).strip().strip('"').replace("_", " ")
        id_to_name[int(idx)] = clean
    if len(id_to_name) != NUM_CLASSES:
        raise RuntimeError(f"ожидалось {NUM_CLASSES} классов, получено {len(id_to_name)}")
    return [id_to_name[i] for i in range(NUM_CLASSES)]


def main() -> int:
    out_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--out" else _default_out_path()
    try:
        names = fetch_id_to_name()
    except Exception as e:
        print(f"❌ не удалось получить/распарсить Kinetics classnames: {e}", file=sys.stderr)
        print("   (нужна сеть; источник:", CLASSNAMES_URL, ")", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")
    os.replace(tmp, out_path)
    print(f"✅ записано {len(names)} меток Kinetics-400 → {out_path}")
    print("   примеры:", names[0], "|", names[199], "|", names[399])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
