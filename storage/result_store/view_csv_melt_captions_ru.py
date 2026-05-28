"""
Краткие русские подписи к именам фич (колонок) в melt-таблице view_csv.py.
Словарь токенов (части snake_case) + переопределения из JSON, иначе сборка.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

# Частые «вне meta_» и горячие meta_*: однозначная фраза.
FULL_KEY_RU: dict[str, str] = {
    "duration_ms": "Сколько миллисекунд длилась обработка шага/модуля.",
    "device_used": "Какой бэкенд/устройство (cpu, cuda, …) указан в отчёте.",
    "manifest_status": "Статус манифеста артефактов (ok, предупреждения, сбой).",
    "manifest_empty_reason": "Почему манифест пуст, если пуст (для сравнения сцен).",
    "npz_error": "Текст ошибки при чтении/записи NPZ, если была.",
    "meta_status": "Краткий итог шага/инференса (ok, fallback, нет данных и т.д.).",
    "meta_device": "Устройство, на котором считались тяжёлые стадии (как в NPZ).",
    "meta_device_used": "Совпадает/отличается с device_used; что реально взяла реализация.",
}

# Суффиксы/фрагменты в имени (в порядке приоритета) → короткая русская затычка, если
# полного ключа нет. Используем только если нет внешнего override.
SUFFIX_RU: tuple[tuple[str, str], ...] = (
    ("_total_frames", "итоговое число кадров/окон"),
    ("_processed_frames", "сколько кадров реально прогнали"),
    ("_min_cutoff", "нижняя граница сглаживания/фильтра"),
    ("_min_scene_seconds", "мин. длина сцены в секундах"),
    ("_max_frames", "ограничение на число кадров/отсечек"),
    ("_confidence", "порог уверенности/оценка"),
    ("_threshold", "порог/чувствительность (см. значение)"),
    ("_count", "счётчик/число элементов"),
    ("_sec", "величина в секундах"),
    ("_seconds", "величина в секундах"),
    ("_ms", "миллисекунды"),
    ("_fps", "частота кадров, к/с"),
    ("_ratio", "доля/отношение (0..1)"),
    ("_window", "размер/радиус окна"),
    ("_radius", "радиус/полуокно вокруг якоря"),
    ("_stride", "шаг/прореживание кадров"),
    ("_batch", "размер батча инференса"),
)

# Слова из snake_case (без цифр) → смысловой глоссарий. Расширяйте по мере новых колонок.
_TOKEN_RU: dict[str, str] = {
    "advanced": "расширенные",
    "alignment": "выровн.",
    "analysis": "анализ",
    "asymmetry": "асимметрия",
    "au": "аудио-юнит",
    "audio": "аудио",
    "average": "среднее",
    "backend": "бэкенд",
    "bands": "полосы",
    "basename": "баз. имя",
    "batch": "батч",
    "bbox": "прямоугольник",
    "beta": "β/бета-коэф.",
    "bins": "корзины",
    "boundaries": "границы",
    "box": "порог бокса (objectness)",
    "brand": "бренд",
    "bursts": "всплески/бёрсты",
    "category": "категория",
    "centered": "по центру",
    "channels": "каналы",
    "chars": "симв.",
    "chroma": "хрома/цв. призн.",
    "clap": "clap-метка",
    "class": "класс",
    "colors": "цвета",
    "compact": "компактн.",
    "components": "ГК/комп.",
    "confidence": "увер.",
    "contrast": "контраст",
    "count": "кол-во",
    "crop": "кроп",
    "curve": "кривая",
    "curves": "кривые",
    "cut": "склейка",
    "cutoff": "срез/част. порог",
    "data": "данные",
    "debug": "отладк.",
    "delta": "дельта",
    "detect": "детект.",
    "detection": "детекция",
    "detections": "детекции",
    "dets": "детекции",
    "device": "устройство",
    "digest": "дайджест/хэш",
    "dim": "разм.",
    "distance": "дистанция",
    "diversity": "разнообразие",
    "dominance": "долина/доминанты",
    "downscale": "даунскейл",
    "duration": "длительн.",
    "embedding": "эмбед.",
    "embeddings": "эмбед.",
    "emotional": "эмоц.",
    "empty": "пусто",
    "enable": "вкл.",
    "enabled": "включено",
    "energy": "энергия",
    "engine": "движок",
    "entropy": "энтропия",
    "events": "события",
    "every": "каждые",
    "export": "экспорт",
    "f0": "F0/тон",
    "face": "лицо",
    "faces": "лиц",
    "facing": "к камере",
    "factor": "коэф.",
    "failed": "сбой",
    "family": "семейство/семья",
    "feature": "признак",
    "features": "признаки",
    "fft": "FFT-разм.",
    "filter": "фильтр",
    "found": "найдено",
    "fps": "к/с",
    "frame": "кадр",
    "frames": "кадров",
    "franchise": "франшиза",
    "franchises": "франшиз",
    "fusion": "слияние",
    "gap": "зазор",
    "gaze": "взгляд",
    "global": "глобал.",
    "groups": "группы",
    "h": "выс. OCR",
    "height": "высота",
    "hist": "гистогр.",
    "histograms": "гистогр.",
    "hits": "сраб.",
    "hop": "шаг (STFT)",
    "hpss": "HPSS",
    "hue": "тон (HSV)",
    "id": "id",
    "img": "изобр.",
    "impl": "реализация",
    "individuality": "индивидуальн.",
    "init": "init",
    "input": "вход",
    "json": "JSON",
    "jump": "скачок",
    "k": "k",
    "keep": "сохр.",
    "key": "ключ/тональн.",
    "keyframes": "K-кадры",
    "kmeans": "k-means",
    "label": "метка",
    "labels": "меток",
    "lang": "язык",
    "language": "язык",
    "length": "длина",
    "loudness": "громкость",
    "low": "низк.",
    "margin": "поле",
    "mask": "маска",
    "max": "макс.",
    "mel": "мел-спектр",
    "mels": "мел-полос",
    "mesh": "меш-лицо",
    "method": "метод",
    "mfcc": "MFCC",
    "microexpr": "микроэмоц.",
    "microexpressions": "микроэмоции",
    "min": "мин.",
    "mode": "режим",
    "model": "модель",
    "motion": "движ.",
    "movement": "движ.",
    "ms": "мс",
    "multi": "неск./multi",
    "n": "N",
    "name": "имя",
    "normalize": "норм.",
    "npy": "NPY-файл",
    "num": "число",
    "objects": "объекты",
    "ocr": "OCR",
    "omitted": "опущено",
    "onset": "onset-врем.",
    "openface": "OpenFace",
    "out": "выходн.",
    "pace": "темп/ритм",
    "pad": "запас/пад",
    "palette": "палитра",
    "pca": "PCA-разм.",
    "peak": "пик",
    "peaks": "пики",
    "per": "на",
    "periodicity": "периодичн.",
    "person": "персона",
    "place": "место (Places365)",
    "places": "Places",
    "places365": "Places365",
    "ppocr": "PaddleOCR",
    "prefer": "предпочт.",
    "present": "есть/наличие",
    "preview": "превью",
    "primary": "основн.",
    "processed": "обр.",
    "processor": "процессор текста",
    "progress": "прогон",
    "prompt": "промпт",
    "prompts": "промпты",
    "proposal": "пропозиция (Класс)",
    "psm": "PSM",
    "quality": "качеств.",
    "radius": "радиус",
    "random": "seed",
    "rate": "част.",
    "ratio": "доля",
    "raw": "сырой",
    "reason": "причина",
    "rec": "модель recog.",
    "repeat": "повтор",
    "require": "нужен",
    "retain": "сохр.",
    "reused": "реюз",
    "runtime": "рантайм/время",
    "sample": "сэмпл-рейт/выбор",
    "sampling": "сэмплинг",
    "scene": "сцена",
    "scores": "скоры",
    "sec": "с",
    "seconds": "сек.",
    "segments": "сегм.",
    "semantic": "семантик.",
    "sep": "разд.",
    "series": "ряд",
    "set": "набор",
    "sharpness": "резкость",
    "shot": "план",
    "sigma": "σ/сглаж.",
    "similarity": "схожесть",
    "size": "размер",
    "smoothing": "сглаж.",
    "spec": "спек.",
    "speed": "скор.",
    "state": "state",
    "status": "статус",
    "store": "сохр.",
    "strength": "сила/вес",
    "stride": "шаг/stride",
    "target": "цель/целев.",
    "tempo": "темп",
    "temporal": "врем.",
    "tesseract": "Tesseract",
    "text": "текст",
    "threshold": "порог",
    "time": "время",
    "times": "моменты/разы",
    "timm": "timm",
    "top": "топ-",
    "top1": "топ-1",
    "topk": "топ-k",
    "total": "всего",
    "track": "трек/цель",
    "tracks": "треки",
    "transition": "переход",
    "transitions": "склейки/переходы",
    "trimmed": "урезан.",
    "tta": "TTA-аугм.",
    "tuning": "тюнинг/подбор",
    "type": "тип",
    "ui": "UI/интерф.",
    "use": "исп.",
    "used": "исп.",
    "version": "версия",
    "w": "шир. OCR",
    "warning": "предупр.",
    "weight": "вес",
    "weights": "веса",
    "width": "ширина",
    "window": "окно",
    "workers": "потоки/воркеры",
    "write": "запис.",
}

_NUM_WORD = re.compile(r"^(\d+)$")

def _token_caption(t: str) -> str:
    if not t:
        return ""
    if t in _TOKEN_RU:
        return _TOKEN_RU[t]
    m = _NUM_WORD.match(t)
    if m:
        return m.group(1)
    low = t.lower()
    if low in _TOKEN_RU:
        return _TOKEN_RU[low]
    return t


def _compose_from_name(feat: str) -> str:
    s = feat[5:] if feat.startswith("meta_") else feat
    for suf, blurb in SUFFIX_RU:
        if s.endswith(suf) and len(s) > len(suf) + 2:
            return f"{s.replace('_', ' ').strip()} — {blurb}."
    parts = [p for p in s.split("_") if p]
    ru = [_token_caption(p) for p in parts]
    body = " · ".join(ru) if ru else s
    if len(body) > 200:
        body = body[:197] + "…"
    return f"{body}."


def _timing_caption(feat: str, timing_prefix: str) -> str:
    if not feat.startswith(timing_prefix):
        return _compose_from_name(feat)
    rest = feat[len(timing_prefix) :]
    if not rest:
        return "Вклад/длительность по внутренней подметке (единица — по значению, часто мс)."
    return (
        f"Время/затраты на подшаг «{rest.replace('_', ' ')}». "
        f"Сравнивайте по видео; единица как в ячейке."
    )


def load_description_overrides(path: Optional[Path]) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw: Any = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    m = raw.get("descriptions")
    if isinstance(m, dict):
        out2: dict[str, str] = {}
        for k, v in m.items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            t = v.strip()
            if t:
                out2[k] = t
        return out2
    out: dict[str, str] = {}
    for k, v in raw.items():
        if k in ("comment", "descriptions") or not isinstance(k, str):
            continue
        if not isinstance(v, str):
            continue
        t = v.strip()
        if t:
            out[k] = t
    return out


def melt_feature_caption_ru(
    feature: str,
    overrides: dict[str, str],
    *,
    timing_prefix: str = "meta_timing_",
) -> str:
    if overrides:
        ov = overrides.get(feature)
        if isinstance(ov, str) and ov.strip():
            return ov.strip()
    o = FULL_KEY_RU.get(feature, "").strip()
    if o:
        return o
    if feature.startswith(timing_prefix):
        if feature.endswith("_ms"):
            return _timing_caption(feature, timing_prefix)
        # meta_timing_* без _ms — счётчики/флаги, не длительности
        return _compose_from_name(feature)
    c = _compose_from_name(feature)
    return c
