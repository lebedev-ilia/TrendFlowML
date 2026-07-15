#!/usr/bin/env python3
"""
Альтернативные backbone для action_recognition — выбираются при запуске анализа (`--backbone`).

SlowFast (заштампован v3) остаётся в `action_recognition_slowfast.py` без изменений — это дефолт.
Здесь — VideoMAE / VideoMAEv2 / Hiera через HuggingFace `transformers`. Каждый backbone
самодостаточен: грузит модель, препроцессит клип, отдаёт (penultimate-эмбеддинг L2, логиты) и СВОЮ
карту классов (id2label модели — порядок индексов у разных моделей РАЗНЫЙ, поэтому метки берём из
самой модели, а не из общего kinetics-файла).

Ленивая загрузка тяжёлых зависимостей — внутри `load()`; при отсутствии модели/пакета бросаем
понятную ошибку (recognizer делает fallback→slowfast).

Контракт:
  bb = build_backbone(name, device, precision, model_id=None)
  bb.load()                              # грузит модель + processor + hook на классификатор
  bb.clip_len          -> int            # сколько кадров в клипе ждёт backbone (VideoMAE/Hiera=16)
  bb.num_classes       -> int
  bb.penultimate_dim   -> int            # известно после load
  bb.class_names()     -> List[str]|None # id->name из модели
  emb, logits = bb.infer_clips(clips, batch_size)   # clips: List[List[np.ndarray RGB uint8]]
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Известные дефолтные чекпоинты (можно переопределить model_id / локальным путём при провижене).
DEFAULT_MODEL_IDS = {
    "videomae":   "MCG-NJU/videomae-base-finetuned-kinetics",     # VideoMAE(v1) K400, 16 кадров
    "videomaev2": "OpenGVLab/VideoMAEv2-Base",                    # v2 (может требовать trust_remote_code)
    "hiera":      "facebook/hiera-base-224-in1k-hf",              # Hiera (по умолчанию image; video — см. ниже)
}


class Backbone:
    name = "base"
    clip_len = 16
    num_classes = 400

    def __init__(self, device: str = "cpu", precision: str = "fp32", model_id: Optional[str] = None):
        self.device = device
        self.precision = precision
        self.model_id = model_id or DEFAULT_MODEL_IDS.get(self.name)
        self.model = None
        self.processor = None
        self.penultimate_dim = 0
        self._penult_buf = None
        self._id2label: Optional[Dict[int, str]] = None

    def load(self) -> None:
        raise NotImplementedError

    def class_names(self) -> Optional[List[str]]:
        if not self._id2label:
            return None
        n = max(self._id2label) + 1
        return [str(self._id2label.get(i, f"action_{i}")).replace("_", " ") for i in range(n)]

    # --- общие утилиты ---
    def _register_penult_hook(self, classifier_module) -> None:
        def _cap(module, inp):
            try:
                x = inp[0]
                if hasattr(x, "dim") and x.dim() > 2:
                    x = x.mean(dim=list(range(1, x.dim() - 1)))
                self._penult_buf = x.detach()
            except Exception:
                self._penult_buf = None
        classifier_module.register_forward_pre_hook(_cap)

    def infer_clips(self, clips: List[List[np.ndarray]], batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class _HFVideoBackbone(Backbone):
    """Общая реализация для transformers-video-классификаторов (VideoMAE/VideoMAEv2)."""
    def load(self) -> None:
        import torch
        from transformers import AutoModelForVideoClassification, AutoImageProcessor
        self._torch = torch
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForVideoClassification.from_pretrained(self.model_id)
        self.model.eval().to(self.device)
        if self.precision == "fp16" and str(self.device).startswith("cuda"):
            self.model = self.model.half()
        cfg = self.model.config
        self.num_classes = int(getattr(cfg, "num_labels", self.num_classes))
        self._id2label = {int(k): v for k, v in getattr(cfg, "id2label", {}).items()} or None
        # clip_len из конфига (num_frames), дефолт 16
        self.clip_len = int(getattr(cfg, "num_frames", 16) or 16)
        # penultimate: вход классификатора
        clf = getattr(self.model, "classifier", None)
        if clf is not None:
            self._register_penult_hook(clf)
            self.penultimate_dim = int(getattr(cfg, "hidden_size", 768) or 768)
        else:
            self.penultimate_dim = int(getattr(cfg, "hidden_size", 768) or 768)

    def _prep_batch(self, clip_batch: List[List[np.ndarray]]):
        # processor ждёт список видео (каждое — список PIL/np кадров). Приводим к clip_len.
        vids = []
        for frames in clip_batch:
            fr = frames
            if len(fr) >= self.clip_len:
                idx = np.linspace(0, len(fr) - 1, self.clip_len).round().astype(int)
                fr = [fr[i] for i in idx]
            else:
                fr = fr + [fr[-1]] * (self.clip_len - len(fr))
            vids.append(list(fr))
        return self.processor(vids, return_tensors="pt")

    def infer_clips(self, clips: List[List[np.ndarray]], batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        import torch.nn.functional as F
        embs, logs = [], []
        for s in range(0, len(clips), batch_size):
            batch = clips[s:s + batch_size]
            self._penult_buf = None
            inputs = self._prep_batch(batch)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                out = self.model(**inputs)
            logits = out.logits.float()
            penult = self._penult_buf
            if penult is not None and penult.dim() == 2 and penult.shape[0] == logits.shape[0]:
                emb = F.normalize(penult.float(), p=2, dim=1)
            else:
                emb = F.normalize(logits, p=2, dim=1)
            embs.append(emb.cpu().numpy().astype(np.float32))
            logs.append(logits.cpu().numpy().astype(np.float32))
        if not embs:
            return (np.zeros((0, self.penultimate_dim), np.float32), np.zeros((0, self.num_classes), np.float32))
        return np.concatenate(embs, 0), np.concatenate(logs, 0)


class VideoMAEBackbone(_HFVideoBackbone):
    name = "videomae"
    clip_len = 16


class VideoMAEv2Backbone(_HFVideoBackbone):
    name = "videomaev2"
    clip_len = 16

    def load(self) -> None:
        # v2-чекпоинты часто требуют trust_remote_code; пробуем, при неудаче — понятная ошибка.
        import torch
        from transformers import AutoModelForVideoClassification, AutoImageProcessor
        self._torch = torch
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForVideoClassification.from_pretrained(self.model_id, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(
                f"VideoMAEv2 '{self.model_id}' не загрузился ({e}). Укажи совместимый чекпоинт "
                f"(model_id) или используй --backbone videomae."
            ) from e
        self.model.eval().to(self.device)
        cfg = self.model.config
        self.num_classes = int(getattr(cfg, "num_labels", 400))
        self._id2label = {int(k): v for k, v in getattr(cfg, "id2label", {}).items()} or None
        self.clip_len = int(getattr(cfg, "num_frames", 16) or 16)
        clf = getattr(self.model, "classifier", None)
        if clf is not None:
            self._register_penult_hook(clf)
        self.penultimate_dim = int(getattr(cfg, "hidden_size", 768) or 768)


class HieraBackbone(_HFVideoBackbone):
    """Hiera video-классификатор. Требует transformers с поддержкой Hiera-video ИЛИ пакет `hiera`."""
    name = "hiera"
    clip_len = 16

    def load(self) -> None:
        import torch
        self._torch = torch
        try:
            from transformers import AutoModelForVideoClassification, AutoImageProcessor
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForVideoClassification.from_pretrained(self.model_id)
        except Exception as e:
            raise RuntimeError(
                f"Hiera '{self.model_id}' как video-классификатор не загрузился ({e}). Нужен "
                f"video-чекпоинт Hiera (K400) и совместимый transformers; иначе --backbone videomae/slowfast."
            ) from e
        self.model.eval().to(self.device)
        cfg = self.model.config
        self.num_classes = int(getattr(cfg, "num_labels", 400))
        self._id2label = {int(k): v for k, v in getattr(cfg, "id2label", {}).items()} or None
        self.clip_len = int(getattr(cfg, "num_frames", 16) or 16)
        clf = getattr(self.model, "classifier", None)
        if clf is not None:
            self._register_penult_hook(clf)
        self.penultimate_dim = int(getattr(cfg, "hidden_size", 768) or 768)


_REGISTRY = {
    "videomae": VideoMAEBackbone,
    "videomaev2": VideoMAEv2Backbone,
    "hiera": HieraBackbone,
}


def build_backbone(name: str, device: str = "cpu", precision: str = "fp32",
                   model_id: Optional[str] = None) -> Backbone:
    key = str(name).lower()
    if key not in _REGISTRY:
        raise ValueError(f"неизвестный backbone '{name}'. Доступно: {sorted(_REGISTRY)} (+ slowfast в основном модуле)")
    return _REGISTRY[key](device=device, precision=precision, model_id=model_id)
