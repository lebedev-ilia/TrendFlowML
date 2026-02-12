from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class TextEncoderSpec:
    """
    Pinned text encoder spec for V3.

    Notes:
    - No-network policy: weights must already be available/cached in the runtime environment.
    - For MVP we default to SBERT MiniLM (fast, multilingual-ish).
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"


class SentenceTransformerEncoder:
    def __init__(self, spec: TextEncoderSpec):
        self.spec = spec
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependency: sentence-transformers. "
                "Install it (and ensure weights are cached) to run V3 text embeddings."
            ) from e

        self._model = SentenceTransformer(spec.model_name, device=spec.device)

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: List[str]) -> "np.ndarray":
        try:
            import numpy as np  # type: ignore
        except Exception as e:
            raise RuntimeError("Text encoding requires numpy.") from e

        # sentence-transformers returns np.ndarray by default
        emb = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(emb, dtype="float32")


