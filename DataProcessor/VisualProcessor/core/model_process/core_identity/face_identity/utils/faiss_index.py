import os

import faiss
import numpy as np


class FaceIndex:
    def __init__(self, dim: int = 512):
        """
        FAISS‑индекс для эмбеддингов лиц.

        dim — размер эмбеддинга (для ArcFace = 512)
        """
        self.dim = dim
        # IndexFlatIP — скалярное произведение (для L2‑нормированных векторов = cosine similarity)
        self.index = faiss.IndexFlatIP(dim)
        self.names: list[str] = []  # имена людей по индексу

    @property
    def is_empty(self) -> bool:
        return self.index.ntotal == 0

    def add(self, embedding: np.ndarray, name: str) -> None:
        """
        embedding — 512-d np.array L2-normalized
        name — имя/label человека
        """
        emb = embedding.reshape(1, -1).astype(np.float32)
        self.index.add(emb)
        self.names.append(name)

    def search(self, embedding: np.ndarray, top_k: int = 1):
        emb = embedding.reshape(1, -1).astype(np.float32)
        sims, ids = self.index.search(emb, top_k)
        return sims[0], ids[0]

    def save(self, folder: str = "face_db") -> None:
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "index.faiss"))
        np.save(os.path.join(folder, "names.npy"), np.array(self.names, dtype=object))

    def load(self, folder: str = "face_db") -> None:
        index_path = os.path.join(folder, "index.faiss")
        names_path = os.path.join(folder, "names.npy")
        if not (os.path.exists(index_path) and os.path.exists(names_path)):
            raise FileNotFoundError(f"Не найдены файлы индекса в папке {folder}")

        self.index = faiss.read_index(index_path)
        self.names = np.load(names_path, allow_pickle=True).tolist()


