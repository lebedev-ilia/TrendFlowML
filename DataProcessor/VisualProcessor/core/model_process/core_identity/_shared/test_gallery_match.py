"""Юнит-тесты для gallery_match (задача 6 / Q4-подготовка). Чистый numpy, без сети.

Запуск: python test_gallery_match.py  (или pytest)
"""
import numpy as np

from gallery_match import aggregate_track_topk, l2_normalize, topk_cosine


def test_topk_identifies_nearest():
    # 3 ортогональных «прототипа» в галерее
    gallery = np.eye(4, dtype=np.float32)[:3]  # (3,4): e0,e1,e2
    queries = np.array([[0.9, 0.1, 0, 0], [0, 0, 1.0, 0]], dtype=np.float32)
    queries = l2_normalize(queries)
    idx, score = topk_cosine(queries, gallery, k=2)
    assert idx.shape == (2, 2) and score.shape == (2, 2)
    assert idx[0, 0] == 0          # первый запрос ближе к e0
    assert idx[1, 0] == 2          # второй — к e2
    assert np.all(np.diff(score, axis=1) <= 1e-6)  # отсортировано по убыванию
    assert 0.0 <= score.max() <= 1.0 + 1e-5


def test_k_clamped_to_gallery_size():
    gallery = l2_normalize(np.random.RandomState(0).randn(2, 8))
    q = l2_normalize(np.random.RandomState(1).randn(5, 8))
    idx, score = topk_cosine(q, gallery, k=10)
    assert idx.shape == (5, 2)     # k обрезан до размера галереи


def _expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("ожидался ValueError")


def test_shape_validation():
    g = np.zeros((3, 4), np.float32)
    _expect_value_error(lambda: topk_cosine(np.zeros((4,), np.float32), g, k=1))        # query не 2D
    _expect_value_error(lambda: topk_cosine(np.zeros((2, 5), np.float32), g, k=1))      # несовпадение D
    _expect_value_error(lambda: topk_cosine(np.zeros((1, 4), np.float32),
                                            np.zeros((0, 4), np.float32), k=1))         # пустая галерея


def test_aggregate_track_topk():
    pf_idx = np.array([[0, 1], [0, 2], [1, 0]])
    pf_sc = np.array([[0.9, 0.5], [0.7, 0.6], [0.8, 0.4]], dtype=np.float32)
    labels, scores = aggregate_track_topk(pf_idx, pf_sc, k=3)
    # label 0: max(0.9,0.7,0.4)=0.9; label1: max(0.5,0.8)=0.8; label2:0.6
    assert labels.tolist() == [0, 1, 2]
    assert np.allclose(scores, [0.9, 0.8, 0.6], atol=1e-6)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok: {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
