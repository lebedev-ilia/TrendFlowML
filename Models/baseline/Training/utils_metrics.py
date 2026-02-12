from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


def mae(y_true: List[float], y_pred: List[float]) -> float:
    n = 0
    s = 0.0
    for a, b in zip(y_true, y_pred):
        if a is None or b is None:
            continue
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        n += 1
        s += abs(a - b)
    return float("nan") if n == 0 else s / n


def _rankdata(x: List[float]) -> List[float]:
    """
    Average ranks for ties. Pure-Python (no scipy).
    """
    # pair value with original index
    pairs = [(v, i) for i, v in enumerate(x)]
    pairs.sort(key=lambda t: t[0])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        # average rank positions are 1-based
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[pairs[k][1]] = avg_rank
        i = j + 1
    return ranks


def spearmanr(y_true: List[float], y_pred: List[float]) -> float:
    xs: List[float] = []
    ys: List[float] = []
    for a, b in zip(y_true, y_pred):
        if a is None or b is None:
            continue
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        xs.append(a)
        ys.append(b)
    if len(xs) < 2:
        return float("nan")

    rx = _rankdata(xs)
    ry = _rankdata(ys)

    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = 0.0
    dx = 0.0
    dy = 0.0
    for a, b in zip(rx, ry):
        xa = a - mx
        yb = b - my
        num += xa * yb
        dx += xa * xa
        dy += yb * yb
    if dx <= 0.0 or dy <= 0.0:
        return float("nan")
    return num / math.sqrt(dx * dy)


def age_bucket(age_hours: float) -> str:
    if not math.isfinite(age_hours):
        return "unknown"
    if age_hours < 24.0:
        return "<24h"
    if age_hours < 24.0 * 30.0:
        return "1-30d"
    return ">30d"


@dataclass(frozen=True)
class MetricRow:
    n: int
    spearman: float
    mae: float


def compute_metrics(y_true: List[float], y_pred: List[float]) -> MetricRow:
    n = 0
    yt: List[float] = []
    yp: List[float] = []
    for a, b in zip(y_true, y_pred):
        if a is None or b is None:
            continue
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        yt.append(a)
        yp.append(b)
        n += 1
    return MetricRow(n=n, spearman=spearmanr(yt, yp), mae=mae(yt, yp))


def compute_metrics_by_bucket(
    *,
    y_true: List[float],
    y_pred: List[float],
    buckets: List[str],
) -> Dict[str, MetricRow]:
    idxs: Dict[str, List[int]] = {}
    for i, b in enumerate(buckets):
        idxs.setdefault(b, []).append(i)
    out: Dict[str, MetricRow] = {}
    for b, is_ in idxs.items():
        yt = [y_true[i] for i in is_]
        yp = [y_pred[i] for i in is_]
        out[b] = compute_metrics(yt, yp)
    return out


