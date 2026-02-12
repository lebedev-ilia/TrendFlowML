from __future__ import annotations

"""
Encoder v0 (deterministic) per `Models/docs/contracts/MODEL_CONTRACTS_V1.md`.

Contract output (per modality):
- global_embedding: (D,)
- summary_tokens: (K, D)
- summary_times_s: (K,) (uniform bin centers on [0..duration_sec])
- summary_mask: (K,) (1 for non-empty bins, else 0)

Complexity:
- O(N) in input sequence length (binning is linear via sorting once).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


def _require_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception as e:
        raise RuntimeError("v1 encoder requires PyTorch. Install torch in your training environment.") from e


@dataclass(frozen=True)
class EncoderV0Config:
    d_model: int = 768
    seed: int = 1337
    # adaptive K rule
    k_short: int = 64
    k_mid: int = 96
    k_long: int = 128
    dur_short_s: float = 90.0
    dur_mid_s: float = 600.0
    # stats set (fixed)
    use_mean: bool = True
    use_max: bool = True
    use_p50: bool = True
    use_p90: bool = True


def adaptive_k(duration_sec: float, cfg: EncoderV0Config) -> int:
    if duration_sec < cfg.dur_short_s:
        return cfg.k_short
    if duration_sec < cfg.dur_mid_s:
        return cfg.k_mid
    return cfg.k_long


class EncoderV0:
    """
    Deterministic encoder that maps (times_s, x[N,Din]) -> (summary_tokens[K,D], summary_times_s[K], mask[K]).

    Implementation notes:
    - uniform bins
    - per-bin stats (mean/max/p50/p90)
    - fixed random projection to d_model (seeded, not trainable)
    """

    def __init__(self, *, cfg: EncoderV0Config, d_in: int):
        torch = _require_torch()
        self.cfg = cfg
        self.d_in = int(d_in)

        stat_mult = 0
        stat_mult += 1 if cfg.use_mean else 0
        stat_mult += 1 if cfg.use_max else 0
        stat_mult += 1 if cfg.use_p50 else 0
        stat_mult += 1 if cfg.use_p90 else 0
        self.d_feat = self.d_in * stat_mult

        g = torch.Generator()
        g.manual_seed(cfg.seed + 17 * d_in)
        # fixed projection (buffer-like): [d_feat, d_model]
        self.W = torch.randn(self.d_feat, cfg.d_model, generator=g) * (1.0 / (self.d_feat**0.5))
        self.b = torch.zeros(cfg.d_model)

    def __call__(self, *, times_s, x, duration_sec: float) -> Dict[str, "torch.Tensor"]:
        torch = _require_torch()
        if duration_sec <= 0:
            raise ValueError("duration_sec must be > 0")

        times = times_s
        if not isinstance(times, torch.Tensor):
            times = torch.as_tensor(times, dtype=torch.float32)
        if times.ndim != 1:
            raise ValueError("times_s must be 1D")

        xx = x
        if not isinstance(xx, torch.Tensor):
            xx = torch.as_tensor(xx, dtype=torch.float32)
        if xx.ndim != 2 or xx.shape[0] != times.shape[0]:
            raise ValueError("x must be (N, D_in) with same N as times_s")
        if xx.shape[1] != self.d_in:
            raise ValueError(f"d_in mismatch: expected {self.d_in}, got {xx.shape[1]}")

        K = adaptive_k(float(duration_sec), self.cfg)
        edges = torch.linspace(0.0, float(duration_sec), K + 1, dtype=torch.float32)
        centers = 0.5 * (edges[:-1] + edges[1:])

        # Sort by time once (O(N log N) worst case; acceptable, but we can keep N small.
        # If strict O(N) is required, upstream should already provide sorted times.
        order = torch.argsort(times)
        times_sorted = times[order]
        x_sorted = xx[order]

        # assign each element to a bin id in [0..K-1]
        # bucketize uses right=False => edges[i-1] <= x < edges[i]
        bin_ids = torch.bucketize(times_sorted, edges[1:-1], right=False)  # shape (N,), values 0..K-1

        summary_mask = torch.zeros(K, dtype=torch.float32)
        tokens_feat = torch.zeros(K, self.d_feat, dtype=torch.float32)

        # per-bin aggregate (linear scan)
        start = 0
        for k in range(K):
            # advance end
            end = start
            while end < bin_ids.numel() and int(bin_ids[end].item()) == k:
                end += 1
            if end == start:
                start = end
                continue
            chunk = x_sorted[start:end]  # (M, D_in)
            feats = []
            if self.cfg.use_mean:
                feats.append(chunk.mean(dim=0))
            if self.cfg.use_max:
                feats.append(chunk.max(dim=0).values)
            if self.cfg.use_p50:
                try:
                    feats.append(torch.quantile(chunk, 0.50, dim=0))
                except Exception:
                    feats.append(chunk.median(dim=0).values)
            if self.cfg.use_p90:
                feats.append(torch.quantile(chunk, 0.90, dim=0))
            vec = torch.cat(feats, dim=0)  # (d_feat,)
            tokens_feat[k] = vec
            summary_mask[k] = 1.0
            start = end

        # projection to d_model
        summary_tokens = tokens_feat @ self.W + self.b  # (K, d_model)
        # zero-out empty bins to keep determinism
        summary_tokens = summary_tokens * summary_mask[:, None]

        # global embedding = mean of non-empty tokens
        denom = summary_mask.sum().clamp(min=1.0)
        global_embedding = summary_tokens.sum(dim=0) / denom

        return {
            "global_embedding": global_embedding,
            "summary_tokens": summary_tokens,
            "summary_times_s": centers,
            "summary_mask": summary_mask,
        }


def validate_encoder_output(out: Dict[str, "torch.Tensor"], *, d_model: int) -> None:
    torch = _require_torch()
    for k in ("global_embedding", "summary_tokens", "summary_times_s", "summary_mask"):
        if k not in out:
            raise ValueError(f"Missing key: {k}")
    ge = out["global_embedding"]
    st = out["summary_tokens"]
    tt = out["summary_times_s"]
    sm = out["summary_mask"]
    if ge.shape != (d_model,):
        raise ValueError(f"global_embedding shape must be ({d_model},), got {tuple(ge.shape)}")
    if st.ndim != 2 or st.shape[1] != d_model:
        raise ValueError(f"summary_tokens must be (K,{d_model}), got {tuple(st.shape)}")
    K = st.shape[0]
    if tt.shape != (K,):
        raise ValueError(f"summary_times_s must be ({K},), got {tuple(tt.shape)}")
    if sm.shape != (K,):
        raise ValueError(f"summary_mask must be ({K},), got {tuple(sm.shape)}")
    if not torch.all((sm == 0.0) | (sm == 1.0)):
        raise ValueError("summary_mask must be binary (0/1)")


