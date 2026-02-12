from __future__ import annotations

"""
Encoder v1 (trainable) per `Models/docs/contracts/MODEL_CONTRACTS_V1.md`.

This is an MVP trainable encoder:
- same outputs as encoder_v0
- adaptive K rule (64/96/128)
- O(N) aggregation via uniform time bins (discrete binning, deterministic)

It is intentionally simple: projection + per-bin pooling + MLP.
"""

from dataclasses import dataclass
from typing import Dict

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


def _require_torch():
    if torch is None or nn is None:  # pragma: no cover
        raise RuntimeError("encoder_v1 requires torch. Install torch in your training environment.")
    return torch, nn


@dataclass(frozen=True)
class EncoderV1Config:
    d_model: int = 768
    # adaptive K rule
    k_short: int = 64
    k_mid: int = 96
    k_long: int = 128
    dur_short_s: float = 90.0
    dur_mid_s: float = 600.0
    dropout: float = 0.1


def adaptive_k(duration_sec: float, cfg: EncoderV1Config) -> int:
    if duration_sec < cfg.dur_short_s:
        return cfg.k_short
    if duration_sec < cfg.dur_mid_s:
        return cfg.k_mid
    return cfg.k_long


class EncoderV1(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, *, cfg: EncoderV1Config, d_in: int):
        torch, nn = _require_torch()
        super().__init__()  # type: ignore[misc]
        self.cfg = cfg
        self.d_in = int(d_in)

        self.in_proj = nn.Sequential(
            nn.Linear(self.d_in, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.token_mlp = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

    def forward(self, *, times_s, x, duration_sec: float) -> Dict[str, "torch.Tensor"]:
        torch, nn = _require_torch()
        if duration_sec <= 0:
            raise ValueError("duration_sec must be > 0")

        times = times_s if isinstance(times_s, torch.Tensor) else torch.as_tensor(times_s, dtype=torch.float32)
        xx = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
        if times.ndim != 1:
            raise ValueError("times_s must be 1D")
        if xx.ndim != 2 or xx.shape[0] != times.shape[0]:
            raise ValueError("x must be (N, D_in) with same N as times_s")
        if xx.shape[1] != self.d_in:
            raise ValueError(f"d_in mismatch: expected {self.d_in}, got {xx.shape[1]}")

        K = adaptive_k(float(duration_sec), self.cfg)
        edges = torch.linspace(0.0, float(duration_sec), K + 1, dtype=torch.float32, device=times.device)
        centers = 0.5 * (edges[:-1] + edges[1:])

        # sort by time
        order = torch.argsort(times)
        times_sorted = times[order]
        x_sorted = xx[order]

        # project
        z = self.in_proj(x_sorted)  # (N, d_model)

        bin_ids = torch.bucketize(times_sorted, edges[1:-1], right=False)  # (N,) in 0..K-1
        summary_mask = torch.zeros(K, dtype=torch.float32, device=times.device)
        pooled_mean = torch.zeros(K, self.cfg.d_model, dtype=torch.float32, device=times.device)
        pooled_max = torch.zeros(K, self.cfg.d_model, dtype=torch.float32, device=times.device)

        start = 0
        for k in range(K):
            end = start
            while end < bin_ids.numel() and int(bin_ids[end].item()) == k:
                end += 1
            if end == start:
                start = end
                continue
            chunk = z[start:end]  # (M, d_model)
            pooled_mean[k] = chunk.mean(dim=0)
            pooled_max[k] = chunk.max(dim=0).values
            summary_mask[k] = 1.0
            start = end

        feat = torch.cat([pooled_mean, pooled_max], dim=1)  # (K, 2*d_model)
        summary_tokens = self.token_mlp(feat)  # (K, d_model)
        summary_tokens = summary_tokens * summary_mask[:, None]

        denom = summary_mask.sum().clamp(min=1.0)
        global_embedding = summary_tokens.sum(dim=0) / denom

        return {
            "global_embedding": global_embedding,
            "summary_tokens": summary_tokens,
            "summary_times_s": centers.to(summary_tokens.dtype),
            "summary_mask": summary_mask,
        }


