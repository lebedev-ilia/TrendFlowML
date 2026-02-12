from __future__ import annotations

"""
V2: v1 model skeleton (minimal but contract-correct interfaces).

Implements:
- token inputs: visual/audio/text/meta
- time encoding: MLP(t_center / duration_sec)
- fusion: transformer encoder over concatenated tokens (note: doc prefers cross-attention; this is the minimal stable base)
- outputs: 6 p50 heads (views/likes × 7/14/21), with masked loss support for 7d
- horizon weighting: base weights + learnable uncertainty weights with cap [0.2..5.0]
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


def _require_torch():
    if torch is None or nn is None:  # pragma: no cover
        raise RuntimeError("v1 skeleton requires PyTorch. Install torch in your training environment.")
    return torch, nn


@dataclass(frozen=True)
class V1SkeletonConfig:
    d_model: int = 768
    n_layers: int = 4
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    # time embedding MLP hidden size
    time_mlp_hidden: int = 128
    # horizon base weights
    w_7d: float = 0.5
    w_14d: float = 1.0
    w_21d: float = 1.0
    # uncertainty weight cap
    w_cap_min: float = 0.2
    w_cap_max: float = 5.0
    # quantiles (V5). If length==1 and equals 0.5 => point-only mode.
    quantiles: tuple[float, ...] = (0.5,)


class V1Skeleton(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, *, cfg: V1SkeletonConfig, d_meta: int):
        torch, nn = _require_torch()
        super().__init__()  # type: ignore[misc]
        self.cfg = cfg

        self.meta_proj = nn.Sequential(
            nn.Linear(d_meta, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(1, cfg.time_mlp_hidden),
            nn.ReLU(),
            nn.Linear(cfg.time_mlp_hidden, cfg.d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        # 6 heads p50 (point prediction on log1p(delta) scale)
        Q = len(cfg.quantiles)
        self.head_views = nn.Linear(cfg.d_model, 3 * Q)  # 7/14/21 × Q
        self.head_likes = nn.Linear(cfg.d_model, 3 * Q)

        # learnable log-weights per horizon (uncertainty weighting), initialized to 0 => weight=1
        # cap is applied to weight = exp(-log_var)
        self.log_var = nn.Parameter(torch.zeros(3))

    def forward(
        self,
        *,
        visual_tokens,  # (B, Kv, D) or None
        visual_times_s,  # (B, Kv) or None
        visual_mask,  # (B, Kv) 1/0 or None
        audio_tokens=None,  # (B, Ka, D) or None
        audio_times_s=None,  # (B, Ka) or None
        audio_mask=None,  # (B, Ka) or None
        text_tokens=None,  # (B, Kc, D) or None
        text_mask=None,  # (B, Kc) or None
        meta_vec=None,  # (B, d_meta)
        duration_sec=None,  # (B,)
    ) -> Dict[str, "torch.Tensor"]:
        torch, nn = _require_torch()
        if meta_vec is None or duration_sec is None:
            raise ValueError("meta_vec and duration_sec are required")

        B = meta_vec.shape[0]
        d = self.cfg.d_model

        # meta token: (B,1,D)
        meta_tok = self.meta_proj(meta_vec).unsqueeze(1)
        meta_time = torch.zeros(B, 1, dtype=torch.float32, device=meta_vec.device)
        meta_mask = torch.ones(B, 1, dtype=torch.float32, device=meta_vec.device)

        tokens = [meta_tok]
        times = [meta_time]
        masks = [meta_mask]

        def _append(tok, t, m):
            if tok is None:
                return
            tokens.append(tok)
            if t is None:
                times.append(torch.zeros(tok.shape[0], tok.shape[1], device=tok.device))
            else:
                times.append(t)
            if m is None:
                masks.append(torch.ones(tok.shape[0], tok.shape[1], device=tok.device))
            else:
                masks.append(m)

        _append(visual_tokens, visual_times_s, visual_mask)
        _append(audio_tokens, audio_times_s, audio_mask)
        _append(text_tokens, None, text_mask)  # text has no explicit time in v1 contract (can be extended)

        x = torch.cat(tokens, dim=1)  # (B, T, D)
        t = torch.cat(times, dim=1)  # (B, T)
        m = torch.cat(masks, dim=1)  # (B, T)

        # time encoding: normalize by duration
        dur = duration_sec.clamp(min=1e-3).unsqueeze(1)  # (B,1)
        t_norm = (t / dur).clamp(0.0, 1.0).unsqueeze(-1)  # (B,T,1)
        t_emb = self.time_mlp(t_norm)  # (B,T,D)
        x = x + t_emb

        # attention mask: transformer expects True for positions to mask (pad)
        key_padding_mask = m <= 0.0  # (B,T) bool
        h = self.fusion(x, src_key_padding_mask=key_padding_mask)

        # pooled representation: mean over non-masked
        denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (h * m.unsqueeze(-1)).sum(dim=1) / denom  # (B,D)

        views = self.head_views(pooled)  # (B,3)
        likes = self.head_likes(pooled)  # (B,3)

        Q = len(self.cfg.quantiles)
        views = views.view(views.shape[0], 3, Q)
        likes = likes.view(likes.shape[0], 3, Q)

        return {"views": views, "likes": likes}

    def loss(
        self,
        *,
        pred: Dict[str, "torch.Tensor"],
        target_views,  # (B,3)
        target_likes,  # (B,3)
        mask_7d,  # (B,) 1/0
    ) -> Dict[str, "torch.Tensor"]:
        torch, nn = _require_torch()
        cfg = self.cfg

        pv = pred["views"]  # (B,3,Q)
        pl = pred["likes"]  # (B,3,Q)
        tv = target_views
        tl = target_likes
        Q = pv.shape[-1]

        # Targets come as (B,3); broadcast to (B,3,Q)
        tvq = tv.unsqueeze(-1).expand(-1, -1, Q)
        tlq = tl.unsqueeze(-1).expand(-1, -1, Q)

        if Q == 1 and abs(float(self.cfg.quantiles[0]) - 0.5) < 1e-9:
            # point mode: per-horizon mse
            mse = nn.MSELoss(reduction="none")
            lv = mse(pv.squeeze(-1), tv)  # (B,3)
            ll = mse(pl.squeeze(-1), tl)  # (B,3)
            l = 0.5 * (lv + ll)  # (B,3)
        else:
            # quantile mode: pinball loss, averaged over quantiles
            qs = torch.tensor(self.cfg.quantiles, device=pv.device, dtype=pv.dtype).view(1, 1, Q)  # (1,1,Q)
            dv = tvq - pv
            dl = tlq - pl
            lv = torch.maximum(qs * dv, (qs - 1.0) * dv)  # (B,3,Q)
            ll = torch.maximum(qs * dl, (qs - 1.0) * dl)  # (B,3,Q)
            l = 0.5 * (lv + ll).mean(dim=-1)  # (B,3)

        # apply 7d mask (horizon 0)
        m7 = mask_7d.float().clamp(0.0, 1.0)
        l[:, 0] = l[:, 0] * m7

        # base horizon weights
        base = torch.tensor([cfg.w_7d, cfg.w_14d, cfg.w_21d], device=l.device, dtype=l.dtype).unsqueeze(0)  # (1,3)
        l = l * base

        # learned uncertainty weights: weight = exp(-log_var) capped
        w = torch.exp(-self.log_var).clamp(cfg.w_cap_min, cfg.w_cap_max)  # (3,)
        l_weighted = l * w.unsqueeze(0) + self.log_var.unsqueeze(0)  # (B,3)

        # mean over batch, and normalize 7d effective count to avoid bias if many missing
        # (simple approach: divide 7d term by max(sum(mask),1) and others by B)
        B = l.shape[0]
        denom7 = m7.sum().clamp(min=1.0)
        loss7 = l_weighted[:, 0].sum() / denom7
        loss14 = l_weighted[:, 1].mean()
        loss21 = l_weighted[:, 2].mean()
        total = loss7 + loss14 + loss21

        return {"loss": total, "loss_7d": loss7, "loss_14d": loss14, "loss_21d": loss21, "w": w.detach()}


