from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


def _parse_iso8601(s: Any) -> Optional[datetime]:
    if not isinstance(s, str) or not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def split_hybrid_time_channel(
    df: "pd.DataFrame",
    *,
    channel_col: str,
    published_col: str,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> "pd.Series":
    """
    Same policy as baseline:
      - per-channel min publishedAt
      - sort channels by that min time
      - assign channels into train/val/test by fractions
    """
    import pandas as pd  # type: ignore

    if channel_col not in df.columns:
        raise ValueError(f"Missing channel column: {channel_col}")
    if published_col not in df.columns:
        raise ValueError(f"Missing publishedAt column: {published_col}")

    pub_dt = df[published_col].apply(_parse_iso8601)
    pub_dt = pub_dt.fillna(datetime(1970, 1, 1, tzinfo=timezone.utc))

    tmp = pd.DataFrame({channel_col: df[channel_col].astype(str), "_pub": pub_dt})
    g = tmp.groupby(channel_col)["_pub"].min().sort_values()
    channels = list(g.index)

    n = len(channels)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_set = set(channels[:n_train])
    val_set = set(channels[n_train : n_train + n_val])

    def _assign(ch: str) -> str:
        if ch in train_set:
            return "train"
        if ch in val_set:
            return "val"
        return "test"

    return df[channel_col].astype(str).apply(_assign)


