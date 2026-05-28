from __future__ import annotations

import contextlib
import os
import socket
from typing import Dict, Iterator, Optional


def _default_hf_home() -> str:
    """Default HuggingFace cache: ~/.cache/huggingface."""
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface")


def _hf_cache_has_hub(cache_dir: str) -> bool:
    """Check if HF cache dir has hub/ with model entries (e.g. models--*)."""
    hub_dir = os.path.join(cache_dir, "hub")
    if not os.path.isdir(hub_dir):
        return False
    try:
        entries = os.listdir(hub_dir)
        return any(e.startswith("models--") for e in entries)
    except OSError:
        return False


def enforce_offline_env(models_root: str) -> Dict[str, str]:
    """
    Best-effort offline enforcement via environment variables.

    This does NOT monkeypatch sockets by default (too intrusive). It is enough to ensure
    HF/Transformers/SentenceTransformers do not attempt downloads.
    
    Note: For demucs models, TORCH_HOME is used for caching, but if the model is not
    in cache and offline mode is enforced, the model loading will fail (expected behavior).
    Models should be pre-downloaded via download scripts.
    
    HF cache: if models_root/hf_cache has hub/ with models, use it; else fall back to
    ~/.cache/huggingface (e.g. emotion_diarization WavLM when bundled_models/hf_cache is empty).
    """
    mr = os.path.abspath(str(models_root))
    hf_cache_bundled = os.path.join(mr, "hf_cache")
    if _hf_cache_has_hub(hf_cache_bundled):
        hf_home = hf_cache_bundled
    else:
        hf_home = _default_hf_home()
    hf_hub = os.path.join(hf_home, "hub")
    env = {
        # HuggingFace offline
        # "HF_HUB_OFFLINE": "1",
        # "TRANSFORMERS_OFFLINE": "1",
        # Common cache roots: use bundled if populated, else default HF cache
        "SENTENCE_TRANSFORMERS_HOME": hf_home,
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": hf_hub,
        "TRANSFORMERS_CACHE": hf_hub,
        # Torch cache (for torchvision weights, torch.hub models, and demucs)
        # demucs.pretrained.get_model() uses torch.hub.load internally, which respects TORCH_HOME
        "TORCH_HOME": os.path.join(mr, "torch_cache"),
        # OpenAI CLIP cache root (our code reads this env var where applicable)
        "DP_CLIP_WEIGHTS_DIR": os.path.join(mr, "clip_cache"),
        # Avoid noisy telemetry
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    # Use os.environ[k] = v instead of setdefault to ensure values are set (not just defaults)
    for k, v in env.items():
        os.environ[k] = v
    return env


def pin_cache_env(models_root: str, *, offline: bool = True) -> Dict[str, str]:
    """
    Pin common cache roots under models_root.

    Use-cases:
    - runtime (offline=True): enforce no-network + pinned caches (ModelManager default)
    - bootstrap / pretriton bench (offline=False): allow network but still write into models_root caches
    """
    mr = os.path.abspath(str(models_root))
    env = {
        # HuggingFace cache roots pinned under models_root
        "SENTENCE_TRANSFORMERS_HOME": os.path.join(mr, "hf_cache"),
        "HF_HOME": os.path.join(mr, "hf_cache"),
        # Torch cache (torch.hub + torchvision weights)
        "TORCH_HOME": os.path.join(mr, "torch_cache"),
        # OpenAI CLIP cache root (used by bootstrap/export helpers and some components)
        "DP_CLIP_WEIGHTS_DIR": os.path.join(mr, "clip_cache"),
        # Avoid noisy telemetry
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    if offline:
        env.update(
            {
                # "HF_HUB_OFFLINE": "1",
                # "TRANSFORMERS_OFFLINE": "1",
            }
        )
    for k, v in env.items():
        os.environ[k] = v
    return env


@contextlib.contextmanager
def network_guard(*, enabled: bool = True) -> Iterator[None]:
    """
    Strict no-network guard for tests.
    Monkeypatches `socket.socket.connect` to always raise.
    """
    if not enabled:
        yield
        return

    orig_connect = socket.socket.connect

    def _blocked_connect(self, address):  # type: ignore[no-untyped-def]
        raise RuntimeError(f"network_forbidden: attempted socket connect to {address!r}")

    socket.socket.connect = _blocked_connect  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = orig_connect  # type: ignore[assignment]


