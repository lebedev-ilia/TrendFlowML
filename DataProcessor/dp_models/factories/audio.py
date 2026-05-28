from __future__ import annotations

# === torchaudio compatibility patch (MUST be at module level, before any speechbrain imports) ===
try:
    import torchaudio  # type: ignore
    if not hasattr(torchaudio, 'list_audio_backends'):
        def _list_audio_backends():
            return ['soundfile', 'sox']
        torchaudio.list_audio_backends = _list_audio_backends
except Exception:
    pass

from typing import Any


def create_whisper_model(*, model_size: str) -> Any:
    """
    Create a Whisper model architecture (no weights) for TorchStateDictProvider.
    
    Uses openai-whisper library to create the model architecture.
    The actual weights will be loaded by TorchStateDictProvider from checkpoint.
    
    Args:
        model_size: Whisper model size: "small", "medium", or "large"
    
    Returns:
        Whisper model instance (without weights loaded)
    """
    try:
        import whisper  # type: ignore
    except Exception as e:
        raise RuntimeError(f"openai-whisper is not installed: {e}") from e
    
    size = str(model_size).strip().lower()
    if size not in ("tiny", "base", "small", "medium", "large"):
        raise ValueError(f"Unsupported Whisper model size: {size}. Expected: tiny|base|small|medium|large")
    
    # Create model architecture using whisper's internal API
    # We create the model structure without loading weights
    from whisper import model as whisper_model  # type: ignore
    
    # Model dimensions for each size (from whisper source)
    dims_map = {
        "tiny": whisper_model.ModelDimensions(
            n_mels=80, n_audio_ctx=1500, n_audio_state=384, n_audio_head=6, n_audio_layer=4,
            n_vocab=51865, n_text_ctx=448, n_text_state=384, n_text_head=6, n_text_layer=4
        ),
        "base": whisper_model.ModelDimensions(
            n_mels=80, n_audio_ctx=1500, n_audio_state=512, n_audio_head=8, n_audio_layer=6,
            n_vocab=51865, n_text_ctx=448, n_text_state=512, n_text_head=8, n_text_layer=6
        ),
        "small": whisper_model.ModelDimensions(
            n_mels=80, n_audio_ctx=1500, n_audio_state=768, n_audio_head=12, n_audio_layer=12,
            n_vocab=51865, n_text_ctx=448, n_text_state=768, n_text_head=12, n_text_layer=12
        ),
        "medium": whisper_model.ModelDimensions(
            n_mels=80, n_audio_ctx=1500, n_audio_state=1024, n_audio_head=16, n_audio_layer=24,
            n_vocab=51865, n_text_ctx=448, n_text_state=1024, n_text_head=16, n_text_layer=24
        ),
        "large": whisper_model.ModelDimensions(
            n_mels=80, n_audio_ctx=1500, n_audio_state=1280, n_audio_head=20, n_audio_layer=32,
            n_vocab=51865, n_text_ctx=448, n_text_state=1280, n_text_head=20, n_text_layer=32
        ),
    }
    
    dims = dims_map.get(size)
    if dims is None:
        raise ValueError(f"Model dimensions not defined for size: {size}")
    
    # Create model with the specified dimensions (no weights loaded yet)
    model = whisper_model.Whisper(dims=dims)
    return model


def create_speaker_diarization_model(*, model_size: str, checkpoint_path: str = None) -> Any:
    """
    Create a speaker diarization embedding model architecture (no weights) for TorchStateDictProvider.
    
    Uses speechbrain library to create ECAPA-TDNN model architecture for speaker embeddings.
    The actual weights will be loaded by TorchStateDictProvider from checkpoint.
    
    If checkpoint_path is provided and checkpoint contains a full model, it will be loaded directly.
    Otherwise, creates model architecture via speechbrain and loads weights from checkpoint.
    
    Args:
        model_size: Model size: "small" or "large"
        checkpoint_path: Optional path to checkpoint file (for loading full model if available)
    
    Returns:
        Speaker embedding model instance (without weights loaded, or with weights if full model in checkpoint)
    """
    # Try to load full model from checkpoint if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            import torch
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(ckpt, dict) and "model" in ckpt:
                # Full model is available in checkpoint
                model = ckpt["model"]
                # Move to CPU and set eval mode
                try:
                    model.eval()
                    model = model.cpu()
                except Exception:
                    pass
                return model
        except Exception as e:
            # If loading full model fails, fall back to creating architecture
            pass
    
    import os
    # === torchaudio compatibility patch (MUST be before speechbrain import) ===
    # Note: Patch is already applied at module level, but we ensure it's applied here too
    try:
        import torchaudio  # type: ignore
        if not hasattr(torchaudio, 'list_audio_backends'):
            # Патч для совместимости с новыми версиями torchaudio
            def _list_audio_backends():
                return ['soundfile', 'sox']
            torchaudio.list_audio_backends = _list_audio_backends
    except Exception:
        pass
    
    # Note: We allow internet access for first-time model structure loading
    # After first load, the model structure will be cached and can be used offline
    import os
    # Don't force offline mode - allow internet for first-time setup
    # HuggingFace will use cache if available, or download if needed
    
    try:
        from speechbrain.inference.speaker import EncoderClassifier  # type: ignore
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError(f"speechbrain and torch are required: {e}") from e
    
    size = str(model_size).strip().lower()
    if size not in ("small", "large"):
        raise ValueError(f"Unsupported speaker diarization model size: {size}. Expected: small|large")
    
    # Create model architecture using speechbrain's ECAPA-TDNN
    # We'll load the model structure from a local checkpoint or create it programmatically
    # The checkpoint will be loaded by TorchStateDictProvider
    
    # For now, we create the model structure by loading from a temporary location
    # The actual weights will be replaced by the checkpoint
    # Note: This requires the model structure to match the checkpoint format
    
    # We use EncoderClassifier which wraps the ECAPA-TDNN encoder
    # The model will be created with the architecture, but weights will come from checkpoint
    try:
        # Try to load hparams from local checkpoint directory first (offline mode)
        # This allows offline operation if model was previously downloaded
        import tempfile
        import os
        from pathlib import Path
        
        # Try to find local hparams.yaml in the same directory as checkpoint
        # The checkpoint path will be resolved by TorchStateDictProvider, but we can't access it here
        # So we'll try to use HF cache or fallback to temp directory
        
        # Try to use HF cache directory if available
        hf_snapshot_dir = None
        try:
            hf_home = os.environ.get("HF_HOME") or os.path.join(Path.home(), ".cache", "huggingface")
            hf_cache_dir = os.path.join(hf_home, "hub", "models--speechbrain--spkrec-ecapa-voxceleb")
            if os.path.exists(hf_cache_dir):
                # Find the snapshot directory
                snapshots_dir = os.path.join(hf_cache_dir, "snapshots")
                if os.path.exists(snapshots_dir):
                    # Get the first (or latest) snapshot
                    snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if snapshots:
                        hf_snapshot_dir = os.path.join(snapshots_dir, snapshots[0])
        except Exception:
            hf_snapshot_dir = None
        
        # Use snapshot directory if found, otherwise use temp directory
        # Copy files from HF cache to temp dir to avoid symlink issues
        if hf_snapshot_dir and os.path.exists(hf_snapshot_dir):
            # Create a temp directory and copy files from snapshot (to avoid symlink issues)
            savedir = tempfile.mkdtemp(prefix="ecapa_model_")
            cleanup_temp = True
            try:
                import shutil
                # Copy all files from snapshot to temp directory
                for item in os.listdir(hf_snapshot_dir):
                    src = os.path.join(hf_snapshot_dir, item)
                    dst = os.path.join(savedir, item)
                    if os.path.islink(src):
                        # Resolve symlink and copy actual file
                        real_path = os.readlink(src)
                        if os.path.isabs(real_path):
                            target = real_path
                        else:
                            target = os.path.join(os.path.dirname(src), real_path)
                        if os.path.exists(target):
                            shutil.copy2(target, dst)
                    elif os.path.isfile(src):
                        shutil.copy2(src, dst)
                    elif os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
            except Exception as copy_error:
                # If copy fails, try using snapshot directly
                savedir = hf_snapshot_dir
                cleanup_temp = False
        else:
            savedir = tempfile.mkdtemp(prefix="temp_ecapa_")
            cleanup_temp = True
        
        try:
            # Create model using HuggingFace source
            # This will use cache if available, or download if needed
            # Note: First-time setup requires internet connection
            # Verify torchaudio patch is applied
            try:
                import torchaudio as ta_check
                if not hasattr(ta_check, 'list_audio_backends'):
                    def _list_audio_backends():
                        return ['soundfile', 'sox']
                    ta_check.list_audio_backends = _list_audio_backends
            except Exception:
                pass
            
            model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=savedir,
                run_opts={"device": "cpu"},
            )
            
            # The model structure is now created
            # TorchStateDictProvider will load the actual weights from checkpoint
            return model
        except Exception as e:
            # If HF cache doesn't work, try to create a minimal model structure
            # This is a fallback - ideally the model should be in HF cache
            error_msg = str(e)
            if "Connection error" in error_msg or "Internet connection" in error_msg:
                raise RuntimeError(
                    f"Failed to create speaker diarization model architecture: {e}. "
                    f"Model structure not found in HuggingFace cache. "
                    f"Please run download_speaker_diarization_models.py first to populate the cache."
                ) from e
            raise
        finally:
            # Clean up temp directory if we created one (not HF cache)
            if cleanup_temp and os.path.exists(savedir):
                try:
                    import shutil
                    shutil.rmtree(savedir)
                except Exception:
                    pass
    except Exception as e:
        raise RuntimeError(f"Failed to create speaker diarization model architecture: {e}") from e


def create_emotion_diarization_model(*, model_size: str, checkpoint_path: str = None) -> Any:
    """
    Create an emotion diarization model architecture (no weights) for TorchStateDictProvider.
    
    This factory function creates the model architecture. The actual weights will be loaded
    by TorchStateDictProvider from checkpoint.
    
    If checkpoint_path is provided and checkpoint contains a full model, it will be loaded directly.
    Otherwise, creates a placeholder model architecture that will be populated by state_dict.
    
    Args:
        model_size: Model size: "small" or "large"
        checkpoint_path: Optional path to checkpoint file (for loading full model if available)
    
    Returns:
        Emotion diarization model instance (without weights loaded, or with weights if full model in checkpoint)
    """
    import os
    
    # Try to load full model from checkpoint if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            import torch
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model" in ckpt:
                # Full model is available in checkpoint
                model = ckpt["model"]
                # Move to CPU and set eval mode
                try:
                    model.eval()
                    model = model.cpu()
                except Exception:
                    pass
                return model
        except Exception as e:
            # If loading full model fails, fall back to creating architecture
            pass
    
    # If no full model in checkpoint, we need to create the architecture
    # For now, we'll create a placeholder that will be populated by state_dict
    # The actual architecture depends on the specific emotion recognition model used
    # This is a generic implementation that works with most PyTorch models
    
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        raise RuntimeError(f"torch is required: {e}") from e
    
    size = str(model_size).strip().lower()
    if size not in ("small", "large"):
        raise ValueError(f"Unsupported emotion diarization model size: {size}. Expected: small|large")
    
    # Create a placeholder model architecture
    # The actual architecture will be determined by the state_dict keys
    # This is a minimal implementation - the model will be fully defined by the checkpoint
    class EmotionDiarizationModel(nn.Module):
        """
        Placeholder emotion diarization model.
        The actual architecture will be determined by the state_dict loaded from checkpoint.
        """
        def __init__(self, model_size: str):
            super().__init__()
            self.model_size = model_size
            # This is a placeholder - actual layers will be loaded from state_dict
            self.placeholder = nn.Parameter(torch.zeros(1))
        
        def forward(self, x):
            # Placeholder forward - will be replaced by actual model
            raise NotImplementedError(
                "EmotionDiarizationModel forward is not implemented. "
                "Model architecture must be fully defined in checkpoint or state_dict."
            )
    
    # Create model instance
    model = EmotionDiarizationModel(model_size=size)
    return model


def create_source_separation_model(*, model_size: str, checkpoint_path: str = None) -> Any:
    """
    Create a source separation model architecture (no weights) for TorchStateDictProvider.
    
    This factory function creates the DemucsEnergyModel architecture. The actual weights will be loaded
    by TorchStateDictProvider from checkpoint state_dict.
    
    The model takes log-mel spectrograms [B, n_mels, T] as input and outputs source energies [B, 4]
    for 4 sources: vocals, drums, bass, other.
    
    Args:
        model_size: Model size: "large"
        checkpoint_path: Optional path to checkpoint file (not used, kept for compatibility)
    
    Returns:
        DemucsEnergyModel instance (without weights loaded, weights will be loaded by TorchStateDictProvider)
    """
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        raise RuntimeError(f"torch is required: {e}") from e
    
    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
    except ImportError as e:
        raise RuntimeError(f"demucs library is required: {e}") from e
    
    size = str(model_size).strip().lower()
    if size not in ("large",):
        raise ValueError(f"Unsupported source separation model size: {size}. Expected: large")
    
    # Demucs model name for large size
    demucs_name = "htdemucs"
    
    # Create DemucsEnergyModel architecture (same as in download script)
    # NOTE: get_model() will load the base Demucs model from cache (TORCH_HOME) if available
    # ModelManager sets TORCH_HOME to models_root/torch_cache via enforce_offline_env()
    # The base Demucs model should be pre-downloaded via download_source_separation_models.py
    # This download happens ONCE during model preparation, not at runtime
    class DemucsEnergyModel(nn.Module):
        """
        In-process Source Separation Energy Extractor
        Input:  log-mel spectrogram [B, n_mels, T]
        Output: energy shares [B, 4] (vocals, drums, bass, other)
        
        Note: Source order matches spec files: ["vocals", "drums", "bass", "other"]
        """
        
        SOURCES = ["vocals", "drums", "bass", "other"]
        
        def __init__(self, demucs_name="htdemucs", samplerate=44100):
            super().__init__()
            
            self.samplerate = samplerate
            # get_model() uses torch.hub.load internally, which respects TORCH_HOME
            # ModelManager sets TORCH_HOME to models_root/torch_cache via enforce_offline_env()
            # However, if the model is not in cache, get_model() will download it from internet
            # This is expected behavior on FIRST run - the model will be cached for subsequent runs
            # The base Demucs model should be pre-downloaded via download_source_separation_models.py
            # to avoid runtime downloads, but if it's not pre-downloaded, it will download once and cache
            try:
                # Check if offline mode is enforced
                import os
                hf_offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
                torch_home = os.environ.get("TORCH_HOME", "")
                
                if hf_offline and not torch_home:
                    # Offline mode but TORCH_HOME not set - this shouldn't happen if ModelManager initialized correctly
                    import warnings
                    warnings.warn(
                        f"source_separation | HF_HUB_OFFLINE=1 but TORCH_HOME not set. "
                        f"ModelManager should set TORCH_HOME. Model may fail to load from cache."
                    )
                
                # Load model (will use cache if available, or download if not in cache and offline=False)
                self.demucs = get_model(demucs_name)
            except Exception as e:
                # If model loading fails (e.g., not in cache and offline mode), provide helpful error
                error_msg = str(e).lower()
                if "offline" in error_msg or "not found" in error_msg or "cache" in error_msg:
                    raise RuntimeError(
                        f"Failed to load Demucs model '{demucs_name}' from cache. "
                        f"The base Demucs model must be pre-downloaded (ONE-TIME setup): "
                        f"python scripts/download_source_separation_models.py --sizes large --models-root <DP_MODELS_ROOT>. "
                        f"This will download the base model to TORCH_HOME cache. "
                        f"Original error: {e}"
                    ) from e
                raise RuntimeError(f"Failed to load Demucs model '{demucs_name}': {e}") from e
            self.demucs.eval()
            self.demucs.requires_grad_(False)
        
        def forward(self, logmel: torch.Tensor) -> torch.Tensor:
            """
            Args:
                logmel: [B, n_mels, T]
            Returns:
                shares: [B, 4] (vocals, drums, bass, other)
            """
            B = logmel.shape[0]
            device = logmel.device
            
            # Approximate inversion for feature-level inference
            wav = self._logmel_to_waveform(logmel)
            
            if wav.shape[1] == 1:
                wav = wav.repeat(1, 2, 1)
            
            with torch.no_grad():
                sources = apply_model(
                    self.demucs,
                    wav,
                    device=device,
                    split=True,
                    overlap=0.25,
                    progress=False,
                )
            
            # sources: [B, 4, C, T] (Demucs order: drums, bass, other, vocals)
            # Reorder to match spec: vocals, drums, bass, other
            # Demucs order: [0=drums, 1=bass, 2=other, 3=vocals]
            # Target order: [0=vocals, 1=drums, 2=bass, 3=other]
            reordered_sources = torch.stack([
                sources[:, 3],  # vocals
                sources[:, 0],  # drums
                sources[:, 1],  # bass
                sources[:, 2],  # other
            ], dim=1)  # [B, 4, C, T]
            
            energies = reordered_sources.pow(2).mean(dim=(2, 3))  # [B, 4]
            shares = energies / (energies.sum(dim=1, keepdim=True) + 1e-8)
            
            return shares
        
        def _logmel_to_waveform(self, logmel: torch.Tensor) -> torch.Tensor:
            """
            Approximate inversion for feature-level inference.
            """
            mel = logmel.exp()
            energy = mel.mean(dim=1)  # [B, T]
            wav = energy.unsqueeze(1)  # [B, 1, T]
            return wav
    
    # Create model instance
    model = DemucsEnergyModel(demucs_name=demucs_name, samplerate=44100)
    return model


def create_pyannote_speaker_diarization_pipeline(*, checkpoint_path: str = None) -> Any:
    """
    Create a pyannote.audio speaker diarization Pipeline.
    
    Args:
        checkpoint_path: Optional path to saved pipeline directory
    
    Returns:
        pyannote.audio Pipeline instance
    """
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except Exception as e:
        raise RuntimeError(f"pyannote.audio is not installed: {e}") from e
    
    # If checkpoint_path is provided, load from local directory
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            try:
                pipeline = Pipeline.from_pretrained(checkpoint_path, local_files_only=True)
            except TypeError:
                pipeline = Pipeline.from_pretrained(checkpoint_path)
            return pipeline
        except Exception as e:
            raise RuntimeError(f"Failed to load pyannote pipeline from {checkpoint_path}: {e}") from e
    
    # Otherwise, create a default pipeline (will need to be loaded from HF or local)
    # This is a fallback - normally checkpoint_path should be provided
    raise RuntimeError(
        "pyannote speaker diarization pipeline requires checkpoint_path. "
        "Please provide a local path to saved pipeline or use ModelManager to resolve it."
    )

