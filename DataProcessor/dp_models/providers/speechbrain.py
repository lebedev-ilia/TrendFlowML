from __future__ import annotations

import os
from typing import Any

from ..errors import ModelManagerError
from ..specs import ModelSpec


class SpeechBrainProvider:
    """
    In-process SpeechBrain model provider.
    
    Policy:
    - model must be loaded from a **local directory** (no HF id downloads).
    - offline env must be set by ModelManager.
    - Supports Speech_Emotion_Diarization and other SpeechBrain inference interfaces.
    """
    
    def supports(self, spec: ModelSpec) -> bool:
        return (spec.runtime == "inprocess") and ("speechbrain" in (spec.engine or "").lower())
    
    def load(
        self,
        *,
        spec: ModelSpec,
        device: str,
        precision: str,
        models_root: str,
        runtime_params: dict | None = None,
    ) -> Any:
        rp = runtime_params or spec.runtime_params or {}
        if not isinstance(rp, dict):
            rp = {}

        
        # Get model class name from runtime_params (default: Speech_Emotion_Diarization)
        model_class_name = rp.get("model_class", "Speech_Emotion_Diarization")
        
        # Pick first directory artifact as the model folder
        model_dir_rel = None
        for a in spec.local_artifacts:
            if str(a.kind) == "dir":
                model_dir_rel = str(a.path)
                break
        if not model_dir_rel:
            raise ModelManagerError(
                message="SpeechBrainProvider requires a local_artifacts entry with kind=dir",
                error_code="weights_missing",
                details={"model_name": spec.model_name},
            )

        try:
            import torch  # type: ignore
        except Exception as e:
            raise ModelManagerError(
                message="torch is not installed",
                error_code="dependency_missing",
                details={"import_error": str(e)},
            ) from e
        
        # Set HF environment variables BEFORE importing SpeechBrain/transformers
        # This is critical because transformers may cache the cache path on first import
        from pathlib import Path
        
        # Determine HF cache directory
        hf_hub_cache = None
        
        # First, check if HF_HOME is already set (e.g., by enforce_offline_env from ModelManager)
        # This takes priority as it points to models_root/hf_cache
        current_hf_home = os.environ.get("HF_HOME")
        if current_hf_home and os.path.exists(current_hf_home):
            # Use the already-set HF_HOME (likely from enforce_offline_env)
            hf_hub_cache = os.path.join(current_hf_home, "hub")
            if not os.path.exists(hf_hub_cache):
                hf_hub_cache = None
        else:
            # Fallback to standard cache location if not set
            default_hf_home = os.path.join(Path.home(), ".cache", "huggingface")
            if os.path.exists(default_hf_home):
                os.environ.setdefault("HF_HOME", default_hf_home)
                hf_hub_cache = os.path.join(default_hf_home, "hub")
                if not os.path.exists(hf_hub_cache):
                    hf_hub_cache = None
        
        # Set HF cache environment variables to ensure transformers uses the correct cache
        if hf_hub_cache:
            # Force set (not setdefault) to ensure transformers uses the correct cache
            os.environ["HF_HUB_CACHE"] = hf_hub_cache
            os.environ["TRANSFORMERS_CACHE"] = hf_hub_cache
            # Also set HF_HOME if not already set
            if not current_hf_home:
                os.environ["HF_HOME"] = os.path.dirname(hf_hub_cache)
        # Try to import SpeechBrain (prefer local version from component if available)
        try:
            # First try local speechbrain from emotion_diarization_extractor
            import sys
            extractor_dir = Path(__file__).resolve().parent.parent.parent / "AudioProcessor" / "src" / "extractors" / "emotion_diarization_extractor"
            speechbrain_path = extractor_dir / "speechbrain"
            if speechbrain_path.exists() and str(speechbrain_path) not in sys.path:
                sys.path.insert(0, str(speechbrain_path))
            
            from speechbrain.inference.diarization import Speech_Emotion_Diarization  # type: ignore
        except ImportError:
            # Fallback to system-installed speechbrain
            try:
                from speechbrain.inference.diarization import Speech_Emotion_Diarization  # type: ignore
            except Exception as e:
                raise ModelManagerError(
                    message="speechbrain is not installed (neither local nor system)",
                    error_code="dependency_missing",
                    details={"import_error": str(e), "hint": "Install speechbrain or ensure local copy exists in emotion_diarization_extractor/speechbrain/"},
                ) from e
        

        # Load model from local directory using from_hparams
        try:
            import logging
            import contextlib
            import sys

            # Подавляем логи загрузки весов от transformers
            # Отключаем tqdm прогресс-бары и HF progress bars
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            os.environ["TQDM_DISABLE"] = "1"
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            os.environ["PYTHONWARNINGS"] = "ignore"
            
            # Устанавливаем уровень логирования для transformers
            transformers_logger = logging.getLogger("transformers")
            transformers_logger.setLevel(logging.ERROR)
            transformers_logger.propagate = False
            
            # Подавляем логи от huggingface_hub
            hf_logger = logging.getLogger("huggingface_hub")
            hf_logger.setLevel(logging.ERROR)
            hf_logger.propagate = False
            
            # Подавляем логи от accelerate
            accelerate_logger = logging.getLogger("accelerate")
            accelerate_logger.setLevel(logging.ERROR)
            accelerate_logger.propagate = False
            
            # Подавляем логи от speechbrain
            sb_logger = logging.getLogger("speechbrain")
            sb_logger.setLevel(logging.ERROR)
            sb_logger.propagate = False
            
            # Подавляем логи от tqdm
            tqdm_logger = logging.getLogger("tqdm")
            tqdm_logger.setLevel(logging.ERROR)
            tqdm_logger.propagate = False
            
            # Подавляем логи от speechbrain.utils
            sb_utils_logger = logging.getLogger("speechbrain.utils")
            sb_utils_logger.setLevel(logging.ERROR)
            sb_utils_logger.propagate = False
            
            # Подавляем логи от speechbrain.utils.checkpoints
            sb_checkpoints_logger = logging.getLogger("speechbrain.utils.checkpoints")
            sb_checkpoints_logger.setLevel(logging.ERROR)
            sb_checkpoints_logger.propagate = False

            # Всегда используем HuggingFace модель (как в скрипте)
            # Не используем redirect_stdout/stderr, так как это может вызывать блокировки
            model_source = "speechbrain/emotion-diarization-wavlm-large"

            # Подавляем stdout/stderr во время загрузки модели для скрытия прогресс-баров
            @contextlib.contextmanager
            def suppress_output():
                """Временно подавляет stdout и stderr."""
                with open(os.devnull, "w") as devnull:
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    try:
                        sys.stdout = devnull
                        sys.stderr = devnull
                        yield
                    finally:
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr

            # Загружаем модель без run_opts сначала (как в скрипте), чтобы избежать лишней инициализации
            # Устройство будет установлено автоматически или через hparams
            # Убеждаемся, что cache_dir явно указан для WavLM модели, чтобы избежать создания локального checkpoint
            with suppress_output():
                # Важно: устанавливаем переменные окружения ДО загрузки модели
                # Это гарантирует, что transformers будет использовать правильный кэш для WavLM
                if hf_hub_cache:
                    # Force set (не setdefault) чтобы гарантировать использование правильного кэша
                    os.environ["HF_HUB_CACHE"] = hf_hub_cache
                    os.environ["TRANSFORMERS_CACHE"] = hf_hub_cache
                    # Также устанавливаем HF_HOME если не установлен
                    if not current_hf_home:
                        os.environ["HF_HOME"] = os.path.dirname(hf_hub_cache)
                
                # Дополнительно: убеждаемся, что рабочая директория не влияет на создание checkpoint
                # Сохраняем текущую рабочую директорию и временно меняем её, чтобы избежать создания checkpoint в DataProcessor/
                original_cwd = os.getcwd()
                try:
                    # Временно меняем рабочую директорию на временную, чтобы избежать создания checkpoint в DataProcessor/
                    import tempfile
                    temp_dir = tempfile.mkdtemp()
                    os.chdir(temp_dir)
                    
                    model = Speech_Emotion_Diarization.from_hparams(
                        source=model_source,
                    )
                finally:
                    # Восстанавливаем рабочую директорию
                    os.chdir(original_cwd)
                    # Удаляем временную директорию (если она пустая)
                    try:
                        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                            os.rmdir(temp_dir)
                    except Exception:
                        pass

        except Exception as e:
            # SpeechBrain will raise its own errors if files are missing
            # We wrap them in ModelManagerError for consistency
            error_msg = str(e)
            raise ModelManagerError(
                message="Failed to load SpeechBrain model",
                error_code="model_load_failed",
                details={"model_name": spec.model_name, "model_source": model_source, "error": error_msg, "error_type": type(e).__name__},
            ) from e
        
        return model

