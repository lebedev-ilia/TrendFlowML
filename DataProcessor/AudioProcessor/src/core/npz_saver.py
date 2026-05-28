"""
Модуль для сохранения NPZ артефактов для различных extractors.
Это самый большой модуль, содержащий логику сохранения для каждого типа extractor'а.
"""
import os
from typing import Any, Dict, Optional

import numpy as np

from ..utils.cli_utils import utc_iso_now, as_float, as_int, atomic_save_npz

# Contract versions for extractors
KEY_CONTRACT_VERSION = "key_contract_v1"
BAND_ENERGY_CONTRACT_VERSION = "band_energy_contract_v1"
SPECTRAL_ENTROPY_CONTRACT_VERSION = "spectral_entropy_contract_v1"


def build_meta(
    *,
    producer: str,
    producer_version: str,
    status: str,
    schema_version: str,
    extra: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Строит метаданные для NPZ файла.
    
    Args:
        producer: Имя продюсера
        producer_version: Версия продюсера
        status: Статус
        schema_version: Версия схемы
        extra: Дополнительные метаданные
    
    Returns:
        numpy array с метаданными
    """
    d = {
        "producer": producer,
        "producer_version": producer_version,
        "schema_version": schema_version,
        "status": status,
        "created_at": utc_iso_now(),
    }
    if extra:
        d.update(extra)
    # PR-3: model system baseline
    try:
        # Import lazily: sys.path is prepared inside main()
        from ...utils.meta_builder import apply_models_meta  # type: ignore

        d = apply_models_meta(d, models_used=d.get("models_used"))
    except Exception:
        # Best-effort: do not crash saving path on missing helper.
        d.setdefault("models_used", [])
        d.setdefault("model_signature", "")
    return np.asarray(d, dtype=object)


def _safe_arr(payload: Dict[str, Any], key: str, *, dtype: Any) -> np.ndarray:
    """
    Безопасное преобразование payload -> np.ndarray.
    Избегает `payload.get(key) or []` который ломается для numpy массивов (ambiguous truth value).
    """
    v = payload.get(key)
    if v is None:
        v = []
    return np.asarray(v, dtype=dtype).reshape(-1)


def save_component_npz(
    *,
    run_rs_path: str,
    component_name: str,
    payload: Optional[Dict[str, Any]],
    status: str,
    error: Optional[str],
    empty_reason: Optional[str],
    producer_version: str,
    schema_version: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Сохраняет NPZ артефакт для компонента.
    
    Это основная функция, которая делегирует сохранение специфичным функциям
    для каждого типа extractor'а. Большая часть логики находится в отдельных модулях.
    """
    comp_dir = os.path.join(run_rs_path, component_name)
    # Baseline: fixed artifact filename (run_id already provides uniqueness in path).
    out_path = os.path.join(comp_dir, f"{component_name}_features.npz")

    payload = payload or {}

    # Store a flexible "feature vector" representation:
    # - feature_names: object array of strings
    # - feature_values: float32 array aligned with names
    feature_names: list[str] = []
    feature_values: list[float] = []

    def add(name: str, value: Any):
        feature_names.append(name)
        feature_values.append(as_float(value))
    
    # Создаем функцию _arr для передачи в саверы
    def _arr(key: str, *, dtype: Any) -> np.ndarray:
        return _safe_arr(payload, key, dtype=dtype)

    # Импортируем специфичные саверы для каждого extractor'а
    # Это позволяет разнести логику по отдельным файлам
    from .npz_savers import (
        save_tempo_npz,
        save_loudness_npz,
        save_clap_npz,
        save_asr_npz,
        save_speaker_diarization_npz,
        save_emotion_diarization_npz,
        save_source_separation_npz,
        save_speech_analysis_npz,
        save_onset_npz,
        save_pitch_npz,
        save_rhythmic_npz,
        save_chroma_npz,
        save_key_npz,
        save_band_energy_npz,
        save_spectral_entropy_npz,
        save_spectral_npz,
        save_quality_npz,
        save_mfcc_npz,
        save_mel_npz,
        save_hpss_npz,
        save_voice_quality_npz,
    )
    
    # Делегируем сохранение специфичным функциям
    saver_map = {
        "tempo_extractor": save_tempo_npz,
        "loudness_extractor": save_loudness_npz,
        "clap_extractor": save_clap_npz,
        "asr_extractor": save_asr_npz,
        "speaker_diarization_extractor": save_speaker_diarization_npz,
        "emotion_diarization_extractor": save_emotion_diarization_npz,
        "source_separation_extractor": save_source_separation_npz,
        "speech_analysis_extractor": save_speech_analysis_npz,
        "onset_extractor": save_onset_npz,
        "pitch_extractor": save_pitch_npz,
        "rhythmic_extractor": save_rhythmic_npz,
        "chroma_extractor": save_chroma_npz,
        "key_extractor": save_key_npz,
        "band_energy_extractor": save_band_energy_npz,
        "spectral_entropy_extractor": save_spectral_entropy_npz,
        "spectral_extractor": save_spectral_npz,
        "quality_extractor": save_quality_npz,
        "mfcc_extractor": save_mfcc_npz,
        "mel_extractor": save_mel_npz,
        "hpss_extractor": save_hpss_npz,
        "voice_quality_extractor": save_voice_quality_npz,
    }
    
    if component_name in saver_map:
        saved_path = saver_map[component_name](
            out_path=out_path,
            payload=payload,
            status=status,
            error=error,
            empty_reason=empty_reason,
            producer_version=producer_version,
            schema_version=schema_version,
            extra_meta=extra_meta,
            run_rs_path=run_rs_path,
            feature_names=feature_names,
            feature_values=feature_values,
            add=add,
            _arr=_arr,
            build_meta=build_meta,
        )
        
        # Generate render files (JSON + HTML) after saving NPZ
        # This is best-effort: don't fail if render generation fails
        if os.path.exists(saved_path):
            try:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Generating render for {component_name} (NPZ saved to {saved_path})")
                from .renderer import render_component
                render_component(
                    npz_path=saved_path,
                    component_name=component_name,
                    output_dir=comp_dir,
                    enable_render=True,
                    enable_html_render=True,
                )
                logger.info(f"Render generation completed for {component_name}")
            except Exception as e:
                # Best-effort: log but don't fail
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not generate render for {component_name}: {e}")
                import traceback
                logger.debug(f"Render generation traceback for {component_name}: {traceback.format_exc()}")
        else:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"NPZ file not found for render generation: {saved_path}")
        
        return saved_path
    
    # Generic fallback: dump scalars into feature vector and store raw payload as object
    for k, v in payload.items():
        if isinstance(v, (int, float, np.integer, np.floating)) or v is None:
            add(str(k), v)
    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        payload=np.asarray(payload, dtype=object),
        meta=build_meta(
            producer=component_name,
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
            },
        ),
    )
    
    # Generate render files (JSON + HTML) after saving NPZ
    # This is best-effort: don't fail if render generation fails
    try:
        from .renderer import render_component
        # Get render flags from config (if available)
        # For now, enable both JSON and HTML by default
        render_component(
            npz_path=out_path,
            component_name=component_name,
            output_dir=comp_dir,
            enable_render=True,
            enable_html_render=True,
        )
    except Exception as e:
        # Best-effort: log but don't fail
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Could not generate render for {component_name}: {e}")
    
    return out_path

