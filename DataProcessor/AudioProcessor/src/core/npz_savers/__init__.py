"""
NPZ саверы для различных extractors.
Каждый савер отвечает за сохранение NPZ артефактов для конкретного типа extractor'а.
"""
from .tempo import save_tempo_npz
from .loudness import save_loudness_npz
from .clap import save_clap_npz
from .asr import save_asr_npz
from .speaker_diarization import save_speaker_diarization_npz
from .emotion_diarization import save_emotion_diarization_npz
from .source_separation import save_source_separation_npz
from .speech_analysis import save_speech_analysis_npz
from .onset import save_onset_npz
from .rhythmic import save_rhythmic_npz
from .chroma import save_chroma_npz
from .key import save_key_npz
from .band_energy import save_band_energy_npz
from .spectral_entropy import save_spectral_entropy_npz
from .spectral import save_spectral_npz
from .quality import save_quality_npz
from .mfcc import save_mfcc_npz
from .mel import save_mel_npz
from .hpss import save_hpss_npz
from .voice_quality import save_voice_quality_npz

__all__ = [
    "save_tempo_npz",
    "save_loudness_npz",
    "save_clap_npz",
    "save_asr_npz",
    "save_speaker_diarization_npz",
    "save_emotion_diarization_npz",
    "save_source_separation_npz",
    "save_speech_analysis_npz",
    "save_onset_npz",
    "save_rhythmic_npz",
    "save_chroma_npz",
    "save_key_npz",
    "save_band_energy_npz",
    "save_spectral_entropy_npz",
    "save_spectral_npz",
    "save_quality_npz",
    "save_mfcc_npz",
    "save_mel_npz",
    "save_hpss_npz",
    "save_voice_quality_npz",
]

