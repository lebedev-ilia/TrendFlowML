"""
Pitch extractor: извлечение основной частоты (f0) с использованием PYIN/YIN/CREPE.
"""

# Backward compatible import path:
#   from src.extractors.pitch_extractor import PitchExtractor

from .main import PitchExtractor

__all__ = ["PitchExtractor"]

