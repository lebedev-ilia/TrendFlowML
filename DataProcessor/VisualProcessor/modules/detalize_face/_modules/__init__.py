"""
Модули для извлечения различных типов фич лица.
"""

from .base_module import FaceModule
from .geometry_module import GeometryModule
from .pose_module import PoseModule
from .quality_module import QualityModule
from .eyes_module import EyesModule
from .motion_module import MotionModule
from .structure_module import StructureModule
from .lip_reading_module import LipReadingModule

# Registry для автоматической загрузки модулей
MODULE_REGISTRY = {
    "geometry": GeometryModule,
    "pose": PoseModule,
    "quality": QualityModule,
    "eyes": EyesModule,
    "motion": MotionModule,
    "structure": StructureModule,
    "lip_reading": LipReadingModule,
}

__all__ = [
    "FaceModule",
    "GeometryModule",
    "PoseModule",
    "QualityModule",
    "EyesModule",
    "MotionModule",
    "StructureModule",
    "LipReadingModule",
    "MODULE_REGISTRY",
]

