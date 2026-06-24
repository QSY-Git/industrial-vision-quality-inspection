"""
核心引擎层

提供缺陷检测统一接口，支持多架构扩展 (YOLO / RT-DETR / YOLOE)。

对外 API:
  - create_detector(arch) → DetectorBase
  - DetectorBase (抽象基类)
  - YOLODetector
  - DetectionResult / Defect / ModelInfo (数据结构)
"""

from app.core.base import (
    BBoxNorm,
    BBoxPixel,
    Defect,
    DetectionResult,
    DetectorBase,
    ModelInfo,
)
from app.core.factory import (
    create_detector,
    detect_arch_from_weights,
    list_available_architectures,
    register_architecture,
)

# YOLODetector 延迟导入 (需要 ultralytics)
def __getattr__(name: str):
    if name == "YOLODetector":
        from app.core.yolo_detector import YOLODetector as _YOLO
        return _YOLO
    if name in globals():
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DetectorBase",
    "YOLODetector",
    "DetectionResult",
    "Defect",
    "BBoxNorm",
    "BBoxPixel",
    "ModelInfo",
    "create_detector",
    "detect_arch_from_weights",
    "list_available_architectures",
    "register_architecture",
]
