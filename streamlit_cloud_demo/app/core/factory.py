"""
检测器工厂

根据架构名称创建 YOLODetector 实例。
当前仅支持 YOLO11/YOLOv8 系列。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.base import DetectorBase


# 架构注册表
ARCH_REGISTRY: dict[str, type[DetectorBase]] = {}

# 需要延迟导入的架构 → (module_path, class_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "yolo":    ("app.core.yolo_detector", "YOLODetector"),
    "yolov8":  ("app.core.yolo_detector", "YOLODetector"),
    "yolov11": ("app.core.yolo_detector", "YOLODetector"),
    "yolo11":  ("app.core.yolo_detector", "YOLODetector"),
}


def create_detector(
    architecture: str = "yolo",
    **kwargs: Any,
) -> DetectorBase:
    """
    工厂函数 — 创建检测器实例

    Args:
        architecture: 架构名称 "yolo" | "yolov11"
        **kwargs: 传递给检测器构造函数的参数

    Returns:
        DetectorBase 实例

    Example:
        detector = create_detector("yolo", device="cuda:0", class_names=[...])
        detector.load("best.pt")
        result = detector.detect("image.jpg")
    """
    arch = architecture.lower().replace("-", "").replace("_", "")

    # 1. 检查已注册
    if arch in ARCH_REGISTRY:
        return ARCH_REGISTRY[arch](**kwargs)

    # 2. 尝试延迟导入
    if arch in _LAZY_IMPORTS:
        module_path, class_name = _LAZY_IMPORTS[arch]
        try:
            import importlib
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            # 导入成功后注册到缓存
            for alias, (mp, cn) in _LAZY_IMPORTS.items():
                if mp == module_path and cn == class_name:
                    ARCH_REGISTRY[alias] = cls
            return cls(**kwargs)
        except ImportError as e:
            raise ImportError(
                f"无法加载 {architecture} 检测器。\n"
                f"请确认已安装: pip install ultralytics torch\n"
                f"原始错误: {e}"
            )

    raise ValueError(
        f"不支持的检测器架构: {architecture}\n"
        f"支持的架构: {list(ARCH_REGISTRY.keys()) + list(_LAZY_IMPORTS.keys())}"
    )


def detect_arch_from_weights(weights_path: str) -> str:
    """
    从 .pt 权重文件自动检测架构类型

    Args:
        weights_path: .pt 文件路径

    Returns:
        架构名称: "yolo"
    """
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    try:
        import torch
    except ImportError:
        # 无法读取 .pt → 默认 yolo
        return "yolo"

    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        try:
            checkpoint = torch.load(str(path), map_location="cpu")
        except Exception:
            return "yolo"

    model_name = ""
    if isinstance(checkpoint, dict):
        model_name = str(checkpoint.get("model_name", "")).lower()
        inner = checkpoint.get("model", {})
        if isinstance(inner, dict):
            yaml_data = inner.get("yaml", {})
            if isinstance(yaml_data, dict):
                model_name = str(yaml_data.get("model_name", model_name)).lower()

    if any(name in model_name for name in ("yolo", "yolov8", "yolov11", "yolo11")):
        return "yolo"

    return "yolo"


def list_available_architectures() -> list[str]:
    """列出所有可用架构 (已注册 + 可延迟导入)"""
    return sorted(set(ARCH_REGISTRY.keys()) | set(_LAZY_IMPORTS.keys()))


def register_architecture(name: str, detector_cls: type[DetectorBase]) -> None:
    """
    注册新的检测器架构 (运行时)

    Args:
        name: 架构标识 (如 "rtdetr")
        detector_cls: 检测器类 (必须继承 DetectorBase)
    """
    if not issubclass(detector_cls, DetectorBase):
        raise TypeError(f"{detector_cls} 必须继承 DetectorBase")
    ARCH_REGISTRY[name.lower()] = detector_cls
