"""
推理引擎 — 抽象基类 & 数据结构

定义:
  - DetectorBase: 所有检测器必须实现的统一接口
  - Defect / DetectionResult / ModelInfo: 推理结果数据结构
  - ImageInput: 统一图像输入类型

扩展:
  新增检测器只需继承 DetectorBase，实现 _load_model / _inference / _postprocess
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image


# ============================================================
# 图像输入类型
# ============================================================
ImageInput = Union[str, Path, np.ndarray, Image.Image]
"""统一图像输入: 文件路径 | numpy数组(H,W,3) | PIL.Image"""


# ============================================================
# 数据结构
# ============================================================

@dataclass
class BBoxNorm:
    """归一化边界框 (YOLO 格式, 0~1)"""
    x_center: float
    y_center: float
    width: float
    height: float

    def to_pixel(self, img_w: int, img_h: int) -> BBoxPixel:
        """转换为像素坐标"""
        x1 = int((self.x_center - self.width / 2) * img_w)
        y1 = int((self.y_center - self.height / 2) * img_h)
        x2 = int((self.x_center + self.width / 2) * img_w)
        y2 = int((self.y_center + self.height / 2) * img_h)
        return BBoxPixel(
            x1=max(0, x1), y1=max(0, y1),
            x2=min(img_w, x2), y2=min(img_h, y2),
        )

    def to_xywh_pixel(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        """转换为 (x, y, w, h) 像素坐标"""
        x = int((self.x_center - self.width / 2) * img_w)
        y = int((self.y_center - self.height / 2) * img_h)
        w = int(self.width * img_w)
        h = int(self.height * img_h)
        return x, y, w, h


@dataclass
class BBoxPixel:
    """像素边界框 (x1,y1)-(x2,y2)"""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


@dataclass
class Defect:
    """单个检测到的缺陷"""
    class_name: str              # NEU 规范英文类别名
    class_id: int                # 类别索引
    confidence: float            # 置信度 0~1
    bbox: BBoxNorm               # 归一化坐标

    @property
    def severity(self) -> float:
        """缺陷严重度 = 面积比例 × 置信度"""
        return self.bbox.width * self.bbox.height * self.confidence


@dataclass
class DetectionResult:
    """单张图像的检测结果"""
    image_id: str                          # 图像标识 (文件名或UUID)
    defects: list[Defect] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)   # 原始图像 (W, H)
    inference_time_ms: float = 0.0
    model_name: str = ""

    @property
    def total_defects(self) -> int:
        return len(self.defects)

    @property
    def is_defect_free(self) -> bool:
        return len(self.defects) == 0

    @property
    def defect_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for d in self.defects:
            dist[d.class_name] = dist.get(d.class_name, 0) + 1
        return dist

    @property
    def max_confidence_defect(self) -> Defect | None:
        if not self.defects:
            return None
        return max(self.defects, key=lambda d: d.confidence)

    def filter_by_confidence(self, threshold: float) -> DetectionResult:
        """按置信度阈值过滤"""
        return DetectionResult(
            image_id=self.image_id,
            defects=[d for d in self.defects if d.confidence >= threshold],
            image_size=self.image_size,
            inference_time_ms=self.inference_time_ms,
            model_name=self.model_name,
        )

    def to_api_dict(self) -> dict:
        """转换为 API 响应格式"""
        return {
            "image_id": self.image_id,
            "status": "defect_detected" if not self.is_defect_free else "defect_free",
            "total_defects": self.total_defects,
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "defects": [
                {
                    "type": d.class_name,
                    "confidence": round(d.confidence, 4),
                    "bbox": {
                        "x": round(d.bbox.x_center, 4),
                        "y": round(d.bbox.y_center, 4),
                        "w": round(d.bbox.width, 4),
                        "h": round(d.bbox.height, 4),
                    },
                }
                for d in self.defects
            ],
            "distribution": self.defect_distribution,
            "inference_time_ms": round(self.inference_time_ms, 1),
            "model": self.model_name,
        }


@dataclass
class ModelInfo:
    """加载的模型元信息"""
    name: str                    # 模型名称
    architecture: str            # yolo | rtdetr | yoloe
    version: str = ""
    class_names: list[str] = field(default_factory=list)
    image_size: tuple[int, int] = (640, 640)
    weights_path: str = ""


# ============================================================
# 抽象基类
# ============================================================

class DetectorBase(ABC):
    """
    缺陷检测器统一抽象基类

    所有检测器 (YOLO / RT-DETR / YOLOE) 必须实现:
      - _load_model()   : 加载模型权重
      - _inference()     : 前向推理
      - _postprocess()   : 后处理 → Defect 列表

    公共模板方法:
      - load()           : 加载权重 + 标记就绪
      - detect()         : 预处理 → 推理 → 后处理 → DetectionResult
      - detect_batch()   : 批量推理
      - warmup()         : 模型预热
    """

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: tuple[int, int] = (640, 640),
        device: str = "cpu",
        class_names: list[str] | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device
        self.class_names = class_names or []

        self._loaded = False
        self._model: Any = None
        self._weights_path: str = ""

    # ----------------------------------------------------------
    # 子类必须实现
    # ----------------------------------------------------------

    @abstractmethod
    def _load_model(self, weights_path: str) -> Any:
        """
        加载模型权重，返回模型实例。

        Args:
            weights_path: 权重文件路径 (.pt / .pth / ...)

        Returns:
            模型对象 (YOLO / RTDETR / ...)
        """
        ...

    @abstractmethod
    def _inference(self, image_tensor: Any) -> Any:
        """
        执行前向推理。

        Args:
            image_tensor: 预处理后的张量

        Returns:
            原始预测输出 (架构相关)
        """
        ...

    @abstractmethod
    def _postprocess(
        self,
        raw_output: Any,
        original_shape: tuple[int, int],
    ) -> list[Defect]:
        """
        后处理: 原始预测 → Defect 列表。

        Args:
            raw_output: _inference() 的返回值
            original_shape: 原始图像 (W, H)

        Returns:
            缺陷列表
        """
        ...

    # ----------------------------------------------------------
    # 子类必须实现的属性
    # ----------------------------------------------------------

    @property
    @abstractmethod
    def model_info(self) -> ModelInfo:
        """返回当前加载的模型元信息"""
        ...

    @property
    @abstractmethod
    def supported_formats(self) -> list[str]:
        """返回支持的图像格式"""
        ...

    # ----------------------------------------------------------
    # 模板方法 (不建议子类覆盖)
    # ----------------------------------------------------------

    def load(self, weights_path: str) -> None:
        """
        加载模型权重

        Args:
            weights_path: 模型权重文件路径
        """
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"模型权重不存在: {weights_path}")

        self._weights_path = str(path.resolve())
        self._model = self._load_model(self._weights_path)
        self._loaded = True

    def detect(self, image: ImageInput) -> DetectionResult:
        """
        单张图像缺陷检测

        Args:
            image: 图像输入 (路径 | ndarray | PIL.Image)

        Returns:
            DetectionResult 含所有检测到的缺陷
        """
        self._ensure_loaded()

        t0 = time.perf_counter()

        # 1. 读取并预处理
        img_array, original_shape = self._read_image(image)
        tensor = self._preprocess(img_array)

        # 2. 推理
        raw_output = self._inference(tensor)

        # 3. 后处理
        defects = self._postprocess(raw_output, original_shape)

        elapsed = (time.perf_counter() - t0) * 1000

        return DetectionResult(
            image_id=self._derive_image_id(image),
            defects=defects,
            image_size=original_shape,
            inference_time_ms=round(elapsed, 1),
            model_name=self.model_info.name,
        )

    def detect_batch(self, images: list[ImageInput]) -> list[DetectionResult]:
        """
        批量推理 (顺序执行，GPU 批处理由子类优化)

        Args:
            images: 图像输入列表

        Returns:
            检测结果列表 (顺序与输入一致)
        """
        self._ensure_loaded()
        return [self.detect(img) for img in images]

    def warmup(self, num_iter: int = 3) -> None:
        """
        模型预热 (消除首次推理冷启动延迟)

        Args:
            num_iter: 预热轮数
        """
        self._ensure_loaded()
        dummy = np.random.randint(0, 255, (*self.image_size[::-1], 3), dtype=np.uint8)
        for _ in range(num_iter):
            tensor = self._preprocess(dummy)
            self._inference(tensor)

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("模型未加载，请先调用 detector.load(weights_path)")

    def _read_image(self, image: ImageInput) -> tuple[np.ndarray, tuple[int, int]]:
        """
        统一读取图像 → RGB numpy array

        Returns:
            (image_array_RGB_HWC, (W, H))
        """
        if isinstance(image, (str, Path)):
            img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            # 推断通道顺序: >=3 通道 → 取前3
            if image.ndim == 3 and image.shape[2] >= 3:
                img_array = image[:, :, :3]
            else:
                img_array = image
            img = Image.fromarray(img_array).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise TypeError(f"不支持图像类型: {type(image)}")

        w, h = img.size
        return np.array(img), (w, h)

    def _preprocess(self, img_array: np.ndarray) -> Any:
        """
        图像预处理: resize → normalize → tensor
        子类可覆盖以适配特定模型
        """
        from app.core.preprocessor import ImagePreprocessor
        preprocessor = ImagePreprocessor(
            target_size=self.image_size,
            device=self.device,
        )
        return preprocessor.process(img_array)

    def _derive_image_id(self, image: ImageInput) -> str:
        """从输入推导 image_id"""
        if isinstance(image, (str, Path)):
            return Path(image).stem
        if isinstance(image, np.ndarray):
            return f"array_{image.shape}"
        if isinstance(image, Image.Image):
            return f"pil_{id(image)}"
        return "unknown"
