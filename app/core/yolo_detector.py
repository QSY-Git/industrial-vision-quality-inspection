"""
YOLODetector — 基于 ultralytics YOLO 的缺陷检测器实现

继承 DetectorBase，实现 _load_model / _inference / _postprocess
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ultralytics import YOLO

from app.core.base import (
    Defect,
    DetectorBase,
    ModelInfo,
)
from app.core.neu_classes import canonical_name
from app.core.postprocessor import ResultPostprocessor

logger = logging.getLogger(__name__)


class YOLODetector(DetectorBase):
    """
    YOLOv11 缺陷检测器

    使用 ultralytics YOLO 进行工业缺陷检测。
    支持 YOLOv8 / YOLOv11 系列权重。

    Usage:
        detector = YOLODetector(
            confidence_threshold=0.25,
            device="cuda:0",
            class_names=["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"],
        )
        detector.load("best.pt")
        result = detector.detect("defect_image.jpg")
    """

    SUPPORTED_ARCHITECTURES = {"yolov8", "yolov11", "yolo11", "yolo"}

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: tuple[int, int] = (640, 640),
        device: str = "cpu",
        class_names: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            image_size=image_size,
            device=device,
            class_names=class_names,
        )
        self._model_meta: ModelInfo | None = None

    # ----------------------------------------------------------
    # 实现抽象方法
    # ----------------------------------------------------------

    def _load_model(self, weights_path: str) -> YOLO:
        """加载 YOLO 模型"""
        model = YOLO(weights_path)

        # 提取元信息
        ckpt = model.ckpt if hasattr(model, "ckpt") else {}
        model_name = ckpt.get("model_name", Path(weights_path).stem)
        model_version = ckpt.get("version", "")

        # 尝试从模型获取类别名，并与传入配置比对。
        model_class_names: list[str] = []
        if hasattr(model, "names"):
            names_dict = model.names
            if isinstance(names_dict, dict):
                model_class_names = [
                    canonical_name(names_dict[i])
                    for i in sorted(names_dict.keys())
                ]

        class_names = self.class_names
        if class_names and model_class_names and class_names != model_class_names:
            logger.warning(
                "Configured class order %s differs from model.names %s; using configured order.",
                class_names,
                model_class_names,
            )
        if not class_names:
            class_names = model_class_names

        self._model_meta = ModelInfo(
            name=model_name,
            architecture="yolo",
            version=str(model_version),
            class_names=class_names,
            image_size=self.image_size,
            weights_path=weights_path,
        )

        # 设置设备
        if self.device != "cpu":
            try:
                model.to(self.device)
            except Exception:
                pass

        return model

    def _inference(self, image_tensor: np.ndarray) -> Any:
        """
        YOLO 前向推理

        Args:
            image_tensor: (1, 3, H, W) float32 numpy (由基类预处理生成)

        Returns:
            ultralytics Results 列表
        """
        model: YOLO = self._model
        results = model.predict(
            source=image_tensor,
            imgsz=self.image_size,
            conf=0.001,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            stream=False,
        )
        return results

    def detect(self, image) -> DetectionResult:
        """
        覆盖基类 detect(): 直接传入图像让 ultralytics 自行预处理。

        基类的 _preprocess → tensor 流程不适合 ultralytics，
        ultralytics 内部有完整的预处理链路。
        """
        self._ensure_loaded()

        import time as _time
        t0 = _time.perf_counter()

        # 读取图像
        img_array, original_shape = self._read_image(image)

        # 直接传给 ultralytics (它会自行 resize/normalize)
        raw_output = self._inference(img_array)

        # 后处理
        defects = self._postprocess(raw_output, original_shape)

        elapsed = (_time.perf_counter() - t0) * 1000

        from app.core.base import DetectionResult as DR
        return DR(
            image_id=self._derive_image_id(image),
            defects=defects,
            image_size=original_shape,
            inference_time_ms=round(elapsed, 1),
            model_name=self.model_info.name,
        )

    # 保留 _load_model / _postprocess / model_info 不变

    def _postprocess(
        self,
        raw_output: Any,
        original_shape: tuple[int, int],
    ) -> list[Defect]:
        """
        ultralytics Results → Defect 列表
        """
        postprocessor = ResultPostprocessor(
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            class_names=self.get_class_names(),
        )
        return postprocessor.process(
            raw_predictions=list(raw_output),
            original_shape=original_shape,
        )

    # ----------------------------------------------------------
    # 属性
    # ----------------------------------------------------------

    @property
    def model_info(self) -> ModelInfo:
        if self._model_meta is None:
            return ModelInfo(name="unknown", architecture="yolo")
        return self._model_meta

    @property
    def supported_formats(self) -> list[str]:
        return [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    # ----------------------------------------------------------
    # 便捷方法
    # ----------------------------------------------------------

    def get_class_names(self) -> list[str]:
        """获取当前模型的类别名列表"""
        return self.class_names or self._model_meta.class_names if self._model_meta else []

    def __repr__(self) -> str:
        loaded = "loaded" if self.is_loaded else "not loaded"
        weights = Path(self._weights_path).name if self._weights_path else "none"
        return (
            f"YOLODetector(device={self.device}, "
            f"conf={self.confidence_threshold}, "
            f"weights={weights}, {loaded})"
        )
