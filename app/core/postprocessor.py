"""
结果后处理器

将 Ultralytics YOLO Results 转换为 Defect 列表:
  Ultralytics Results → 置信度过滤 → 坐标格式转换 → Defect 对象

注意: Ultralytics 内部已完成 NMS，此后处理器不再重复执行 NMS。
"""

from __future__ import annotations

import numpy as np

from app.core.base import BBoxNorm, Defect
from app.core.neu_classes import canonical_name, class_name_for_id


class ResultPostprocessor:
    """
    Ultralytics Results 后处理

    从 Ultralytics 推理结果提取缺陷列表:
      1. 读取 Ultralytics Results (已完成 NMS)
      2. 置信度筛选
      3. xyxy → 归一化中心坐标转换
      4. class_id → class_name 映射
      5. 构造 Defect 列表
    """

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,        # 保留参数兼容性，NMS 由 Ultralytics 内部处理
        class_names: list[str] | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        _ = iou_threshold                    # Ultralytics 内部 NMS 已处理
        self.class_names = class_names or []

    # --------------------------------------------------------
    # 公开方法
    # --------------------------------------------------------

    def process(
        self,
        raw_predictions: list,           # ultralytics Results 对象列表
        original_shape: tuple[int, int], # (W, H)
    ) -> list[Defect]:
        """
        后处理主入口

        Args:
            raw_predictions: ultralytics model.predict() 返回的 Results 列表
            original_shape: 原始图像 (W, H)

        Returns:
            Defect 列表
        """
        if not raw_predictions:
            return []

        result = raw_predictions[0]    # 单张图像，取第一个

        if result.boxes is None:
            return []

        # 提取数据
        boxes = result.boxes

        if len(boxes) == 0:
            return []

        # 坐标 → numpy
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
        cls_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)

        # 置信度筛选
        keep = confs >= self.confidence_threshold
        xyxy = xyxy[keep]
        confs = confs[keep]
        cls_ids = cls_ids[keep]

        if len(xyxy) == 0:
            return []

        # 坐标归一化 (Ultralytics 已完成 NMS)
        ow, oh = original_shape
        defects: list[Defect] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            cls_id = int(cls_ids[i])
            cls_name = self._get_class_name(cls_id)

            # 转 YOLO 归一化格式
            x_center = ((x1 + x2) / 2) / ow
            y_center = ((y1 + y2) / 2) / oh
            width = (x2 - x1) / ow
            height = (y2 - y1) / oh

            # 限制在 [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            defects.append(Defect(
                class_name=cls_name,
                class_id=cls_id,
                confidence=round(float(confs[i]), 4),
                bbox=BBoxNorm(
                    x_center=round(float(x_center), 6),
                    y_center=round(float(y_center), 6),
                    width=round(float(width), 6),
                    height=round(float(height), 6),
                ),
            ))

        return defects

    # --------------------------------------------------------
    # 工具
    # --------------------------------------------------------

    def _get_class_name(self, class_id: int) -> str:
        """class_id → class_name"""
        if self.class_names and 0 <= class_id < len(self.class_names):
            return canonical_name(self.class_names[class_id])
        return class_name_for_id(class_id)
