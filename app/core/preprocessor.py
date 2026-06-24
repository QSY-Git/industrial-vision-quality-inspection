"""
图像预处理器

将图像转换为模型输入格式:
  RGB numpy → Resize (保持宽高比 + padding) → Normalize → CHW Tensor → Batch dim
"""

from __future__ import annotations

import numpy as np


class ImagePreprocessor:
    """
    独立图像预处理模块

    处理链路:
      np.ndarray (H,W,3) uint8 → letterbox resize → normalize → (1,3,H,W) float32 tensor
    """

    # ImageNet 均值 & 标准差 (YOLO 默认)
    MEAN = (0.0, 0.0, 0.0)
    STD = (255.0, 255.0, 255.0)

    def __init__(
        self,
        target_size: tuple[int, int] = (640, 640),
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        device: str = "cpu",
    ) -> None:
        self.target_size = target_size
        self.mean = mean or self.MEAN
        self.std = std or self.STD
        self.device = device

    # --------------------------------------------------------
    # 主要入口
    # --------------------------------------------------------

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        完整预处理流程

        Args:
            image: RGB uint8 numpy array (H, W, 3)

        Returns:
            float32 (1, 3, H, W) 归一化张量 (numpy)
        """
        # 1. letterbox resize
        resized, ratio, pad = self.letterbox(image)

        # 2. HWC → CHW + normalize → float32
        tensor = self.to_tensor(resized)

        # 3. 添加 batch 维度
        tensor = np.expand_dims(tensor, axis=0)

        return tensor

    def process_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """批量预处理 → (B, 3, H, W)"""
        tensors = [self.process(img) for img in images]
        return np.concatenate(tensors, axis=0)

    # --------------------------------------------------------
    # 子步骤
    # --------------------------------------------------------

    def letterbox(
        self,
        image: np.ndarray,
        color: tuple[int, int, int] = (114, 114, 114),
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        Resize + padding 保持宽高比

        仿 ultralytics letterbox 行为

        Args:
            image: (H, W, 3) uint8
            color: padding 填充色

        Returns:
            (resized_image, scale_ratio, (pad_left, pad_top))
        """
        h, w = image.shape[:2]
        tw, th = self.target_size

        # 缩放比例: 适应最长边
        r = min(tw / w, th / h)
        new_w, new_h = int(round(w * r)), int(round(h * r))

        # 缩放
        if (new_w, new_h) != (w, h):
            # 使用 PIL 进行高质量缩放
            from PIL import Image
            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            resized = np.array(pil_img)
        else:
            resized = image

        # padding 到目标尺寸
        pad_w = tw - new_w
        pad_h = th - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2

        padded = np.full((th, tw, 3), color, dtype=np.uint8)
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w, :] = resized

        return padded, r, (pad_left, pad_top)

    def to_tensor(self, image: np.ndarray) -> np.ndarray:
        """
        HWC uint8 → CHW float32 归一化

        Args:
            image: (H, W, 3) uint8

        Returns:
            (3, H, W) float32 in [0, 1]
        """
        # HWC → CHW
        tensor = np.transpose(image, (2, 0, 1)).astype(np.float32)

        # Normalize: (pixel / 255)
        tensor /= 255.0

        return tensor

    # --------------------------------------------------------
    # 反向变换 (像素坐标还原)
    # --------------------------------------------------------

    def recover_bbox_pixel(
        self,
        bbox: list[float],          # [x_center, y_center, width, height] 归一化
        original_shape: tuple[int, int],  # (W, H)
        ratio: float,
        pad: tuple[int, int],       # (pad_left, pad_top)
    ) -> tuple[int, int, int, int]:
        """
        将归一化 bbox 还原到原始图像像素坐标

        Args:
            bbox: [x_center, y_center, width, height] 基于 letterbox 后的尺寸
            original_shape: 原始图像 (W, H)
            ratio: 缩放比例
            pad: (pad_left, pad_top)

        Returns:
            (x1, y1, x2, y2) 像素坐标
        """
        x_center, y_center, w, h = bbox
        tw, th = self.target_size
        ow, oh = original_shape

        # 去除 padding 影响
        x_center = (x_center * tw - pad[0]) / ratio
        y_center = (y_center * th - pad[1]) / ratio
        w = (w * tw) / ratio
        h = (h * th) / ratio

        # 转 (x1, y1, x2, y2) 并限制边界
        x1 = max(0, int(x_center - w / 2))
        y1 = max(0, int(y_center - h / 2))
        x2 = min(ow, int(x_center + w / 2))
        y2 = min(oh, int(y_center + h / 2))

        return x1, y1, x2, y2
