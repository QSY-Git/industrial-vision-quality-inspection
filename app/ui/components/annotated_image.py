"""
标注图像渲染器

在检测结果图像上绘制中文标签 + 颜色区分 + 置信度。
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

from app.ui.font_utils import load_label_font
from app.ui.labels import cn, rgb


def draw_annotations(
    image: Image.Image,
    defects: list[dict],
    thickness: int = 3,
    font_size: int = 16,
) -> Image.Image:
    """
    在 PIL 图像上绘制缺陷标注框 (中文标签)

    Args:
        image: 原始 RGB PIL.Image
        defects: [{"class_name": str, "confidence": float, "bbox": {"x","y","w","h"}}]
        thickness: 边框线宽
        font_size: 文字大小

    Returns:
        带标注的 RGB PIL.Image
    """
    if not defects:
        return image

    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    w, h = image.size

    font, _, supports_chinese = load_label_font(font_size)

    for d in defects:
        class_name = d.get("class_name", "unknown")
        confidence = d.get("confidence", 0.0)
        bbox = d.get("bbox", {})

        # 归一化 → 像素
        x_center = bbox.get("x", 0.5) * w
        y_center = bbox.get("y", 0.5) * h
        bw = bbox.get("w", 0.1) * w
        bh = bbox.get("h", 0.1) * h

        x1 = _clamp_int(x_center - bw / 2, 0, max(w - 1, 0))
        y1 = _clamp_int(y_center - bh / 2, 0, max(h - 1, 0))
        x2 = _clamp_int(x_center + bw / 2, 0, max(w - 1, 0))
        y2 = _clamp_int(y_center + bh / 2, 0, max(h - 1, 0))
        if x2 <= x1 or y2 <= y1:
            continue

        # 中文标签 + 颜色；若没有中文字体，回退英文避免方块乱码。
        label_name = cn(class_name) if supports_chinese else str(class_name)
        label_text = f"{label_name} {float(confidence):.1%}"
        line_color = rgb(class_name)
        text_color = _best_text_color(line_color)

        # 绘制矩形
        draw.rectangle([x1, y1, x2, y2], outline=line_color, width=thickness)

        # 标签背景 + 文字
        text_bbox = draw.textbbox((x1, y1), label_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        pad_x = 4
        pad_y = 3
        label_w = min(w, text_w + pad_x * 2)
        label_h = min(h, text_h + pad_y * 2)
        label_x = _clamp_int(x1, 0, max(w - label_w, 0))
        if y1 - label_h >= 0:
            label_y = y1 - label_h
        elif y2 + label_h <= h:
            label_y = y2
        else:
            label_y = y1
        label_y = _clamp_int(label_y, 0, max(h - label_h, 0))

        draw.rectangle(
            [label_x, label_y, label_x + label_w, label_y + label_h],
            fill=line_color,
        )
        draw.text(
            (label_x + pad_x, label_y + pad_y),
            label_text,
            fill=text_color,
            font=font,
        )

    return draw_img


def draw_annotations_bgr(
    image_bgr,
    defects: list[dict],
    thickness: int = 3,
    font_size: int = 16,
):
    """Draw labels on an OpenCV BGR image and return BGR."""
    import numpy as np

    rgb_array = image_bgr[:, :, ::-1]
    pil_img = Image.fromarray(rgb_array.astype("uint8"), mode="RGB")
    annotated = draw_annotations(pil_img, defects, thickness=thickness, font_size=font_size)
    return np.array(annotated)[:, :, ::-1]


def pil_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
    """PIL.Image → bytes"""
    buf = io.BytesIO()
    image.save(buf, format=format, quality=90)
    return buf.getvalue()


def _best_text_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = bg
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 150 else (255, 255, 255)


def _clamp_int(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, round(float(value)))))
