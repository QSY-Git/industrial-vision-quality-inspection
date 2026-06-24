"""Font resolution helpers for drawing Chinese labels on images."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

logger = logging.getLogger(__name__)

CHINESE_PROBE_TEXT = "龟裂夹杂斑块麻点氧化皮划痕"

WINDOWS_CJK_FONTS = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
)

LINUX_CJK_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.otf",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
)


def get_configured_font_path() -> str | None:
    """Read optional ui.font_path from config/settings.yaml without requiring PyYAML."""
    config_path = Path("config/settings.yaml")
    if not config_path.exists():
        return None

    in_ui = False
    for raw in config_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw.startswith((" ", "\t")) and stripped.endswith(":"):
            in_ui = stripped[:-1] == "ui"
            continue
        if in_ui and stripped.startswith("font_path:"):
            _, value = stripped.split(":", 1)
            value = value.strip().strip("\"'")
            return value or None
    return None


def iter_font_candidates() -> list[Path]:
    candidates: list[str | None] = [
        os.getenv("LABEL_FONT_PATH"),
        get_configured_font_path(),
        *WINDOWS_CJK_FONTS,
        *LINUX_CJK_FONTS,
    ]
    result: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


@lru_cache(maxsize=8)
def load_label_font(font_size: int) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, Path | None, bool]:
    """Return a cached font, its path, and whether it supports Chinese labels."""
    for path in iter_font_candidates():
        if not path.exists():
            continue
        try:
            font = ImageFont.truetype(str(path), font_size)
        except OSError as exc:
            logger.warning("Unable to load label font %s: %s", path, exc)
            continue
        if _font_can_render(font, CHINESE_PROBE_TEXT):
            return font, path, True
        logger.warning("Label font does not appear to support Chinese: %s", path)

    logger.warning(
        "No Chinese-capable label font found. Set LABEL_FONT_PATH or ui.font_path; "
        "annotation labels will fall back to English class names."
    )
    return ImageFont.load_default(), None, False


def _font_can_render(font: ImageFont.ImageFont, text: str) -> bool:
    try:
        bbox = font.getbbox(text)
    except Exception:
        return False
    return (bbox[2] - bbox[0]) > 0 and (bbox[3] - bbox[1]) > 0
