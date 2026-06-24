"""UI helpers for NEU defect labels.

The canonical class configuration lives in app.core.neu_classes.
"""

from __future__ import annotations

from app.core.neu_classes import cn_to_en, cn_options
from app.core.neu_classes import color as _color
from app.core.neu_classes import icon as _icon
from app.core.neu_classes import zh_name


def cn(en_name: str) -> str:
    """英文类别名 -> 中文标签；未知类别保留原名。"""
    return zh_name(en_name)


def color(en_name: str) -> str:
    return _color(en_name)


def icon(en_name: str) -> str:
    return _icon(en_name)


def rgb(en_name: str) -> tuple[int, int, int]:
    hex_str = color(en_name).lstrip("#")
    return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))


def all_cn_names() -> list[str]:
    return cn_options()[1:]
