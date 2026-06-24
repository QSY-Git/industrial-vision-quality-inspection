"""NEU Surface Defect class configuration.

This is the single source of truth for class order, canonical storage names,
Chinese display labels, and model-name aliases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NeuClass:
    id: int
    name: str
    zh: str
    color: str
    icon: str
    model_name: str | None = None


NEU_CLASSES: tuple[NeuClass, ...] = (
    NeuClass(0, "crazing", "龟裂", "#D72638", "🔴"),
    NeuClass(1, "inclusion", "夹杂", "#7D3C98", "🟣"),
    NeuClass(2, "patches", "斑块", "#F49D37", "🟠"),
    NeuClass(3, "pitted_surface", "麻点", "#2E86AB", "🔵"),
    NeuClass(4, "rolled_in_scale", "氧化皮", "#566573", "⚙️", "rolled-in_scale"),
    NeuClass(5, "scratches", "划痕", "#1B998B", "🟢"),
)

CANONICAL_NAMES: list[str] = [c.name for c in NEU_CLASSES]
MODEL_NAMES: list[str] = [c.model_name or c.name for c in NEU_CLASSES]
ZH_NAMES: list[str] = [c.zh for c in NEU_CLASSES]

_BY_NAME = {c.name: c for c in NEU_CLASSES}
_ALIASES = {c.name: c.name for c in NEU_CLASSES}
_ALIASES.update({c.model_name: c.name for c in NEU_CLASSES if c.model_name})


def canonical_name(raw_name: str) -> str:
    """Return canonical English storage name, warning when unknown."""
    name = str(raw_name)
    canonical = _ALIASES.get(name)
    if canonical is None:
        logger.warning("Unknown defect class name from model/API: %s", name)
        return name
    return canonical


def class_name_for_id(class_id: int) -> str:
    """Return canonical class name for a model class id."""
    if 0 <= class_id < len(NEU_CLASSES):
        return NEU_CLASSES[class_id].name
    logger.warning("Unknown defect class id from model: %s", class_id)
    return str(class_id)


def is_neu_class(name: str) -> bool:
    return canonical_name(name) in _BY_NAME


def zh_name(name: str) -> str:
    canonical = canonical_name(name)
    cls = _BY_NAME.get(canonical)
    return cls.zh if cls else name


def color(name: str) -> str:
    canonical = canonical_name(name)
    cls = _BY_NAME.get(canonical)
    return cls.color if cls else "#888888"


def icon(name: str) -> str:
    canonical = canonical_name(name)
    cls = _BY_NAME.get(canonical)
    return cls.icon if cls else "?"


def cn_options() -> list[str]:
    return ["全部"] + ZH_NAMES


def cn_to_en(cn_name: str) -> str | None:
    if cn_name == "全部":
        return None
    for cls in NEU_CLASSES:
        if cls.zh == cn_name:
            return cls.name
    logger.warning("Unknown Chinese defect label: %s", cn_name)
    return None
