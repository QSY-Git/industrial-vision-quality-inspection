"""
指标卡片组件

用于首页展示关键数值。
"""

from __future__ import annotations

import streamlit as st


def metric_card(
    label: str,
    value,
    icon: str = "",
    delta=None,
    key: str = "",
) -> None:
    """
    渲染一个指标卡片

    Args:
        label: 指标标签
        value: 数值 (int/float/str)
        icon: 图标 emoji
        delta: 变化量 (可选)
        key: 唯一 key
    """
    display_label = f"{icon} {label}" if icon else label
    st.metric(
        label=display_label,
        value=value,
        delta=delta,
    )
