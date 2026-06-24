"""
缺陷记录卡片组件

用于历史记录页展示每条检测结果。
"""

from __future__ import annotations

import streamlit as st

from app.ui.local_client import LocalClient as APIClient
from app.ui.labels import cn, color, icon as _icon


def defect_card(item: dict, api: APIClient | None = None) -> None:
    """
    渲染一张缺陷记录卡片

    Args:
        item: { id, image_name, defect_type, confidence, bbox, detect_time }
        api: API 客户端 (删除功能需要)
    """
    defect_type = item.get("defect_type", "unknown")
    bbox = item.get("bbox", {})

    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.markdown(
                f"{_icon(defect_type)} **{cn(defect_type)}** "
                f"`{item.get('image_name', '?')}`"
            )
            st.caption(
                f"bbox: ({bbox.get('x',0):.2f}, {bbox.get('y',0):.2f}, "
                f"{bbox.get('w',0):.2f}, {bbox.get('h',0):.2f})"
            )

        with col2:
            conf = item.get("confidence", 0)
            st.metric(label="置信度", value=f"{conf:.2%}")

        with col3:
            st.caption(item.get("detect_time", "?")[:19])
