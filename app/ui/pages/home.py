"""
首页 — 系统概览仪表板
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui.local_client import LocalClient as APIClient
from app.ui.components.metric_card import metric_card
from app.ui.labels import cn, color, icon as _icon


def render_home(api: APIClient) -> None:
    st.title("钢板表面缺陷检测系统")
    st.caption("基于 YOLO11 的 NEU-DET 六类缺陷检测与质量分析")

    stats_data = _safe_fetch(api.stats, days=1)
    history_data = _safe_fetch(api.history, page=1, size=5)
    health_data = _safe_fetch(api.health)

    st.markdown("### 关键指标")
    overview = stats_data.get("data", {}).get("overview", {}) if stats_data else {}
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("今日检测", overview.get("today", 0), icon="📸")
    with c2:
        metric_card("检测图片数", overview.get("total", 0), icon="✅")
    with c3:
        dist_list = stats_data.get("data", {}).get("defect_distribution", []) if stats_data else []
        metric_card("今日缺陷框", overview.get("today_defect_total", 0), icon="📊")
    with c4:
        metric_card("无缺陷率", f"{overview.get('defect_free_rate', 0):.1%}", icon="🔍")

    st.markdown("---")
    cl, cr = st.columns([3, 2])

    with cl:
        st.markdown("### 最近检测")
        items = history_data.get("data", {}).get("items", []) if history_data else []
        if items:
            df = pd.DataFrame([{
                "图像": it.get("image_name", "?")[:30],
                "缺陷": cn(it.get("defect_type", "?")),
                "置信度": f"{it.get('confidence', 0):.2%}",
                "时间": it.get("detect_time", "?")[:19],
            } for it in items[:5]])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("暂无检测记录")

    with cr:
        st.markdown("### 缺陷分布")
        if dist_list:
            dist_df = pd.DataFrame([{
                "类型": cn(d.get("type", "?")),
                "数量": d.get("count", 0),
                "占比": f"{d.get('percentage', 0):.1%}",
            } for d in dist_list])
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
            for d in dist_list:
                t = d.get("type", "")
                pct = d.get("percentage", 0)
                bar = "█" * int(pct * 30)
                st.markdown(
                    f"{_icon(t)} **{cn(t)}** "
                    f'<span style="color:{color(t)}">{bar}</span> `{pct:.1%}`',
                    unsafe_allow_html=True,
                )
        else:
            st.info("暂无数据")

    st.markdown("---")
    st.markdown("### 模型状态")
    if health_data:
        if health_data.get("model_loaded", False):
            st.success("模型已加载，推理就绪")
        else:
            st.warning("模型未加载")
    else:
        st.error("无法获取 API 状态")


def _safe_fetch(func, **kwargs):
    try:
        return func(**kwargs)
    except Exception:
        return None
