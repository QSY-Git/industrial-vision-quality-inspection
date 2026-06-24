"""
统计分析页 — 图表驱动的数据看板
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui.local_client import LocalClient as APIClient
from app.ui.components.metric_card import metric_card
from app.ui.labels import cn, color, icon as _icon


def render_statistics(api: APIClient) -> None:
    st.title("统计分析")
    st.caption("缺陷检测数据多维分析")

    days = st.selectbox(
        "时间范围", options=[1, 3, 7, 14, 30, 90], index=2,
        format_func=lambda d: f"最近 {d} 天",
    )

    try:
        result = api.stats(days=days)
    except Exception as e:
        st.error(f"无法连接 API: {e}")
        return

    if result.get("status") == "error":
        st.error(result.get("error", {}).get("message", "获取统计失败"))
        return

    data = result.get("data", {})
    overview = data.get("overview", {})
    dist = data.get("defect_distribution", [])
    trend = data.get("daily_trend", [])

    st.markdown("### 概览")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("检测图片数", overview.get("total", 0), icon="✅")
    with c2:
        metric_card("今日检测", overview.get("today", 0), icon="📸")
    with c3:
        metric_card("无缺陷率", f"{overview.get('defect_free_rate', 0):.1%}", icon="🔍")
    with c4:
        metric_card("缺陷框总数", overview.get("defect_total", 0), icon="⚠️")

    st.markdown("---")
    col_chart1, col_chart2 = st.columns([3, 2])

    with col_chart1:
        st.markdown("#### 每日检测趋势")
        if trend:
            df_trend = pd.DataFrame(trend)
            rename = {k: cn(k) for k in df_trend.columns if k in ["crazing","inclusion","patches","pitted_surface","rolled_in_scale","scratches"]}
            rename["date"] = "日期"
            rename["total"] = "检测图片数"
            rename["defect_images"] = "有缺陷图片数"
            rename["defect_boxes"] = "缺陷框总数"
            rename["defect_free_images"] = "无缺陷图片数"
            df_trend = df_trend.rename(columns=rename)
            df_trend.set_index("日期", inplace=True)
            st.bar_chart(df_trend, use_container_width=True)
        else:
            st.info("暂无趋势数据")

    with col_chart2:
        st.markdown("#### 缺陷类型分布")
        if dist:
            df_dist = pd.DataFrame([{"类型": cn(d.get("type", "?")), "数量": d.get("count", 0)} for d in dist])
            st.bar_chart(df_dist.set_index("类型"), use_container_width=True, horizontal=True)
            for d in dist:
                t = d.get("type", "?")
                cnt = d.get("count", 0)
                pct = d.get("percentage", 0)
                bar = "█" * int(pct * 40)
                empty = "░" * (40 - int(pct * 40))
                st.markdown(
                    f'`{cn(t):<6}` '
                    f'<span style="color:{color(t)}">{bar}</span>{empty} '
                    f"**{cnt}** ({pct:.1%})",
                    unsafe_allow_html=True,
                )
        else:
            st.info("暂无分布数据")

    if trend:
        st.markdown("---")
        st.markdown("### 详细数据")
        df_detail = pd.DataFrame(trend)
        cols_rename = {}
        for c in df_detail.columns:
            cols_rename[c] = cn(c) if c in ["crazing","inclusion","patches","pitted_surface","rolled_in_scale","scratches"] else c
        cols_rename["date"] = "日期"
        cols_rename["total"] = "检测图片数"
        cols_rename["defect_images"] = "有缺陷图片数"
        cols_rename["defect_boxes"] = "缺陷框总数"
        cols_rename["defect_free_images"] = "无缺陷图片数"
        df_detail = df_detail.rename(columns=cols_rename)
        st.dataframe(df_detail, use_container_width=True, hide_index=True)
