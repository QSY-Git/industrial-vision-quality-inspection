"""
统一侧边栏 — 导航 + 模型状态 (云端演示版)

与原始 sidebar 的区别:
  - 无 API URL 输入框（本地推理，不依赖 HTTP API）
  - 显示本地模型状态
  - 提示在线演示数据为临时存储
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_sidebar(client) -> str:
    """
    渲染侧边栏，返回选中的页面名称。

    参数 client 为 LocalClient 实例（兼容 APIClient 的 health/is_connected 方法）。

    Returns:
        "home" | "inspection" | "history" | "statistics"
    """
    with st.sidebar:
        st.markdown("## 🏭 工业视觉质检")
        st.caption("云端演示版 — 本地推理")

        # ── 导航 ──
        st.markdown("---")
        page = st.radio(
            "导航",
            options=["🏠 首页", "📸 图片检测", "📋 历史记录", "📊 统计分析"],
            format_func=lambda x: x,
            label_visibility="collapsed",
        )

        # ── 模型状态 ──
        st.markdown("---")
        st.markdown("### ⚙️ 运行状态")

        if client.is_connected():
            st.success("🟢 模型就绪")
            try:
                health = client.health()
                loaded = health.get("model_loaded", False)
                model_path = health.get("model_path", "")
                if loaded:
                    st.info(f"🤖 已加载: {Path(model_path).name if model_path else 'N/A'}")
                else:
                    st.warning("⚠️ 模型未加载")
            except Exception:
                st.warning("⚠️ 状态获取失败")
        else:
            st.error("🔴 模型未就绪")

        # ── 数据库说明 ──
        st.markdown("---")
        st.markdown("### 💾 数据说明")
        st.caption(
            "在线演示使用临时数据库 `/tmp/inspection.db`。"
            "应用休眠、重启或重新部署后，检测记录将被清空。"
        )

        # ── 底部 ──
        st.markdown("---")
        st.caption("v1.0.0 | Streamlit Cloud Demo")

    # 映射页面名
    page_map = {
        "🏠 首页": "home",
        "📸 图片检测": "inspection",
        "📋 历史记录": "history",
        "📊 统计分析": "statistics",
    }
    return page_map.get(page, "home")
