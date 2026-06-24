"""
历史记录页 — 分页查询 + 筛选
"""

from __future__ import annotations

import streamlit as st

from app.ui.local_client import LocalClient as APIClient
from app.ui.components.defect_card import defect_card
from app.ui.labels import cn, cn_options, cn_to_en


def render_history(api: APIClient) -> None:
    st.title("检测历史")
    st.caption("浏览和筛选所有检测记录")

    st.markdown("### 筛选条件")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        cn_selected = st.selectbox("缺陷类型", options=cn_options(), index=0)

    with c2:
        image_name = st.text_input("图像名称", placeholder="输入图像名...")

    with c3:
        min_confidence = st.number_input("最小置信度", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

    with c4:
        page_size = st.selectbox("每页条数", options=[10, 20, 50], index=1)

    page = st.session_state.get("history_page", 1)
    query_defect = cn_to_en(cn_selected)
    query_image = image_name.strip() or None
    query_conf = min_confidence if min_confidence > 0 else None

    col_s, col_r = st.columns([1, 5])
    with col_s:
        if st.button("搜索", type="primary"):
            st.session_state["history_page"] = 1
            st.rerun()
    with col_r:
        if st.button("重置"):
            st.session_state["history_page"] = 1
            st.rerun()

    try:
        result = api.history(
            page=page, size=page_size, image_name=query_image,
            defect_type=query_defect, min_confidence=query_conf,
        )
        if result.get("status") == "error":
            st.error(result.get("error", {}).get("message", "查询失败"))
            return

        data = result.get("data", {})
        items = data.get("items", [])
        total = data.get("total", 0)

        st.markdown("---")
        total_pages = max(1, (total + page_size - 1) // page_size)
        st.caption(f"共 **{total}** 条 | 第 **{page}**/**{total_pages}** 页")

        if not items:
            st.info("暂无记录")
        else:
            for item in items:
                defect_card(item, api=api)

        if total > page_size:
            st.markdown("---")
            p1, p2, p3, p4, p5 = st.columns([1, 1, 2, 1, 1])
            with p1:
                if st.button("首页", disabled=page <= 1):
                    st.session_state["history_page"] = 1; st.rerun()
            with p2:
                if st.button("上一页", disabled=page <= 1):
                    st.session_state["history_page"] = max(1, page - 1); st.rerun()
            with p3:
                st.markdown(f"<div style='text-align:center'>{page}/{total_pages}</div>", unsafe_allow_html=True)
            with p4:
                if st.button("下一页", disabled=page >= total_pages):
                    st.session_state["history_page"] = min(total_pages, page + 1); st.rerun()
            with p5:
                if st.button("末页", disabled=page >= total_pages):
                    st.session_state["history_page"] = total_pages; st.rerun()

    except Exception as e:
        st.error(f"无法连接 API: {e}")
