"""
Streamlit Cloud 演示版入口

启动:
  streamlit run streamlit_app.py

架构:
  UI Page → LocalClient → Service → Core Engine / DB Repository
  无 FastAPI / HTTP 依赖，单一 Streamlit 进程。
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── 路径修复 ──
# 确保 streamlit_cloud_demo/ 在 sys.path 中，
# 同时避免脚本目录遮蔽 app 包。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEMO_DIR = _PROJECT_ROOT / "streamlit_cloud_demo"
_SCRIPT_DIR = str(Path(__file__).resolve().parent)

# 移除脚本目录避免遮蔽
sys.path = [p for p in sys.path if Path(p).resolve() != Path(_SCRIPT_DIR).resolve()]
# 确保项目根和 demo 目录在 path 中
for p in (str(_PROJECT_ROOT), str(_DEMO_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

from app.core.neu_classes import CANONICAL_NAMES
from app.ui.local_client import LocalClient
from app.ui.components.sidebar import render_sidebar
from app.ui.pages.home import render_home
from app.ui.pages.inspection import render_inspection
from app.ui.pages.history import render_history
from app.ui.pages.statistics import render_statistics


# ── 模型路径 ──
_MODEL_PATH = _PROJECT_ROOT / "models" / "neu" / "best.pt"


@st.cache_resource
def get_local_client() -> LocalClient:
    """
    使用 st.cache_resource 缓存 LocalClient，
    确保模型只加载一次，所有用户会话共享。
    """
    return LocalClient(
        model_path=str(_MODEL_PATH),
        db_path="/tmp/inspection.db",
        device="cpu",
        class_names=CANONICAL_NAMES,
        confidence_threshold=0.25,
    )


def main() -> None:
    """Streamlit Cloud 演示版主入口"""

    # ── 页面配置 ──
    st.set_page_config(
        page_title="钢板表面缺陷检测平台",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── 初始化本地客户端 (cached) ──
    client = get_local_client()

    # ── 侧边栏 ──
    page = render_sidebar(client)

    # ── 页面路由 ──
    try:
        if page == "home":
            render_home(client)
        elif page == "inspection":
            render_inspection(client)
        elif page == "history":
            render_history(client)
        elif page == "statistics":
            render_statistics(client)
    except Exception as e:
        st.error(f"页面加载异常: {e}")
        st.info("请确认模型文件存在: models/neu/best.pt")


if __name__ == "__main__":
    main()
