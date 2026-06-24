"""
streamlit_app.py — Streamlit Community Cloud 入口

Streamlit Cloud 自动检测仓库根目录下的 streamlit_app.py 作为应用入口。
部署时无需额外配置，Streamlit Cloud 会自动安装 requirements.txt 中的依赖。

架构:
  Streamlit Cloud 单一进程 → LocalClient → Service → Core / DB
  无 FastAPI / Uvicorn 后台进程。
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── 将 streamlit_cloud_demo/ 加入 sys.path ──
_REPO_ROOT = Path(__file__).resolve().parent
_DEMO_DIR = _REPO_ROOT / "streamlit_cloud_demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

# ── 导入并启动云端演示版应用 ──
from app.ui.app import main

if __name__ == "__main__":
    main()
