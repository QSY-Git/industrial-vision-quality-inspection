"""
streamlit_app.py — Streamlit Community Cloud 入口

将此目录作为 GitHub 仓库根目录上传后，Streamlit Cloud 自动
识别此文件作为应用入口。

启动:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── 仓库根 = 此文件所在目录 ──
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── 导入并启动 ──
from app.ui.app import main

if __name__ == "__main__":
    main()
