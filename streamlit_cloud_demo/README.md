# 钢板表面缺陷检测平台 — Streamlit Cloud 演示版

基于 YOLO11 的 NEU-DET 六类钢板表面缺陷检测与质量分析系统云端演示版。

## 本地安装

```bash
# 1. 克隆仓库
git clone <repo-url>
cd <repo>

# 2. 安装依赖 (Python 3.11)
pip install -r streamlit_cloud_demo/requirements.txt

# 3. 确认模型文件存在
ls -lh models/neu/best.pt
```

## 本地启动

```bash
streamlit run streamlit_app.py
```

应用将在 `http://localhost:8501` 启动。

## Streamlit Cloud 部署

1. 将仓库推送到 GitHub
2. 在 [share.streamlit.io](https://share.streamlit.io) 中连接 GitHub 仓库
3. Streamlit Cloud 自动识别根目录的 `streamlit_app.py` 作为入口
4. **入口文件**: `streamlit_app.py`
5. **Python 版本**: 3.11

## 在线数据库说明

在线演示使用 **临时数据库** (`/tmp/inspection.db`)。应用休眠、重启或重新部署后，所有检测记录将被清空。

## 项目结构 (部署版)

```
streamlit_cloud_demo/
├── app/
│   ├── core/              # 推理引擎 (YOLODetector)
│   ├── db/                # SQLite Repository
│   ├── services/          # 业务编排 (Inspection/History/Statistics)
│   └── ui/
│       ├── app.py         # Streamlit 入口
│       ├── local_client.py # 本地客户端 (替代 HTTP API)
│       ├── components/    # UI 组件
│       └── pages/         # 页面 (首页/检测/历史/统计)
├── review_samples/
│   └── images/            # 示例图片目录
├── requirements.txt
├── packages.txt
└── README.md
```

## 技术架构

```
Streamlit UI → LocalClient → InspectionService/HistoryService/StatisticsService
                                   ↓
                            YOLODetector + SQLite Repository
```

- 单一 Streamlit 进程，无 FastAPI/Uvicorn 后台服务
- `st.cache_resource` 缓存模型，所有会话共享
- CPU 推理，模型路径: `models/neu/best.pt`
- 六类 NEU 缺陷: 龟裂 / 夹杂 / 斑块 / 麻点 / 氧化皮 / 划痕
