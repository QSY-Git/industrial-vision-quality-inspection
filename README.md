# 钢板表面缺陷检测平台 — Streamlit Cloud 演示版

基于 YOLO11 的 NEU-DET 六类钢板表面缺陷检测与质量分析系统。

## 本地安装

```bash
pip install -r requirements.txt
```

## 本地启动

```bash
streamlit run streamlit_app.py
```

## Streamlit Cloud 部署

将此目录作为 GitHub 仓库根目录上传，Streamlit Cloud 自动完成部署：

1. 创建 GitHub 仓库，将本目录内容推送到仓库根目录
2. 在 [share.streamlit.io](https://share.streamlit.io) 连接仓库
3. **入口文件**: `streamlit_app.py`（自动识别）
4. **Python 版本**: 3.11

## 在线数据库说明

⚠️ **在线演示使用临时数据库** (`/tmp/inspection.db`)。应用休眠、重启或重新部署后，所有检测记录将被清空。

## 目录结构

```
<repo>/
├── streamlit_app.py          ← Streamlit Cloud 入口
├── requirements.txt          ← Python 依赖
├── packages.txt              ← 系统级依赖
├── models/
│   └── neu/
│       └── best.pt           ← YOLO11 权重
├── review_samples/
│   └── images/               ← 放入示例图片即可在检测页选择
└── app/
    ├── core/                 ← 推理引擎
    ├── db/                   ← SQLite Repository
    ├── services/             ← 业务编排
    └── ui/                   ← Streamlit 前端
```

## 六类 NEU 缺陷

| 英文 | 中文 | 颜色 |
|---|---|---|
| crazing | 龟裂 | 🔴 |
| inclusion | 夹杂 | 🟣 |
| patches | 斑块 | 🟠 |
| pitted_surface | 麻点 | 🔵 |
| rolled_in_scale | 氧化皮 | ⚙️ |
| scratches | 划痕 | 🟢 |
