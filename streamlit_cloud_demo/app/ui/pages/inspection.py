"""
图片检测页 — 上传 + 示例图片 + 推理 + 结果展示
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from app.ui.local_client import LocalClient as APIClient
from app.ui.components.annotated_image import draw_annotations
from app.ui.labels import cn, color, icon, cn_options, cn_to_en


# 示例图片目录 (相对于仓库根目录)
_SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "review_samples" / "images"


def _list_sample_images() -> list[Path]:
    """列出 review_samples/images/ 中的所有图片文件"""
    if not _SAMPLES_DIR.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(
        p for p in _SAMPLES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def render_inspection(api: APIClient) -> None:
    st.title("图片缺陷检测")
    st.caption("上传工业产品图像，或使用内置示例图片，自动检测缺陷")

    # ── 示例图片选择 ──
    sample_images = _list_sample_images()
    if sample_images:
        st.markdown("### 📁 使用示例图片")
        sample_names = [p.name for p in sample_images]
        selected_sample = st.selectbox(
            "选择示例图片（六类 NEU 缺陷样本）",
            options=["— 不上传，使用下方文件上传 —"] + sample_names,
            help="从 review_samples/images/ 中选择示例图片直接测试",
        )
        if selected_sample and not selected_sample.startswith("—"):
            sample_path = _SAMPLES_DIR / selected_sample
            if sample_path.exists():
                image_bytes = sample_path.read_bytes()
                filename = sample_path.name
                preview_img = Image.open(BytesIO(image_bytes)).convert("RGB")
                preview_img.thumbnail((400, 400))
                st.image(preview_img, caption=f"示例: {filename}", width=400)

                if st.button("检测示例图片", type="primary", use_container_width=True):
                    with st.spinner("检测中 ..."):
                        result = api.predict(image_bytes, filename, 0.25)

                    if result.get("status") == "error":
                        st.error(f"检测失败: {result.get('error', {}).get('message', 'unknown')}")
                    else:
                        st.session_state["last_result"] = result.get("data", {})
                        st.session_state["last_preview"] = Image.open(BytesIO(image_bytes)).convert("RGB")
                st.markdown("---")
    else:
        st.info(
            "💡 **提示**: 将六类 NEU 缺陷样本图片放入 `review_samples/images/` 目录，"
            "即可在检测页直接选择示例图片测试。"
        )

    # ── 文件上传 ──
    st.markdown("### 📤 上传图片")
    col_upload, col_param = st.columns([3, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "选择或拖拽图像文件",
            type=["jpg", "jpeg", "png", "bmp"],
            help="支持 JPG / PNG / BMP 格式",
        )

    with col_param:
        confidence = st.slider(
            "置信度阈值",
            min_value=0.0, max_value=1.0, value=0.25, step=0.05,
            help="低于此阈值的检测结果将被过滤",
        )

    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        filename = uploaded.name or "image.jpg"
        preview_img = Image.open(BytesIO(image_bytes)).convert("RGB")
        preview_img.thumbnail((400, 400))

        if st.button("开始检测", type="primary", use_container_width=True):
            with st.spinner("检测中 ..."):
                result = api.predict(image_bytes, filename, confidence)

            if result.get("status") == "error":
                st.error(f"检测失败: {result.get('error', {}).get('message', 'unknown')}")
            else:
                st.session_state["last_result"] = result.get("data", {})
                st.session_state["last_preview"] = preview_img

    if "last_result" in st.session_state:
        data = st.session_state["last_result"]
        preview = st.session_state.get("last_preview")
        _render_result(data, preview)


def _render_result(data: dict, original_image: Image.Image | None = None) -> None:
    st.markdown("---")
    st.markdown("### 检测结果")

    col_img, col_info = st.columns([1, 1])

    with col_img:
        if original_image:
            defects = data.get("defects", [])
            annotated = draw_annotations(original_image, defects)
            st.image(annotated, caption="标注结果", use_container_width=True)
        else:
            st.info("请上传图像")

    with col_info:
        st.metric("检测到缺陷", data.get("total_defects", 0))
        st.metric("推理耗时", f"{data.get('inference_time_ms', 0):.0f} ms")
        st.caption(f"模型: {data.get('model', '?')}")
        if data.get("is_defect_free", True):
            st.success("该产品无缺陷")
        else:
            st.warning(f"检测到 {data.get('total_defects', 0)} 个缺陷")

    defects = data.get("defects", [])
    if defects:
        st.markdown("---")
        cols = st.columns(min(len(defects), 3))
        for i, d in enumerate(defects):
            with cols[i % len(cols)]:
                with st.container(border=True):
                    cls_name = d.get("class_name", "?")
                    conf = d.get("confidence", 0)
                    bbox = d.get("bbox", {})
                    st.markdown(f"{icon(cls_name)} **{cn(cls_name)}**")
                    st.progress(min(float(conf), 1.0), text=f"{conf:.2%}")
                    st.caption(
                        f"x={bbox.get('x',0):.3f} y={bbox.get('y',0):.3f} "
                        f"w={bbox.get('w',0):.3f} h={bbox.get('h',0):.3f}"
                    )
