"""
InspectionService — 检测编排服务

职责:
  1. 图像预处理 (bytes → PIL)
  2. 调用 Core Engine 推理
  3. 结果持久化到 Repository
  4. 返回结构化结果

不含: HTTP 处理 / AI 逻辑 / SQL
"""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image

from app.core.base import (
    BBoxNorm,
    Defect,
    DetectionResult,
    DetectorBase,
)
from app.db.repository import BBoxRecord, DetectionRecord, DetectionRepository
from app.db.repository import InspectionRecord


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


@dataclass
class PredictResult:
    """检测服务返回值 (非 Pydantic, 不含 HTTP 概念)"""
    inspection_id: str
    image_id: str
    image_size: tuple[int, int]
    defects: list[Defect]
    total_defects: int
    is_defect_free: bool
    distribution: dict[str, int]
    inference_time_ms: float
    model: str
    saved_to_db: bool
    saved_records: list[DetectionRecord] = field(default_factory=list)


class InspectionService:
    """
    缺陷检测编排服务

    依赖 (构造函数注入):
      - detector: DetectorBase
      - repository: DetectionRepository
    """

    def __init__(
        self,
        detector: DetectorBase,
        repository: DetectionRepository,
    ) -> None:
        self._detector = detector
        self._repo = repository

    # --------------------------------------------------------
    # 检测
    # --------------------------------------------------------

    def predict(
        self,
        image_bytes: bytes,
        filename: str,
        confidence: float = 0.25,
    ) -> PredictResult:
        """
        执行缺陷检测

        Args:
            image_bytes: 图像文件原始字节
            filename: 原始文件名
            confidence: 置信度阈值

        Returns:
            PredictResult

        Raises:
            ValueError: 文件格式不支持 / 文件过大
            RuntimeError: 推理失败
        """
        # 1. 校验
        self._validate_file(image_bytes, filename)

        # 2. bytes → PIL.Image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"无法读取图像: {e}")

        # 3. 暂时覆盖置信度阈值
        original_conf = self._detector.confidence_threshold
        self._detector.confidence_threshold = confidence
        try:
            detection: DetectionResult = self._detector.detect(image)
        finally:
            self._detector.confidence_threshold = original_conf

        # 4. 生成 image_id
        image_id = self._generate_image_id(filename)
        inspection_id = image_id

        # 5. 持久化。无缺陷图片也必须保存 inspection 级记录。
        saved_records = self._persist_defects(
            inspection_id=inspection_id,
            image_id=image_id,
            original_filename=filename,
            detection=detection,
        )

        return PredictResult(
            inspection_id=inspection_id,
            image_id=image_id,
            image_size=detection.image_size,
            defects=detection.defects,
            total_defects=detection.total_defects,
            is_defect_free=detection.is_defect_free,
            distribution=detection.defect_distribution,
            inference_time_ms=detection.inference_time_ms,
            model=detection.model_name,
            saved_to_db=True,
            saved_records=saved_records,
        )

    def predict_batch(
        self,
        images: list[tuple[bytes, str]],
        confidence: float = 0.25,
    ) -> list[PredictResult]:
        """批量检测"""
        results: list[PredictResult] = []
        for image_bytes, filename in images:
            results.append(self.predict(image_bytes, filename, confidence))
        return results

    # --------------------------------------------------------
    # 内部
    # --------------------------------------------------------

    @staticmethod
    def _validate_file(data: bytes, filename: str) -> None:
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件格式: {ext or '未知'}。"
                f"支持: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        if len(data) > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"文件过大: {len(data) / 1024 / 1024:.1f}MB。"
                f"限制: {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f}MB"
            )

    def _persist_defects(
        self,
        inspection_id: str,
        image_id: str,
        original_filename: str,
        detection: DetectionResult,
    ) -> list[DetectionRecord]:
        """DetectionResult → DetectionRecord 列表 → 持久化"""
        now = DetectionRecord.now_iso()
        self._repo.save_inspection(
            InspectionRecord(
                inspection_id=inspection_id,
                image_name=original_filename,
                detect_time=now,
                total_defects=detection.total_defects,
                model=detection.model_name,
            )
        )

        if not detection.defects:
            return []

        records = [
            DetectionRecord(
                image_name=image_id,
                defect_type=d.class_name,
                confidence=d.confidence,
                bbox=BBoxRecord(
                    x=d.bbox.x_center,
                    y=d.bbox.y_center,
                    w=d.bbox.width,
                    h=d.bbox.height,
                ),
                detect_time=now,
                inspection_id=inspection_id,
            )
            for d in detection.defects
        ]

        return self._repo.save_batch(records)

    @staticmethod
    def _generate_image_id(filename: str) -> str:
        """生成唯一 image_id"""
        uid = uuid.uuid4().hex[:12]
        safe_name = "".join(c for c in filename if c.isalnum() or c in "._-")[:48]
        return f"{safe_name}_{uid}"
