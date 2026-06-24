"""
LocalClient — Streamlit Cloud 本地客户端

替代 APIClient，直接调用本地 Service 层，无需 FastAPI HTTP 服务。
对外接口与 APIClient 完全兼容，页面无需修改。

架构:
  UI Page → LocalClient → Service → Core Engine / DB Repository
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.base import DetectorBase
from app.core.factory import create_detector
from app.core.neu_classes import CANONICAL_NAMES
from app.db import create_repository
from app.db.repository import DetectionRepository
from app.services.inspection import InspectionService
from app.services.history import HistoryService
from app.services.statistics import StatisticsService


class LocalClient:
    """
    本地推理客户端 — 兼容 APIClient 接口

    Usage:
        client = LocalClient(
            model_path="models/neu/best.pt",
            db_path="/tmp/inspection.db",
            device="cpu",
        )
        health = client.health()
        result = client.predict(image_bytes, "img.jpg", 0.25)
        history = client.history(page=1, size=20)
        stats = client.stats(days=7)
    """

    def __init__(
        self,
        model_path: str = "models/neu/best.pt",
        db_path: str = "/tmp/inspection.db",
        device: str = "cpu",
        class_names: list[str] | None = None,
        confidence_threshold: float = 0.25,
    ) -> None:
        self._model_path = str(Path(model_path))
        self._db_path = str(Path(db_path))
        self._device = device
        self._class_names = class_names or CANONICAL_NAMES

        # ── 数据层 ──
        self._repository: DetectionRepository = create_repository(self._db_path)

        # ── 推理引擎 ──
        self._detector: DetectorBase = create_detector(
            "yolo",
            device=self._device,
            class_names=self._class_names,
            confidence_threshold=confidence_threshold,
        )
        self._load_model()

        # ── Service 层 ──
        self._inspection_service = InspectionService(
            detector=self._detector,
            repository=self._repository,
        )
        self._history_service = HistoryService(repository=self._repository)
        self._statistics_service = StatisticsService(repository=self._repository)

    # ============================================================
    # 健康检查
    # ============================================================

    def health(self) -> dict[str, Any]:
        """返回与 FastAPI /health 相同的字典结构"""
        return {
            "status": "healthy",
            "model_loaded": self._detector.is_loaded,
            "db_initialized": True,
            "model_path": self._detector.model_info.weights_path,
            "class_names": self._detector.model_info.class_names,
        }

    # ============================================================
    # 检测
    # ============================================================

    def predict(
        self,
        image_bytes: bytes,
        filename: str,
        confidence: float = 0.25,
    ) -> dict[str, Any]:
        """
        执行缺陷检测 — 返回与 POST /api/v1/predict 相同的字典结构
        """
        try:
            result = self._inspection_service.predict(image_bytes, filename, confidence)
        except ValueError as e:
            return {
                "status": "error",
                "error": {"code": "INVALID_FILE", "message": str(e)},
            }
        except RuntimeError as e:
            return {
                "status": "error",
                "error": {"code": "INFERENCE_ERROR", "message": str(e)},
            }
        except Exception as e:
            return {
                "status": "error",
                "error": {"code": "INTERNAL_ERROR", "message": f"检测失败: {e}"},
            }

        return {
            "status": "success",
            "data": {
                "inspection_id": result.inspection_id,
                "image_id": result.image_id,
                "image_size": {
                    "width": result.image_size[0],
                    "height": result.image_size[1],
                },
                "defects": [
                    {
                        "class_name": d.class_name,
                        "confidence": d.confidence,
                        "bbox": {
                            "x": d.bbox.x_center,
                            "y": d.bbox.y_center,
                            "w": d.bbox.width,
                            "h": d.bbox.height,
                        },
                    }
                    for d in result.defects
                ],
                "total_defects": result.total_defects,
                "is_defect_free": result.is_defect_free,
                "distribution": result.distribution,
                "inference_time_ms": result.inference_time_ms,
                "model": result.model,
                "saved_to_db": result.saved_to_db,
            },
        }

    # ============================================================
    # 历史
    # ============================================================

    def history(
        self,
        page: int = 1,
        size: int = 20,
        image_name: str | None = None,
        defect_type: str | None = None,
        min_confidence: float | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        """
        分页查询检测历史 — 返回与 GET /api/v1/history 相同的字典结构
        """
        try:
            result = self._history_service.query(
                page=page,
                size=size,
                image_name=image_name,
                defect_type=defect_type,
                min_confidence=min_confidence,
                date_from=date_from,
                date_to=date_to,
            )
        except Exception as e:
            return {
                "status": "error",
                "error": {"code": "QUERY_ERROR", "message": str(e)},
            }

        return {
            "status": "success",
            "data": {
                "total": result.total,
                "page": result.page,
                "size": result.size,
                "items": [
                    {
                        "id": r.id or 0,
                        "inspection_id": r.inspection_id,
                        "image_name": r.image_name,
                        "defect_type": r.defect_type,
                        "confidence": r.confidence,
                        "bbox": {
                            "x": r.bbox.x,
                            "y": r.bbox.y,
                            "w": r.bbox.w,
                            "h": r.bbox.h,
                        },
                        "detect_time": r.detect_time,
                    }
                    for r in result.items
                ],
            },
        }

    # ============================================================
    # 统计
    # ============================================================

    def stats(self, days: int = 7) -> dict[str, Any]:
        """
        获取统计概览 — 返回与 GET /api/v1/stats 相同的字典结构
        """
        try:
            result = self._statistics_service.get_stats(days=days)
        except Exception as e:
            return {
                "status": "error",
                "error": {"code": "STATS_ERROR", "message": str(e)},
            }

        return {
            "status": "success",
            "data": {
                "overview": {
                    "total": result.total,
                    "today": result.today,
                    "today_defect_total": result.today_defect_total,
                    "today_defect_images": result.today_defect_images,
                    "today_defect_free_images": result.today_defect_free_images,
                    "defect_total": result.defect_total,
                    "defect_images": result.defect_images,
                    "defect_free_images": result.defect_free_images,
                    "defect_free_rate": result.defect_free_rate,
                },
                "defect_distribution": [
                    {
                        "type": d.type,
                        "count": d.count,
                        "percentage": d.percentage,
                    }
                    for d in result.defect_distribution
                ],
                "daily_trend": [
                    {
                        "date": d.date,
                        "total": d.total,
                        "defect_images": d.defect_images,
                        "defect_boxes": d.defect_boxes,
                        "defect_free_images": d.defect_free_images,
                        "crazing": d.crazing,
                        "inclusion": d.inclusion,
                        "patches": d.patches,
                        "pitted_surface": d.pitted_surface,
                        "rolled_in_scale": d.rolled_in_scale,
                        "scratches": d.scratches,
                    }
                    for d in result.daily_trend
                ],
            },
        }

    # ============================================================
    # 模型管理
    # ============================================================

    def load_model(self, weights_path: str | None = None) -> dict[str, Any]:
        """
        加载/切换模型 — 返回与 POST /api/v1/models/load 相同的字典结构
        """
        path = weights_path or self._model_path
        p = Path(path)
        if not p.exists():
            return {
                "status": "error",
                "message": f"模型文件不存在: {path}",
            }
        try:
            self._detector.load(str(p))
            self._model_path = str(p)
            return {
                "status": "loaded",
                "model": self._detector.model_info.name,
                "class_names": self._detector.model_info.class_names,
                "weights_path": self._detector.model_info.weights_path,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"模型加载失败: {e}",
            }

    # ============================================================
    # 工具
    # ============================================================

    def is_connected(self) -> bool:
        """本地客户端始终可用"""
        return self._detector.is_loaded

    # ============================================================
    # 内部
    # ============================================================

    def _load_model(self) -> None:
        """加载模型权重"""
        p = Path(self._model_path)
        if p.exists():
            self._detector.load(str(p))
