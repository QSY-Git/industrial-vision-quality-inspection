"""
Repository 抽象接口 & 数据传输对象 (DTO)

Repository 模式:
  - 业务层只依赖 DetectionRepository 抽象
  - 具体实现 (SQLite) 与业务层完全解耦
  - 数据通过 DTO 传输, 不含任何 SQL
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ============================================================
# DTO (Data Transfer Objects)
# ============================================================

@dataclass
class BBoxRecord:
    """标注边界框 (归一化坐标)"""
    x: float     # x_center 0~1
    y: float     # y_center 0~1
    w: float     # width    0~1
    h: float     # height   0~1

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @classmethod
    def from_dict(cls, d: dict) -> BBoxRecord:
        return cls(
            x=float(d.get("x", 0)),
            y=float(d.get("y", 0)),
            w=float(d.get("w", 0)),
            h=float(d.get("h", 0)),
        )


@dataclass
class DetectionRecord:
    """检测记录 DTO — Repository 与业务层之间的传输对象"""
    image_name: str
    defect_type: str
    confidence: float
    bbox: BBoxRecord
    detect_time: str           # ISO 8601
    inspection_id: str | None = None
    id: int | None = None
    created_at: str | None = None

    @classmethod
    def from_row(cls, row: dict) -> DetectionRecord:
        """从数据库行字典构造"""
        import json
        bbox_data = json.loads(row["bbox"]) if isinstance(row["bbox"], str) else row["bbox"]
        return cls(
            id=row.get("id"),
            image_name=row["image_name"],
            defect_type=row["defect_type"],
            confidence=row["confidence"],
            bbox=BBoxRecord.from_dict(bbox_data),
            detect_time=row["detect_time"],
            inspection_id=row.get("inspection_id"),
            created_at=row.get("created_at"),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "image_name": self.image_name,
            "defect_type": self.defect_type,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict(),
            "detect_time": self.detect_time,
            "inspection_id": self.inspection_id,
            "created_at": self.created_at,
        }

    @staticmethod
    def now_iso() -> str:
        """生成当前 ISO 8601 时间戳"""
        return datetime.now(timezone.utc).isoformat()


@dataclass
class DefectTypeCount:
    """缺陷类型统计"""
    defect_type: str
    count: int


@dataclass
class InspectionRecord:
    """一次图片检测请求记录。"""
    inspection_id: str
    image_name: str
    detect_time: str
    total_defects: int
    model: str = ""
    is_legacy: bool = False
    created_at: str | None = None


@dataclass
class DailyStats:
    """每日统计"""
    date: str
    total: int
    defect_images: int = 0
    defect_boxes: int = 0
    defect_free_images: int = 0
    by_type: dict[str, int] = field(default_factory=dict)


@dataclass
class StatsSummary:
    """Inspection-level summary statistics."""
    inspected_images: int
    defect_boxes: int
    defect_images: int
    defect_free_images: int


@dataclass
class DetectionFilters:
    """查询筛选条件"""
    image_name: str | None = None
    defect_type: str | None = None
    min_confidence: float | None = None
    max_confidence: float | None = None
    date_from: str | None = None
    date_to: str | None = None
    limit: int = 100
    offset: int = 0


# ============================================================
# Repository 抽象接口
# ============================================================

class DetectionRepository(ABC):
    """
    检测记录 Repository 抽象接口

    所有数据访问必须通过此接口，禁止业务层直接操作数据库。
    """

    # ---- 写入 ----

    @abstractmethod
    def save(self, record: DetectionRecord) -> DetectionRecord:
        """
        保存单条检测记录

        Args:
            record: 检测记录 (id=None 则新建)

        Returns:
            带 id 的记录
        """
        ...

    @abstractmethod
    def save_batch(self, records: list[DetectionRecord]) -> list[DetectionRecord]:
        """
        批量保存 (单事务)

        Args:
            records: 检测记录列表

        Returns:
            带 id 的记录列表
        """
        ...

    @abstractmethod
    def save_inspection(self, inspection: InspectionRecord) -> InspectionRecord:
        """保存一次图片检测请求。"""
        ...

    # ---- 查询 ----

    @abstractmethod
    def find_by_id(self, record_id: int) -> DetectionRecord | None:
        """按 ID 查询"""
        ...

    @abstractmethod
    def find_by_image(self, image_name: str) -> list[DetectionRecord]:
        """
        按图像名查询全部缺陷

        Args:
            image_name: 图像文件名

        Returns:
            该图像的所有缺陷记录
        """
        ...

    @abstractmethod
    def find_by_defect_type(
        self, defect_type: str, limit: int = 100
    ) -> list[DetectionRecord]:
        """按缺陷类型查询"""
        ...

    @abstractmethod
    def find_recent(self, limit: int = 20) -> list[DetectionRecord]:
        """查询最近记录"""
        ...

    @abstractmethod
    def search(self, filters: DetectionFilters) -> list[DetectionRecord]:
        """组合条件筛选"""
        ...

    # ---- 统计 ----

    @abstractmethod
    def count_by_type(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[DefectTypeCount]:
        """按缺陷类型统计数量"""
        ...

    @abstractmethod
    def stats_summary(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> StatsSummary:
        """按检测请求统计图片数、缺陷图片数、无缺陷图片数和缺陷框数。"""
        ...

    @abstractmethod
    def daily_stats(self, days: int = 7) -> list[DailyStats]:
        """每日统计"""
        ...

    @abstractmethod
    def total_count(self) -> int:
        """记录总数"""
        ...

    # ---- 删除 ----

    @abstractmethod
    def delete_by_id(self, record_id: int) -> bool:
        """
        删除单条记录

        Returns:
            True 删除成功, False 记录不存在
        """
        ...

    @abstractmethod
    def delete_older_than(self, before_date: str) -> int:
        """
        删除过期记录

        Args:
            before_date: ISO 8601 日期，此日期之前的记录将被删除

        Returns:
            删除的记录数
        """
        ...
