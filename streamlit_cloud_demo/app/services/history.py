"""
HistoryService — 历史查询编排服务

职责:
  1. 将查询参数转换为 Repository 调用
  2. 分页计算
  3. 返回结构化结果

不含: HTTP 处理 / SQL
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.db.repository import (
    BBoxRecord,
    DetectionFilters,
    DetectionRecord,
    DetectionRepository,
)


@dataclass
class HistoryResult:
    """历史查询服务返回值"""
    total: int
    page: int
    size: int
    items: list[DetectionRecord]

    @property
    def total_pages(self) -> int:
        return max(1, math.ceil(self.total / self.size))


class HistoryService:
    """
    历史查询编排服务

    依赖:
      - repository: DetectionRepository
    """

    def __init__(self, repository: DetectionRepository) -> None:
        self._repo = repository

    def query(
        self,
        page: int = 1,
        size: int = 20,
        image_name: str | None = None,
        defect_type: str | None = None,
        min_confidence: float | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> HistoryResult:
        """
        分页查询检测历史

        Args:
            page: 页码 (1-based)
            size: 每页条数
            image_name: 图像名筛选
            defect_type: 缺陷类型筛选
            min_confidence: 最小置信度
            date_from: 起始日期
            date_to: 终止日期

        Returns:
            HistoryResult
        """
        # 构建筛选器
        filters = DetectionFilters(
            image_name=image_name,
            defect_type=defect_type,
            min_confidence=min_confidence,
            date_from=date_from,
            date_to=date_to,
            limit=size,
            offset=(page - 1) * size,
        )

        # 查询当前页
        items = self._repo.search(filters)

        # 查询总数 (用无分页的 search)
        count_filters = DetectionFilters(
            image_name=image_name,
            defect_type=defect_type,
            min_confidence=min_confidence,
            date_from=date_from,
            date_to=date_to,
            limit=100000,
            offset=0,
        )
        total = len(self._repo.search(count_filters))

        return HistoryResult(
            total=total,
            page=page,
            size=size,
            items=items,
        )

    def get_by_id(self, record_id: int) -> DetectionRecord | None:
        """按 ID 查询单条"""
        return self._repo.find_by_id(record_id)

    def delete(self, record_id: int) -> bool:
        """删除单条"""
        return self._repo.delete_by_id(record_id)
