"""
StatisticsService — 统计编排服务

职责:
  1. 聚合多个 Repository 查询
  2. 计算百分比 / 缺陷率
  3. 返回结构化统计结果

不含: HTTP 处理 / SQL
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.db.repository import (
    DefectTypeCount,
    DailyStats,
    DetectionRepository,
)
from app.core.neu_classes import CANONICAL_NAMES


@dataclass
class DefectDistribution:
    type: str
    count: int
    percentage: float


@dataclass
class DailyTrend:
    date: str
    total: int
    defect_images: int = 0
    defect_boxes: int = 0
    defect_free_images: int = 0
    crazing: int = 0
    inclusion: int = 0
    patches: int = 0
    pitted_surface: int = 0
    rolled_in_scale: int = 0
    scratches: int = 0


@dataclass
class StatsResult:
    """统计服务返回值"""
    total: int
    today: int
    today_defect_total: int
    today_defect_images: int
    today_defect_free_images: int
    defect_total: int
    defect_images: int
    defect_free_images: int
    defect_free_rate: float
    defect_distribution: list[DefectDistribution]
    daily_trend: list[DailyTrend]


class StatisticsService:
    """
    统计编排服务

    依赖:
      - repository: DetectionRepository
    """

    ALL_DEFECT_TYPES = CANONICAL_NAMES

    def __init__(self, repository: DetectionRepository) -> None:
        self._repo = repository

    def get_stats(self, days: int = 7) -> StatsResult:
        """
        获取统计概览

        Args:
            days: 统计最近 N 天

        Returns:
            StatsResult
        """
        # 1. inspection-level overview
        summary = self._repo.stats_summary()
        total = summary.inspected_images

        # 2. 缺陷类型分布
        type_counts = self._repo.count_by_type()
        distribution = self._build_distribution(type_counts)

        # 3. 每日趋势
        daily = self._repo.daily_stats(days=days)
        trend = self._build_trend(daily)

        # 4. 今日 + 无缺陷率
        today = trend[-1] if trend else None
        today_count = today.total if today else 0
        defect_free_rate = (
            summary.defect_free_images / total
            if total > 0 else 0.0
        )

        return StatsResult(
            total=total,
            today=today_count,
            today_defect_total=today.defect_boxes if today else 0,
            today_defect_images=today.defect_images if today else 0,
            today_defect_free_images=today.defect_free_images if today else 0,
            defect_total=summary.defect_boxes,
            defect_images=summary.defect_images,
            defect_free_images=summary.defect_free_images,
            defect_free_rate=round(defect_free_rate, 4),
            defect_distribution=distribution,
            daily_trend=trend,
        )

    # --------------------------------------------------------
    # 内部
    # --------------------------------------------------------

    def _build_distribution(
        self,
        counts: list[DefectTypeCount],
    ) -> list[DefectDistribution]:
        """构建缺陷分布"""
        count_map = {c.defect_type: c.count for c in counts}
        total_defects = sum(count_map.values()) or 1

        return [
            DefectDistribution(
                type=t,
                count=count_map.get(t, 0),
                percentage=round(count_map.get(t, 0) / total_defects, 4),
            )
            for t in self.ALL_DEFECT_TYPES
            if count_map.get(t, 0) > 0
        ]

    @staticmethod
    def _build_trend(daily: list[DailyStats]) -> list[DailyTrend]:
        """构建每日趋势"""
        return [
            DailyTrend(
                date=d.date,
                total=d.total,
                defect_images=d.defect_images,
                defect_boxes=d.defect_boxes,
                defect_free_images=d.defect_free_images,
                crazing=d.by_type.get("crazing", 0),
                inclusion=d.by_type.get("inclusion", 0),
                patches=d.by_type.get("patches", 0),
                pitted_surface=d.by_type.get("pitted_surface", 0),
                rolled_in_scale=d.by_type.get("rolled_in_scale", 0),
                scratches=d.by_type.get("scratches", 0),
            )
            for d in daily
        ]
