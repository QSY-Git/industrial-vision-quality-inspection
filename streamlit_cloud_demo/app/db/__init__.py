"""
数据库层 — Repository 模式封装 SQLite

对外 API:
  - create_repository(db_path) → DetectionRepository
  - DetectionRepository (ABC) — 业务层依赖此抽象
  - DetectionRecord, DetectionFilters, DefectTypeCount, DailyStats (DTO)
"""

from app.db.connection import DatabaseConnection
from app.db.repository import (
    BBoxRecord,
    DailyStats,
    DefectTypeCount,
    DetectionFilters,
    DetectionRecord,
    DetectionRepository,
)


def create_repository(db_path: str = "data/database/inspection.db") -> DetectionRepository:
    """
    工厂函数 — 创建 Repository 实例

    Args:
        db_path: SQLite 数据库文件路径

    Returns:
        DetectionRepository 实例 (默认 SqliteDetectionRepo)
    """
    from app.db.sqlite_repo import SqliteDetectionRepo
    db = DatabaseConnection(db_path)
    return SqliteDetectionRepo(db)


__all__ = [
    "DetectionRepository",
    "DetectionRecord",
    "BBoxRecord",
    "DetectionFilters",
    "DefectTypeCount",
    "DailyStats",
    "DatabaseConnection",
    "create_repository",
]
