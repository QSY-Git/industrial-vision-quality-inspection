"""
数据库迁移 (DDL)

幂等建表 — 重复执行不会报错
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.db.connection import DatabaseConnection

VERSION = 2

DDL_DETECTION_HISTORY = """
CREATE TABLE IF NOT EXISTS detection_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    image_name  TEXT    NOT NULL,
    defect_type TEXT    NOT NULL,
    confidence  REAL    NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    bbox        TEXT    NOT NULL,
    detect_time TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);
"""

DDL_INSPECTIONS = """
CREATE TABLE IF NOT EXISTS inspections (
    inspection_id TEXT PRIMARY KEY,
    image_name    TEXT    NOT NULL,
    detect_time   TEXT    NOT NULL,
    total_defects INTEGER NOT NULL CHECK (total_defects >= 0),
    model         TEXT    NOT NULL DEFAULT '',
    is_legacy     INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_image_name ON detection_history(image_name);",
    "CREATE INDEX IF NOT EXISTS idx_defect_type ON detection_history(defect_type);",
    "CREATE INDEX IF NOT EXISTS idx_detect_time ON detection_history(detect_time DESC);",
    "CREATE INDEX IF NOT EXISTS idx_image_defect ON detection_history(image_name, defect_type);",
    "CREATE INDEX IF NOT EXISTS idx_history_inspection ON detection_history(inspection_id);",
    "CREATE INDEX IF NOT EXISTS idx_inspection_time ON inspections(detect_time DESC);",
    "CREATE INDEX IF NOT EXISTS idx_inspection_legacy ON inspections(is_legacy);",
]


def run_migrations(db: DatabaseConnection) -> None:
    """
    执行所有迁移 (幂等)

    Args:
        db: DatabaseConnection 实例
    """
    with db.get_connection() as conn:
        # 建表
        conn.execute(DDL_INSPECTIONS)
        conn.execute(DDL_DETECTION_HISTORY)
        _ensure_column(conn, "detection_history", "inspection_id", "TEXT")

        # 建索引
        for index_sql in INDEXES:
            try:
                conn.execute(index_sql)
            except sqlite3.OperationalError:
                pass


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    """Add a column if it does not already exist."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {row[1] for row in rows}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
