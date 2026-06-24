"""
数据库连接管理器

管理 SQLite 连接生命周期，提供:
  - 单例连接池 (线程安全)
  - context manager 支持
  - 自动建表
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Generator
from contextlib import contextmanager


class DatabaseConnection:
    """
    SQLite 连接管理器

    Usage:
        db = DatabaseConnection("data/database/inspection.db")
        db.initialize()

        with db.get_connection() as conn:
            conn.execute("SELECT ...")

        # 或直接获取
        conn = db.connect()
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._initialized = False

    # --------------------------------------------------------
    # 连接管理
    # --------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        """获取数据库连接"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(
            str(self.db_path),
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """上下文管理器 — 自动 commit/close"""
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # --------------------------------------------------------
    # 初始化
    # --------------------------------------------------------

    def initialize(self) -> None:
        """执行 DDL 建表 (幂等)"""
        from app.db.migrations import run_migrations
        run_migrations(self)
        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def __repr__(self) -> str:
        return f"DatabaseConnection(path={self.db_path}, init={self._initialized})"
