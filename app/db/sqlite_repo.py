"""
SQLite Repository 实现

DetectionRepository 的具体 SQLite 实现。
所有 SQL 逻辑封装在此，业务层不可见。
"""

from __future__ import annotations

import json

from app.db.connection import DatabaseConnection
from app.db.repository import (
    BBoxRecord,
    DailyStats,
    DefectTypeCount,
    DetectionFilters,
    DetectionRecord,
    DetectionRepository,
    InspectionRecord,
    StatsSummary,
)
from app.core.neu_classes import CANONICAL_NAMES


class SqliteDetectionRepo(DetectionRepository):
    """
    SQLite 检测记录 Repository

    封装全部 SQL，通过 DatabaseConnection 获取连接。
    业务层通过 DetectionRepository 接口调用，不感知 SQL。
    """

    TABLE = "detection_history"

    def __init__(self, db: DatabaseConnection) -> None:
        self._db = db
        if not db.is_initialized:
            db.initialize()

    # ============================================================
    # 写入
    # ============================================================

    def save(self, record: DetectionRecord) -> DetectionRecord:
        with self._db.get_connection() as conn:
            cursor = conn.execute(
                f"""INSERT INTO {self.TABLE}
                   (image_name, defect_type, confidence, bbox, detect_time, inspection_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    record.image_name,
                    record.defect_type,
                    record.confidence,
                    json.dumps(record.bbox.to_dict()),
                    record.detect_time or DetectionRecord.now_iso(),
                    record.inspection_id,
                ),
            )
            record.id = cursor.lastrowid
            record.created_at = DetectionRecord.now_iso()
            return record

    def save_batch(self, records: list[DetectionRecord]) -> list[DetectionRecord]:
        if not records:
            return records

        with self._db.get_connection() as conn:
            now = DetectionRecord.now_iso()
            sql = (
                f"INSERT INTO {self.TABLE}"
                "(image_name, defect_type, confidence, bbox, detect_time, inspection_id) "
                "VALUES (?, ?, ?, ?, ?, ?)"
            )
            for r in records:
                cursor = conn.execute(
                    sql,
                    (
                        r.image_name,
                        r.defect_type,
                        r.confidence,
                        json.dumps(r.bbox.to_dict()),
                        r.detect_time or now,
                        r.inspection_id,
                    ),
                )
                r.id = cursor.lastrowid
                r.created_at = now
            return records

    def save_inspection(self, inspection: InspectionRecord) -> InspectionRecord:
        with self._db.get_connection() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO inspections
                   (inspection_id, image_name, detect_time, total_defects, model, is_legacy)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    inspection.inspection_id,
                    inspection.image_name,
                    inspection.detect_time,
                    inspection.total_defects,
                    inspection.model,
                    1 if inspection.is_legacy else 0,
                ),
            )
            row = conn.execute(
                "SELECT created_at FROM inspections WHERE inspection_id = ?",
                (inspection.inspection_id,),
            ).fetchone()
            inspection.created_at = row["created_at"] if row else inspection.created_at
            return inspection

    # ============================================================
    # 查询
    # ============================================================

    def find_by_id(self, record_id: int) -> DetectionRecord | None:
        with self._db.get_connection() as conn:
            row = conn.execute(
                f"SELECT * FROM {self.TABLE} WHERE id = ?",
                (record_id,),
            ).fetchone()
            return self._row_to_record(row)

    def find_by_image(self, image_name: str) -> list[DetectionRecord]:
        with self._db.get_connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM {self.TABLE} WHERE image_name = ? "
                f"ORDER BY confidence DESC",
                (image_name,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def find_by_defect_type(
        self, defect_type: str, limit: int = 100
    ) -> list[DetectionRecord]:
        with self._db.get_connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM {self.TABLE} WHERE defect_type = ? "
                f"ORDER BY detect_time DESC LIMIT ?",
                (defect_type, limit),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def find_recent(self, limit: int = 20) -> list[DetectionRecord]:
        with self._db.get_connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM {self.TABLE} "
                f"ORDER BY detect_time DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def search(self, filters: DetectionFilters) -> list[DetectionRecord]:
        conditions: list[str] = []
        params: list = []

        if filters.image_name:
            conditions.append("image_name = ?")
            params.append(filters.image_name)
        if filters.defect_type:
            conditions.append("defect_type = ?")
            params.append(filters.defect_type)
        if filters.min_confidence is not None:
            conditions.append("confidence >= ?")
            params.append(filters.min_confidence)
        if filters.max_confidence is not None:
            conditions.append("confidence <= ?")
            params.append(filters.max_confidence)
        if filters.date_from:
            conditions.append("detect_time >= ?")
            params.append(filters.date_from)
        if filters.date_to:
            conditions.append("detect_time <= ?")
            params.append(filters.date_to)

        where = (" AND ".join(conditions)) if conditions else "1=1"

        sql = (
            f"SELECT * FROM {self.TABLE} WHERE {where} "
            f"ORDER BY detect_time DESC LIMIT ? OFFSET ?"
        )
        params.extend([filters.limit, filters.offset])

        with self._db.get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    # ============================================================
    # 统计
    # ============================================================

    def count_by_type(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[DefectTypeCount]:
        conditions, params = self._date_conditions("i.detect_time", date_from, date_to)
        conditions.append("i.is_legacy = 0")
        placeholders = ",".join("?" for _ in CANONICAL_NAMES)
        conditions.append(f"h.defect_type IN ({placeholders})")
        params.extend(CANONICAL_NAMES)
        where = " AND ".join(conditions)

        with self._db.get_connection() as conn:
            rows = conn.execute(
                f"""SELECT h.defect_type, COUNT(*) as cnt
                    FROM {self.TABLE} h
                    JOIN inspections i ON i.inspection_id = h.inspection_id
                    WHERE {where}
                    GROUP BY h.defect_type
                    ORDER BY cnt DESC""",
                params,
            ).fetchall()
        return [DefectTypeCount(defect_type=r["defect_type"], count=r["cnt"]) for r in rows]

    def stats_summary(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> StatsSummary:
        conditions, params = self._date_conditions("detect_time", date_from, date_to)
        conditions.append("is_legacy = 0")
        where = " AND ".join(conditions)

        with self._db.get_connection() as conn:
            row = conn.execute(
                f"""SELECT
                        COUNT(*) AS inspected_images,
                        COALESCE(SUM(total_defects), 0) AS defect_boxes,
                        COALESCE(SUM(CASE WHEN total_defects > 0 THEN 1 ELSE 0 END), 0) AS defect_images,
                        COALESCE(SUM(CASE WHEN total_defects = 0 THEN 1 ELSE 0 END), 0) AS defect_free_images
                    FROM inspections
                    WHERE {where}""",
                params,
            ).fetchone()

        return StatsSummary(
            inspected_images=int(row["inspected_images"] or 0),
            defect_boxes=int(row["defect_boxes"] or 0),
            defect_images=int(row["defect_images"] or 0),
            defect_free_images=int(row["defect_free_images"] or 0),
        )

    def daily_stats(self, days: int = 7) -> list[DailyStats]:
        with self._db.get_connection() as conn:
            rows = conn.execute(
                """SELECT DATE(detect_time) as date,
                          COUNT(*) as total,
                          COALESCE(SUM(total_defects), 0) as defect_boxes,
                          COALESCE(SUM(CASE WHEN total_defects > 0 THEN 1 ELSE 0 END), 0) as defect_images,
                          COALESCE(SUM(CASE WHEN total_defects = 0 THEN 1 ELSE 0 END), 0) as defect_free_images
                    FROM inspections
                    WHERE is_legacy = 0
                      AND detect_time >= DATE('now', ? || ' days')
                    GROUP BY DATE(detect_time)
                    ORDER BY date ASC""",
                (f"-{days}",),
            ).fetchall()

            results: list[DailyStats] = []
            for row in rows:
                date_str = row["date"]
                placeholders = ",".join("?" for _ in CANONICAL_NAMES)
                type_rows = conn.execute(
                    f"""SELECT h.defect_type, COUNT(*) as cnt
                        FROM {self.TABLE} h
                        JOIN inspections i ON i.inspection_id = h.inspection_id
                        WHERE i.is_legacy = 0
                          AND DATE(i.detect_time) = ?
                          AND h.defect_type IN ({placeholders})
                        GROUP BY h.defect_type""",
                    (date_str, *CANONICAL_NAMES),
                ).fetchall()
                by_type = {r["defect_type"]: r["cnt"] for r in type_rows}

                results.append(DailyStats(
                    date=date_str,
                    total=row["total"],
                    defect_images=int(row["defect_images"] or 0),
                    defect_boxes=int(row["defect_boxes"] or 0),
                    defect_free_images=int(row["defect_free_images"] or 0),
                    by_type=by_type,
                ))

        return results

    def total_count(self) -> int:
        with self._db.get_connection() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) as cnt FROM {self.TABLE}"
            ).fetchone()
            return row["cnt"] if row else 0

    # ============================================================
    # 删除
    # ============================================================

    def delete_by_id(self, record_id: int) -> bool:
        with self._db.get_connection() as conn:
            cursor = conn.execute(
                f"DELETE FROM {self.TABLE} WHERE id = ?",
                (record_id,),
            )
            return cursor.rowcount > 0

    def delete_older_than(self, before_date: str) -> int:
        with self._db.get_connection() as conn:
            cursor = conn.execute(
                f"DELETE FROM {self.TABLE} WHERE detect_time < ?",
                (before_date,),
            )
            return cursor.rowcount

    # ============================================================
    # 内部
    # ============================================================

    @staticmethod
    def _row_to_record(row) -> DetectionRecord | None:
        """sqlite3.Row → DetectionRecord"""
        if row is None:
            return None
        return DetectionRecord.from_row(dict(row))

    @staticmethod
    def _date_conditions(
        column: str,
        date_from: str | None,
        date_to: str | None,
    ) -> tuple[list[str], list]:
        conditions: list[str] = []
        params: list = []
        if date_from:
            conditions.append(f"{column} >= ?")
            params.append(date_from)
        if date_to:
            conditions.append(f"{column} <= ?")
            params.append(date_to)
        return conditions, params
