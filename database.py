"""
数据库操作模块 — 支持 PostgreSQL（生产）和 SQLite（本地开发）
存储和检索每日新闻联播摘要，自动清理过期记录
"""

import json
import logging
from datetime import date, datetime, timedelta
from typing import Optional

from config import DATABASE_PATH, DATABASE_URL, RETENTION_DAYS

logger = logging.getLogger(__name__)

# ============================================================
# 后端检测 & 适配
# ============================================================
_USE_PG = bool(DATABASE_URL)

if _USE_PG:
    import psycopg2
    import psycopg2.extras

    def _sql(query: str) -> str:
        """将 SQLite 风格的 ? 占位符转换为 PostgreSQL 的 %s。"""
        return query.replace("?", "%s")

    def get_connection():
        conn = psycopg2.connect(DATABASE_URL)
        return conn

    def _fetchone(cursor):
        row = cursor.fetchone()
        return dict(row) if row else None

    def _fetchall(cursor):
        return [dict(row) for row in cursor.fetchall()]

    def _execute(cursor, query: str, params: tuple = ()):
        cursor.execute(_sql(query), params)

    def _last_rowid(cursor) -> int:
        """PG 下 SERIAL 列用 RETURNING 获取，此函数仅 SQLite 调用。"""
        return 0

    logger.info("数据库后端: PostgreSQL")
else:
    import sqlite3

    def _sql(query: str) -> str:
        return query

    def get_connection():
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _fetchone(cursor):
        row = cursor.fetchone()
        return dict(row) if row else None

    def _fetchall(cursor):
        return [dict(row) for row in cursor.fetchall()]

    def _execute(cursor, query: str, params: tuple = ()):
        cursor.execute(_sql(query), params)

    def _last_rowid(cursor) -> int:
        return cursor.lastrowid

    logger.info(f"数据库后端: SQLite ({DATABASE_PATH})")


# ============================================================
# 初始化
# ============================================================

def init_db() -> None:
    """初始化数据库表结构（兼容 PG 和 SQLite）。"""
    conn = get_connection()
    try:
        if _USE_PG:
            cur = conn.cursor()
            cur.execute(_sql("""
                CREATE TABLE IF NOT EXISTS daily_summaries (
                    id SERIAL PRIMARY KEY,
                    date TEXT UNIQUE NOT NULL,
                    raw_text TEXT,
                    segments_json TEXT,
                    summary_json TEXT,
                    source TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            cur.execute(_sql("""
                CREATE INDEX IF NOT EXISTS idx_daily_summaries_date
                ON daily_summaries(date DESC)
            """))
            cur.close()
        else:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE NOT NULL,
                    raw_text TEXT,
                    segments_json TEXT,
                    summary_json TEXT,
                    source TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_daily_summaries_date
                ON daily_summaries(date DESC)
            """)
        conn.commit()
        logger.info("数据库初始化完成")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise
    finally:
        conn.close()


# ============================================================
# 清理过期记录
# ============================================================

def cleanup_old_records(retention_days: int = RETENTION_DAYS) -> int:
    """
    删除超过保留天数的记录。

    Args:
        retention_days: 保留天数，默认使用配置值

    Returns:
        int: 删除的记录数
    """
    cutoff_date = (date.today() - timedelta(days=retention_days)).isoformat()
    conn = get_connection()
    try:
        if _USE_PG:
            cur = conn.cursor()
            cur.execute(_sql("DELETE FROM daily_summaries WHERE date < ?"), (cutoff_date,))
            deleted = cur.rowcount
            cur.close()
        else:
            cur = conn.cursor()
            cur.execute("DELETE FROM daily_summaries WHERE date < ?", (cutoff_date,))
            deleted = cur.rowcount
            cur.close()
        conn.commit()
        if deleted > 0:
            logger.info(f"清理完成: 删除 {deleted} 条 {cutoff_date} 之前的记录")
        return deleted
    except Exception as e:
        logger.error(f"清理失败: {e}")
        return 0
    finally:
        conn.close()


# ============================================================
# CRUD 操作
# ============================================================

def save_raw_transcript(target_date: date, raw_text: str, source: str,
                        segments: list | None = None) -> bool:
    """保存原始文字稿及分段数据。"""
    conn = get_connection()
    try:
        segments_json = json.dumps(segments, ensure_ascii=False) if segments else None
        if _USE_PG:
            cur = conn.cursor()
            cur.execute(_sql(
                """
                INSERT INTO daily_summaries (date, raw_text, segments_json, source, status)
                VALUES (?, ?, ?, ?, 'pending')
                ON CONFLICT(date) DO UPDATE SET
                    raw_text = EXCLUDED.raw_text,
                    segments_json = EXCLUDED.segments_json,
                    source = EXCLUDED.source,
                    updated_at = CURRENT_TIMESTAMP
                """
            ), (target_date.isoformat(), raw_text, segments_json, source))
            cur.close()
        else:
            conn.execute(
                """
                INSERT INTO daily_summaries (date, raw_text, segments_json, source, status)
                VALUES (?, ?, ?, ?, 'pending')
                ON CONFLICT(date) DO UPDATE SET
                    raw_text = excluded.raw_text,
                    segments_json = excluded.segments_json,
                    source = excluded.source,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (target_date.isoformat(), raw_text, segments_json, source),
            )
        conn.commit()
        logger.info(f"已保存 {target_date.isoformat()} 的原始文字稿 (来源: {source})")
        return True
    except Exception as e:
        logger.error(f"保存原始文字稿失败: {e}")
        return False
    finally:
        conn.close()


def save_summary(target_date: date, summary: dict) -> bool:
    """保存 AI 摘要，成功后触发过期清理。"""
    conn = get_connection()
    success = False
    try:
        summary_json = json.dumps(summary, ensure_ascii=False)
        if _USE_PG:
            cur = conn.cursor()
            cur.execute(_sql(
                """
                UPDATE daily_summaries
                SET summary_json = ?,
                    status = 'summarized',
                    updated_at = CURRENT_TIMESTAMP
                WHERE date = ?
                """
            ), (summary_json, target_date.isoformat()))
            cur.close()
        else:
            conn.execute(
                """
                UPDATE daily_summaries
                SET summary_json = ?,
                    status = 'summarized',
                    updated_at = CURRENT_TIMESTAMP
                WHERE date = ?
                """,
                (summary_json, target_date.isoformat()),
            )
        conn.commit()
        logger.info(f"已保存 {target_date.isoformat()} 的 AI 摘要")
        success = True
    except Exception as e:
        logger.error(f"保存摘要失败: {e}")
    finally:
        conn.close()

    # 保存成功后清理过期记录
    if success:
        cleanup_old_records()

    return success


def mark_failed(target_date: date) -> None:
    """标记某日为获取/摘要失败。"""
    conn = get_connection()
    try:
        if _USE_PG:
            cur = conn.cursor()
            cur.execute(_sql(
                """
                INSERT INTO daily_summaries (date, status)
                VALUES (?, 'failed')
                ON CONFLICT(date) DO UPDATE SET
                    status = 'failed',
                    updated_at = CURRENT_TIMESTAMP
                """
            ), (target_date.isoformat(),))
            cur.close()
        else:
            conn.execute(
                """
                INSERT INTO daily_summaries (date, status)
                VALUES (?, 'failed')
                ON CONFLICT(date) DO UPDATE SET
                    status = 'failed',
                    updated_at = CURRENT_TIMESTAMP
                """,
                (target_date.isoformat(),),
            )
        conn.commit()
    finally:
        conn.close()


def get_latest_summary() -> Optional[dict]:
    """获取最新一期已摘要的记录。"""
    conn = get_connection()
    try:
        if _USE_PG:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(_sql(
                """
                SELECT * FROM daily_summaries
                WHERE status = 'summarized'
                ORDER BY date DESC
                LIMIT 1
                """
            ))
            row = cur.fetchone()
            cur.close()
        else:
            row = conn.execute(
                """
                SELECT * FROM daily_summaries
                WHERE status = 'summarized'
                ORDER BY date DESC
                LIMIT 1
                """
            ).fetchone()

        if not row:
            return None
        return _row_to_dict(dict(row) if _USE_PG else row)
    finally:
        conn.close()


def get_summary_by_date(target_date: date) -> Optional[dict]:
    """获取指定日期的摘要。"""
    conn = get_connection()
    try:
        if _USE_PG:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(_sql(
                "SELECT * FROM daily_summaries WHERE date = ?"
            ), (target_date.isoformat(),))
            row = cur.fetchone()
            cur.close()
        else:
            row = conn.execute(
                "SELECT * FROM daily_summaries WHERE date = ?",
                (target_date.isoformat(),),
            ).fetchone()

        if not row:
            return None
        return _row_to_dict(dict(row) if _USE_PG else row)
    finally:
        conn.close()


def get_history(limit: int = 30, offset: int = 0) -> list[dict]:
    """获取历史摘要列表（分页），不含 raw_text。"""
    conn = get_connection()
    try:
        if _USE_PG:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(_sql(
                """
                SELECT id, date, source, status, created_at, updated_at,
                       summary_json
                FROM daily_summaries
                WHERE status = 'summarized'
                ORDER BY date DESC
                LIMIT ? OFFSET ?
                """
            ), (limit, offset))
            rows = cur.fetchall()
            cur.close()
        else:
            rows = conn.execute(
                """
                SELECT id, date, source, status, created_at, updated_at,
                       summary_json
                FROM daily_summaries
                WHERE status = 'summarized'
                ORDER BY date DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

        results = []
        for row in rows:
            d = dict(row) if _USE_PG else dict(row)
            if d.get("summary_json"):
                summary = json.loads(d["summary_json"])
                # overview 已移除，取 top_news 头条标题作为摘要预览
                top_news = summary.get("top_news", [])
                if top_news:
                    d["overview"] = "、".join(item["title"] for item in top_news[:3])
                else:
                    d["overview"] = ""
                d["keywords"] = summary.get("keywords", [])
                del d["summary_json"]
            results.append(d)

        return results
    finally:
        conn.close()


def get_history_count() -> int:
    """获取已摘要的总天数。"""
    conn = get_connection()
    try:
        if _USE_PG:
            cur = conn.cursor()
            cur.execute(_sql(
                "SELECT COUNT(*) as cnt FROM daily_summaries WHERE status = 'summarized'"
            ))
            row = cur.fetchone()
            cur.close()
            return row[0] if row else 0
        else:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM daily_summaries WHERE status = 'summarized'"
            ).fetchone()
            return row["cnt"] if row else 0
    finally:
        conn.close()


def get_db_stats() -> dict:
    """获取数据库统计信息（用于监控）。"""
    conn = get_connection()
    try:
        if _USE_PG:
            cur = conn.cursor()
            cur.execute(_sql("SELECT COUNT(*) as cnt FROM daily_summaries"))
            total = cur.fetchone()[0]
            cur.execute(_sql(
                "SELECT COUNT(*) as cnt FROM daily_summaries WHERE status = 'summarized'"
            ))
            summarized = cur.fetchone()[0]
            cur.execute(_sql(
                "SELECT COUNT(*) as cnt FROM daily_summaries WHERE status = 'failed'"
            ))
            failed = cur.fetchone()[0]
            cur.execute(_sql(
                "SELECT MIN(date) as d FROM daily_summaries WHERE status = 'summarized'"
            ))
            oldest = cur.fetchone()[0]
            cur.close()
        else:
            total = conn.execute("SELECT COUNT(*) as cnt FROM daily_summaries").fetchone()["cnt"]
            summarized = conn.execute(
                "SELECT COUNT(*) as cnt FROM daily_summaries WHERE status = 'summarized'"
            ).fetchone()["cnt"]
            failed = conn.execute(
                "SELECT COUNT(*) as cnt FROM daily_summaries WHERE status = 'failed'"
            ).fetchone()["cnt"]
            oldest_row = conn.execute(
                "SELECT MIN(date) as d FROM daily_summaries WHERE status = 'summarized'"
            ).fetchone()
            oldest = oldest_row["d"] if oldest_row else None

        return {
            "backend": "postgresql" if _USE_PG else "sqlite",
            "total_records": total,
            "summarized": summarized,
            "failed": failed,
            "oldest_date": oldest,
            "retention_days": RETENTION_DAYS,
        }
    finally:
        conn.close()


# ============================================================
# 内部工具
# ============================================================

def _row_to_dict(row: dict) -> dict:
    """将数据库行转换为统一格式，解析 JSON 字段。"""
    d = dict(row)
    if d.get("summary_json"):
        d["summary"] = json.loads(d["summary_json"])
        del d["summary_json"]
    if d.get("segments_json"):
        d["segments"] = json.loads(d["segments_json"])
        del d["segments_json"]
    return d
