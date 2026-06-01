"""
SQLite 数据库操作模块
存储和检索每日新闻联播摘要
"""

import json
import logging
import sqlite3
from datetime import date, datetime
from typing import Optional

from config import DATABASE_PATH

logger = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    """获取数据库连接。"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """初始化数据库表结构。"""
    conn = get_connection()
    try:
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
        # 为日期创建索引，加速查询
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_daily_summaries_date
            ON daily_summaries(date DESC)
        """)
        conn.commit()
        logger.info("数据库初始化完成")
    finally:
        conn.close()


def save_raw_transcript(target_date: date, raw_text: str, source: str,
                        segments: list | None = None) -> bool:
    """
    保存原始文字稿及分段数据（获取成功后立即存储）。

    Args:
        target_date: 日期
        raw_text: 原始文字稿
        source: 数据来源
        segments: 单条新闻分段列表 [{title, content}, ...]

    Returns:
        bool: 是否保存成功
    """
    conn = get_connection()
    try:
        segments_json = json.dumps(segments, ensure_ascii=False) if segments else None
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
    """
    保存 AI 摘要。

    Args:
        target_date: 日期
        summary: 结构化摘要 dict

    Returns:
        bool: 是否保存成功
    """
    conn = get_connection()
    try:
        summary_json = json.dumps(summary, ensure_ascii=False)
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
        return True
    except Exception as e:
        logger.error(f"保存摘要失败: {e}")
        return False
    finally:
        conn.close()


def mark_failed(target_date: date) -> None:
    """标记某日为获取/摘要失败。"""
    conn = get_connection()
    try:
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
    """
    获取最新一期已摘要的记录。

    Returns:
        dict: {date, raw_text, summary: dict, source, status, created_at} 或 None
    """
    conn = get_connection()
    try:
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

        return _row_to_dict(row)
    finally:
        conn.close()


def get_summary_by_date(target_date: date) -> Optional[dict]:
    """
    获取指定日期的摘要。

    Args:
        target_date: 目标日期

    Returns:
        dict 或 None
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM daily_summaries WHERE date = ?",
            (target_date.isoformat(),),
        ).fetchone()

        if not row:
            return None

        return _row_to_dict(row)
    finally:
        conn.close()


def get_history(limit: int = 30, offset: int = 0) -> list[dict]:
    """
    获取历史摘要列表（分页）。

    Args:
        limit: 每页条数
        offset: 偏移量

    Returns:
        list[dict]: 摘要列表（不含 raw_text 以节省内存）
    """
    conn = get_connection()
    try:
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
            d = dict(row)
            if d.get("summary_json"):
                summary = json.loads(d["summary_json"])
                # 只保留 overview + keywords，减少传输量
                d["overview"] = summary.get("overview", "")
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
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM daily_summaries WHERE status = 'summarized'"
        ).fetchone()
        return row["cnt"] if row else 0
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    """将数据库行转换为 dict，解析 summary_json 和 segments_json。"""
    d = dict(row)
    if d.get("summary_json"):
        d["summary"] = json.loads(d["summary_json"])
    del d["summary_json"]
    if d.get("segments_json"):
        d["segments"] = json.loads(d["segments_json"])
    del d["segments_json"]
    return d
