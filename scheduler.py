"""
弹性轮询调度器
每日 19:30 起每 10 分钟尝试获取新闻联播文字稿，直到成功或超过截止时间
"""

import logging
from datetime import date, datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from config import (
    POLL_DEADLINE_HOUR,
    POLL_DEADLINE_MINUTE,
    POLL_INTERVAL_MINUTES,
    POLL_START_HOUR,
    POLL_START_MINUTE,
)
from database import get_summary_by_date, mark_failed, save_raw_transcript, save_summary
from fetcher import fetch_daily_transcript
from summarizer import generate_summary

logger = logging.getLogger(__name__)

# 全局调度器实例
_scheduler: BackgroundScheduler | None = None

# 记录今日已完成的轮询，避免重复执行
_today_done: str | None = None  # 格式: "YYYY-MM-DD"


def _poll_and_summarize() -> None:
    """
    单次轮询：尝试获取 + 摘要 + 存储。
    如果今日已成功则跳过。
    """
    global _today_done

    today = date.today()
    today_str = today.isoformat()

    # 检查是否超过截止时间
    now = datetime.now()
    deadline = now.replace(hour=POLL_DEADLINE_HOUR, minute=POLL_DEADLINE_MINUTE, second=0, microsecond=0)
    if now > deadline:
        logger.info(f"已超过截止时间 {POLL_DEADLINE_HOUR:02d}:{POLL_DEADLINE_MINUTE:02d}，停止轮询")
        _stop_polling_if_active()
        return

    # 今日已完成则跳过
    if _today_done == today_str:
        logger.info(f"今日 ({today_str}) 已完成摘要，跳过本次轮询")
        _stop_polling_if_active()
        return

    # 检查数据库中是否已有今日摘要
    existing = get_summary_by_date(today)
    if existing and existing.get("status") == "summarized":
        logger.info(f"今日 ({today_str}) 已有摘要记录，标记完成")
        _today_done = today_str
        _stop_polling_if_active()
        return

    # 尝试获取文字稿
    logger.info(f"[轮询] 尝试获取 {today_str} 的文字稿...")
    result = fetch_daily_transcript(today)

    if not result:
        logger.info(f"[轮询] {today_str} 文字稿暂未就绪，{POLL_INTERVAL_MINUTES} 分钟后重试")
        return

    # 获取成功 → 保存原始稿
    logger.info(f"[轮询] 成功获取 {today_str} 文字稿 (来源: {result['source']})")
    save_raw_transcript(today, result["raw_text"], result["source"])

    # 生成 AI 摘要
    logger.info(f"[轮询] 开始为 {today_str} 生成 AI 摘要...")
    summary = generate_summary(result["raw_text"], today)

    if summary:
        save_summary(today, summary)
        logger.info(f"✅ {today_str} 摘要已生成并保存！")
        _today_done = today_str
        _stop_polling_if_active()
    else:
        logger.error(f"[轮询] {today_str} AI 摘要生成失败")
        mark_failed(today)
        _today_done = today_str
        _stop_polling_if_active()


def _stop_polling_if_active() -> None:
    """如果轮询任务在运行，移除之（今日已完成或超时）。"""
    global _scheduler
    if _scheduler and _scheduler.get_job("poll_xwlb"):
        _scheduler.remove_job("poll_xwlb")
        logger.info("已停止今日轮询任务")


def start_scheduler() -> None:
    """启动弹性轮询调度器。"""
    global _scheduler

    if _scheduler and _scheduler.running:
        logger.info("调度器已在运行中")
        return

    _scheduler = BackgroundScheduler(
        timezone="Asia/Shanghai",  # 北京时间
        job_defaults={"coalesce": True, "max_instances": 1},
    )

    # 添加轮询任务：每日指定时间开始，间隔执行
    _scheduler.add_job(
        _poll_and_summarize,
        trigger=CronTrigger(
            hour=f"{POLL_START_HOUR}-{POLL_DEADLINE_HOUR}",
            minute=f"*/{POLL_INTERVAL_MINUTES}",
            timezone="Asia/Shanghai",
        ),
        id="poll_xwlb",
        name="新闻联播弹性轮询",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info(
        f"调度器已启动 — 每日 {POLL_START_HOUR:02d}:{POLL_START_MINUTE:02d} "
        f"开始轮询，间隔 {POLL_INTERVAL_MINUTES} 分钟，"
        f"截止 {POLL_DEADLINE_HOUR:02d}:{POLL_DEADLINE_MINUTE:02d}"
    )


def stop_scheduler() -> None:
    """停止调度器。"""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("调度器已停止")
        _scheduler = None
