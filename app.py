"""
新闻联播每日总结 — Flask Web 应用
"""

import logging
from datetime import date, datetime

from flask import Flask, jsonify, render_template, request

from config import DATABASE_PATH, FLASK_DEBUG, FLASK_HOST, FLASK_PORT, LOG_LEVEL, SECRET_KEY
from database import (
    get_history, get_history_count, get_latest_summary, get_summary_by_date,
    init_db, mark_failed, save_raw_transcript, save_summary,
)
from summarizer import generate_summary, match_segments_to_summary
from fetcher import fetch_daily_transcript

# ============================================================
# 初始化
# ============================================================
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY

# 应用启动时初始化数据库
init_db()

# ============================================================
# 页面路由
# ============================================================


@app.route("/")
def index():
    """首页 — 展示最新一期摘要。"""
    latest = get_latest_summary()
    return render_template("index.html", summary=latest, today=date.today())


@app.route("/history")
def history():
    """历史记录页面。"""
    page = request.args.get("page", 1, type=int)
    per_page = 20
    offset = (page - 1) * per_page

    total = get_history_count()
    items = get_history(limit=per_page, offset=offset)
    total_pages = max(1, (total + per_page - 1) // per_page)

    return render_template(
        "history.html",
        items=items,
        page=page,
        total_pages=total_pages,
        total=total,
    )


@app.route("/detail/<string:date_str>")
def detail(date_str: str):
    """单日详情页。"""
    try:
        target_date = date.fromisoformat(date_str)
    except ValueError:
        return "日期格式错误，请使用 YYYY-MM-DD", 400

    record = get_summary_by_date(target_date)
    if not record:
        return render_template("detail.html", date_str=date_str, record=None)

    return render_template("detail.html", date_str=date_str, record=record)


# ============================================================
# API 路由
# ============================================================


@app.route("/api/summary/latest")
def api_latest():
    """API — 获取最新摘要 JSON。"""
    latest = get_latest_summary()
    if not latest:
        return jsonify({"error": "暂无摘要数据"}), 404

    # 移除 raw_text 以减少 API 响应体积
    if "raw_text" in latest:
        del latest["raw_text"]
    return jsonify(latest)


@app.route("/api/summary/<string:date_str>")
def api_summary_by_date(date_str: str):
    """API — 获取指定日期摘要 JSON。"""
    try:
        target_date = date.fromisoformat(date_str)
    except ValueError:
        return jsonify({"error": "日期格式错误，请使用 YYYY-MM-DD"}), 400

    record = get_summary_by_date(target_date)
    if not record:
        return jsonify({"error": f"未找到 {date_str} 的摘要"}), 404

    if "raw_text" in record:
        del record["raw_text"]
    return jsonify(record)


@app.route("/api/fetch-now", methods=["POST"])
def api_fetch_now():
    """手动触发抓取 + 摘要生成。支持 ?date=YYYY-MM-DD 指定日期，默认今天。"""
    date_str = request.args.get("date", "").strip()
    if date_str:
        try:
            target_date = date.fromisoformat(date_str)
            if target_date > date.today():
                return jsonify({"error": "不能抓取未来日期"}), 400
        except ValueError:
            return jsonify({"error": "日期格式错误，请使用 YYYY-MM-DD"}), 400
    else:
        target_date = date.today()

    logger.info(f"手动触发: {target_date.isoformat()} 的抓取与摘要")

    # Step 1: 获取文字稿
    result = fetch_daily_transcript(target_date)
    if not result:
        mark_failed(target_date)
        return jsonify({"error": f"无法获取 {target_date.isoformat()} 的文字稿，请稍后重试"}), 503

    # Step 2: 保存原始文字稿（含分段）
    segments = result.get("segments", [])
    save_raw_transcript(target_date, result["raw_text"], result["source"], segments)

    # Step 3: 生成 AI 摘要（传入分段以启用 segment_id 标注）
    summary = generate_summary(result["raw_text"], target_date, segments=segments)
    if not summary:
        mark_failed(target_date)
        return jsonify({"error": "AI 摘要生成失败"}), 500

    # 匹配原始分段
    if segments:
        summary = match_segments_to_summary(summary, segments)

    # Step 4: 保存摘要
    save_summary(target_date, summary)

    return jsonify({
        "status": "ok",
        "date": target_date.isoformat(),
        "source": result["source"],
        "message": "摘要已生成",
    })


@app.route("/api/status")
def api_status():
    """API — 获取系统状态。"""
    latest = get_latest_summary()
    from config import DATABASE_URL
    import os
    return jsonify({
        "status": "running",
        "latest_date": latest["date"] if latest else None,
        "total_summaries": get_history_count(),
        "db_backend": "PostgreSQL" if DATABASE_URL else "SQLite",
        "has_database_url": bool(DATABASE_URL),
        "has_pg_vars": bool(os.environ.get("PGHOST") or os.environ.get("PGDATABASE") or os.environ.get("POSTGRES_URL")),
        "server_time": datetime.now().isoformat(),
    })


# ============================================================
# 启动定时调度器
# ============================================================
import atexit
from scheduler import start_scheduler, stop_scheduler

start_scheduler()
atexit.register(stop_scheduler)

# ============================================================
# 启动入口（本地开发用）
# ============================================================

if __name__ == "__main__":
    logger.info(f"新闻联播每日总结启动 → http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
