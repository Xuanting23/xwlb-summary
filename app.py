"""
新闻联播每日总结 — Flask Web 应用
"""

import logging
from datetime import date, datetime

from flask import Flask, jsonify, render_template, request

from config import (
    DATABASE_PATH, FLASK_DEBUG, FLASK_HOST, FLASK_PORT,
    LOG_LEVEL, SECRET_KEY,
)
from database import (
    get_history, get_history_count, get_latest_summary, get_summary_by_date,
    init_db,
)

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
app.config["TEMPLATES_AUTO_RELOAD"] = True

# 应用启动时初始化数据库
init_db()

# ============================================================
# 页面路由
# ============================================================


@app.route("/")
def index():
    """首页 — 仅展示今日摘要，无数据则显示空状态。"""
    latest = get_latest_summary()
    today = date.today()
    # 最新摘要如果不是今天的，就不显示（避免跨天后展示旧闻）
    if latest and str(latest.get("date", "")) != str(today):
        latest = None
    return render_template("index.html", summary=latest, today=today)


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
