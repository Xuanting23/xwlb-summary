"""
新闻联播每日总结 — 配置文件
"""

import os

# ============================================================
# DeepSeek API 配置
# ============================================================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-8718db41c5c649c9a61c78afd614ec75")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"  # DeepSeek V4 Flash

# ============================================================
# 调度配置
# ============================================================
# 首次尝试时间 (北京时间, 24小时制)
POLL_START_HOUR = 19
POLL_START_MINUTE = 30

# 轮询间隔 (分钟)
POLL_INTERVAL_MINUTES = 10

# 最后截止时间 (北京时间)
POLL_DEADLINE_HOUR = 21
POLL_DEADLINE_MINUTE = 0

# ============================================================
# 数据库配置
# ============================================================
# Railway 云端优先使用 /data/ 持久挂载目录，本地开发降级到项目 data/ 目录
_DATA_DIR = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", os.path.join(os.path.dirname(__file__), "data"))
os.makedirs(_DATA_DIR, exist_ok=True)
DATABASE_PATH = os.path.join(_DATA_DIR, "xwlb.db")

# ============================================================
# Flask 配置
# ============================================================
FLASK_HOST = "0.0.0.0"
FLASK_PORT = int(os.environ.get("PORT", 5000))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production")

# ============================================================
# 日志配置
# ============================================================
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
