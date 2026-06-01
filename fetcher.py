"""
新闻联播文字稿获取模块
首选 AkShare，备选 cn.govopendata.com
"""

import logging
from datetime import date, datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def fetch_via_akshare(target_date: date) -> Optional[dict]:
    """
    通过 AkShare 获取新闻联播文字稿。

    Args:
        target_date: 目标日期

    Returns:
        dict: {date, raw_text, source, fetch_time} 或 None
    """
    try:
        import akshare as ak

        date_str = target_date.strftime("%Y%m%d")
        logger.info(f"[AkShare] 正在获取 {date_str} 的文字稿...")

        df = ak.news_cctv(date=date_str)

        if df is None or df.empty:
            logger.warning(f"[AkShare] {date_str} 返回空数据（可能停播）")
            return None

        # 拼接标题+内容为完整文字稿
        parts = []
        for _, row in df.iterrows():
            title = str(row.get("title", "")).strip()
            content = str(row.get("content", "")).strip()
            if title:
                parts.append(f"【{title}】")
            if content:
                parts.append(content)
            if title or content:
                parts.append("")  # 空行分隔

        raw_text = "\n".join(parts).strip()

        if not raw_text:
            logger.warning(f"[AkShare] {date_str} 文字稿内容为空")
            return None

        logger.info(f"[AkShare] 成功获取 {date_str} 的文字稿，共 {len(raw_text)} 字")
        return {
            "date": target_date.isoformat(),
            "raw_text": raw_text,
            "source": "akshare",
            "fetch_time": datetime.now().isoformat(),
        }

    except ImportError:
        logger.error("[AkShare] 未安装 akshare 库，请运行 pip install akshare")
        return None
    except Exception as e:
        logger.warning(f"[AkShare] 获取失败: {e}")
        return None


def fetch_via_govopendata(target_date: date) -> Optional[dict]:
    """
    通过 cn.govopendata.com 获取新闻联播文字稿（备选方案）。

    Args:
        target_date: 目标日期

    Returns:
        dict: {date, raw_text, source, fetch_time} 或 None
    """
    try:
        date_str = target_date.strftime("%Y%m%d")
        url = f"https://cn.govopendata.com/xinwenlianbo/{date_str}/"

        logger.info(f"[govopendata] 正在请求 {url} ...")

        resp = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
            timeout=15,
        )

        if resp.status_code == 404:
            logger.warning(f"[govopendata] {date_str} 页面不存在（可能停播或尚未更新）")
            return None

        if resp.status_code != 200:
            logger.warning(f"[govopendata] HTTP {resp.status_code}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # 提取正文内容（移除 script/style 标签）
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        body = soup.find("body")
        if not body:
            body = soup

        text = body.get_text(separator="\n", strip=True)

        if not text or len(text) < 100:
            logger.warning(f"[govopendata] {date_str} 提取的文字过短")
            return None

        logger.info(f"[govopendata] 成功获取 {date_str} 的文字稿，共 {len(text)} 字")
        return {
            "date": target_date.isoformat(),
            "raw_text": text,
            "source": "govopendata",
            "fetch_time": datetime.now().isoformat(),
        }

    except requests.RequestException as e:
        logger.warning(f"[govopendata] 网络请求失败: {e}")
        return None
    except Exception as e:
        logger.warning(f"[govopendata] 解析失败: {e}")
        return None


def fetch_daily_transcript(target_date: date) -> Optional[dict]:
    """
    获取指定日期的新闻联播文字稿。
    首选 AkShare，失败则降级到 cn.govopendata.com。

    Args:
        target_date: 目标日期

    Returns:
        dict: {date, raw_text, source, fetch_time} 或 None
    """
    logger.info(f"===== 开始获取 {target_date.isoformat()} 的新闻联播文字稿 =====")

    # 方案 1: AkShare
    result = fetch_via_akshare(target_date)
    if result:
        return result

    # 方案 2: cn.govopendata.com
    logger.info("AkShare 未获取到数据，尝试备选方案 govopendata...")
    result = fetch_via_govopendata(target_date)
    if result:
        return result

    logger.warning(f"所有数据源均无法获取 {target_date.isoformat()} 的文字稿")
    return None
