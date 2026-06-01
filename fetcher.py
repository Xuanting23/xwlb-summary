"""
新闻联播文字稿获取模块
首选 AkShare，备选 cn.govopendata.com
返回完整文字稿 + 单条新闻分段列表
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
        dict: {date, raw_text, segments, source, fetch_time} 或 None
    """
    try:
        import akshare as ak

        date_str = target_date.strftime("%Y%m%d")
        logger.info(f"[AkShare] 正在获取 {date_str} 的文字稿...")

        df = ak.news_cctv(date=date_str)

        if df is None or df.empty:
            logger.warning(f"[AkShare] {date_str} 返回空数据（可能停播）")
            return None

        # 拼接标题+内容为完整文字稿，同时保留分段
        parts = []
        segments = []
        for _, row in df.iterrows():
            title = str(row.get("title", "")).strip()
            content = str(row.get("content", "")).strip()
            if title:
                parts.append(f"【{title}】")
            if content:
                parts.append(content)
            if title or content:
                parts.append("")  # 空行分隔
            if title and content:
                segments.append({"title": title, "content": content})

        raw_text = "\n".join(parts).strip()

        if not raw_text:
            logger.warning(f"[AkShare] {date_str} 文字稿内容为空")
            return None

        logger.info(
            f"[AkShare] 成功获取 {date_str} 的文字稿，"
            f"共 {len(raw_text)} 字，{len(segments)} 条新闻"
        )
        return {
            "date": target_date.isoformat(),
            "raw_text": raw_text,
            "segments": segments,
            "source": "akshare",
            "fetch_time": datetime.now().isoformat(),
        }

    except ImportError:
        logger.error("[AkShare] 未安装 akshare 库，请运行 pip install akshare")
        return None
    except Exception as e:
        logger.warning(f"[AkShare] 获取失败: {e}")
        return None


def fetch_via_cctv(target_date: date) -> Optional[dict]:
    """
    通过 CCTV 官网获取新闻联播文字稿（最权威来源）。

    1. 访问节目单页 https://tv.cctv.com/lm/xwlb/day/{YYYYMMDD}.shtml
    2. 提取每条新闻的独立页面链接
    3. 逐条抓取标题 + 正文
    """
    try:
        from bs4 import BeautifulSoup

        date_compact = target_date.strftime("%Y%m%d")
        list_url = f"https://tv.cctv.com/lm/xwlb/day/{date_compact}.shtml"

        logger.info(f"[CCTV] 正在获取节目单: {list_url}")

        list_resp = requests.get(
            list_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/130.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "zh-CN,zh;q=0.9",
            },
            timeout=15,
        )

        if list_resp.status_code != 200:
            logger.warning(f"[CCTV] 节目单页 HTTP {list_resp.status_code}")
            return None

        list_soup = BeautifulSoup(list_resp.text, "lxml")

        # 提取所有新闻链接（<li> 内嵌 <a> 标签，href 含 VIDE）
        news_links = []
        for li in list_soup.find_all("li"):
            a_tags = li.find_all("a", href=True)
            for a in a_tags:
                href = a["href"]
                if "VIDE" in href and "tv.cctv.com" in href:
                    # 取纯文本标题
                    title = a.get_text(separator=" ", strip=True)
                    # 去掉可能的前缀标记
                    title = title.replace("[视频]", "").strip()
                    if title and href not in [n["url"] for n in news_links]:
                        news_links.append({"url": href, "title": title})

        if not news_links:
            logger.warning(f"[CCTV] 未找到新闻链接，可能页面结构已变更")
            return None

        logger.info(f"[CCTV] 找到 {len(news_links)} 条新闻链接，开始逐条抓取...")

        segments = []
        for idx, nl in enumerate(news_links):
            try:
                item_resp = requests.get(
                    nl["url"],
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/130.0.0.0 Safari/537.36"
                        ),
                    },
                    timeout=15,
                )
                item_resp.encoding = "utf-8"
                item_soup = BeautifulSoup(item_resp.text, "lxml")

                # 提取标题
                h3 = item_soup.find("h3")
                title = h3.get_text(strip=True) if h3 else nl["title"]
                title = title.replace("[视频]", "").strip()

                # 提取正文（多种可能的容器）
                content_div = (
                    item_soup.find("div", class_="cnt_bd") or
                    item_soup.find("div", class_="content_area") or
                    item_soup.find("div", class_="allcontent") or
                    item_soup.find("div", class_="video-content") or
                    item_soup.find("article") or
                    item_soup.find("div", class_="text")
                )
                if content_div:
                    for tag in content_div(["script", "style"]):
                        tag.decompose()
                    content = content_div.get_text(separator="\n", strip=True)
                    content = content.strip()

                    # 如果包含"主要内容"标记，截取该标记之后的部分
                    main_marker = content.find("主要内容")
                    if main_marker >= 0:
                        content = content[main_marker + 4:].strip()

                    for prefix in ["央视网消息（新闻联播）：", "央视网消息(新闻联播)：",
                                   "央视网消息（新闻联播）", "央视网消息(新闻联播)",
                                   "央视网消息:"]:
                        if content.startswith(prefix):
                            content = content[len(prefix):].strip()
                else:
                    # 回退：取所有 p 标签文本
                    paras = item_soup.find_all("p")
                    content = "\n".join(p.get_text(strip=True) for p in paras if p.get_text(strip=True))
                    if not content:
                        body = item_soup.find("body")
                        if body:
                            for tag in body(["script", "style", "nav", "header", "footer"]):
                                tag.decompose()
                        content = body.get_text(separator="\n", strip=True) if body else ""

                # 过滤掉过短的内容（可能是纯视频无文字稿）
                if len(content) < 20:
                    continue

                if content:
                    segments.append({"title": title, "content": content})

            except Exception as e:
                logger.warning(f"[CCTV] 抓取 {nl['url'][-30:]} 失败: {e}")
                continue

        if not segments:
            logger.warning("[CCTV] 未能提取任何分段内容")
            return None

        # 拼接完整文字稿
        parts = []
        for seg in segments:
            parts.append(f"【{seg['title']}】")
            parts.append(seg["content"])
            parts.append("")
        raw_text = "\n".join(parts).strip()

        total_chars = sum(len(s["content"]) for s in segments)
        logger.info(
            f"[CCTV] 成功获取 {date_compact} 的文字稿，"
            f"共 {len(raw_text)} 字（纯正文 {total_chars} 字），"
            f"{len(segments)} 条新闻"
        )

        return {
            "date": target_date.isoformat(),
            "raw_text": raw_text,
            "segments": segments,
            "source": "cctv",
            "fetch_time": datetime.now().isoformat(),
        }

    except requests.RequestException as e:
        logger.warning(f"[CCTV] 网络请求失败: {e}")
        return None
    except Exception as e:
        logger.warning(f"[CCTV] 解析失败: {e}")
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
            "segments": [],  # govopendata 无法提取分段
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

    # 方案 1: CCTV 官网（最权威，优先）
    result = fetch_via_cctv(target_date)
    if result:
        return result

    # 方案 2: AkShare
    logger.info("CCTV 未获取到数据，尝试 AkShare...")
    result = fetch_via_akshare(target_date)
    if result:
        return result

    # 方案 3: cn.govopendata.com
    logger.info("AkShare 未获取到数据，尝试备选方案 govopendata...")
    result = fetch_via_govopendata(target_date)
    if result:
        return result

    logger.warning(f"所有数据源均无法获取 {target_date.isoformat()} 的文字稿")
    return None
