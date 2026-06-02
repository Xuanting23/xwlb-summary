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

        # 最低分段数阈值：不足3条视为数据尚未就绪，返回 None 触发重试
        if len(segments) < 3:
            logger.warning(
                f"[AkShare] {date_str} 有效分段不足 ({len(segments)} 条 < 3)，"
                f"数据可能尚未就绪，触发重试"
            )
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
        skipped = []  # 记录跳过的链接及原因
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

                # 提取标题（尝试多种选择器）
                title = nl["title"]
                for title_sel in ["h3", "h1", "h2", ".title", ".cnt_title", "title"]:
                    h_tag = item_soup.find(title_sel) if title_sel.startswith(".") else item_soup.find(title_sel)
                    if not h_tag and title_sel.startswith("."):
                        h_tag = item_soup.find(class_=title_sel[1:])
                    if h_tag:
                        t = h_tag.get_text(strip=True)
                        if t and len(t) >= 4:
                            title = t
                            break
                title = title.replace("[视频]", "").strip()

                # 提取正文（多种可能的容器，按优先级尝试）
                content_div_selectors = [
                    ("div", "cnt_bd"),
                    ("div", "content_area"),
                    ("div", "allcontent"),
                    ("div", "video-content"),
                    ("div", "video_content"),
                    ("div", "content"),
                    ("div", "text"),
                    ("div", "article-content"),
                    ("div", "main_content"),
                    ("div", "detail_content"),
                    ("div", "post_content"),
                    ("div", "entry-content"),
                    ("div", "post-body"),
                    ("div", "article-body"),
                    ("div", "news-content"),
                    ("div", "news_content"),
                    ("div", "body"),              # 备选
                    ("div", "main"),              # 备选
                    ("article", None),
                    ("section", "content"),
                    ("section", "article"),
                ]
                content_div = None
                for tag_name, cls in content_div_selectors:
                    if cls:
                        content_div = item_soup.find(tag_name, class_=cls)
                    else:
                        content_div = item_soup.find(tag_name)
                    if content_div:
                        break

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
                                   "央视网消息:", "央视网消息（"]:
                        if content.startswith(prefix):
                            content = content[len(prefix):].strip()

                    # 如果 div 提取的内容也过短，尝试用 p 标签
                    if len(content) < 20:
                        paras = item_soup.find_all("p")
                        p_text = "\n".join(p.get_text(strip=True) for p in paras if p.get_text(strip=True))
                        if len(p_text) > len(content):
                            content = p_text.strip()
                else:
                    # 回退：取所有 p 标签文本
                    paras = item_soup.find_all("p")
                    content = "\n".join(p.get_text(strip=True) for p in paras if p.get_text(strip=True))
                    if not content:
                        # 进一步回退：取 body 文本
                        body = item_soup.find("body")
                        if body:
                            for tag in body(["script", "style", "nav", "header", "footer"]):
                                tag.decompose()
                            content = body.get_text(separator="\n", strip=True) if body else ""
                        else:
                            content = ""

                # 记录跳过的链接及原因
                if len(content) < 20:
                    reason = f"内容过短({len(content)}字)"
                    skipped.append((nl["title"][:40], nl["url"][-50:], reason))
                    logger.warning(f"[CCTV] 跳过 [{idx}] {nl['title'][:40]}: {reason}")
                    continue

                # 去重：仅按 URL 去重（链接提取阶段已做），不再按内容相似度判断
                # 不同 URL 返回相似内容属于央视自身问题，不应由我们过滤
                url_already_seen = any(s.get("_url") == nl["url"] for s in segments)
                if url_already_seen:
                    reason = "URL 重复"
                    skipped.append((nl["title"][:40], nl["url"][-50:], reason))
                    logger.warning(f"[CCTV] 跳过 [{idx}] {nl['title'][:40]}: {reason}")
                    continue

                if content:
                    segments.append({"title": title, "content": content, "_url": nl["url"]})

            except requests.RequestException as e:
                reason = f"网络错误: {e}"
                skipped.append((nl["title"][:40], nl["url"][-50:], reason))
                logger.warning(f"[CCTV] 抓取 [{idx}] {nl['title'][:40]} 失败: {reason}")
                continue
            except Exception as e:
                reason = f"解析异常: {e}"
                skipped.append((nl["title"][:40], nl["url"][-50:], reason))
                logger.warning(f"[CCTV] 抓取 [{idx}] {nl['title'][:40]} 失败: {reason}")
                continue

        # 汇总日志
        if skipped:
            logger.warning(
                f"[CCTV] 抓取汇总: {len(segments)}/{len(news_links)} 成功, "
                f"{len(skipped)} 条被跳过"
            )
            for i, (title, url_tail, reason) in enumerate(skipped):
                logger.warning(f"[CCTV]   跳过[{i}]: {title} | {reason} | ...{url_tail}")

        if not segments:
            logger.warning("[CCTV] 未能提取任何分段内容")
            return None

        # 过滤掉非新闻内容的分段（节目片头、纯导航页等）
        filtered_segments = []
        for seg in segments:
            title = seg.get("title", "")
            content = seg.get("content", "")

            # 跳过节目片头/内容提要（如"《新闻联播》 20260601 19:00"）
            # 标志：标题含"新闻联播"且内容以"本期节目主要内容"开头
            if "新闻联播" in title and "本期节目主要内容" in content[:100]:
                logger.info(f"[CCTV] 过滤节目片头/提要: {title[:50]}")
                continue

            # 跳过明显不是新闻内容的纯导航/广告文字
            if len(content) < 10:
                logger.info(f"[CCTV] 过滤过短分段: {title[:50]} ({len(content)}字)")
                continue

            # 清洗正文中的网页UI残留文字
            for boilerplate in [
                "查看更多评论", "京ICP备", "中央广播电视总台央视网版权所有",
                "央视网版权所有", "返回顶部", "扫一扫", "分享到",
            ]:
                pos = content.find(boilerplate)
                if pos >= 0:
                    content = content[:pos].strip()

            # 如果清洗后内容过短则跳过
            if len(content) < 10:
                logger.info(f"[CCTV] 过滤清洗后过短分段: {title[:50]}")
                continue

            seg["content"] = content
            filtered_segments.append(seg)

        if len(filtered_segments) < len(segments):
            logger.info(
                f"[CCTV] 过滤后: {len(filtered_segments)}/{len(segments)} 条有效分段"
            )
        segments = filtered_segments

        if not segments:
            logger.warning("[CCTV] 过滤后无有效分段")
            return None

        # 最低分段数阈值：不足3条视为抓取失败，触发后续回退
        if len(segments) < 3:
            logger.warning(
                f"[CCTV] 有效分段不足 ({len(segments)} 条 < 3)，视为抓取失败，触发回退"
            )
            return None

        # 清理内部标记字段
        for seg in segments:
            seg.pop("_url", None)

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
    优先 AkShare，其次 CCTV 官网（≥3 条分段才算成功），最后尝试 govopendata。

    Args:
        target_date: 目标日期

    Returns:
        dict: {date, raw_text, source, fetch_time} 或 None
    """
    logger.info(f"===== 开始获取 {target_date.isoformat()} 的新闻联播文字稿 =====")

    # 方案 1: AkShare（最可靠，优先）
    result = fetch_via_akshare(target_date)
    if result:
        return result

    # 方案 2: CCTV 官网（备选，有分段数阈值兜底）
    logger.info("AkShare 未获取到数据，尝试 CCTV 官网...")
    result = fetch_via_cctv(target_date)
    if result:
        return result

    # 方案 3: cn.govopendata.com（最后备选）
    logger.info("CCTV 未获取到数据，尝试备选方案 govopendata...")
    result = fetch_via_govopendata(target_date)
    if result:
        return result

    logger.warning(f"所有数据源均无法获取 {target_date.isoformat()} 的文字稿")
    return None
