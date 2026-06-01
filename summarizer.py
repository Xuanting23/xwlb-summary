"""
AI 摘要生成模块
使用 DeepSeek V4 Flash API 将新闻联播文字稿转换为结构化摘要
包含标题模糊匹配，将 AI 摘要关联到原始新闻分段
"""

import json
import logging
import re
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Optional

from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

logger = logging.getLogger(__name__)

# 初始化 DeepSeek 客户端（兼容 OpenAI SDK）
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

SYSTEM_PROMPT = """你是一位资深的新闻编辑，擅长对《新闻联播》内容进行专业、精炼的摘要整理。

你的任务：分析提供的新闻联播文字稿（每条新闻已用 [S0], [S1] 等编号标注），输出一份结构化的 JSON 摘要。

## 输出要求

请严格按照以下 JSON 结构输出：

{
  "overview": "用 2-3 句话概括今日新闻联播的总体主题和基调",
  "top_news": [
    {
      "title": "新闻标题（简洁准确）",
      "summary": "一句话概括该条新闻的核心内容（30-50字）",
      "category": "政治/经济/科技/民生/外交/军事/文化/教育/生态",
      "importance": 5,
      "segment_id": 0
    }
  ],
  "domestic_briefs_summary": "用一段话概括所有国内联播快讯的主要内容（2-4句话）。如果没有快讯则用空字符串 \"\"。",
  "international_news": [
    {
      "title": "国际新闻标题",
      "summary": "一句话概括核心内容（30-50字）",
      "region": "地区（如：中东/欧洲/亚太/北美/非洲等）",
      "segment_id": 8
    }
  ],
  "international_briefs_summary": "用一段话概括所有国际快讯的内容（1-3句话）。如果没有则用空字符串 \"\"。",
  "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]
}

## 规则

1. **top_news**: 挑选 3-5 条最重要的国内要闻，按重要性排序。importance 为 1-5 星。
2. **domestic_briefs_summary**: 将所有国内联播快讯合并写成一段话综述，不分条列举。
3. **international_news**: 国际新闻的独立摘要，通常 1-4 条。
4. **international_briefs_summary**: 将所有国际快讯合并写成一段话综述，不分条列举。
5. **keywords**: 5 个最具代表性的关键词。
6. **segment_id**: top_news 和 international_news 的每条必须填写对应分段编号 [S0][S1]...。务必准确！
7. 只输出 JSON，不要包含任何其他文字。"""


def _clean_json_response(text: str) -> str:
    """清理模型返回的文本，提取纯 JSON 部分。"""
    # 移除可能的 markdown 代码块标记
    text = text.strip()

    # 匹配 ```json ... ``` 或 ``` ... ```
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    # 尝试找到第一个 { 和最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return text


def _format_segments_for_prompt(segments: list) -> str:
    """将分段列表格式化为带编号的 prompt 文本。"""
    if not segments:
        return ""
    lines = []
    for i, seg in enumerate(segments):
        title = seg.get("title", "")
        content = seg.get("content", "")
        lines.append(f"[S{i}] {title}")
        lines.append(content)
        lines.append("")
    return "\n".join(lines)


def generate_summary(raw_text: str, target_date: date,
                     segments: list | None = None) -> Optional[dict]:
    """
    调用 DeepSeek API 生成结构化摘要。

    Args:
        raw_text: 新闻联播原始文字稿
        target_date: 日期
        segments: 原始分段列表（可选，用于 segment_id 标注）

    Returns:
        dict: 结构化摘要，含元数据；失败返回 None
    """
    if not raw_text or len(raw_text.strip()) < 50:
        logger.warning("输入文字稿过短，跳过摘要生成")
        return None

    # 构建带有分段编号的输入文本
    if segments:
        input_text = _format_segments_for_prompt(segments)
        logger.info(f"[DeepSeek] 使用分段模式，共 {len(segments)} 条")
    else:
        input_text = raw_text[:15000]

    for attempt in range(2):  # 最多尝试 2 次
        try:
            logger.info(f"[DeepSeek] 摘要生成尝试 {attempt + 1}/2...")

            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"以下是 {target_date.isoformat()} 新闻联播的文字稿，请生成结构化摘要：\n\n{input_text}"},
                ],
                temperature=0.3,  # 低温度保证输出稳定
                max_tokens=4096,
                response_format={"type": "json_object"},  # DeepSeek 支持 JSON 模式
            )

            content = response.choices[0].message.content
            usage = response.usage

            # 清理并解析 JSON
            cleaned = _clean_json_response(content)
            summary = json.loads(cleaned)

            # 补充元数据
            summary["_meta"] = {
                "date": target_date.isoformat(),
                "model": DEEPSEEK_MODEL,
                "tokens_input": usage.prompt_tokens if usage else 0,
                "tokens_output": usage.completion_tokens if usage else 0,
                "generated_at": datetime.now().isoformat(),
            }

            logger.info(
                f"[DeepSeek] 摘要生成成功 "
                f"(input: {summary['_meta']['tokens_input']} tokens, "
                f"output: {summary['_meta']['tokens_output']} tokens)"
            )
            return summary

        except json.JSONDecodeError as e:
            logger.warning(f"[DeepSeek] JSON 解析失败 (尝试 {attempt + 1}): {e}")
            if attempt == 1:
                logger.error("[DeepSeek] 两次尝试均失败，返回 None")
                return None

        except Exception as e:
            logger.error(f"[DeepSeek] API 调用失败 (尝试 {attempt + 1}): {e}")
            if attempt == 1:
                return None

    return None


def summary_to_html(summary: dict) -> str:
    """
    将结构化摘要转换为简单的 HTML 片段（用于模板内嵌展示）。
    可选工具函数，方便在模板中直接渲染。

    Args:
        summary: generate_summary 返回的 dict

    Returns:
        str: HTML 字符串
    """
    if not summary:
        return "<p class='no-data'>暂无摘要数据</p>"

    html_parts = []

    # 概述
    overview = summary.get("overview", "")
    if overview:
        html_parts.append(f'<div class="overview"><p class="overview-text">{overview}</p></div>')

    # Top 要闻
    top_news = summary.get("top_news", [])
    if top_news:
        html_parts.append('<section class="news-section"><h2>📌 今日要闻</h2><div class="news-list">')
        for item in top_news:
            stars = "⭐" * item.get("importance", 3)
            category = item.get("category", "")
            html_parts.append(
                f'<div class="news-card top-news">'
                f'<h3>{item["title"]}</h3>'
                f'<p class="summary">{item["summary"]}</p>'
                f'<div class="meta"><span class="category-tag">{category}</span><span class="stars">{stars}</span></div>'
                f'</div>'
            )
        html_parts.append('</div></section>')

    # 国内快讯
    briefs = summary.get("domestic_briefs", [])
    if briefs:
        html_parts.append('<section class="news-section"><h2>🇨🇳 国内快讯</h2><ul class="brief-list">')
        for item in briefs:
            html_parts.append(f'<li><strong>{item["title"]}</strong>：{item["summary"]}</li>')
        html_parts.append('</ul></section>')

    # 国际新闻
    intl = summary.get("international_news", [])
    if intl:
        html_parts.append('<section class="news-section"><h2>🌍 国际新闻</h2><div class="news-list">')
        for item in intl:
            region = item.get("region", "")
            html_parts.append(
                f'<div class="news-card intl-news">'
                f'<h3>{item["title"]}</h3>'
                f'<p class="summary">{item["summary"]}</p>'
                f'<div class="meta"><span class="region-tag">{region}</span></div>'
                f'</div>'
            )
        html_parts.append('</div></section>')

    # 国际快讯
    intl_briefs = summary.get("international_briefs", [])
    if intl_briefs:
        html_parts.append('<section class="news-section"><h2>🌐 国际快讯</h2><ul class="brief-list">')
        for item in intl_briefs:
            html_parts.append(f'<li><strong>{item["title"]}</strong>：{item["summary"]}</li>')
        html_parts.append('</ul></section>')

    # 关键词
    keywords = summary.get("keywords", [])
    if keywords:
        tags = "".join(f'<span class="keyword-tag">{kw}</span>' for kw in keywords)
        html_parts.append(f'<section class="keywords-section"><h2>🔑 关键词</h2><div class="keywords">{tags}</div></section>')

    return "\n".join(html_parts)


def _title_similarity(a: str, b: str) -> float:
    """计算两个标题的字符串相似度 (0~1)。"""
    return SequenceMatcher(None, a, b).ratio()


def _keyword_overlap(ai_title: str, seg_title: str, seg_content: str) -> float:
    """
    计算 AI 标题中的关键词在原始分段标题+内容中的覆盖率。
    提取 AI 标题中的核心词（长度>=2），看有多少出现在分段中。
    """
    # 从 AI 标题中提取有效片段（用常见标点分割）
    import re
    parts = re.split(r"[，,、\s]+", ai_title)
    if not parts:
        return 0

    combined = seg_title + seg_content
    scores = []
    for part in parts:
        part = part.strip()
        if len(part) >= 2:
            if part in combined:
                scores.append(1.0)
            else:
                # 部分匹配：检查逐字覆盖率
                chars = [c for c in part if c in combined]
                scores.append(len(chars) / max(len(part), 1))

    return sum(scores) / max(len(scores), 1)


def _composite_score(ai_item: dict, seg: dict) -> float:
    """
    综合评分：标题相似度 40% + 关键词覆盖率 35% + 摘要文本匹配 25%
    """
    ai_title = ai_item.get("title", "")
    ai_summary = ai_item.get("summary", "")
    seg_title = seg.get("title", "")
    seg_content = seg.get("content", "")

    # 1. 标题相似度
    title_score = _title_similarity(ai_title, seg_title)

    # 2. 关键词覆盖率（分别在标题和内容中检查）
    kw_score = _keyword_overlap(ai_title, seg_title, seg_content)

    # 3. AI 摘要文本是否出现在分段内容中
    if ai_summary and len(ai_summary) >= 8:
        # 取摘要的前15个字符，在分段内容中查找
        snippet = ai_summary[:15]
        if snippet in seg_content:
            summary_score = 1.0
        else:
            summary_score = _title_similarity(ai_summary[:30], seg_content[:200])
    else:
        summary_score = 0

    return title_score * 0.40 + kw_score * 0.35 + summary_score * 0.25


def _is_brief_segment(seg: dict) -> bool:
    """判断一个分段是否属于快讯类（标题短、或含快讯关键词）。"""
    title = seg.get("title", "")
    # 标题含"快讯"二字
    if "快讯" in title:
        return True
    # 标题很短（典型的快讯汇总标题）
    if len(title) <= 15 and len(seg.get("content", "")) > 200:
        return True
    return False


def match_segments_to_summary(summary: dict, segments: list) -> dict:
    """
    为 AI 摘要关联原始分段。

    - top_news / international_news: 通过 AI 的 segment_id 精确匹配 + 模糊兜底
    - briefs_summary: 找到原文中的快讯类分段，拼接全文

    Args:
        summary: AI 生成的摘要 dict
        segments: 原始分段列表 [{title, content}, ...]

    Returns:
        更新后的 summary dict，条目新增 original_content / briefs_original 字段
    """
    if not segments:
        return summary

    used_ids = set()
    segment_hits = 0
    fuzzy_hits = 0

    # ===== 1. 精确条目匹配：top_news + international_news =====
    for cat in ["top_news", "international_news"]:
        for item in summary.get(cat, []):
            seg_id = item.pop("segment_id", None)
            if seg_id is not None and isinstance(seg_id, int) and 0 <= seg_id < len(segments):
                item["original_content"] = segments[seg_id]["content"]
                used_ids.add(seg_id)
                segment_hits += 1
                continue

            # 模糊兜底
            best_score, best_idx, best_content = 0, -1, None
            for i, seg in enumerate(segments):
                score = _composite_score(item, seg)
                if score > best_score:
                    best_score, best_idx, best_content = score, i, seg["content"]
            if best_score >= 0.22:
                item["original_content"] = best_content
                used_ids.add(best_idx)
                fuzzy_hits += 1
                continue

            # 终极兜底：子串搜索
            title = item.get("title", "")
            for seg in segments:
                if title[:4] in seg.get("content", "") or title[-4:] in seg.get("content", ""):
                    item["original_content"] = seg["content"]
                    fuzzy_hits += 1
                    break

    # ===== 2. 快讯综述：找到快讯类分段拼成原文 =====
    brief_segments = [
        seg for i, seg in enumerate(segments)
        if _is_brief_segment(seg) and i not in used_ids
    ]
    # 如果按标题没找到快讯段，用最后 1/3 的分段作为快讯段
    if not brief_segments:
        n = max(len(segments) // 3, 1)
        brief_segments = [
            seg for i, seg in enumerate(segments)
            if i >= len(segments) - n and i not in used_ids
        ]

    briefs_text = "\n\n".join(
        f"【{seg['title']}】\n{seg['content']}"
        for seg in brief_segments
    )

    if summary.get("domestic_briefs_summary"):
        summary["domestic_briefs_original"] = briefs_text

    if summary.get("international_briefs_summary"):
        # 国际快讯段：标题中含"国际"或靠后的分段
        intl_segs = [
            seg for i, seg in enumerate(segments)
            if ("国际" in seg.get("title", "")) and i not in used_ids
        ]
        intl_text = "\n\n".join(
            f"【{seg['title']}】\n{seg['content']}"
            for seg in (intl_segs or brief_segments)
        )
        summary["international_briefs_original"] = intl_text

    top_count = sum(1 for it in summary.get("top_news", []) if it.get("original_content"))
    top_total = len(summary.get("top_news", []))
    intl_count = sum(1 for it in summary.get("international_news", []) if it.get("original_content"))
    intl_total = len(summary.get("international_news", []))

    logger.info(
        f"分段匹配: AI标注 {segment_hits} + 模糊 {fuzzy_hits} "
        f"(要闻 {top_count}/{top_total}, 国际 {intl_count}/{intl_total}, "
        f"快讯原文 {len(brief_segments)} 段, 国际快讯原文 {len(intl_segs) if 'intl_segs' in dir() else 0} 段)"
    )

    return summary
