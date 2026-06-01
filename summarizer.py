"""
AI 摘要生成模块
使用 DeepSeek V4 Flash API 将新闻联播文字稿转换为结构化摘要
"""

import json
import logging
import re
from datetime import date, datetime
from typing import Optional

from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

logger = logging.getLogger(__name__)

# 初始化 DeepSeek 客户端（兼容 OpenAI SDK）
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

SYSTEM_PROMPT = """你是一位资深的新闻编辑，擅长对《新闻联播》内容进行专业、精炼的摘要整理。

你的任务：分析提供的新闻联播文字稿，输出一份结构化的 JSON 摘要。

## 输出要求

请严格按照以下 JSON 结构输出：

{
  "overview": "用 2-3 句话概括今日新闻联播的总体主题和基调",
  "top_news": [
    {
      "title": "新闻标题（简洁准确）",
      "summary": "一句话概括该条新闻的核心内容（30-50字）",
      "category": "政治/经济/科技/民生/外交/军事/文化/教育/生态",
      "importance": 5
    }
  ],
  "domestic_briefs": [
    {
      "title": "快讯标题",
      "summary": "一句话概括（20-30字）"
    }
  ],
  "international_news": [
    {
      "title": "国际新闻标题",
      "summary": "一句话概括核心内容（30-50字）",
      "region": "地区（如：中东/欧洲/亚太/北美/非洲等）"
    }
  ],
  "international_briefs": [
    {
      "title": "国际快讯标题",
      "summary": "一句话概括（20-30字）"
    }
  ],
  "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]
}

## 规则

1. **top_news**: 挑选 3-5 条最重要的国内要闻，按重要性排序。importance 为 1-5 星（5=最重要）。
2. **domestic_briefs**: 国内联播快讯，每条 1-2 句话，通常 3-8 条。如果没有则返回空数组 []。
3. **international_news**: 国际新闻详细摘要，通常 2-5 条。如果没有则返回空数组 []。
4. **international_briefs**: 国际快讯，通常 2-5 条。如果没有则返回空数组 []。
5. **keywords**: 5 个最具代表性的关键词，用于快速了解今日焦点。
6. 只输出 JSON，不要包含任何其他文字（不要 markdown 代码块标记）。
7. 标题和摘要务必简洁，避免冗余修饰词。"""


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


def generate_summary(raw_text: str, target_date: date) -> Optional[dict]:
    """
    调用 DeepSeek API 生成结构化摘要。

    Args:
        raw_text: 新闻联播原始文字稿
        target_date: 日期

    Returns:
        dict: 结构化摘要，含元数据；失败返回 None
    """
    if not raw_text or len(raw_text.strip()) < 50:
        logger.warning("输入文字稿过短，跳过摘要生成")
        return None

    # 限制输入长度（DeepSeek 128K 上下文足够，但控制成本）
    # 新闻联播每期约 5000-8000 字，保留完整内容
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
