"""
AI 摘要生成模块
使用 DeepSeek V4 Pro API 将新闻联播文字稿转换为结构化摘要
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

你的任务：分析提供的新闻联播文字稿（每条新闻已用 [S0], [S1] 等编号标注），输出一份结构化的 JSON 摘要，并为每条摘要标注对应的分段编号。

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
  "domestic_briefs": [
    {
      "title": "快讯标题",
      "summary": "一句话概括（20-30字）",
      "segment_id": 6
    }
  ],
  "international_news": [
    {
      "title": "国际新闻标题",
      "summary": "一句话概括核心内容（30-50字）",
      "region": "地区（如：中东/欧洲/亚太/北美/非洲等）",
      "segment_id": 8
    }
  ],
  "international_briefs": [
    {
      "title": "国际快讯标题",
      "summary": "一句话概括（20-30字）",
      "segment_id": 10
    }
  ],
  "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]
}

## 规则

**最重要原则：你必须逐条列出原文中出现的每一条新闻，不允许跳过、合并、遗漏任何一条。新闻联播是国家权威新闻节目，每一条都重要。**

1. **top_news**: 国内要闻。每个 [SX] 编号对应一条独立的新闻，一条对应一个 JSON 对象，严禁合并。按播出顺序排列。importance 为 1-5 星。有几条输出几条。
2. **domestic_briefs**: 国内「联播快讯」。该板块会将多条快讯放在同一个分段中，请根据标题变化和话题切换将其拆分为独立的条目，逐一列出，严禁合并。每条都要写。
3. **international_news**: 国际新闻。注意：国际新闻经常被合并在同一个分段中，请仔细阅读内容，根据话题切换、国家/地区变化、事件变化将其拆分为独立条目。例如「A 国称…… B 国计划……」应拆为两条。每条都要写，逐一列出，严禁合并。
4. **international_briefs**: 国际快讯。同 domestic_briefs，多条快讯常在一个分段中，必须根据标题和话题变化拆分为独立条目。每条都要写。
5. **keywords**: 5 个最具代表性的关键词。
6. **segment_id**: 每条摘要填写对应的 [SX] 编号。如果从一个分段中拆出多条新闻，则这几条共用同一个 segment_id。
7. **严禁合并**: 四个板块中，每个 JSON 对象只描述一条新闻。绝对禁止把多条新闻写在一个对象里。有多少条就输出多少个 JSON 对象。
8. 只输出 JSON，不要包含任何其他文字。"""


def _clean_json_response(text: str) -> str:
    """清理模型返回的文本，提取纯 JSON 部分。"""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
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
    """调用 DeepSeek API 生成结构化摘要。"""
    if not raw_text or len(raw_text.strip()) < 50:
        logger.warning("输入文字稿过短，跳过摘要生成")
        return None

    if segments:
        input_text = _format_segments_for_prompt(segments)
        logger.info(f"[DeepSeek] 使用分段模式，共 {len(segments)} 条")
    else:
        input_text = raw_text[:15000]

    for attempt in range(2):
        try:
            logger.info(f"[DeepSeek] 摘要生成尝试 {attempt + 1}/2...")
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"以下是 {target_date.isoformat()} 新闻联播的文字稿，请生成结构化摘要：\n\n{input_text}"},
                ],
                temperature=0.3,
                max_tokens=8192,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            usage = response.usage
            cleaned = _clean_json_response(content)
            summary = json.loads(cleaned)
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
            logger.warning(f"[DeepSeek] JSON 解析失败 (attempt {attempt + 1}): {e}")
            if attempt == 1:
                return None
        except Exception as e:
            logger.error(f"[DeepSeek] API 调用失败 (attempt {attempt + 1}): {e}")
            if attempt == 1:
                return None
    return None


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _keyword_overlap(ai_title: str, seg_title: str, seg_content: str) -> float:
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
                chars = [c for c in part if c in combined]
                scores.append(len(chars) / max(len(part), 1))
    return sum(scores) / max(len(scores), 1)


def _composite_score(ai_item: dict, seg: dict) -> float:
    ai_title = ai_item.get("title", "")
    ai_summary = ai_item.get("summary", "")
    seg_title = seg.get("title", "")
    seg_content = seg.get("content", "")
    title_score = _title_similarity(ai_title, seg_title)
    kw_score = _keyword_overlap(ai_title, seg_title, seg_content)
    if ai_summary and len(ai_summary) >= 8:
        snippet = ai_summary[:15]
        if snippet in seg_content:
            summary_score = 1.0
        else:
            summary_score = _title_similarity(ai_summary[:30], seg_content[:200])
    else:
        summary_score = 0
    return title_score * 0.40 + kw_score * 0.35 + summary_score * 0.25


def _extract_relevant_passage(content: str, title: str, summary: str, window: int = 300) -> str:
    """从分段内容中精准摘取与摘要条目最相关的段落。"""
    if not content:
        return ""
    clean_title = re.sub(r'[，,、。！？\s\d]+', '', title)
    fragments = [clean_title[i:i+6] for i in range(0, len(clean_title)-3, 2)]
    fragments = [f for f in fragments if len(f) >= 4]

    if not fragments:
        return content

    best_pos = -1
    for frag in fragments[:3]:
        pos = content.find(frag)
        if pos != -1:
            best_pos = pos
            break

    if best_pos == -1:
        clean_sum = re.sub(r'[，,、。！？\s\d]+', '', summary)
        for i in range(0, min(len(clean_sum)-4, 30), 3):
            frag = clean_sum[i:i+6]
            if len(frag) >= 4:
                pos = content.find(frag)
                if pos != -1:
                    best_pos = pos
                    break

    if best_pos == -1:
        return content

    start = max(0, best_pos - 50)
    end = min(len(content), best_pos + window)

    left_period = content.rfind('。', start, best_pos)
    if left_period != -1:
        start = left_period + 1
    else:
        left_nl = content.rfind('\n', start, best_pos)
        if left_nl != -1:
            start = left_nl + 1

    right_period = content.find('。', best_pos + 20, end)
    if right_period != -1:
        end = right_period + 1

    passage = content[start:end].strip()
    return passage if passage else content


def match_segments_to_summary(summary: dict, segments: list) -> dict:
    """
    为 AI 摘要每个条目独立匹配原始分段，不做排他限制。

    匹配策略（按优先级）：
    1. AI 标注的 segment_id → 直接查分段
    2. 综合模糊评分 → 标题+关键词+摘要文本
    3. 内容子串搜索 → 在分段正文中查找标题关键词

    快讯类条目额外做精准摘取，只取相关段落而非整个分段。
    """
    if not segments:
        return summary

    categories = ["top_news", "international_news", "domestic_briefs", "international_briefs"]
    all_items = []
    for cat in categories:
        for item in summary.get(cat, []):
            all_items.append((cat, item))

    segment_hits = 0
    fuzzy_hits = 0
    content_hits = 0

    for category, item in all_items:
        # ---- 策略 1: AI 的 segment_id ----
        seg_id = item.pop("segment_id", None)
        if seg_id is not None and isinstance(seg_id, int) and 0 <= seg_id < len(segments):
            item["original_content"] = segments[seg_id]["content"]
            segment_hits += 1
            continue

        # ---- 策略 2: 综合模糊评分 ----
        best_score, best_content = 0, None
        for seg in segments:
            score = _composite_score(item, seg)
            if score > best_score:
                best_score = score
                best_content = seg["content"]

        threshold = 0.20 if category in ("top_news", "international_news") else 0.10
        if best_score >= threshold:
            item["original_content"] = best_content
            fuzzy_hits += 1
            continue

        # ---- 策略 3: 内容子串搜索 ----
        title = item.get("title", "")
        keywords = [title[i:i+3] for i in range(0, max(len(title)-2, 0), 2) if len(title[i:i+3]) >= 3]
        for seg in segments:
            content = seg.get("content", "")
            if any(kw in content for kw in keywords):
                item["original_content"] = seg["content"]
                content_hits += 1
                break

    # ---- 精准摘取：从整段原文中定位每条摘要的对应段落 ----
    for cat in ["domestic_briefs", "international_briefs", "international_news"]:
        for item in summary.get(cat, []):
            raw = item.get("original_content", "")
            if raw and len(raw) > 300:
                item["original_content"] = _extract_relevant_passage(
                    raw, item.get("title", ""), item.get("summary", "")
                )

    # 统计
    counts = {}
    for cat in categories:
        items = summary.get(cat, [])
        ok = sum(1 for it in items if it.get("original_content"))
        counts[cat] = f"{ok}/{len(items)}"

    logger.info(
        f"分段匹配: AI标注 {segment_hits} + 模糊 {fuzzy_hits} + 子串 {content_hits} = "
        f"{segment_hits + fuzzy_hits + content_hits} 条 "
        f"(要闻 {counts['top_news']}, 国际 {counts['international_news']}, "
        f"国内快讯 {counts['domestic_briefs']}, 国际快讯 {counts['international_briefs']})"
    )

    return summary
