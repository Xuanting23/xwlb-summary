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

你的任务：分析提供的新闻联播文字稿（每条新闻已用 [S0], [S1] 等编号标注），将文字稿中每一句话、每一个字都覆盖到，输出一份结构化的 JSON 摘要，并为每条摘要标注对应的分段编号。

## 核心原则：逐字全覆盖

**这是最重要的要求：你的摘要 + 原文必须覆盖文字稿的每一个字。不允许跳过、遗漏、合并任何内容。从 [S0] 开始到最后一个分段结束，原稿中的每一个字都必须出现在某条摘要的 original_content 中。**

## 输出要求

请严格按照以下 JSON 结构输出：

{
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

**第一条（最重要）：你必须逐条列出原文中出现的每一条新闻，不允许跳过、合并、遗漏任何一条。新闻联播是国家权威新闻节目，每一条都重要。文字稿中的每一个字都必须属于某条摘要。**

1. **top_news（今日要闻）**: 国内重要新闻。报道篇幅较长、涉及国家领导人/重大政策/经济形势/重要事件的归入此类。每个 [SX] 编号对应一条独立的新闻，按播出顺序排列。importance 为 1-5 星。篇幅较短的例行通报、数据发布、一般性活动报道等不属于要闻。
2. **domestic_briefs（国内快讯）**: 国内简短新闻。篇幅较短（原文通常少于 300 字）、内容为数据通报/一般性活动/常规工作部署等。这些常出现在联播快讯板块。注意区分：短新闻归快讯，长报道归要闻。
3. **international_news（国际新闻）**: 涉及外国/国际组织的新闻。按播出顺序列出。注意区分：涉及中国的对外交往（如中国领导人会见外国政要）属于要闻，纯外国/国际事务属于国际新闻。
4. **international_briefs（国际快讯）**: 短篇国际新闻，原文少于 300 字。
5. **keywords**: 5 个最具代表性的关键词。
6. **segment_id**: 每条摘要填写对应的 [SX] 编号。如果从一个分段中拆出多条新闻，则这几条共用同一个 segment_id。
7. **严禁合并**: 四个板块中，每个 JSON 对象只描述一条新闻。绝对禁止把多条新闻写在一个对象里。有多少条就输出多少个 JSON 对象。
8. **全覆盖验证**: 输出前请自检——原文的每个 [SX] 分段是否都被至少一条摘要引用？如果某个分段没有被引用，说明有遗漏，必须补上。
9. 只输出 JSON，不要包含任何其他文字。"""


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
                max_tokens=16384,
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


def _find_title_boundaries(content: str, items: list) -> list[dict]:
    """在分段原文中按标题/关键信息定位每条摘要的起止边界，返回排序后的切片。"""
    boundaries = []
    for idx, item in enumerate(items):
        title = item.get("title", "")
        best_pos = -1
        best_len = 0

        # 策略 A: 标题子串精确匹配（从长到短）
        for length in range(len(title), 2, -1):
            for offset in range(0, len(title) - length + 1):
                sub = title[offset:offset + length]
                pos = content.find(sub)
                if pos != -1 and length > best_len:
                    best_pos = pos
                    best_len = length

        # 策略 B: 字符级模糊匹配 — 取标题每个字，在原文中找首次出现的聚集位置
        if best_pos < 0:
            chars = [ch for ch in title if len(ch.strip()) >= 0]
            first_pos = -1
            for ch in chars[:6]:  # 用前6个字定位
                p = content.find(ch)
                if p != -1 and (first_pos < 0 or p < first_pos):
                    first_pos = p
            if first_pos >= 0:
                # 从 first_pos 向前找句子开头
                for ch2 in chars[:3]:
                    p2 = content.rfind(ch2, 0, first_pos + 10)
                    if p2 >= 0 and p2 < first_pos:
                        first_pos = p2
                best_pos = first_pos
                best_len = 1

        # 策略 C: 摘要关键词搜索
        if best_pos < 0:
            summary = item.get("summary", "")
            for length in range(min(len(summary), 40), 3, -1):
                for offset in range(0, len(summary) - length + 1):
                    sub = summary[offset:offset + length]
                    pos = content.find(sub)
                    if pos != -1:
                        best_pos = pos
                        best_len = length
                        break
                if best_pos >= 0:
                    break

        if best_pos >= 0:
            boundaries.append({"idx": idx, "pos": best_pos,
                               "match": content[best_pos:best_pos+min(best_len,40)],
                               "len": best_len})

    # 按位置排序
    boundaries.sort(key=lambda b: b["pos"])
    return boundaries


def _split_shared_segment(content: str, items: list) -> None:
    """
    将共享同一个分段的多个条目按原标题边界精确切分，
    每条只取自己对应的原文片段。原地修改 item['original_content']。
    多条拼接后 = 分段全文，一字不漏。
    """
    if len(items) <= 1:
        for item in items:
            item["original_content"] = content
        return

    boundaries = _find_title_boundaries(content, items)
    if len(boundaries) < 2:
        # 无法可靠切分，保持完整分段（降级）
        for item in items:
            item["original_content"] = content
        return

    # 为每条分配片段：从自己的边界到下一条的边界（第一条从 0 开始，最后一条到文末）
    for bi, b in enumerate(boundaries):
        start = 0 if bi == 0 else b["pos"]
        end = boundaries[bi + 1]["pos"] if bi + 1 < len(boundaries) else len(content)
        snippet = content[start:end]
        items[b["idx"]]["original_content"] = snippet

    # 未匹配到的条目用完整分段兜底
    matched_indices = {b["idx"] for b in boundaries}
    for i, item in enumerate(items):
        if i not in matched_indices or not item.get("original_content"):
            item["original_content"] = content


def match_segments_to_summary(summary: dict, segments: list) -> dict:
    """
    为 AI 摘要每个条目匹配原始分段原文。

    匹配策略（按优先级）：
    1. AI 标注的 segment_id → 直接取该分段完整原文
    2. 综合模糊评分 → 标题+关键词+摘要文本
    3. 内容子串搜索 → 在分段正文中查找标题关键词

    关键：当一个分段被多条摘要共享时，按标题边界精确切分，
    每条只拿自己的片段。拼接后 = 分段全文，一字不漏。
    """
    if not segments:
        return summary

    categories = ["top_news", "international_news", "domestic_briefs", "international_briefs"]
    all_items = []
    for cat in categories:
        for item in summary.get(cat, []):
            all_items.append((cat, item))

    # ---- 阶段 1: 初始匹配（整段分配） ----
    segment_hits = 0
    fuzzy_hits = 0
    content_hits = 0

    # 记录每个分段被哪些条目标注（用于后续切分）
    seg_to_items: dict[int, list[dict]] = {i: [] for i in range(len(segments))}

    for category, item in all_items:
        matched_seg = None

        # 策略 1: AI 的 segment_id
        seg_id = item.pop("segment_id", None)
        if seg_id is not None and isinstance(seg_id, int) and 0 <= seg_id < len(segments):
            matched_seg = seg_id
            segment_hits += 1

        # 策略 2: 综合模糊评分
        if matched_seg is None:
            best_score, best_seg_idx = 0, -1
            for i, seg in enumerate(segments):
                score = _composite_score(item, seg)
                if score > best_score:
                    best_score = score
                    best_seg_idx = i
            threshold = 0.20 if category in ("top_news", "international_news") else 0.10
            if best_score >= threshold and best_seg_idx >= 0:
                matched_seg = best_seg_idx
                fuzzy_hits += 1

        # 策略 3: 内容子串搜索
        if matched_seg is None:
            title = item.get("title", "")
            keywords = [title[i:i+3] for i in range(0, max(len(title)-2, 0), 2) if len(title[i:i+3]) >= 3]
            for seg_idx, seg in enumerate(segments):
                if any(kw in seg["content"] for kw in keywords):
                    matched_seg = seg_idx
                    content_hits += 1
                    break

        if matched_seg is not None:
            seg_to_items[matched_seg].append(item)

    # ---- 阶段 2: 按分段切分原文 ----
    for seg_idx, items in seg_to_items.items():
        if not items:
            continue
        content = segments[seg_idx]["content"]
        _split_shared_segment(content, items)

    # ---- 阶段 3: 覆盖率验证 ----
    total_chars = sum(len(seg["content"]) for seg in segments)
    referenced = [False] * len(segments)
    for seg_idx, items in seg_to_items.items():
        if items:
            referenced[seg_idx] = True

    covered_segments = sum(referenced)
    covered_chars = sum(len(seg["content"]) for i, seg in enumerate(segments) if referenced[i])
    coverage_pct = round(covered_chars / max(total_chars, 1) * 100, 1)
    uncovered = [i for i, ref in enumerate(referenced) if not ref]

    # 计算每个分段内切分的片段总长度
    for seg_idx in range(len(segments)):
        items = seg_to_items.get(seg_idx, [])
        if len(items) > 1:
            split_total = sum(len(it.get("original_content", "")) for it in items)
            seg_total = len(segments[seg_idx]["content"])
            if split_total < seg_total * 0.5:
                logger.warning(
                    f"S{seg_idx} 切分可能不完整: {len(items)}条, "
                    f"切分总长={split_total}/{seg_total}"
                )

    # 统计
    counts = {}
    for cat in categories:
        items = summary.get(cat, [])
        ok = sum(1 for it in items if it.get("original_content"))
        counts[cat] = f"{ok}/{len(items)}"

    items_with_shared = sum(1 for seg_idx, items in seg_to_items.items() if len(items) > 1)
    logger.info(
        f"分段匹配: AI标注 {segment_hits} + 模糊 {fuzzy_hits} + 子串 {content_hits} = "
        f"{segment_hits + fuzzy_hits + content_hits} 条 "
        f"(要闻 {counts['top_news']}, 国际 {counts['international_news']}, "
        f"国内快讯 {counts['domestic_briefs']}, 国际快讯 {counts['international_briefs']})"
    )
    logger.info(
        f"分段切分: {items_with_shared} 个共享分段已切分, "
        f"覆盖率 {covered_segments}/{len(segments)} 分段, "
        f"{covered_chars}/{total_chars} 字符 ({coverage_pct}%)"
    )
    if uncovered:
        logger.warning(f"未覆盖的分段: S{uncovered}，原文可能有遗漏！")

    return summary
