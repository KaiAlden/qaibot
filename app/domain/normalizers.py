# 数据清洗与标准化工具模块，负责将用户输入的各种"非标准"表达方式转换为系统内部统一的"标准"格式。

from __future__ import annotations

import math
import re

from app.domain.constants import (
    AREA_ALIAS,
    CATEGORY_KEYWORDS,
    MONTH_TO_SEASON,
    SOLAR_TERM_TO_SEASON,
    SYMPTOM_HINTS,
    VALID_AREAS,
    VALID_CONSTITUTIONS,
)


QUESTION_WORDS = ["请问", "是不是", "是否", "什么", "哪种", "怎么", "如何", "可以", "应该", "有什么", "吗", "？", "?"]
SYMPTOM_FRAGMENT_HINTS = ["痛", "冷", "热", "汗", "干", "乏", "困", "闷", "胀", "黏", "虚", "油"]


def clean_text(value: object) -> str:
    """
        __作用__：去除文本中的各种"杂质"。
        处理内容：
            - __NaN 值__：Excel 单元格读到的 NaN 转成空字符串
            - __零宽字符__：去掉 `\u200b` 等不可见字符
            - __换行规范化__：`\r\n` → `\n`，`\r` → `\n`
            - __多余空行压缩__：3 个以上换行 → 2 个
    """
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = "" if value is None else str(value)
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def normalize_constitution(value: str | None) -> str | None:
    """
    - 先清洗文本
    - 在 `VALID_CONSTITUTIONS`（标准体质列表）中匹配
    - 匹配不到但以"体质"结尾 → 直接返回
    - 匹配不到也不以"体质"结尾 → 自动补上"体质"
    """
    if not value:
        return None
    text = clean_text(value)
    for name in VALID_CONSTITUTIONS:
        if name in text or name.replace("体质", "") in text:
            return name
    return text if text.endswith("体质") else f"{text}体质"


def normalize_area(value: str | None) -> str | None:
    """
    - 先清洗文本
    - 在 `VALID_AREAS`（标准地区列表）中匹配
    - 匹配不到 → 使用 `AREA_ALIAS` 进行别名转换
    """
    if not value:
        return None
    text = clean_text(value)
    if text in VALID_AREAS:
        return text
    for alias, area in AREA_ALIAS.items():
        if alias in text:
            return area
    return text


def normalize_term(value: str | None) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    text = clean_text(value)
    for term, season in SOLAR_TERM_TO_SEASON.items():
        if term in text:
            return term, season
    return text, SOLAR_TERM_TO_SEASON.get(text)


def detect_advice_type(text: str) -> str | None:
    for category, keywords in CATEGORY_KEYWORDS.items():
        if category == "diet_principle":
            continue
        if category in text or any(keyword in text for keyword in keywords):
            return category
    return None


def current_season(month: int) -> str:
    return MONTH_TO_SEASON[month]


def extract_symptoms(text: str) -> list[str]:
    """
    两阶段：
    1. 先用 `SYMPTOM_HINTS`（症状提示词列表）进行关键词匹配，快速捕获明显的症状线索---> 准确率
    2. 再用SYMPTOM_FRAGMENT_HINTS 特征字，对剩余文本进行切分，捕获更隐晦的症状描述片段---> 召回率
    3. 最后去重并限制数量，确保输出的症状关键词既丰富又相关
    """
    symptom_text = _remove_non_symptom_slots(clean_text(text)) # 第1步：清洗文本 + 移除噪音词
    hits = [keyword for keyword in SYMPTOM_HINTS if keyword in symptom_text] # 第2步：从预定义关键词列表匹配
    fragments = re.split(r"[，,。；;、\s]+", symptom_text) #第3步：按标点分割成片段
    for fragment in fragments:  # 第4步：用特征字捕获未预定义的症状
        if len(fragment) <= 1:
            continue
        if any(hint in fragment for hint in SYMPTOM_FRAGMENT_HINTS):
            hits.append(fragment)
    return list(dict.fromkeys([hit for hit in hits if hit]))[:12] # 第5步：去重 + 限制数量


def _remove_non_symptom_slots(text: str) -> str:
    """
     去掉已知的体质名、地区名、节气名、疑问词等"噪音"，
     留下更纯粹的症状描述文本，方便后续的症状关键词提取。
    """
    cleaned = text
    removable: list[str] = []
    for constitution in VALID_CONSTITUTIONS:
        removable.append(constitution)
        removable.append(constitution.replace("体质", ""))
    removable.extend(VALID_AREAS)
    removable.extend(AREA_ALIAS.keys())
    removable.extend(SOLAR_TERM_TO_SEASON.keys())
    for category, keywords in CATEGORY_KEYWORDS.items():
        removable.append(category)
        removable.extend(keywords)
    removable.extend(QUESTION_WORDS)

    for item in sorted(set(removable), key=len, reverse=True):
        if item:
            cleaned = cleaned.replace(item, "")
    return cleaned
