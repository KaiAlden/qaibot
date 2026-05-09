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
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = "" if value is None else str(value)
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def normalize_constitution(value: str | None) -> str | None:
    if not value:
        return None
    text = clean_text(value)
    for name in VALID_CONSTITUTIONS:
        if name in text or name.replace("体质", "") in text:
            return name
    return text if text.endswith("体质") else f"{text}体质"


def normalize_area(value: str | None) -> str | None:
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
    symptom_text = _remove_non_symptom_slots(clean_text(text))
    hits = [keyword for keyword in SYMPTOM_HINTS if keyword in symptom_text]
    fragments = re.split(r"[，,。；;、\s]+", symptom_text)
    for fragment in fragments:
        if len(fragment) <= 1:
            continue
        if any(hint in fragment for hint in SYMPTOM_FRAGMENT_HINTS):
            hits.append(fragment)
    return list(dict.fromkeys([hit for hit in hits if hit]))[:12]


def _remove_non_symptom_slots(text: str) -> str:
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
