from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.domain import (
    AREA_ALIAS,
    CATEGORY_KEYWORDS,
    CONSTITUTIONS,
    SEASON_BY_TERM,
    SYMPTOM_HINTS,
    VALID_AREAS,
)


GeneralIntent = Literal[
    "greeting",
    "self_intro",
    "capability_intro",
    "company_intro",
    "external_realtime",
    "thanks",
    "goodbye",
    "unknown",
]

UNSUPPORTED_ANSWER = "我目前暂不支持这方面的回答。如果你需要的话，我可以帮你做体质识别、体质特点说明、饮食建议和调理建议。"


@dataclass(frozen=True)
class GeneralIntentResult:
    intent: GeneralIntent
    answer: str | None = None


class GeneralIntentParser:
    def parse(self, message: str) -> GeneralIntentResult:
        text = message.strip().lower()
        if not text or self._has_domain_signal(message):
            return GeneralIntentResult("unknown")

        if self._contains_any(text, ["谢谢", "感谢", "多谢", "辛苦了", "thank"]):
            return GeneralIntentResult("thanks", "不客气。你可以继续描述症状，或询问某种体质的特点、饮食和调理建议。")

        if self._contains_any(text, ["再见", "拜拜", "bye", "回头见"]):
            return GeneralIntentResult("goodbye", "好的，再见。后续如果想了解体质、饮食或调理建议，随时可以继续问我。")

        if self._contains_any(text, ["你好", "您好", "hello", "hi", "早上好", "下午好", "晚上好", "在吗"]):
            return GeneralIntentResult(
                "greeting",
                "你好，我是中医体质饮食与调理问答助手。你可以描述最近的身体表现，或直接询问某种体质的特点、饮食和调理建议。",
            )

        if self._contains_any(text, ["你是谁", "介绍一下你自己", "你是什么", "你叫什么"]):
            return GeneralIntentResult(
                "self_intro",
                "我是一个中医体质饮食与调理问答助手，可以根据体质、地区、季节和症状，提供饮食原则、运动、起居、情绪、穴位、药浴等方面的建议。",
            )

        if self._contains_any(text, ["你能做什么", "有什么功能", "可以帮我什么", "怎么使用", "如何使用","能干什么","能力是什么","有什么用"]):
            return GeneralIntentResult(
                "capability_intro",
                "我可以帮你做体质识别、体质特点说明，提供饮食建议和运动、起居、情绪、穴位、药浴等方面的调理建议。你可以描述症状，也可以直接问“阳虚体质春季怎么吃”“痰湿体质怎么运动”等问题。",
            )

        if self._contains_any(text, ["公司", "你们公司", "介绍一下公司"]):
            return GeneralIntentResult("company_intro", UNSUPPORTED_ANSWER)

        if self._contains_any(text, ["天气", "新闻", "几点", "股票", "价格", "汇率"]):
            return GeneralIntentResult("external_realtime", UNSUPPORTED_ANSWER)

        return GeneralIntentResult("unknown")

    @staticmethod
    def _contains_any(text: str, keywords: list[str]) -> bool:
        return any(keyword.lower() in text for keyword in keywords)

    def _has_domain_signal(self, text: str) -> bool:
        domain_terms: list[str] = []
        for constitution in CONSTITUTIONS:
            domain_terms.append(constitution)
            domain_terms.append(constitution.replace("体质", ""))
        domain_terms.extend(VALID_AREAS)
        domain_terms.extend(AREA_ALIAS.keys())
        domain_terms.extend(SEASON_BY_TERM.keys())
        domain_terms.extend(SYMPTOM_HINTS)
        for category, keywords in CATEGORY_KEYWORDS.items():
            domain_terms.append(category)
            domain_terms.extend(keywords)
        domain_terms.extend(["体质", "调理", "症状", "表现", "特点", "特征", "饮食"])
        return any(term and term in text for term in domain_terms)
