from __future__ import annotations

from datetime import datetime

from app.domain import (
    AREA_ALIAS,
    CATEGORY_KEYWORDS,
    CONSTITUTIONS,
    SEASON_BY_TERM,
    VALID_AREAS,
    current_season,
    detect_advice_type,
    detect_advice_types,
    extract_symptoms,
    normalize_area,
    normalize_constitution,
    normalize_term,
)
from app.schemas import ParsedIntent


class IntentParser:
    def parse(self, message: str, session: dict) -> ParsedIntent:
        constitution = self._extract_constitution(message) or session.get("constitution")
        area = self._extract_area(message) or session.get("area")
        _, season = self._extract_term(message)
        season = season or session.get("season")
        advice_types = detect_advice_types(message)
        advice_type = advice_types[0] if advice_types else detect_advice_type(message)
        symptoms = extract_symptoms(message)
        intent = self._infer_intent(message, symptoms, constitution, advice_types, session)

        if not season and intent in {"diet_advice", "conditioning_advice", "mixed", "general_followup"}:
            season = current_season(datetime.now().month)

        return ParsedIntent(
            intent=intent,
            symptoms=symptoms,
            constitution=constitution,
            area=area,
            season=season,
            advice_type=advice_type,
            advice_types=advice_types,
        )

    def _infer_intent(
        self,
        message: str,
        symptoms: list[str],
        constitution: str | None,
        advice_types: list[str],
        session: dict,
    ) -> str:
        has_diet_question = self._contains_any(message, CATEGORY_KEYWORDS["diet_principle"])
        has_conditioning_question = bool(advice_types)
        asks_identity = self._contains_any(message, ["什么体质", "哪种体质", "判断体质", "识别体质"])

        asks_constitution_explain = constitution and self._contains_any(
            message,
            ["特点", "特征", "表现", "症状", "是什么", "什么意思", "有哪些", "怎么判断"],
        )

        if asks_constitution_explain and not has_diet_question and not has_conditioning_question:
            return "constitution_explain"
        if symptoms and (has_diet_question or has_conditioning_question):
            return "mixed"
        if asks_identity or (symptoms and not constitution):
            return "identify_constitution"
        if has_conditioning_question:
            return "conditioning_advice"
        if has_diet_question:
            return "diet_advice"
        if constitution or session.get("constitution"):
            return "general_followup"
        return "irrelevant"

    @staticmethod
    def _extract_constitution(text: str) -> str | None:
        for constitution in CONSTITUTIONS:
            short = constitution.replace("体质", "")
            if constitution in text or short in text:
                return constitution
        return normalize_constitution(text) if text.strip() in CONSTITUTIONS else None

    @staticmethod
    def _extract_area(text: str) -> str | None:
        for area in VALID_AREAS:
            if area in text:
                return area
        for alias in AREA_ALIAS:
            if alias in text:
                return normalize_area(alias)
        return None

    @staticmethod
    def _extract_term(text: str) -> tuple[str | None, str | None]:
        for term in SEASON_BY_TERM:
            if term in text:
                return normalize_term(term)
        return None, None

    @staticmethod
    def _contains_any(text: str, keywords: list[str]) -> bool:
        return any(keyword in text for keyword in keywords)
