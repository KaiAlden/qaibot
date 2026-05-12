from __future__ import annotations

from app.schemas import ParsedIntent


class ClarificationDecider:
    def decide(self, parsed: ParsedIntent, session: dict) -> str | None:
        if parsed.intent == "irrelevant" and not session.get("constitution"):
            return "为了给出更准确的体质饮食和调理建议，请先描述一下你最近的身体表现，比如怕冷怕热、出汗、睡眠、消化和情绪状态。"

        if not parsed.constitution and parsed.intent in {"diet_advice", "conditioning_advice", "general_followup"}:
            return "我还不知道你的体质。请描述一下你最近的主要症状，比如怕冷、乏力、口干、胸闷、睡眠和大便情况。"

        if not parsed.area and parsed.intent in {"diet_advice", "conditioning_advice", "mixed"}:
            return "请告诉我你所在的地区或省份，比如广东、北京、四川等，我会按地区匹配饮食和调理建议。"

        return None
