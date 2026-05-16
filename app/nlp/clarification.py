from __future__ import annotations

from app.schemas import ParsedIntent


class ClarificationDecider:
    def decide(self, parsed: ParsedIntent, session: dict) -> str | None:
        known_constitution = (
            parsed.constitution
            or (session.get("target_constitutions") or [None])[0]
            or session.get("user_constitution")
        )

        if parsed.intent == "irrelevant" and not known_constitution:
            return "为了给出更准确的体质饮食和调理建议，请先描述一下你最近的身体表现，比如怕冷怕热、出汗、睡眠、消化和情绪状态。"

        if not known_constitution and parsed.intent in {"diet_advice", "conditioning_advice", "general_followup"}:
            return "我还不知道你的体质。请描述一下你最近的主要症状，比如怕冷、乏力、口干、胸闷、睡眠和大便情况。"

        if not known_constitution and not parsed.symptoms and parsed.intent == "mixed":
            return "为了给出更准确的综合建议，请先告诉我你的体质，或补充几个主要身体表现，比如怕冷怕热、出汗、口干口苦、睡眠和消化情况。"

        if not parsed.area and parsed.intent in {"diet_advice", "conditioning_advice", "mixed"}:
            return "请告诉我你所在的地区或省份，比如广东、北京、四川等，我会按地区匹配饮食和调理建议。"

        return None
