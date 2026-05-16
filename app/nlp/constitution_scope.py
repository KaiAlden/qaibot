from __future__ import annotations

from app.domain import CONSTITUTIONS
from app.schemas import ParsedIntent, RoutedTask, TurnContext


STRONG_FOLLOWUP_MAX_TURNS = 5
FOLLOWUP_MAX_TURNS = 2
NON_TCM_SWITCH_LIMIT = 1

SELF_REFERENCE_SIGNALS = [
    "我是",
    "我属于",
    "我这种",
    "我这个",
    "我的体质",
    "我自己的体质",
    "我的情况",
    "我这种情况",
    "按我的情况",
    "按照我的情况",
    "像我这样",
    "像我这种",
    "我自己",
    "我测出来",
    "我测的是",
    "医生说我是",
    "我被诊断",
    "我该",
    "我能",
    "我适合",
    "本人",
    "自己的体质",
]

TOPIC_REFERENCE_SIGNALS = [
    "该体质",
    "这种",
    "这种体质",
    "这个体质",
    "刚才那个体质",
    "上面说的体质",
    "前面说的体质",
    "之前说的体质",
    "刚才",
    "上面",
    "前面",
]

FOLLOWUP_SIGNALS = [
    "那",
    "饮食呢",
    "运动呢",
    "起居呢",
    "情绪呢",
    "怎么吃",
    "怎么调",
]

STRONG_FOLLOWUP_SIGNALS = ["刚才", "上面", "前面", "之前说的", "刚才那个", "上面说的"]
COMPARISON_SIGNALS = ["区别", "对比", "比较", "哪个更", "分别", "不同", "共同点", "一样吗"]

SPECIAL_TOPIC_CONSTITUTIONS = {
    "敏感肌": "特禀体质",
    "过敏体质": "特禀体质",
    "鼻炎": "特禀体质",
}


class ConstitutionScopeResolver:
    """Resolve whether a constitution is the user profile, current topic, or follow-up context."""

    def resolve(self, message: str, parsed: ParsedIntent, routed: RoutedTask, session: dict) -> TurnContext:
        mentioned = self._extract_constitutions(message)
        if parsed.constitution and parsed.constitution not in mentioned and self._constitution_appears(message, parsed.constitution):
            mentioned.append(parsed.constitution)

        special = self._special_topic(message)
        if special and special not in mentioned:
            mentioned.append(special)

        user_constitution = session.get("user_constitution")
        last_topics = list(session.get("last_topic_constitutions") or [])
        self_reference = self._has_self_reference(message)
        topic_reference = self._has_topic_reference(message)
        comparison = len(mentioned) > 1 or self._contains_any(message, COMPARISON_SIGNALS)
        followup = topic_reference or self._is_followup(message, parsed)

        context = TurnContext(
            user_constitution=user_constitution,
            secondary_constitution=session.get("secondary_constitution"),
            last_topic_constitutions=last_topics,
            mentioned_constitutions=mentioned,
        )

        if parsed.intent == "mixed":
            context.target_constitutions = mentioned or ([user_constitution] if user_constitution else [])
            context.scope_type = "identify" if parsed.symptoms or not user_constitution else "self_query"
            context.should_update_user_profile = False
            context.allow_user_profile_in_answer = True
            context.reason = "混合意图是当前轮完整任务，不作为模糊追问处理"
            return context

        if parsed.intent == "identify_constitution":
            context.scope_type = "identify"
            context.target_constitutions = mentioned
            context.should_update_user_profile = False
            context.allow_user_profile_in_answer = True
            context.reason = "用户请求体质识别"
            return context

        if self_reference and topic_reference:
            context.needs_scope_clarification = True
            context.clarification_question = "你是想了解你自己的体质，还是刚才讨论的那种体质？"
            context.reason = "同时出现本人指代和最近话题指代，需要确认"
            return context

        if mentioned:
            context.target_constitutions = mentioned
            context.should_update_user_profile = self_reference and len(mentioned) == 1
            context.scope_type = "comparison" if comparison else ("self_query" if self_reference else "topic_query")
            context.allow_user_profile_in_answer = self_reference or mentioned == [user_constitution]
            context.reason = "当前轮明确提到体质"
            return context

        if self_reference and user_constitution:
            context.target_constitutions = [user_constitution]
            context.scope_type = "self_query"
            context.allow_user_profile_in_answer = True
            context.reason = "本人指代优先使用用户画像"
            return context

        if self_reference and not user_constitution:
            context.scope_type = "self_query"
            context.needs_scope_clarification = True
            context.clarification_question = "我还不知道你的体质。请先描述一下你最近的主要身体表现，我再帮你判断。"
            context.reason = "本人指代但缺少用户画像"
            return context

        if followup:
            inherited = self._inheritable_last_topics(message, session)
            context.turns_since_last_topic = self._turns_since_last_topic(session)
            if len(inherited) == 1:
                context.target_constitutions = inherited
                context.scope_type = "followup"
                context.allow_user_profile_in_answer = inherited[0] == user_constitution
                context.reason = "继承最近体质话题"
                return context
            if len(inherited) > 1:
                context.scope_type = "followup"
                context.needs_scope_clarification = True
                context.clarification_question = (
                    "你是想继续比较"
                    + "、".join(inherited)
                    + "，还是只想了解其中某一种体质的建议？"
                )
                context.reason = "最近话题包含多个体质，需要确认"
                return context
            context.scope_type = "followup"
            context.needs_scope_clarification = True
            context.clarification_question = "你说的“这种体质”具体指哪一种体质？可以告诉我是气虚、阳虚、湿热等哪一类。"
            context.reason = "最近体质话题已失效或不存在"
            return context

        if parsed.intent in {"diet_advice", "conditioning_advice", "general_followup"} and user_constitution:
            context.target_constitutions = [user_constitution]
            context.scope_type = "self_query"
            context.allow_user_profile_in_answer = True
            context.reason = "未明确新话题，使用用户画像"
            return context

        context.scope_type = "unknown"
        context.reason = "未识别出体质作用域"
        return context

    @staticmethod
    def _contains_any(text: str, keywords: list[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    def _has_self_reference(self, text: str) -> bool:
        return self._contains_any(text, SELF_REFERENCE_SIGNALS)

    def _has_topic_reference(self, text: str) -> bool:
        topic_text = text
        for self_phrase in ["我这种", "我这个", "像我这种", "像我这样"]:
            topic_text = topic_text.replace(self_phrase, "")
        if self._contains_any(topic_text, TOPIC_REFERENCE_SIGNALS):
            return True
        if "体质" in topic_text and self._contains_any(topic_text, ["该", "这种", "这个", "刚才", "上面", "前面"]):
            return True
        return False

    def _extract_constitutions(self, text: str) -> list[str]:
        found: list[str] = []
        for constitution in CONSTITUTIONS:
            short = constitution.replace("体质", "")
            if constitution in text or short in text:
                found.append(constitution)
        return list(dict.fromkeys(found))

    @staticmethod
    def _constitution_appears(text: str, constitution: str) -> bool:
        short = constitution.replace("体质", "")
        return constitution in text or short in text

    @staticmethod
    def _special_topic(text: str) -> str | None:
        for keyword, constitution in SPECIAL_TOPIC_CONSTITUTIONS.items():
            if keyword in text:
                return constitution
        return None

    def _is_followup(self, message: str, parsed: ParsedIntent) -> bool:
        if self._contains_any(message, FOLLOWUP_SIGNALS):
            return True
        return parsed.intent == "general_followup" and not self._extract_constitutions(message)

    def _inheritable_last_topics(self, message: str, session: dict) -> list[str]:
        topics = list(session.get("last_topic_constitutions") or [])
        if not topics:
            return []
        turns_since = self._turns_since_last_topic(session)
        if turns_since is None:
            return []
        non_tcm_turns = int(session.get("non_tcm_turns_since_topic") or 0)
        if non_tcm_turns > NON_TCM_SWITCH_LIMIT:
            return []
        if turns_since <= FOLLOWUP_MAX_TURNS:
            return topics
        if turns_since <= STRONG_FOLLOWUP_MAX_TURNS and self._contains_any(message, STRONG_FOLLOWUP_SIGNALS):
            return topics
        return []

    @staticmethod
    def _turns_since_last_topic(session: dict) -> int | None:
        last_turn = session.get("last_topic_turn_index")
        if last_turn is None:
            return None
        return max(int(session.get("turn_index") or 0) + 1 - int(last_turn), 0)
