from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from app.config import Settings
from app.domain import ADVICE_TYPES, CONSTITUTIONS, VALID_AREAS, VALID_SEASONS, normalize_area
from app.schemas import ParsedIntent, RoutedTask


TCM_GUARD_TERMS = [
    "中医",
    "体质",
    "养生",
    "饮食",
    "调理",
    "节气",
    "立春",
    "立夏",
    "立秋",
    "立冬",
    "春季",
    "夏季",
    "秋季",
    "冬季",
    "起居",
    "情绪",
    "运动",
    "穴位",
    "药浴",
    "症状",
    "睡眠",
    "脾胃",
    "阳虚",
    "阴虚",
    "气虚",
    "痰湿",
    "湿热",
    "血瘀",
    "气郁",
    "特禀",
    "平和",
    "上火",
    "去火",
    "降火",
    "清热",
    "口干",
    "口苦",
    "咽干",
    "便秘",
    "腹泻",
    "胃胀",
    "胃寒",
    "脾虚",
    "湿气",
    "舌苔",
    "枸杞",
    "黄芪",
    "红枣",
    "姜茶",
    "菊花",
    "薏米",
]

WEATHER_TERMS = ["天气", "气温", "温度", "下雨", "降雨", "空气质量", "湿度", "风力", "台风", "冷不冷", "热不热"]
MUSIC_TERMS = ["播放", "听歌", "听音乐", "音乐", "歌曲", "暂停", "切歌"]
WEB_SEARCH_TERMS = ["搜索", "搜一下", "查一下", "新闻", "价格", "政策", "官网", "联网", "网上"]
SEASONAL_TERMS = ["节气", "立春", "立夏", "立秋", "立冬", "春季", "夏季", "秋季", "冬季"]
DIET_TERMS = ["饮食", "吃", "忌口", "食养", "食疗", "食物", "食材", "喝", "枸杞", "红枣", "黄芪"]
EXPLAIN_TERMS = ["特征", "特点", "表现", "是什么", "什么意思", "情况", "原因"]
CONDITIONING_TERMS = ["调理", "改善", "怎么办", "怎么缓解", "起居", "运动", "情绪", "穴位", "药浴"]


class TaskRouter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = OpenAI(
            api_key=settings.llm_api_key or "EMPTY",
            base_url=settings.llm_base_url or None,
            timeout=settings.router_llm_timeout,
        )

    def route(self, message: str, session: dict, runtime_context: dict[str, Any]) -> RoutedTask:
        fallback = self._fallback_route(message, session)
        if fallback.confidence == "high":
            return self._guardrail(message, fallback)

        try:
            response = self.llm.chat.completions.create(
                model=self.settings.llm_model,
                messages=self._messages(message, session, runtime_context),
                temperature=0,
                max_tokens=300,
                timeout=self.settings.router_llm_timeout,
            )
            content = response.choices[0].message.content or ""
            task = RoutedTask.model_validate(self._extract_json(content))
        except Exception:  # noqa: BLE001
            task = fallback
        return self._guardrail(message, task)

    def _messages(self, message: str, session: dict, runtime_context: dict[str, Any]) -> list[dict[str, str]]:
        prompt = f"""你是严格的任务路由器,只负责分类和抽取槽位,不回答用户问题。

可选 route:
- tcm_health:中医体质、养生、饮食、起居、情绪、运动、穴位、药浴、节气健康、症状调理。
- weather:天气、气温、降雨、空气质量、风力、湿度等实时气象查询。
- music:播放、暂停、搜索歌曲、听音乐、切歌等音乐操作。
- web_search:除中医养生健康外的开放知识、新闻、政策、价格、地点、人物、产品等联网查询。
- smalltalk:寒暄、感谢、再见、自我介绍、能力介绍。
- unsupported:不支持或高风险请求。

强规则:
1. 凡是中医养生、体质、饮食、调理、节气健康相关问题,一律 route=tcm_health。
2. “查一下、搜索、最近、今天、2026年、当前”不能单独决定 web_search。
3. 只有非中医健康领域且需要外部知识/实时信息时,才 route=web_search。
4. 天气查询 route=weather;音乐操作 route=music。
5. 如果一个问题同时包含天气和养生,但核心诉求是“怎么养生/怎么调理”,route=tcm_health。
6. 如果一个问题同时要求“查天气并给养生建议”,route=weather,并在 tool_args 中设置 followup_domain=tcm_health。

tcm_health 下 intent 只能是:
- identify_constitution:用户想检测/判断/识别/评估自己的体质。
- constitution_explain:用户想了解某种体质特点、表现、含义。
- diet_advice:用户问饮食、怎么吃、忌口、食养。
- conditioning_advice:用户问运动、起居、情绪、穴位、药浴等调理。
- seasonal_health_advice:用户问节气/季节养生健康建议。
- mixed:同时包含症状和建议需求,或多个健康诉求混合。
- general_followup:结合已有体质/上下文继续追问。
- irrelevant:不属于 tcm_health 时使用。

易错例子:
- “查一下立夏养生注意事项” => route=tcm_health, intent=seasonal_health_advice
- “2026年立夏适合怎么吃?” => route=tcm_health, intent=diet_advice
- “今天杭州天气怎么样?” => route=weather, tool_name=weather
- “今天杭州天气这么热,怎么养生？” => route=tcm_health, intent=seasonal_health_advice
- “帮我搜一下杭州今天有什么新闻” => route=web_search, tool_name=web_search
- “播放一首轻音乐” => route=music, tool_name=music
- “你好,我想检测一下体质” => route=tcm_health, intent=identify_constitution, need_clarification=true

当前会话:
- 已知体质:{session.get("constitution")}
- 已知兼夹体质:{session.get("secondary_constitution")}
- 已知地区:{session.get("area")}
- 已知季节:{session.get("season")}
- 运行时上下文:{json.dumps(runtime_context, ensure_ascii=False)}

用户输入:{message}

只输出 JSON,字段固定如下:
{{
  "route": "tcm_health",
  "intent": "identify_constitution",
  "symptoms": [],
  "constitution": null,
  "area": null,
  "season": null,
  "advice_types": [],
  "tool_name": null,
  "tool_args": {{}},
  "need_clarification": false,
  "clarification_question": null,
  "confidence": "high",
  "reason": "简短中文理由",
  "response_text": null
}}"""
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        return json.loads(text)

    def _guardrail(self, message: str, task: RoutedTask) -> RoutedTask:
        if self._has_tcm_signal(message) and task.route in {"web_search", "unsupported"}:
            task.route = "tcm_health"
            if task.intent == "irrelevant":
                task.intent = "general_followup"

        if task.route == "tcm_health":
            task.intent = task.intent if task.intent != "irrelevant" else "general_followup"
            task.tool_name = None
            task.tool_args = {}
            task.advice_types = [item for item in task.advice_types if item in ADVICE_TYPES]
            if task.constitution not in CONSTITUTIONS:
                task.constitution = None
            if task.area:
                task.area = normalize_area(task.area)
            if task.season not in VALID_SEASONS:
                task.season = None
            if task.intent == "identify_constitution" and not task.symptoms:
                task.need_clarification = True
                task.clarification_question = (
                    task.clarification_question
                    or "可以的。请描述一下你最近的主要身体表现,比如怕冷怕热、出汗、睡眠、消化、大便、口干口苦和情绪状态。"
                )
        elif task.route == "weather":
            task.intent = "irrelevant"
            task.tool_name = task.tool_name or "weather"
        elif task.route == "music":
            task.intent = "irrelevant"
            task.tool_name = task.tool_name or "music"
        elif task.route == "web_search":
            task.intent = "irrelevant"
            task.tool_name = task.tool_name or "web_search"
            if not task.tool_args:
                task.tool_args = {"query": message}
        return task

    def _fallback_route(self, message: str, session: dict) -> RoutedTask:
        if self._has_tcm_signal(message):
            if "体质" in message and any(term in message for term in ["检测", "判断", "识别", "评估", "测"]):
                intent = "identify_constitution"
            elif any(constitution in message for constitution in CONSTITUTIONS) and any(term in message for term in EXPLAIN_TERMS):
                intent = "constitution_explain"
            elif any(term in message for term in DIET_TERMS):
                intent = "diet_advice"
            elif any(term in message for term in SEASONAL_TERMS):
                intent = "seasonal_health_advice"
            elif any(term in message for term in CONDITIONING_TERMS):
                intent = "conditioning_advice"
            else:
                intent = "general_followup"
            return RoutedTask(route="tcm_health", intent=intent, confidence="high")
        if any(term in message for term in WEATHER_TERMS):
            return RoutedTask(route="weather", tool_name="weather", tool_args={"query": message}, confidence="high")
        if any(term in message for term in MUSIC_TERMS):
            return RoutedTask(route="music", tool_name="music", tool_args={"query": message}, confidence="high")
        if any(term in message for term in WEB_SEARCH_TERMS):
            return RoutedTask(route="web_search", tool_name="web_search", tool_args={"query": message}, confidence="high")
        if session.get("constitution"):
            return RoutedTask(route="tcm_health", intent="general_followup", confidence="low")
        return RoutedTask(route="unsupported", intent="irrelevant", confidence="low")

    @staticmethod
    def _has_tcm_signal(message: str) -> bool:
        return any(term in message for term in TCM_GUARD_TERMS)


def parsed_from_task(task: RoutedTask) -> ParsedIntent:
    return task.to_parsed_intent()
