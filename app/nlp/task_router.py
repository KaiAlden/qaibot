from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from app.domain import (
    ADVICE_TYPES,
    CONSTITUTIONS,
    VALID_AREAS,
    VALID_SEASONS,
    extract_symptoms,
    normalize_area,
    normalize_constitution,
    normalize_term,
)
from app.schemas import ParsedIntent, RoutedTask

if TYPE_CHECKING:
    from app.config import Settings


ROUTES = {"tcm_health", "weather", "music", "web_search", "smalltalk", "unsupported"}
TCM_INTENTS = {
    "identify_constitution",
    "constitution_explain",
    "diet_advice",
    "conditioning_advice",
    "seasonal_health_advice",
    "mixed",
    "general_followup",
    "irrelevant",
}
CONFIDENCES = {"high", "medium", "low"}
DIET_PRINCIPLE = "季节饮食原则"
ADVICE_TYPE_ALIASES = {
    "diet": DIET_PRINCIPLE,
    "food": DIET_PRINCIPLE,
    "diet_principle": DIET_PRINCIPLE,
    "饮食": DIET_PRINCIPLE,
    "食养": DIET_PRINCIPLE,
    "吃什么": DIET_PRINCIPLE,
    "忌口": "忌食清单",
    "avoid_food": "忌食清单",
    "exercise": "运动推荐",
    "运动": "运动推荐",
    "sleep": "起居建议",
    "daily_living": "起居建议",
    "作息": "起居建议",
    "起居": "起居建议",
    "emotion": "情绪调节",
    "情绪": "情绪调节",
    "acupoint": "穴位保健",
    "穴位": "穴位保健",
    "medicated_bath": "药浴调理",
    "药浴": "药浴调理",
}

HIGH_RISK_MEDICAL_TERMS = [
    "确诊",
    "诊断",
    "开方",
    "处方药",
    "癌症",
    "高烧",
    "发烧四十度",
    "发烧40",
    "胸口疼",
    "胸痛",
    "心脏不舒服",
    "替代手术",
    "保证治好",
    "偏方",
    "孕妇自行用药",
    "孕妇药浴",
    "急救",
    "要不要去医院",
]
NEGATED_TCM_TERMS = [
    "和中医无关",
    "和养生无关",
    "不是中医",
    "不问体质",
    "不需要养生",
]


class TaskRouter:
    """以 LLM 为主的路由器：模型给出路由与意图；本地代码负责安全、枚举合法性与工具字段规范化。"""

    def __init__(self, settings: Settings):
        """读取配置并初始化 OpenAI 兼容客户端与路由超时。"""
        from openai import OpenAI

        self.settings = settings
        self.router_timeout = max(float(settings.router_llm_timeout), 20.0)
        self.llm = OpenAI(
            api_key=settings.llm_api_key or "EMPTY",
            base_url=settings.llm_base_url or None,
            timeout=self.router_timeout,
        )

    def route(self, message: str, session: dict, runtime_context: dict[str, Any]) -> RoutedTask:
        """对用户消息做路由：安全优先，其次 LLM JSON，失败则规则兜底，最后经护栏返回。"""
        safety = self._safety_override(message)
        if safety:
            return safety

        try:
            response = self._create_completion(message, session, runtime_context)
            content = response.choices[0].message.content or ""
            task = self._validate_result(self._extract_json(content), message)
        except Exception as exc:  # noqa: BLE001
            task = self._llm_failure_fallback(message, exc)
        return self._guardrail(message, task)

    def _create_completion(self, message: str, session: dict, runtime_context: dict[str, Any]) -> Any:
        """调用聊天补全 API；优先请求 JSON 对象格式，不支持则降级为普通调用。"""
        params = {
            "model": self.settings.llm_model,
            "messages": self._messages(message, session, runtime_context),
            "temperature": 0,
            "max_tokens": 1200,
            "timeout": self.router_timeout,
        }
        try:
            return self.llm.chat.completions.create(
                **params,
                response_format={"type": "json_object"},
            )
        except TypeError:
            return self.llm.chat.completions.create(**params)

    def _messages(self, message: str, session: dict, runtime_context: dict[str, Any]) -> list[dict[str, str]]:
        """拼装路由用的 system/user 消息（含会话上下文与用户原文）。"""
        context = {
            "user_constitution": session.get("user_constitution"),
            "secondary_constitution": session.get("secondary_constitution"),
            "last_topic_constitutions": session.get("last_topic_constitutions") or [],
            "last_intent": session.get("last_intent"),
            "last_advice_types": session.get("last_advice_types") or [],
            "area": session.get("area"),
            "season": session.get("season"),
            "pending_clarification": session.get("pending_clarification"),
            "runtime_context": runtime_context,
        }
        system = """你是一个面向真实业务系统的意图路由器,只负责把用户消息分类为结构化 JSON。

必须只输出 JSON,不要输出 Markdown,不要解释,不要展示推理过程,不要输出 Thinking Process,不要输出 <think>。

一级 route 只能是:
- tcm_health:中医体质、体质辨识、体质解释、饮食调养、运动作息情志穴位药浴、节气季节养生等。
- weather:天气、温度、空气、降雨等实时天气查询。
- music:播放、暂停、上一首、下一首、换一首、来一首歌等音乐控制。
- web_search:明确要求查新闻、查资料、查最新信息、查网页、查百科等外部实时信息。
- smalltalk:问候、感谢、告别、询问你能做什么等普通寒暄。
- unsupported:系统不支持或不应处理的请求。

二级 intent 只用于 tcm_health 内部细分；非 tcm_health 路由的 intent 必须是 irrelevant。

tcm_health 的 intent 只能是:
- identify_constitution:用户想判断自己或他人是什么体质。即使没有直接说“体质”,只要描述症状并询问自己属于哪类、怎么回事,也可以判断为体质辨识。
- constitution_explain:询问某个体质是什么、特点、表现、判断标准、形成原因、与症状的关系。
- diet_advice:重点询问饮食、吃什么、忌口、食材、食养。
- conditioning_advice:重点询问运动、作息、睡眠、情绪、穴位、药浴、防护、日常调理。
- seasonal_health_advice:重点询问节气或季节养生,且没有明确饮食或调理细分主诉。
- mixed:同时包含多个中医任务,例如“症状 + 饮食/运动/调理建议”、“是什么体质 + 怎么调理”、“同时问饮食和运动”。
- general_followup:只有在上下文已经明确,且本轮只是“继续说说、还要注意什么、展开讲讲”等模糊追问时使用。
- irrelevant:进入 tcm_health 但无法对应任何中医健康任务时使用。

安全要求:
- 涉及确诊、诊断疾病、开处方、处方药、急症、胸痛、高烧、癌症治疗、替代手术、保证治好、孕妇自行用药等,route=unsupported, intent=irrelevant。
- 用户明确说“和中医无关/和养生无关/不是中医/不问体质”时,不要强行归入 tcm_health。

字段要求:
- confidence 只能是 high、medium、low。
- symptoms 填用户明确描述的症状词,没有则 []。
- constitution 必须是九种体质之一或 null。
- advice_types 填用户明确询问的调理类型,如 diet、exercise、sleep、emotion、acupoint、medicated_bath 等；没有则 []。
- route 为 weather/music/web_search 时 tool_name 分别填 weather/music/web_search,tool_args 至少包含 {"query": 原始问题}。
- route 为 smalltalk 时 response_text 给出简短自然回复。
- need_clarification 只在低置信度或确实缺少必要信息时为 true。

输出格式:
{
  "route": "tcm_health",
  "intent": "mixed",
  "symptoms": [],
  "constitution": null,
  "area": null,
  "season": null,
  "advice_types": [],
  "tool_name": null,
  "tool_args": {},
  "need_clarification": false,
  "clarification_question": null,
  "confidence": "high",
  "reason": "一句话说明分类依据",
  "response_text": null
}"""
        user = f"""会话上下文:
{json.dumps(context, ensure_ascii=False)}

用户消息:
{message}

/no_think"""
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any]:
        """从模型原始文本中提取首个 JSON 对象并反序列化为字典。"""
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        return json.loads(text)

    def _validate_result(self, data: dict[str, Any], message: str) -> RoutedTask:
        """将 LLM 输出的字典校验并规范化为 RoutedTask。"""
        route = data.get("route")
        intent = data.get("intent")
        if route not in ROUTES:
            route = "unsupported"
        if route != "tcm_health":
            intent = "irrelevant"
        elif intent not in TCM_INTENTS:
            intent = "general_followup" if data.get("need_clarification") else "irrelevant"

        confidence = data.get("confidence")
        if confidence not in CONFIDENCES:
            confidence = "low"

        task = RoutedTask(
            route=route,
            intent=intent,
            symptoms=self._normalize_symptoms(data.get("symptoms"), message),
            constitution=self._normalize_constitution(data.get("constitution")),
            area=self._normalize_area(data.get("area")),
            season=self._normalize_season(data.get("season")),
            advice_types=self._normalize_advice_types(data.get("advice_types")),
            tool_name=data.get("tool_name"),
            tool_args=data.get("tool_args") if isinstance(data.get("tool_args"), dict) else {},
            need_clarification=bool(data.get("need_clarification")),
            clarification_question=data.get("clarification_question") or None,
            confidence=confidence,
            reason=data.get("reason") or None,
            response_text=data.get("response_text") or None,
        )
        return self._normalize_tool_fields(task, message)

    def _guardrail(self, message: str, task: RoutedTask) -> RoutedTask:
        """路由后护栏：再次安全检查，并按业务规则修正工具字段、寒暄回复与澄清提示。"""
        safety = self._safety_override(message)
        if safety:
            return safety

        task = self._normalize_tool_fields(task, message)
        if task.route == "tcm_health":
            task.tool_name = None
            task.tool_args = {}
            if task.intent == "irrelevant" and task.confidence == "low":
                task.need_clarification = True
                task.clarification_question = task.clarification_question or "你是想判断体质、了解某种体质,还是想咨询饮食/调理建议？"
        elif task.route == "smalltalk":
            task.intent = "irrelevant"
            task.response_text = task.response_text or "你好,我可以帮你做体质辨识,也可以提供饮食、调理和节气养生建议。"
        elif task.route == "unsupported":
            task.intent = "irrelevant"
            task.tool_name = None
            task.tool_args = {}
        return task

    @staticmethod
    def _normalize_tool_fields(task: RoutedTask, message: str) -> RoutedTask:
        """为天气/音乐/联网搜索类路由补齐 tool_name 与默认 query 参数。"""
        if task.route in {"weather", "music", "web_search"}:
            task.intent = "irrelevant"
            task.tool_name = task.route
            if not task.tool_args:
                task.tool_args = {"query": message}
        return task

    @staticmethod
    def _safety_override(message: str) -> RoutedTask | None:
        """命中高风险医疗用语或用户排除中医时，返回固定的 unsupported 任务；否则为 None。"""
        if any(term in message for term in HIGH_RISK_MEDICAL_TERMS):
            return RoutedTask(
                route="unsupported",
                intent="irrelevant",
                confidence="high",
                reason="命中高风险医疗请求,禁止进入普通 RAG。",
            )
        if any(term in message for term in NEGATED_TCM_TERMS):
            return RoutedTask(
                route="unsupported",
                intent="irrelevant",
                confidence="high",
                reason="用户明确排除中医/养生领域。",
            )
        return None

    def _llm_failure_fallback(self, message: str, exc: Exception) -> RoutedTask:
        """LLM 调用或解析失败时，用语义关键词与规则猜测路由并标记低置信。"""
        reason = f"router_llm_failed: {exc.__class__.__name__}"
        if self._contains_any(message, ["天气", "气温", "温度", "下雨", "降雨", "空气质量"]):
            return RoutedTask(
                route="weather",
                intent="irrelevant",
                tool_name="weather",
                tool_args={"query": message},
                confidence="low",
                reason=reason,
            )
        if self._contains_any(message, ["播放", "放一首", "来一首", "下一首", "上一首", "暂停", "继续播放", "停止播放"]):
            return RoutedTask(
                route="music",
                intent="irrelevant",
                tool_name="music",
                tool_args={"query": message},
                confidence="low",
                reason=reason,
            )
        if self._contains_any(message, ["你好", "您好", "hello", "hi", "谢谢", "再见", "拜拜", "你能做什么", "你是谁"]):
            return RoutedTask(
                route="smalltalk",
                intent="irrelevant",
                confidence="low",
                response_text="你好,我可以帮你做体质辨识,也可以提供饮食、调理和节气养生建议。",
                reason=reason,
            )
        if self._looks_like_tcm_health(message):
            return RoutedTask(
                route="tcm_health",
                intent=self._fallback_tcm_intent(message),
                symptoms=self._fallback_symptoms(message),
                advice_types=self._fallback_advice_types(message),
                confidence="low",
                reason=reason,
            )
        return RoutedTask(route="unsupported", intent="irrelevant", confidence="low", reason=reason)

    @staticmethod
    def _normalize_area(value: Any) -> str | None:
        """将地区字段规范为合法取值，非法则返回 None。"""
        if not value:
            return None
        normalized = normalize_area(str(value))
        return normalized if normalized in VALID_AREAS else None

    @staticmethod
    def _normalize_constitution(value: Any) -> str | None:
        """将体质字段规范为九种体质之一，非法则返回 None。"""
        if not value:
            return None
        normalized = normalize_constitution(str(value))
        return normalized if normalized in CONSTITUTIONS else None

    @staticmethod
    def _normalize_season(value: Any) -> str | None:
        """将季节/节气相关表述规范为合法季节枚举，非法则返回 None。"""
        if not value:
            return None
        _, season = normalize_term(str(value))
        candidate = season or str(value)
        return candidate if candidate in VALID_SEASONS else None

    @classmethod
    def _normalize_advice_types(cls, value: Any) -> list[str]:
        """将 advice_types 列表经别名映射后过滤为系统允许的调理类型，并去重。"""
        normalized: list[str] = []
        allowed = {*ADVICE_TYPES, DIET_PRINCIPLE}
        for item in cls._string_list(value):
            candidate = ADVICE_TYPE_ALIASES.get(item, item)
            if candidate in allowed:
                normalized.append(candidate)
        return list(dict.fromkeys(normalized))

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        """将任意值转为非空字符串列表（仅当值为 list 时处理）。"""
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @classmethod
    def _normalize_symptoms(cls, value: Any, message: str) -> list[str]:
        """合并模型给出的症状与用户原文推断的症状，去重并限制最多 12 条。"""
        symptoms = cls._string_list(value)
        inferred = extract_symptoms(message)
        return list(dict.fromkeys([*symptoms, *inferred]))[:12]

    @staticmethod
    def _contains_any(text: str, terms: list[str]) -> bool:
        """判断文本（忽略大小写）是否包含任意给定子串。"""
        return any(term and term.lower() in text.lower() for term in terms)

    def _looks_like_tcm_health(self, message: str) -> bool:
        """用语义关键词或症状抽取判断消息是否像中医健康类询问。"""
        if extract_symptoms(message):
            return True
        return self._contains_any(
            message,
            [
                "体质",
                "中医",
                "养生",
                "调理",
                "饮食",
                "忌口",
                "运动",
                "起居",
                "睡眠",
                "情绪",
                "穴位",
                "药浴",
                "节气",
                "怕冷",
                "乏力",
                "口干",
                "口苦",
                "舌苔",
                "胸闷",
                "痰多",
                "湿气",
                "上火",
            ],
        )

    def _fallback_tcm_intent(self, message: str) -> str:
        """无 LLM 时根据关键词推断 tcm_health 下的 intent 字符串。"""
        has_diet = self._contains_any(message, ["饮食", "吃", "忌口", "食养", "食材"])
        has_conditioning = self._contains_any(message, ["调理", "运动", "起居", "睡眠", "情绪", "穴位", "药浴"])
        has_symptom = bool(self._fallback_symptoms(message))
        if (has_diet and has_conditioning) or (has_symptom and (has_diet or has_conditioning)):
            return "mixed"
        if self._contains_any(message, ["是什么体质", "什么体质", "哪种体质", "判断", "辨识", "识别"]):
            return "identify_constitution"
        if self._contains_any(message, ["是什么", "特点", "特征", "表现", "怎么回事", "判断标准"]):
            return "constitution_explain"
        if has_diet:
            return "diet_advice"
        if has_conditioning:
            return "conditioning_advice"
        if self._contains_any(message, ["春季", "夏季", "秋季", "冬季", "节气", "三伏", "冬至", "立春", "立夏"]):
            return "seasonal_health_advice"
        return "general_followup"

    def _fallback_advice_types(self, message: str) -> list[str]:
        """无 LLM 时根据关键词推断调理类型列表，并经 _normalize_advice_types 规范化。"""
        raw: list[str] = []
        if self._contains_any(message, ["饮食", "吃", "食养", "食材"]):
            raw.append("diet")
        if self._contains_any(message, ["忌口"]):
            raw.append("忌口")
        if self._contains_any(message, ["运动"]):
            raw.append("exercise")
        if self._contains_any(message, ["起居", "睡眠", "作息"]):
            raw.append("sleep")
        if self._contains_any(message, ["情绪"]):
            raw.append("emotion")
        if self._contains_any(message, ["穴位"]):
            raw.append("acupoint")
        if self._contains_any(message, ["药浴"]):
            raw.append("medicated_bath")
        return self._normalize_advice_types(raw)

    def _fallback_symptoms(self, message: str) -> list[str]:
        """从用户原文抽取症状词列表（兜底路径）。"""
        return extract_symptoms(message)


def parsed_from_task(task: RoutedTask) -> ParsedIntent:
    """将 RoutedTask 转为下游使用的 ParsedIntent。"""
    return task.to_parsed_intent()
