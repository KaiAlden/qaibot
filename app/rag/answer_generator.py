from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from app.config import Settings
from app.rag.thinking import StreamPart, ThinkingStreamParser, parse_model_output, summarize_thinking
from app.schemas import TurnContext


class AnswerGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = OpenAI(
            api_key=settings.llm_api_key or "EMPTY",
            base_url=settings.llm_base_url or None,
            timeout=settings.llm_request_timeout,
        )

    def generate(
        self,
        message: str,
        session: dict,
        retrieved: list[dict],
        identification: dict | None = None,
        turn_context: TurnContext | None = None,
    ) -> str:
        prompt = self._build_prompt(message, session, retrieved, identification, turn_context)
        response = self.llm.chat.completions.create(
            model=self.settings.llm_model,
            messages=self._messages(prompt),
            temperature=self.settings.llm_temperature,
        )
        message_obj = response.choices[0].message
        content = message_obj.content or ""
        reasoning = getattr(message_obj, "reasoning_content", None) or ""
        parsed = parse_model_output(
            content,
            self.settings.thinking_start_tag,
            self.settings.thinking_end_tag,
            self.settings.thinking_answer_start_tag,
            self.settings.thinking_answer_end_tag,
        )

        if reasoning and not parsed.thinking:
            parsed_answer = parsed.answer or content.strip()
            return parsed_answer
        return parsed.answer if (parsed.answer or parsed.thinking) else content.strip()

    def generate_stream(
        self,
        message: str,
        session: dict,
        retrieved: list[dict],
        identification: dict | None = None,
        turn_context: TurnContext | None = None,
    ) -> Iterator[StreamPart]:
        prompt = self._build_prompt(message, session, retrieved, identification, turn_context)
        response = self.llm.chat.completions.create(
            model=self.settings.llm_model,
            messages=self._messages(prompt),
            temperature=self.settings.llm_temperature,
            stream=True,
        )
        parser = ThinkingStreamParser(
            self.settings.thinking_start_tag,
            self.settings.thinking_end_tag,
            self.settings.thinking_answer_start_tag,
            self.settings.thinking_answer_end_tag,
            self.settings.thinking_stream_buffer_chars,
        )
        for chunk in response:
            if not chunk.choices:
                continue
            delta_obj = chunk.choices[0].delta
            reasoning_delta = getattr(delta_obj, "reasoning_content", None)
            if reasoning_delta:
                yield StreamPart("thinking", reasoning_delta)
            content_delta = delta_obj.content
            if content_delta:
                yield from parser.feed(content_delta)
        yield from parser.finish()

    def prompt_size(
        self,
        message: str,
        session: dict,
        retrieved: list[dict],
        identification: dict | None = None,
        turn_context: TurnContext | None = None,
    ) -> int:
        return len(self._build_prompt(message, session, retrieved, identification, turn_context))

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "你必须使用中文回答。"
                    "如果需要输出思考过程，只能输出简短中文思考摘要，并且必须完整放在 "
                    f"{self.settings.thinking_start_tag} 和 {self.settings.thinking_end_tag} 之间。"
                    "最终给用户看的正式回答必须完整放在 "
                    f"{self.settings.thinking_answer_start_tag} 和 {self.settings.thinking_answer_end_tag} 之间。"
                    "不要输出英文 Thought Process、Analyze the Request、Final Output、Final Answer 等标题。"
                    "不要把思考过程混入正式回答。"
                ),
            },
            {"role": "user", "content": prompt},
        ]

    def format_thinking(self, text: str) -> str:
        if self.settings.thinking_display_mode == "raw":
            return text
        if self.settings.thinking_display_mode == "summary":
            return summarize_thinking(text, self.settings.thinking_summary_max_chars)
        return ""

    def _build_prompt(
        self,
        message: str,
        session: dict,
        retrieved: list[dict],
        identification: dict | None = None,
        turn_context: TurnContext | None = None,
    ) -> str:
        context = self._context_text(retrieved)
        runtime_context_text = self._runtime_context_text(session.get("_runtime_context") or {})
        if runtime_context_text:
            context = f"{runtime_context_text}\n\n---\n\n{context}"
        history_text = self._history_text(session.get("history", []))
        fallback_note = self._fallback_note(retrieved)
        identify_note = ""
        if identification and identification.get("primary_constitution"):
            matched = "、".join(identification.get("matched_symptoms") or [])
            identify_note = (
                f"本轮初步识别体质为{identification['primary_constitution']}。"
                f"匹配症状：{matched or '未明确列出'}。"
                f"{identification.get('reasoning', '')}"
            )
        scope_text = self._scope_text(session, turn_context)

        return f"""你是专业、温和、谨慎的中医体质饮食调理助手。

用户问题：
{message}

当前会话信息：
- 用户长期体质画像：{session.get("user_constitution")}
- 兼夹体质：{session.get("secondary_constitution")}
- 地区：{session.get("area")}
- 季节：{session.get("season")}

本轮体质作用域：
{scope_text}

最近对话历史：
{history_text}

体质识别说明：
{identify_note}

检索资料：
{context}

检索降级说明：
{fallback_note}

请基于检索资料回答。要求：
1. 回答具体、可操作，结合当地的天气、节气等情况输出个性化的养生建议，不编造资料中没有的食材或调理法。
2. 如果使用了季节或地区降级资料，要自然说明“未找到完全匹配资料，参考了更宽泛条件”。
3. 涉及药浴、穴位、明显不适或长期症状时，提醒咨询专业医师。
4. 不要把自己称为医生，不做疾病诊断。
5. 用中文，长度适中。"""

    def _context_text(self, retrieved: list[dict]) -> str:
        chunks = []
        max_chars = self.settings.rag_chunk_max_chars
        for item in retrieved:
            content = str(item["payload"].get("content", "")).strip()
            target = item.get("target_constitution") or item["payload"].get("target_constitution")
            if target and content:
                content = f"【本轮检索对象：{target}】\n{content}"
            if max_chars > 0 and len(content) > max_chars:
                content = content[:max_chars].rstrip() + "..."
            if content:
                chunks.append(content)
        return "\n\n---\n\n".join(chunks) if chunks else "无"

    @staticmethod
    def _scope_text(session: dict, turn_context: TurnContext | None) -> str:
        if not turn_context:
            return (
                f"- 本轮讨论对象：{session.get('target_constitution') or '未明确'}\n"
                "- 说明：未提供结构化作用域，请不要主动扩大用户体质画像。"
            )
        targets = "、".join(turn_context.target_constitutions) or "未明确"
        last_topics = "、".join(turn_context.last_topic_constitutions) or "无"
        user_constitution = turn_context.user_constitution or "未明确"
        allow = "可以" if turn_context.allow_user_profile_in_answer else "不可以"
        lines = [
            f"- 用户长期体质画像：{user_constitution}",
            f"- 本轮讨论对象：{targets}",
            f"- 最近话题体质：{last_topics}",
            f"- 本轮作用域类型：{turn_context.scope_type}",
            f"- 是否可把用户长期体质作为主体回答依据：{allow}",
        ]
        if turn_context.target_constitutions and turn_context.user_constitution:
            if turn_context.user_constitution not in turn_context.target_constitutions:
                lines.append("- 注意：用户长期体质和本轮讨论对象不同，主体回答必须围绕本轮讨论对象。")
                lines.append("- 如需提到用户长期体质，只能作为补充提醒，并明确二者不是同一件事。")
        if turn_context.scope_type == "comparison":
            lines.append("- 比较类问题请按体质分组说明，再总结共同点和差异点，避免混合成一种建议。")
        return "\n".join(lines)

    @staticmethod
    def _runtime_context_text(runtime_context: dict) -> str:
        if not runtime_context:
            return ""

        labels = {
            "location": "当前地点",
            "current_time": "当前时间",
            "time": "当前时间",
            "solar_term": "当前节气",
            "area": "检索地区",
            "season": "检索季节",
        }
        lines = []
        for key in ("location", "current_time", "solar_term", "area", "season"):
            value = runtime_context.get(key)
            if value:
                lines.append(f"- {labels[key]}：{value}")

        extra_lines = []
        for key, value in runtime_context.items():
            if key not in labels and value not in (None, "", [], {}):
                extra_lines.append(f"- {key}：{value}")
        lines.extend(extra_lines)

        if not lines:
            return ""
        return "当前运行时上下文（请结合这些信息生成本轮回答）：\n" + "\n".join(lines)

    @staticmethod
    def _fallback_note(retrieved: list[dict]) -> str:
        levels = {item.get("fallback_level") for item in retrieved if item.get("fallback_level")}
        if not levels or levels == {"area_season_constitution"}:
            return "使用精确匹配资料。"
        return "部分资料来自降级检索：" + "、".join(sorted(levels))

    def _history_text(self, history: list[dict]) -> str:
        if not history:
            return "无"

        max_items = max(self.settings.rag_history_turns, 0) * 2
        if max_items:
            history = history[-max_items:]

        lines = []
        for item in history:
            role = "用户" if item.get("role") == "user" else "助手"
            content = str(item.get("content", "")).strip()
            if content:
                lines.append(f"{role}：{content}")
        return "\n".join(lines) if lines else "无"
