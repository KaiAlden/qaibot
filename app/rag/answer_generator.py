from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from app.config import Settings


class AnswerGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or None,
            timeout=settings.llm_request_timeout,
        )

    def generate(
        self,
        message: str,
        session: dict,
        retrieved: list[dict],
        identification: dict | None = None,
    ) -> str:
        prompt = self._build_prompt(message, session, retrieved, identification)
        response = self.llm.chat.completions.create(
            model=self.settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.settings.llm_temperature,
        )
        return response.choices[0].message.content or ""

    def generate_stream(
        self,
        message: str,
        session: dict,
        retrieved: list[dict],
        identification: dict | None = None,
    ) -> Iterator[str]:
        prompt = self._build_prompt(message, session, retrieved, identification)
        response = self.llm.chat.completions.create(
            model=self.settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.settings.llm_temperature,
            stream=True,
        )
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def prompt_size(
        self,
        message: str,
        session: dict,
        retrieved: list[dict],
        identification: dict | None = None,
    ) -> int:
        return len(self._build_prompt(message, session, retrieved, identification))

    def _build_prompt(
        self,
        message: str,
        session: dict,
        retrieved: list[dict],
        identification: dict | None = None,
    ) -> str:
        context = self._context_text(retrieved)
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

        return f"""你是专业、温和、谨慎的中医体质饮食调理助手。

用户问题：
{message}

当前会话信息：
- 体质：{session.get("constitution")}
- 兼夹体质：{session.get("secondary_constitution")}
- 地区：{session.get("area")}
- 季节：{session.get("season")}

最近对话历史：
{history_text}

体质识别说明：
{identify_note}

检索资料：
{context}

检索降级说明：
{fallback_note}

请基于检索资料回答。要求：
1. 回答具体、可操作，不编造资料中没有的食材或调理法。
2. 如果使用了季节或地区降级资料，要自然说明“未找到完全匹配资料，参考了更宽泛条件”。
3. 涉及药浴、穴位、明显不适或长期症状时，提醒咨询专业医师。
4. 不要把自己称为医生，不做疾病诊断。
5. 用中文，长度适中。"""

    def _context_text(self, retrieved: list[dict]) -> str:
        chunks = []
        max_chars = self.settings.rag_chunk_max_chars
        for item in retrieved:
            content = str(item["payload"].get("content", "")).strip()
            if max_chars > 0 and len(content) > max_chars:
                content = content[:max_chars].rstrip() + "..."
            if content:
                chunks.append(content)
        return "\n\n---\n\n".join(chunks) if chunks else "无"

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
