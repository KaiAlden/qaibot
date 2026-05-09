from __future__ import annotations

import json

from openai import OpenAI

from app.config import Settings
from app.domain import CONSTITUTIONS
from app.rag.qdrant_store import QdrantStore


class ConstitutionIdentifier:
    def __init__(self, settings: Settings, store: QdrantStore):
        self.settings = settings
        self.store = store
        self.llm = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url or None)

    def identify(self, message: str) -> dict:
        candidates = self.store.search(
            self.settings.qdrant_constitution_collection,
            message,
            filters={"type": "constitution_identify"},
            limit=4,
        )
        context = "\n\n---\n\n".join(item["payload"].get("content", "") for item in candidates)
        if not context:
            return {
                "primary_constitution": None,
                "secondary_constitution": None,
                "confidence": "low",
                "matched_symptoms": [],
                "reasoning": "没有检索到可用的体质识别资料。",
            }

        prompt = f"""你是中医体质识别助手。请只根据候选资料和用户描述做初步体质判断，不要做诊断。

用户描述：
{message}

候选资料：
{context}

可选体质只能来自：{", ".join(CONSTITUTIONS)}

请输出 JSON：
{{
  "primary_constitution": "阳虚体质",
  "secondary_constitution": null,
  "confidence": "high/medium/low",
  "matched_symptoms": ["怕冷", "手脚冰凉"],
  "reasoning": "简短说明"
}}"""

        response = self.llm.chat.completions.create(
            model=self.settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        try:
            result = json.loads(response.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            result = {}

        primary = result.get("primary_constitution")
        secondary = result.get("secondary_constitution")
        if primary not in CONSTITUTIONS:
            primary = None
        if secondary not in CONSTITUTIONS:
            secondary = None

        return {
            "primary_constitution": primary,
            "secondary_constitution": secondary,
            "confidence": result.get("confidence", "low"),
            "matched_symptoms": result.get("matched_symptoms", []),
            "reasoning": result.get("reasoning", ""),
        }
