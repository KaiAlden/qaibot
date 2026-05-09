from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Intent = Literal[
    "identify_constitution",
    "diet_advice",
    "conditioning_advice",
    "mixed",
    "general_followup",
    "irrelevant",
]


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class RetrievedChunk(BaseModel):
    chunk_id: str
    score: float
    type: str
    fallback_level: str | None = None


class SessionState(BaseModel):
    constitution: str | None = None
    secondary_constitution: str | None = None
    area: str | None = None
    season: str | None = None
    last_intent: str | None = None
    last_advice_type: str | None = None


class ChatResponse(BaseModel):
    answer: str
    need_clarification: bool
    clarification_question: str | None
    session_state: SessionState
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)


class ParsedIntent(BaseModel):
    intent: Intent
    symptoms: list[str] = Field(default_factory=list)
    constitution: str | None = None
    area: str | None = None
    season: str | None = None
    advice_type: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class KnowledgeChunk(BaseModel):
    chunk_id: str
    type: Literal["constitution_identify", "diet_principle", "suggestion"]
    content: str
    area: str | None = None
    season: str | None = None
    constitution: str | None = None
    suggestion_name: str | None = None

    def payload(self) -> dict[str, Any]:
        data = self.model_dump()
        return {k: v for k, v in data.items() if v not in (None, [], "")}
