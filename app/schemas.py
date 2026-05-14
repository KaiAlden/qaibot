from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Intent = Literal[
    "identify_constitution",
    "constitution_explain",
    "diet_advice",
    "conditioning_advice",
    "seasonal_health_advice",
    "mixed",
    "general_followup",
    "irrelevant",
]

Route = Literal["tcm_health", "weather", "music", "web_search", "smalltalk", "unsupported"]
ToolStatus = Literal["pending", "success", "failed", "not_configured"]


class RuntimeContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    location: str | None = None
    current_time: str | None = None
    time: str | None = None
    solar_term: str | None = None


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    runtime_context: RuntimeContext | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    score: float
    type: str
    fallback_level: str | None = None


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    status: ToolStatus
    result: dict[str, Any] | str | None = None


class SessionState(BaseModel):
    constitution: str | None = None
    secondary_constitution: str | None = None
    area: str | None = None
    season: str | None = None
    last_intent: str | None = None
    last_advice_types: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    need_clarification: bool
    clarification_question: str | None
    session_state: SessionState
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    route: Route | None = None
    tool_call: ToolCall | None = None


class ParsedIntent(BaseModel):
    intent: Intent
    symptoms: list[str] = Field(default_factory=list)
    constitution: str | None = None
    area: str | None = None
    season: str | None = None
    advice_type: str | None = None
    advice_types: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class RoutedTask(BaseModel):
    route: Route
    intent: Intent = "irrelevant"
    symptoms: list[str] = Field(default_factory=list)
    constitution: str | None = None
    area: str | None = None
    season: str | None = None
    advice_types: list[str] = Field(default_factory=list)
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    need_clarification: bool = False
    clarification_question: str | None = None
    confidence: Literal["high", "medium", "low"] = "low"
    reason: str | None = None
    response_text: str | None = None

    def to_parsed_intent(self) -> ParsedIntent:
        advice_types = list(dict.fromkeys(self.advice_types))
        return ParsedIntent(
            intent=self.intent,
            symptoms=self.symptoms,
            constitution=self.constitution,
            area=self.area,
            season=self.season,
            advice_type=advice_types[0] if advice_types else None,
            advice_types=advice_types,
            raw=self.model_dump(),
        )


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
