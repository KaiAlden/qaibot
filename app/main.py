from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.config import load_settings
from app.schemas import ChatRequest, ChatResponse
from app.services.chat_service import ChatService


app = FastAPI(title="QAibot TCM Constitution RAG")


@lru_cache
def get_service() -> ChatService:
    return ChatService(load_settings())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return get_service().chat(request)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        get_service().chat_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
