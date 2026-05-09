from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI

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
