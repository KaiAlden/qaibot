from __future__ import annotations

# 函数缓存装饰器，性能优化工具，把函数的返回结果存起来，下次用相同参数调用时，直接返回缓存的结果，不用重新计算。
from functools import lru_cache

from fastapi import FastAPI
# StreamingResponse 是 FastAPI 实现「实时流式输出」的工具,接收一个 生成器函数（带 yield 的函数）
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
