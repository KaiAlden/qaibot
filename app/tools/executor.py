from __future__ import annotations

from app.schemas import ToolCall


class ToolExecutor:
    def execute(self, name: str, args: dict) -> ToolCall:
        return ToolCall(
            name=name,
            args=args,
            status="not_configured",
            result="外部工具接口已预留，当前尚未接入具体工具服务。",
        )
