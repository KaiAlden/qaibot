"""定义项目中所有核心数据的结构和格式
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

"""
Any：万能类型，啥都能接
Literal：限定变量只能用固定值
BaseModel：Pydantic 数据模型基类，自动校验数据
Field：给字段加规则、描述、约束
"""


# 定义用户意图的枚举类型
Intent = Literal[
    "identify_constitution", # 识别体质
    "diet_advice", # 饮食建议
    "conditioning_advice", # 调理建议
    "mixed", # 混合意图，既有饮食又有调理
    "general_followup", # 一般跟进，用户没有明确问饮食或调理，但有症状或体质相关问题
    "irrelevant", # 无关意图
]

# 定义API接口接收到的请求体的格式
class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    message: str = Field(min_length=1)

# 定义从向量数据库检索到的知识片段的格式,包括片段ID、相似度分数、类型和是否是回退级别的标记
class RetrievedChunk(BaseModel):
    chunk_id: str
    score: float
    type: str
    fallback_level: str | None = None


# 追踪多轮对话中的状态信息，如用户体质、上次意图等，帮助模型更好地理解上下文和生成回答
class SessionState(BaseModel):
    constitution: str | None = None
    secondary_constitution: str | None = None
    area: str | None = None
    season: str | None = None
    last_intent: str | None = None
    last_advice_type: str | None = None


# 定义API接口返回的响应体的格式,包括最终回答、是否需要澄清、澄清问题、会话状态和检索到的知识片段列表
class ChatResponse(BaseModel):
    answer: str
    need_clarification: bool
    clarification_question: str | None
    session_state: SessionState
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)

# 定义解析用户意图后的结构化结果，包括意图类型、相关症状、体质、区域、季节、建议类型和原始解析结果等信息
class ParsedIntent(BaseModel):
    intent: Intent
    symptoms: list[str] = Field(default_factory=list)
    constitution: str | None = None
    area: str | None = None
    season: str | None = None
    advice_type: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

# 定义知识库中的知识块，表示向量数据库中存储的一条知识记录
class KnowledgeChunk(BaseModel):
    chunk_id: str
    type: Literal["constitution_identify", "diet_principle", "suggestion"]
    content: str
    area: str | None = None
    season: str | None = None
    constitution: str | None = None
    suggestion_name: str | None = None

    # 将数据转为字典，同时自动过滤掉 `None`、空列表、空字符串，保持存储的整洁
    def payload(self) -> dict[str, Any]:
        data = self.model_dump() # Pydantic 提供的实例方法，用于将 Pydantic 模型对象转换为普通的 Python 字典。
        return {k: v for k, v in data.items() if v not in (None, [], "")}
