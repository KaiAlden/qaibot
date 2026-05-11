from __future__ import annotations

from collections.abc import Iterator
import json
from queue import Empty, Queue
from threading import Thread
from time import perf_counter

from app.config import Settings
from app.nlp.clarification import ClarificationDecider
from app.nlp.intent_parser import IntentParser
from app.rag.answer_generator import AnswerGenerator
from app.rag.constitution_identifier import ConstitutionIdentifier
from app.rag.qdrant_store import QdrantStore
from app.rag.retriever import KnowledgeRetriever
from app.schemas import ChatRequest, ChatResponse, RetrievedChunk
from app.session_store import SessionStore


DIET_PRINCIPLE = "季节饮食原则"

# 核心业务逻辑编排
class ChatService:
    def __init__(self, settings: Settings):
        self.settings = settings # 应用配置，包含模型参数、数据库连接信息、默认值等
        self.store = QdrantStore(settings) # 向量数据库接口，负责存储和检索饮食和调理建议的向量表示
        self.sessions = SessionStore(settings) # 管理用户会话状态和历史
        self.parser = IntentParser() # 解析用户输入，识别意图、症状、体质等关键信息
        self.clarifier = ClarificationDecider() # 判断是否需要澄清问题，生成澄清问题
        self.identifier = ConstitutionIdentifier(settings, self.store) # 根据用户描述的症状和体质资料，判断体质倾向
        self.retriever = KnowledgeRetriever(settings, self.store) # 根据用户查询、体质、地区、季节等信息，从向量数据库中检索相关的饮食和调理建议
        self.generator = AnswerGenerator(settings) # 根据用户查询、会话状态、检索到的资料和体质判断结果，生成个性化的回答

    # 处理一次完整的非流式问答交互
    def chat(self, request: ChatRequest) -> ChatResponse:
        state = self.sessions.get(request.user_id, request.conversation_id) 
        parsed = self.parser.parse(request.message, state)
        self._merge_parsed_state(state, parsed)

        identification = None
        if parsed.intent in {"identify_constitution", "mixed"} or (parsed.symptoms and not state.get("constitution")):
            identification = self.identifier.identify(request.message)
            if identification.get("primary_constitution"):
                state["constitution"] = identification["primary_constitution"]
                state["secondary_constitution"] = identification.get("secondary_constitution")

        if parsed.intent == "identify_constitution" and state.get("constitution"):
            answer = self._identification_answer(identification or {}, state)
            state["last_intent"] = parsed.intent
            self._save_turn(request, state, answer)
            return ChatResponse(
                answer=answer,
                need_clarification=False,
                clarification_question=None,
                session_state=self.sessions.to_public_state(state),
                retrieved_chunks=[],
            )

        # 判断是否需要澄清问题，如果需要，优先进行澄清，等待用户补充信息后再继续后续流程
        clarification = self.clarifier.decide(parsed, state)
        if clarification:
            state["last_intent"] = parsed.intent
            self._save_turn(request, state, clarification)
            return ChatResponse(
                answer=clarification,
                need_clarification=True,
                clarification_question=clarification,
                session_state=self.sessions.to_public_state(state),
                retrieved_chunks=[],
            )
        
        # 如果还无法判断体质，优先引导用户补充症状信息，而不是直接给出建议
        if not state.get("constitution"):
            question = "目前还无法判断体质。请补充几个主要症状，比如怕冷怕热、出汗、口干口苦、睡眠、消化和大便情况。"
            self._save_turn(request, state, question)
            return ChatResponse(
                answer=question,
                need_clarification=True,
                clarification_question=question,
                session_state=self.sessions.to_public_state(state),
                retrieved_chunks=[],
            )

        advice_type = self._advice_type(parsed)
        retrieved = self.retriever.retrieve(
            query=request.message,
            constitution=state["constitution"],
            area=state.get("area") or self.settings.default_area,
            season=state.get("season"),
            advice_type=advice_type,
        )

        answer = self.generator.generate(request.message, state, retrieved, identification)
        state["last_intent"] = parsed.intent
        state["last_advice_type"] = advice_type
        self._save_turn(request, state, answer)

        return ChatResponse(
            answer=answer,
            need_clarification=False,
            clarification_question=None,
            session_state=self.sessions.to_public_state(state),
            retrieved_chunks=self._retrieved_chunks(retrieved),
        )

    def chat_stream(self, request: ChatRequest) -> Iterator[str]:
        started_at = perf_counter() # 记录开始时间，用于后续计算整个流程的耗时，帮助性能监控和优化
        metrics: dict[str, int] = {} # 用于收集各个阶段的耗时指标，最终会发送给前端展示，帮助用户了解系统响应的性能表现

        stage_at = perf_counter() # 记录当前阶段的开始时间，用于计算该阶段的耗时，帮助性能监控和优化
        yield self._sse("status", {"text": "正在理解你的问题..."})
        state = self.sessions.get(request.user_id, request.conversation_id)
        parsed = self.parser.parse(request.message, state)
        self._merge_parsed_state(state, parsed)
        metrics["parse_ms"] = self._elapsed_ms(stage_at) # 解析用户输入的时间，包括识别意图、症状、体质等关键信息的时间

        identification = None
        if parsed.intent in {"identify_constitution", "mixed"} or (parsed.symptoms and not state.get("constitution")):
            stage_at = perf_counter()
            yield self._sse("status", {"text": "正在结合症状分析体质倾向..."})
            identification = self.identifier.identify(request.message)
            metrics["identify_ms"] = self._elapsed_ms(stage_at) 
            if identification.get("primary_constitution"):
                state["constitution"] = identification["primary_constitution"]
                state["secondary_constitution"] = identification.get("secondary_constitution")
        else:
            metrics["identify_ms"] = 0

        if parsed.intent == "identify_constitution" and state.get("constitution"):
            answer = self._identification_answer(identification or {}, state)
            state["last_intent"] = parsed.intent
            self._save_turn(request, state, answer)
            metrics["total_ms"] = self._elapsed_ms(started_at)
            yield self._sse("delta", {"text": answer})
            yield self._sse_done(state, [], False, None, metrics)
            return

        clarification = self.clarifier.decide(parsed, state)
        if clarification:
            state["last_intent"] = parsed.intent
            self._save_turn(request, state, clarification)
            metrics["total_ms"] = self._elapsed_ms(started_at)
            yield self._sse("delta", {"text": clarification})
            yield self._sse_done(state, [], True, clarification, metrics)
            return

        if not state.get("constitution"):
            question = "目前还无法判断体质。请补充几个主要症状，比如怕冷怕热、出汗、口干口苦、睡眠、消化和大便情况。"
            self._save_turn(request, state, question)
            metrics["total_ms"] = self._elapsed_ms(started_at)
            yield self._sse("delta", {"text": question})
            yield self._sse_done(state, [], True, question, metrics)
            return

        advice_type = self._advice_type(parsed)

        stage_at = perf_counter()
        yield self._sse("status", {"text": "正在检索相关饮食和调理资料..."})
        retrieved = self.retriever.retrieve(
            query=request.message,
            constitution=state["constitution"],
            area=state.get("area") or self.settings.default_area,
            season=state.get("season"),
            advice_type=advice_type,
        )
        metrics["retrieve_ms"] = self._elapsed_ms(stage_at)
        metrics["retrieved_chunks"] = len(retrieved)
        metrics["prompt_chars"] = self.generator.prompt_size(request.message, state, retrieved, identification)

        yield self._sse(
            "status",
            {
                "text": "正在组织回答，请稍等...",
                "metrics": {
                    "retrieve_ms": metrics["retrieve_ms"],
                    "prompt_chars": metrics["prompt_chars"],
                },
            },
        )

        answer_parts = []
        token_queue: Queue = Queue()
        llm_started_at = perf_counter()
        first_token_seen = False

        def produce_tokens() -> None:
            try:
                for delta in self.generator.generate_stream(request.message, state, retrieved, identification):
                    token_queue.put(("delta", delta))
                token_queue.put(("done", None))
            except Exception as exc:  # noqa: BLE001
                token_queue.put(("error", str(exc)))

        Thread(target=produce_tokens, daemon=True).start()

        while True:
            try:
                kind, payload = token_queue.get(timeout=self.settings.stream_heartbeat_seconds)
            except Empty:
                waited_seconds = int(perf_counter() - llm_started_at)
                if first_token_seen:
                    yield self._sse_ping(waited_seconds)
                    continue

                if waited_seconds >= self.settings.llm_first_token_timeout:
                    yield self._sse(
                        "error",
                        {
                            "message": "模型首 token 等待超时，请稍后重试，或切换更快的模型服务。",
                            "metrics": {
                                **metrics,
                                "llm_first_token_timeout_ms": waited_seconds * 1000,
                            },
                        },
                    )
                    return

                yield self._sse(
                    "status",
                    {
                        "text": f"模型仍在生成中，已等待 {waited_seconds} 秒...",
                        "metrics": {"llm_wait_ms": waited_seconds * 1000},
                    },
                )
                continue

            if kind == "delta":
                if not first_token_seen:
                    first_token_seen = True
                    metrics["llm_first_token_ms"] = self._elapsed_ms(llm_started_at)
                answer_parts.append(payload)
                yield self._sse("delta", {"text": payload})
            elif kind == "done":
                break
            elif kind == "error":
                yield self._sse("error", {"message": "回答生成失败，请稍后重试。", "detail": payload})
                return

        metrics["llm_total_ms"] = self._elapsed_ms(llm_started_at)
        answer = "".join(answer_parts)
        state["last_intent"] = parsed.intent
        state["last_advice_type"] = advice_type
        self._save_turn(request, state, answer)

        metrics["total_ms"] = self._elapsed_ms(started_at)
        yield self._sse_done(state, retrieved, False, None, metrics)

    # 保存用户和助手的对话历史到会话状态中，便于后续解析和生成使用，同时也方便后续查询和分析用户的历史交互记录
    def _save_turn(self, request: ChatRequest, state: dict, answer: str) -> None:
        self.sessions.append_history(state, "user", request.message)
        self.sessions.append_history(state, "assistant", answer)
        self.sessions.save(request.user_id, request.conversation_id, state)

    
    @staticmethod 
    def _advice_type(parsed) -> str | None:
        if parsed.intent == "diet_advice":
            return DIET_PRINCIPLE
        return parsed.advice_type

    # 将向量数据库检索到的原始结果转换为 API 定义的 RetrievedChunk 列表格式，便于前端展示和后续处理使用
    @staticmethod
    def _retrieved_chunks(retrieved: list[dict]) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id=item["payload"].get("chunk_id", ""),
                score=item["score"],
                type=item["payload"].get("type", ""),
                fallback_level=item.get("fallback_level"),
            )
            for item in retrieved
        ]

    # 生成 SSE 格式的最终事件，包含是否需要澄清、澄清问题、会话状态、检索到的知识片段和性能指标等信息，便于前端展示和调试使用
    def _sse_done(
        self,
        state: dict,
        retrieved: list[dict],
        need_clarification: bool,
        clarification_question: str | None,
        metrics: dict[str, int] | None = None,
    ) -> str:
        return self._sse(
            "done",
            {
                "need_clarification": need_clarification,
                "clarification_question": clarification_question,
                "session_state": self.sessions.to_public_state(state).model_dump(),
                "retrieved_chunks": [
                    {
                        "chunk_id": item["payload"].get("chunk_id", ""),
                        "score": item["score"],
                        "type": item["payload"].get("type", ""),
                        "fallback_level": item.get("fallback_level"),
                    }
                    for item in retrieved
                ],
                "metrics": metrics or {},
            },
        )

    # 被_sse_done调用，包含事件类型和数据内容，便于前端解析和展示使用
    @staticmethod
    def _sse(event: str, data: dict) -> str:
        payload = json.dumps(data, ensure_ascii=False)
        return f"event: {event}\ndata: {payload}\n\n"

    # 生成 SSE 格式的心跳事件，包含已等待的秒数，便于前端展示和调试使用
    @staticmethod
    def _sse_ping(waited_seconds: int) -> str:
        return f"event: ping\ndata: {json.dumps({'waited_seconds': waited_seconds})}\n\n"

    # 生成 SSE 格式的错误事件，包含错误信息和性能指标，便于前端展示和调试使用
    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return int((perf_counter() - started_at) * 1000)

    # 将解析后的状态信息合并到会话状态中
    def _merge_parsed_state(self, state: dict, parsed) -> None:
        if parsed.constitution:
            state["constitution"] = parsed.constitution
        if parsed.area:
            state["area"] = parsed.area
        if parsed.season:
            state["season"] = parsed.season

    # 根据体质判断结果生成针对性的回答，便于用户理解和后续交互使用
    @staticmethod
    def _identification_answer(identification: dict, state: dict) -> str:
        primary = state.get("constitution")
        secondary = state.get("secondary_constitution")
        matched = "、".join(identification.get("matched_symptoms") or [])
        reasoning = identification.get("reasoning") or "这是基于你描述的症状和体质资料做出的初步判断。"
        secondary_text = f"，兼有{secondary}倾向" if secondary else ""
        matched_text = f"匹配到的表现包括：{matched}。" if matched else ""
        return (
            f"根据你描述的情况，初步判断你的体质偏向{primary}{secondary_text}。"
            f"{matched_text}{reasoning}\n\n"
            "体质判断不能替代面诊。如果你还想了解饮食或调理建议，请继续告诉我你所在的地区。"
        )
