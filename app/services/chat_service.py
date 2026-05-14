from __future__ import annotations

from collections.abc import Iterator
import json
from queue import Empty, Queue
from threading import Thread
from time import perf_counter
from typing import Any

from app.config import Settings
from app.domain import normalize_area, normalize_term
from app.nlp.clarification import ClarificationDecider
from app.nlp.general_intent import UNSUPPORTED_ANSWER, GeneralIntentParser
from app.nlp.intent_parser import IntentParser
from app.nlp.task_router import TaskRouter
from app.rag.answer_generator import AnswerGenerator
from app.rag.constitution_identifier import ConstitutionIdentifier
from app.rag.qdrant_store import QdrantStore
from app.rag.retriever import KnowledgeRetriever
from app.schemas import ChatRequest, ChatResponse, RetrievedChunk, RoutedTask, ToolCall
from app.session_store import SessionStore
from app.tools import ToolExecutor


DIET_PRINCIPLE = "季节饮食原则"


class ChatService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = QdrantStore(settings)
        self.sessions = SessionStore(settings)
        self.general_parser = GeneralIntentParser()
        self.parser = IntentParser()
        self.router = TaskRouter(settings)
        self.clarifier = ClarificationDecider()
        self.identifier = ConstitutionIdentifier(settings, self.store)
        self.retriever = KnowledgeRetriever(settings, self.store)
        self.generator = AnswerGenerator(settings)
        self.tools = ToolExecutor()

    def chat(self, request: ChatRequest) -> ChatResponse:
        state = self.sessions.get(request.user_id, request.conversation_id)
        runtime_context = self._runtime_context(request)
        general = self.general_parser.parse(request.message)
        if general.answer and general.intent != "external_realtime":
            state["last_intent"] = general.intent
            state["last_advice_types"] = []
            self._save_turn(request, state, general.answer)
            return ChatResponse(
                answer=general.answer,
                need_clarification=False,
                clarification_question=None,
                session_state=self.sessions.to_public_state(state),
                retrieved_chunks=[],
                route=self._general_route(general.answer),
            )

        routed = self.router.route(request.message, state, runtime_context)
        tool_response = self._tool_response(request, state, routed)
        if tool_response:
            return tool_response

        parsed = self._parsed_from_route_or_rules(routed, request.message, state)
        self._merge_parsed_state(state, parsed)
        effective_state = self._effective_state(state, runtime_context)
        self._apply_runtime_context_to_parsed(parsed, effective_state)

        identification = None
        if parsed.intent in {"identify_constitution", "mixed"} or (parsed.symptoms and not state.get("constitution")):
            identification = self.identifier.identify(request.message)
            if identification.get("primary_constitution"):
                state["constitution"] = identification["primary_constitution"]
                state["secondary_constitution"] = identification.get("secondary_constitution")
                effective_state = self._effective_state(state, runtime_context)

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
                route="tcm_health",
            )

        if parsed.intent == "constitution_explain" and state.get("constitution"):
            retrieved = self.identifier.retrieve_explanation(request.message, effective_state["constitution"])
            answer = self.generator.generate(request.message, effective_state, retrieved, identification)
            state["last_intent"] = parsed.intent
            state["last_advice_types"] = []
            self._save_turn(request, state, answer)
            return ChatResponse(
                answer=answer,
                need_clarification=False,
                clarification_question=None,
                session_state=self.sessions.to_public_state(state),
                retrieved_chunks=self._retrieved_chunks(retrieved),
                route="tcm_health",
            )

        if parsed.intent == "irrelevant":
            state["last_intent"] = parsed.intent
            state["last_advice_types"] = []
            self._save_turn(request, state, UNSUPPORTED_ANSWER)
            return ChatResponse(
                answer=UNSUPPORTED_ANSWER,
                need_clarification=False,
                clarification_question=None,
                session_state=self.sessions.to_public_state(state),
                retrieved_chunks=[],
                route="unsupported",
            )

        if routed.need_clarification and routed.clarification_question:
            state["last_intent"] = parsed.intent
            self._save_turn(request, state, routed.clarification_question)
            return ChatResponse(
                answer=routed.clarification_question,
                need_clarification=True,
                clarification_question=routed.clarification_question,
                session_state=self.sessions.to_public_state(state),
                retrieved_chunks=[],
                route=routed.route,
            )

        clarification = self.clarifier.decide(parsed, effective_state)
        if clarification:
            state["last_intent"] = parsed.intent
            self._save_turn(request, state, clarification)
            return ChatResponse(
                answer=clarification,
                need_clarification=True,
                clarification_question=clarification,
                session_state=self.sessions.to_public_state(state),
                retrieved_chunks=[],
                route="tcm_health",
            )

        if not effective_state.get("constitution") and parsed.intent == "identify_constitution":
            question = "目前还无法判断体质。请补充几个主要症状，比如怕冷怕热、出汗、口干口苦、睡眠、消化和大便情况。"
            self._save_turn(request, state, question)
            return ChatResponse(
                answer=question,
                need_clarification=True,
                clarification_question=question,
                session_state=self.sessions.to_public_state(state),
                retrieved_chunks=[],
                route="tcm_health",
            )

        advice_type = self._advice_type(parsed)
        advice_types = self._advice_types(parsed)
        retrieved = self.retriever.retrieve(
            query=request.message,
            constitution=effective_state.get("constitution"),
            area=effective_state.get("area") or self.settings.default_area,
            season=effective_state.get("season"),
            advice_type=advice_type,
            advice_types=advice_types,
        )

        answer = self.generator.generate(request.message, effective_state, retrieved, identification)
        state["last_intent"] = parsed.intent
        state["last_advice_types"] = advice_types
        self._save_turn(request, state, answer)

        return ChatResponse(
            answer=answer,
            need_clarification=False,
            clarification_question=None,
            session_state=self.sessions.to_public_state(state),
            retrieved_chunks=self._retrieved_chunks(retrieved),
            route="tcm_health",
        )

    def chat_stream(self, request: ChatRequest) -> Iterator[str]:
        started_at = perf_counter()
        metrics: dict[str, object] = {}

        stage_at = perf_counter()
        yield self._sse("status", {"text": "正在理解你的问题..."})
        state = self.sessions.get(request.user_id, request.conversation_id)
        runtime_context = self._runtime_context(request)
        general = self.general_parser.parse(request.message)
        if general.answer and general.intent != "external_realtime":
            state["last_intent"] = general.intent
            state["last_advice_types"] = []
            self._save_turn(request, state, general.answer)
            metrics["total_ms"] = self._elapsed_ms(started_at)
            yield self._sse("delta", {"text": general.answer})
            yield self._sse_done(state, [], False, None, metrics, route=self._general_route(general.answer))
            return

        routed = self.router.route(request.message, state, runtime_context)
        tool_response = self._tool_response(request, state, routed)
        if tool_response:
            metrics["total_ms"] = self._elapsed_ms(started_at)
            yield self._sse("delta", {"text": tool_response.answer})
            yield self._sse_done(
                state,
                [],
                tool_response.need_clarification,
                tool_response.clarification_question,
                metrics,
                route=tool_response.route,
                tool_call=tool_response.tool_call,
            )
            return

        parsed = self._parsed_from_route_or_rules(routed, request.message, state)
        self._merge_parsed_state(state, parsed)
        effective_state = self._effective_state(state, runtime_context)
        self._apply_runtime_context_to_parsed(parsed, effective_state)
        metrics["parse_ms"] = self._elapsed_ms(stage_at)

        identification = None
        if parsed.intent in {"identify_constitution", "mixed"} or (parsed.symptoms and not state.get("constitution")):
            stage_at = perf_counter()
            yield self._sse("status", {"text": "正在结合症状分析体质倾向..."})
            identification = self.identifier.identify(request.message)
            metrics["identify_ms"] = self._elapsed_ms(stage_at)
            if identification.get("primary_constitution"):
                state["constitution"] = identification["primary_constitution"]
                state["secondary_constitution"] = identification.get("secondary_constitution")
                effective_state = self._effective_state(state, runtime_context)
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

        if parsed.intent == "constitution_explain" and state.get("constitution"):
            stage_at = perf_counter()
            retrieved = self.identifier.retrieve_explanation(request.message, effective_state["constitution"])
            metrics["retrieve_ms"] = self._elapsed_ms(stage_at)
            answer = self.generator.generate(request.message, effective_state, retrieved, identification)
            state["last_intent"] = parsed.intent
            state["last_advice_types"] = []
            self._save_turn(request, state, answer)
            metrics["total_ms"] = self._elapsed_ms(started_at)
            yield self._sse("delta", {"text": answer})
            yield self._sse_done(state, retrieved, False, None, metrics, route="tcm_health")
            return

        if parsed.intent == "irrelevant":
            state["last_intent"] = parsed.intent
            state["last_advice_types"] = []
            self._save_turn(request, state, UNSUPPORTED_ANSWER)
            metrics["total_ms"] = self._elapsed_ms(started_at)
            yield self._sse("delta", {"text": UNSUPPORTED_ANSWER})
            yield self._sse_done(state, [], False, None, metrics, route="unsupported")
            return

        if routed.need_clarification and routed.clarification_question:
            state["last_intent"] = parsed.intent
            self._save_turn(request, state, routed.clarification_question)
            metrics["total_ms"] = self._elapsed_ms(started_at)
            yield self._sse("delta", {"text": routed.clarification_question})
            yield self._sse_done(state, [], True, routed.clarification_question, metrics, route=routed.route)
            return

        clarification = self.clarifier.decide(parsed, effective_state)
        if clarification:
            state["last_intent"] = parsed.intent
            self._save_turn(request, state, clarification)
            metrics["total_ms"] = self._elapsed_ms(started_at)
            yield self._sse("delta", {"text": clarification})
            yield self._sse_done(state, [], True, clarification, metrics, route="tcm_health")
            return

        if not effective_state.get("constitution") and parsed.intent == "identify_constitution":
            question = "目前还无法判断体质。请补充几个主要症状，比如怕冷怕热、出汗、口干口苦、睡眠、消化和大便情况。"
            self._save_turn(request, state, question)
            metrics["total_ms"] = self._elapsed_ms(started_at)
            yield self._sse("delta", {"text": question})
            yield self._sse_done(state, [], True, question, metrics, route="tcm_health")
            return

        advice_type = self._advice_type(parsed)
        advice_types = self._advice_types(parsed)

        stage_at = perf_counter()
        yield self._sse("status", {"text": "正在检索相关饮食和调理资料..."})
        retrieved = self.retriever.retrieve(
            query=request.message,
            constitution=effective_state.get("constitution"),
            area=effective_state.get("area") or self.settings.default_area,
            season=effective_state.get("season"),
            advice_type=advice_type,
            advice_types=advice_types,
        )
        metrics["retrieve_ms"] = self._elapsed_ms(stage_at)
        metrics["retrieved_chunks"] = len(retrieved)
        metrics["prompt_chars"] = self.generator.prompt_size(request.message, effective_state, retrieved, identification)

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
        thinking_parts = []
        thinking_chars = 0
        token_queue: Queue = Queue()
        llm_started_at = perf_counter()
        first_token_seen = False

        def produce_tokens() -> None:
            try:
                for part in self.generator.generate_stream(request.message, effective_state, retrieved, identification):
                    token_queue.put((part.kind, part.text))
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

            if kind in {"answer", "thinking", "thinking_done"}:
                if not first_token_seen:
                    first_token_seen = True
                    metrics["llm_first_token_ms"] = self._elapsed_ms(llm_started_at)
                if kind == "thinking":
                    thinking_chars += len(payload)
                    if self.settings.thinking_display_mode == "summary":
                        thinking_parts.append(payload)
                    else:
                        thinking_text = self.generator.format_thinking(payload)
                        if thinking_text:
                            yield self._sse("thinking", {"text": thinking_text})
                elif kind == "thinking_done":
                    if self.settings.thinking_display_mode == "summary" and thinking_parts:
                        thinking_text = self.generator.format_thinking("".join(thinking_parts))
                        thinking_parts = []
                        if thinking_text:
                            yield self._sse("thinking", {"text": thinking_text})
                else:
                    if self.settings.thinking_display_mode == "summary" and thinking_parts:
                        thinking_text = self.generator.format_thinking("".join(thinking_parts))
                        thinking_parts = []
                        if thinking_text:
                            yield self._sse("thinking", {"text": thinking_text})
                    answer_parts.append(payload)
                    yield self._sse("delta", {"text": payload})
            elif kind == "done":
                break
            elif kind == "error":
                yield self._sse("error", {"message": "回答生成失败，请稍后重试。", "detail": payload})
                return

        metrics["llm_total_ms"] = self._elapsed_ms(llm_started_at)
        if self.settings.thinking_display_mode == "summary" and thinking_parts:
            thinking_text = self.generator.format_thinking("".join(thinking_parts))
            if thinking_text:
                yield self._sse("thinking", {"text": thinking_text})
        answer = "".join(answer_parts)
        state["last_intent"] = parsed.intent
        state["last_advice_types"] = advice_types
        self._save_turn(request, state, answer)

        metrics["total_ms"] = self._elapsed_ms(started_at)
        metrics["thinking_chars"] = thinking_chars
        yield self._sse_done(state, retrieved, False, None, metrics, route="tcm_health")

    def _save_turn(self, request: ChatRequest, state: dict, answer: str) -> None:
        self.sessions.append_history(state, "user", request.message)
        self.sessions.append_history(state, "assistant", answer)
        self.sessions.save(request.user_id, request.conversation_id, state)

    def _tool_response(self, request: ChatRequest, state: dict, routed: RoutedTask) -> ChatResponse | None:
        if routed.route == "tcm_health":
            return None

        if routed.route in {"weather", "music", "web_search"}:
            tool_name = routed.tool_name or routed.route
            tool_args = routed.tool_args or {"query": request.message}
            tool_call = self.tools.execute(tool_name, tool_args)
            answer = str(tool_call.result or "外部工具接口已预留，当前尚未接入具体工具服务。")
        elif routed.route == "smalltalk":
            tool_call = None
            answer = routed.response_text or "你好，我可以帮你做中医体质识别、体质特点说明、饮食建议和调理建议，也预留了天气、音乐和联网搜索工具接口。"
        else:
            tool_call = None
            answer = UNSUPPORTED_ANSWER

        state["last_intent"] = routed.route
        state["last_advice_types"] = []
        self._save_turn(request, state, answer)
        return ChatResponse(
            answer=answer,
            need_clarification=False,
            clarification_question=None,
            session_state=self.sessions.to_public_state(state),
            retrieved_chunks=[],
            route=routed.route,
            tool_call=tool_call,
        )

    def _parsed_from_route_or_rules(self, routed: RoutedTask, message: str, state: dict):
        if routed.route == "tcm_health":
            parsed = routed.to_parsed_intent()
            legacy = self.parser.parse(message, state)
            if parsed.intent in {"general_followup", "irrelevant"} and legacy.intent != "irrelevant":
                parsed.intent = legacy.intent
            if not parsed.symptoms:
                parsed.symptoms = legacy.symptoms
            if not parsed.constitution:
                parsed.constitution = legacy.constitution
            if not parsed.area:
                parsed.area = legacy.area
            if not parsed.season:
                parsed.season = legacy.season
            merged_advice_types = list(dict.fromkeys([*parsed.advice_types, *legacy.advice_types]))
            if not merged_advice_types and legacy.advice_type:
                merged_advice_types = [legacy.advice_type]
            parsed.advice_types = merged_advice_types
            parsed.advice_type = parsed.advice_type or (merged_advice_types[0] if merged_advice_types else None)
            return parsed
        return self.parser.parse(message, state)

    @staticmethod
    def _general_route(answer: str) -> str:
        return "unsupported" if answer == UNSUPPORTED_ANSWER else "smalltalk"

    @staticmethod
    def _advice_type(parsed) -> str | None:
        if parsed.intent == "diet_advice":
            return DIET_PRINCIPLE
        return parsed.advice_type

    @staticmethod
    def _advice_types(parsed) -> list[str]:
        if parsed.intent == "diet_advice":
            return []
        return list(dict.fromkeys(parsed.advice_types or ([parsed.advice_type] if parsed.advice_type else [])))

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

    def _sse_done(
        self,
        state: dict,
        retrieved: list[dict],
        need_clarification: bool,
        clarification_question: str | None,
        metrics: dict[str, object] | None = None,
        route: str | None = None,
        tool_call: ToolCall | None = None,
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
                "metrics": {
                    **(metrics or {}),
                    "thinking_mode": self.settings.thinking_display_mode,
                },
                "route": route,
                "tool_call": tool_call.model_dump() if tool_call else None,
            },
        )

    @staticmethod
    def _sse(event: str, data: dict) -> str:
        payload = json.dumps(data, ensure_ascii=False)
        return f"event: {event}\ndata: {payload}\n\n"

    @staticmethod
    def _sse_ping(waited_seconds: int) -> str:
        return f"event: ping\ndata: {json.dumps({'waited_seconds': waited_seconds})}\n\n"

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return int((perf_counter() - started_at) * 1000)

    def _runtime_context(self, request: ChatRequest) -> dict[str, Any]:
        context = request.runtime_context
        if not context:
            return {}

        data = {
            key: value
            for key, value in context.model_dump().items()
            if value not in (None, "", [], {})
        }
        if data.get("time") and not data.get("current_time"):
            data["current_time"] = data["time"]

        location = data.get("location")
        if location:
            area = normalize_area(str(location))
            if area:
                data["area"] = area

        solar_term = data.get("solar_term")
        if solar_term:
            term, season = normalize_term(str(solar_term))
            if term:
                data["solar_term"] = term
            if season:
                data["season"] = season
        return data

    @staticmethod
    def _effective_state(state: dict, runtime_context: dict[str, Any]) -> dict:
        effective = {**state}
        if runtime_context:
            effective["_runtime_context"] = runtime_context
            if runtime_context.get("area"):
                effective["area"] = runtime_context["area"]
            if runtime_context.get("season"):
                effective["season"] = runtime_context["season"]
        return effective

    @staticmethod
    def _apply_runtime_context_to_parsed(parsed, effective_state: dict) -> None:
        runtime_context = effective_state.get("_runtime_context") or {}
        if runtime_context.get("area"):
            parsed.area = effective_state.get("area")
        if runtime_context.get("season"):
            parsed.season = effective_state.get("season")

    def _merge_parsed_state(self, state: dict, parsed) -> None:
        if parsed.constitution:
            state["constitution"] = parsed.constitution
        if parsed.area:
            state["area"] = parsed.area
        if parsed.season:
            state["season"] = parsed.season

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
