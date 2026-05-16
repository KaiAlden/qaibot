from __future__ import annotations

from collections.abc import Iterator
import json
from queue import Empty, Queue
from threading import Thread
from time import perf_counter
from typing import Any

from app.config import Settings
from app.domain import VALID_AREAS, normalize_area, normalize_term
from app.nlp.clarification import ClarificationDecider
from app.nlp.constitution_scope import ConstitutionScopeResolver
from app.nlp.task_router import TaskRouter
from app.rag.answer_generator import AnswerGenerator
from app.rag.constitution_identifier import ConstitutionIdentifier
from app.rag.qdrant_store import QdrantStore
from app.rag.retriever import KnowledgeRetriever
from app.schemas import ChatRequest, ChatResponse, RetrievedChunk, RoutedTask, ToolCall, TurnContext
from app.session_store import SessionStore
from app.tools import ToolExecutor


DIET_PRINCIPLE = "季节饮食原则"


UNSUPPORTED_ANSWER = "抱歉，这个问题超出了当前中医体质养生助手的处理范围。"


class ChatService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = QdrantStore(settings)
        self.sessions = SessionStore(settings)
        self.router = TaskRouter(settings)
        self.scope_resolver = ConstitutionScopeResolver()
        self.clarifier = ClarificationDecider()
        self.identifier = ConstitutionIdentifier(settings, self.store)
        self.retriever = KnowledgeRetriever(settings, self.store)
        self.generator = AnswerGenerator(settings)
        self.tools = ToolExecutor()

    def chat(self, request: ChatRequest) -> ChatResponse:
        state = self.sessions.get(request.user_id, request.conversation_id)
        runtime_context = self._runtime_context(request)
        active_message = request.message
        resumed = self._resume_pending_clarification(request.message, state)
        if resumed:
            routed = resumed["routed"]
            active_message = resumed["message"]
        else:
            routed = self.router.route(request.message, state, runtime_context)
        tool_response = self._tool_response(request, state, routed)
        if tool_response:
            return tool_response

        parsed = routed.to_parsed_intent()
        self._merge_parsed_state(state, parsed)
        effective_state = self._effective_state(state, runtime_context)
        self._apply_runtime_context_to_parsed(parsed, effective_state)
        self._merge_parsed_state(state, parsed)
        turn_context = self.scope_resolver.resolve(active_message, parsed, routed, state)
        self._apply_turn_context_to_effective(effective_state, turn_context)

        identification = None
        if parsed.intent in {"identify_constitution", "mixed"} or (parsed.symptoms and not state.get("user_constitution")):
            identification = self.identifier.identify(active_message)
            if identification.get("error"):
                answer = identification.get("reasoning") or "体质识别服务暂时不可用，请稍后重试。"
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
            if identification.get("primary_constitution"):
                state["user_constitution"] = identification["primary_constitution"]
                state["secondary_constitution"] = identification.get("secondary_constitution")
                turn_context.user_constitution = identification["primary_constitution"]
                turn_context.secondary_constitution = identification.get("secondary_constitution")
                turn_context.target_constitutions = [identification["primary_constitution"]]
                turn_context.should_update_user_profile = True
                turn_context.scope_type = "identify"
                turn_context.allow_user_profile_in_answer = True
                effective_state = self._effective_state(state, runtime_context)
                self._apply_turn_context_to_effective(effective_state, turn_context)

        if parsed.intent == "identify_constitution" and state.get("user_constitution"):
            answer = self._identification_answer(identification or {}, state)
            state["last_intent"] = parsed.intent
            self._remember_topic(state, turn_context)
            self._clear_pending_clarification(state)
            self._save_turn(request, state, answer)
            return ChatResponse(
                answer=answer,
                need_clarification=False,
                clarification_question=None,
                session_state=self.sessions.to_public_state(state),
                retrieved_chunks=[],
                route="tcm_health",
            )

        if turn_context.needs_scope_clarification and turn_context.clarification_question:
            state["last_intent"] = parsed.intent
            self._save_turn(request, state, turn_context.clarification_question)
            return ChatResponse(
                answer=turn_context.clarification_question,
                need_clarification=True,
                clarification_question=turn_context.clarification_question,
                session_state=self.sessions.to_public_state(state),
                retrieved_chunks=[],
                route="tcm_health",
            )

        if (
            parsed.intent == "constitution_explain"
            and len(turn_context.target_constitutions) == 1
            and turn_context.scope_type != "comparison"
        ):
            target = turn_context.target_constitutions[0]
            retrieved = self.identifier.retrieve_explanation(active_message, target)
            self._annotate_target(retrieved, target)
            answer = self.generator.generate(active_message, effective_state, retrieved, identification, turn_context)
            state["last_intent"] = parsed.intent
            state["last_advice_types"] = []
            self._remember_topic(state, turn_context)
            self._clear_pending_clarification(state)
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
            self._mark_non_tcm_turn(state)
            self._clear_pending_clarification(state)
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
            self._set_pending_clarification(state, routed, active_message, parsed, routed.clarification_question)
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
            self._set_pending_clarification(state, routed, active_message, parsed, clarification)
            self._save_turn(request, state, clarification)
            return ChatResponse(
                answer=clarification,
                need_clarification=True,
                clarification_question=clarification,
                session_state=self.sessions.to_public_state(state),
                retrieved_chunks=[],
                route="tcm_health",
            )

        if not effective_state.get("user_constitution") and parsed.intent == "identify_constitution":
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
        retrieved = self.retriever.retrieve_for_targets(
            query=active_message,
            target_constitutions=turn_context.target_constitutions,
            area=effective_state.get("area") or self.settings.default_area,
            season=effective_state.get("season"),
            advice_type=advice_type,
            advice_types=advice_types,
            include_comparison_context=turn_context.scope_type == "comparison",
        )

        answer = self.generator.generate(active_message, effective_state, retrieved, identification, turn_context)
        state["last_intent"] = parsed.intent
        state["last_advice_types"] = advice_types
        self._remember_topic(state, turn_context)
        self._clear_pending_clarification(state)
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
        active_message = request.message
        resumed = self._resume_pending_clarification(request.message, state)
        if resumed:
            routed = resumed["routed"]
            active_message = resumed["message"]
        else:
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

        parsed = routed.to_parsed_intent()
        self._merge_parsed_state(state, parsed)
        effective_state = self._effective_state(state, runtime_context)
        self._apply_runtime_context_to_parsed(parsed, effective_state)
        self._merge_parsed_state(state, parsed)
        turn_context = self.scope_resolver.resolve(active_message, parsed, routed, state)
        self._apply_turn_context_to_effective(effective_state, turn_context)
        metrics["parse_ms"] = self._elapsed_ms(stage_at)

        identification = None
        if parsed.intent in {"identify_constitution", "mixed"} or (parsed.symptoms and not state.get("user_constitution")):
            stage_at = perf_counter()
            yield self._sse("status", {"text": "正在结合症状分析体质倾向..."})
            identification = self.identifier.identify(active_message)
            metrics["identify_ms"] = self._elapsed_ms(stage_at)
            if identification.get("error"):
                answer = identification.get("reasoning") or "体质识别服务暂时不可用，请稍后重试。"
                state["last_intent"] = parsed.intent
                self._save_turn(request, state, answer)
                metrics["total_ms"] = self._elapsed_ms(started_at)
                yield self._sse(
                    "error",
                    {
                        "message": answer,
                        "metrics": metrics,
                    },
                )
                yield self._sse_done(state, [], False, None, metrics, route="tcm_health")
                return
            if identification.get("primary_constitution"):
                state["user_constitution"] = identification["primary_constitution"]
                state["secondary_constitution"] = identification.get("secondary_constitution")
                turn_context.user_constitution = identification["primary_constitution"]
                turn_context.secondary_constitution = identification.get("secondary_constitution")
                turn_context.target_constitutions = [identification["primary_constitution"]]
                turn_context.should_update_user_profile = True
                turn_context.scope_type = "identify"
                turn_context.allow_user_profile_in_answer = True
                effective_state = self._effective_state(state, runtime_context)
                self._apply_turn_context_to_effective(effective_state, turn_context)
        else:
            metrics["identify_ms"] = 0

        if parsed.intent == "identify_constitution" and state.get("user_constitution"):
            answer = self._identification_answer(identification or {}, state)
            state["last_intent"] = parsed.intent
            self._remember_topic(state, turn_context)
            self._clear_pending_clarification(state)
            self._save_turn(request, state, answer)
            metrics["total_ms"] = self._elapsed_ms(started_at)
            yield self._sse("delta", {"text": answer})
            yield self._sse_done(state, [], False, None, metrics)
            return

        if turn_context.needs_scope_clarification and turn_context.clarification_question:
            state["last_intent"] = parsed.intent
            self._save_turn(request, state, turn_context.clarification_question)
            metrics["total_ms"] = self._elapsed_ms(started_at)
            yield self._sse("delta", {"text": turn_context.clarification_question})
            yield self._sse_done(state, [], True, turn_context.clarification_question, metrics, route="tcm_health")
            return

        if (
            parsed.intent == "constitution_explain"
            and len(turn_context.target_constitutions) == 1
            and turn_context.scope_type != "comparison"
        ):
            stage_at = perf_counter()
            target = turn_context.target_constitutions[0]
            retrieved = self.identifier.retrieve_explanation(active_message, target)
            self._annotate_target(retrieved, target)
            metrics["retrieve_ms"] = self._elapsed_ms(stage_at)
            answer = self.generator.generate(active_message, effective_state, retrieved, identification, turn_context)
            state["last_intent"] = parsed.intent
            state["last_advice_types"] = []
            self._remember_topic(state, turn_context)
            self._clear_pending_clarification(state)
            self._save_turn(request, state, answer)
            metrics["total_ms"] = self._elapsed_ms(started_at)
            yield self._sse("delta", {"text": answer})
            yield self._sse_done(state, retrieved, False, None, metrics, route="tcm_health")
            return

        if parsed.intent == "irrelevant":
            state["last_intent"] = parsed.intent
            state["last_advice_types"] = []
            self._mark_non_tcm_turn(state)
            self._clear_pending_clarification(state)
            self._save_turn(request, state, UNSUPPORTED_ANSWER)
            metrics["total_ms"] = self._elapsed_ms(started_at)
            yield self._sse("delta", {"text": UNSUPPORTED_ANSWER})
            yield self._sse_done(state, [], False, None, metrics, route="unsupported")
            return

        if routed.need_clarification and routed.clarification_question:
            state["last_intent"] = parsed.intent
            self._set_pending_clarification(state, routed, active_message, parsed, routed.clarification_question)
            self._save_turn(request, state, routed.clarification_question)
            metrics["total_ms"] = self._elapsed_ms(started_at)
            yield self._sse("delta", {"text": routed.clarification_question})
            yield self._sse_done(state, [], True, routed.clarification_question, metrics, route=routed.route)
            return

        clarification = self.clarifier.decide(parsed, effective_state)
        if clarification:
            state["last_intent"] = parsed.intent
            self._set_pending_clarification(state, routed, active_message, parsed, clarification)
            self._save_turn(request, state, clarification)
            metrics["total_ms"] = self._elapsed_ms(started_at)
            yield self._sse("delta", {"text": clarification})
            yield self._sse_done(state, [], True, clarification, metrics, route="tcm_health")
            return

        if not effective_state.get("user_constitution") and parsed.intent == "identify_constitution":
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
        retrieved = self.retriever.retrieve_for_targets(
            query=active_message,
            target_constitutions=turn_context.target_constitutions,
            area=effective_state.get("area") or self.settings.default_area,
            season=effective_state.get("season"),
            advice_type=advice_type,
            advice_types=advice_types,
            include_comparison_context=turn_context.scope_type == "comparison",
        )
        metrics["retrieve_ms"] = self._elapsed_ms(stage_at)
        metrics["retrieved_chunks"] = len(retrieved)
        metrics["prompt_chars"] = self.generator.prompt_size(active_message, effective_state, retrieved, identification, turn_context)

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
                for part in self.generator.generate_stream(active_message, effective_state, retrieved, identification, turn_context):
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
        self._remember_topic(state, turn_context)
        self._clear_pending_clarification(state)
        self._save_turn(request, state, answer)

        metrics["total_ms"] = self._elapsed_ms(started_at)
        metrics["thinking_chars"] = thinking_chars
        yield self._sse_done(state, retrieved, False, None, metrics, route="tcm_health")

    def _resume_pending_clarification(self, message: str, state: dict) -> dict[str, Any] | None:
        pending = state.get("pending_clarification") or {}
        if not pending:
            return None

        slot = pending.get("slot")
        if slot == "area":
            area = normalize_area(message)
            if area not in VALID_AREAS:
                return None
            state["area"] = area
        else:
            return None

        routed_data = pending.get("routed") or {}
        routed = RoutedTask.model_validate(routed_data)
        routed.area = state.get("area") or routed.area
        routed.need_clarification = False
        routed.clarification_question = None
        state.pop("pending_clarification", None)
        return {
            "routed": routed,
            "message": pending.get("query") or message,
        }

    def _set_pending_clarification(
        self,
        state: dict,
        routed: RoutedTask,
        message: str,
        parsed: Any,
        question: str,
    ) -> None:
        slot = self._pending_slot(parsed, question)
        if not slot:
            state.pop("pending_clarification", None)
            return
        state["pending_clarification"] = {
            "slot": slot,
            "query": message,
            "routed": routed.model_dump(),
            "intent": parsed.intent,
            "question": question,
            "created_turn_index": int(state.get("turn_index") or 0),
        }

    @staticmethod
    def _pending_slot(parsed: Any, question: str) -> str | None:
        if not parsed.area and parsed.intent in {"diet_advice", "conditioning_advice", "mixed"}:
            return "area"
        if "地区" in question or "省份" in question:
            return "area"
        return None

    @staticmethod
    def _clear_pending_clarification(state: dict) -> None:
        state.pop("pending_clarification", None)

    def _save_turn(self, request: ChatRequest, state: dict, answer: str) -> None:
        state["turn_index"] = int(state.get("turn_index") or 0) + 1
        state.pop("constitution", None)
        self.sessions.append_history(state, "user", request.message)
        self.sessions.append_history(state, "assistant", answer)
        self.sessions.save(request.user_id, request.conversation_id, state)

    def _remember_topic(self, state: dict, turn_context: TurnContext) -> None:
        targets = list(dict.fromkeys(turn_context.target_constitutions))
        state["target_constitutions"] = targets
        if targets:
            state["last_topic_constitutions"] = targets
            state["last_topic_turn_index"] = int(state.get("turn_index") or 0) + 1
            state["non_tcm_turns_since_topic"] = 0
        if turn_context.should_update_user_profile and turn_context.user_constitution:
            state["user_constitution"] = turn_context.user_constitution

    @staticmethod
    def _annotate_target(retrieved: list[dict], target: str | None) -> None:
        for item in retrieved:
            item["target_constitution"] = target
            item["payload"] = {**item.get("payload", {}), "target_constitution": target}

    @staticmethod
    def _mark_non_tcm_turn(state: dict) -> None:
        if state.get("last_topic_constitutions"):
            state["non_tcm_turns_since_topic"] = int(state.get("non_tcm_turns_since_topic") or 0) + 1

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
        self._mark_non_tcm_turn(state)
        self._clear_pending_clarification(state)
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
                target_constitution=item.get("target_constitution") or item["payload"].get("target_constitution"),
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
                        "target_constitution": item.get("target_constitution")
                        or item["payload"].get("target_constitution"),
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
    def _apply_turn_context_to_effective(effective_state: dict, turn_context: TurnContext) -> None:
        effective_state["target_constitutions"] = list(turn_context.target_constitutions)
        if turn_context.target_constitutions:
            effective_state["target_constitution"] = turn_context.target_constitutions[0]

    @staticmethod
    def _apply_runtime_context_to_parsed(parsed, effective_state: dict) -> None:
        runtime_context = effective_state.get("_runtime_context") or {}
        if runtime_context.get("area"):
            parsed.area = effective_state.get("area")
        if runtime_context.get("season"):
            parsed.season = effective_state.get("season")

    def _merge_parsed_state(self, state: dict, parsed) -> None:
        if parsed.area:
            state["area"] = parsed.area
        if parsed.season:
            state["season"] = parsed.season

    @staticmethod
    def _identification_answer(identification: dict, state: dict) -> str:
        primary = state.get("user_constitution")
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
