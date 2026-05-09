from __future__ import annotations

from app.config import Settings
from app.nlp.clarification import ClarificationDecider
from app.nlp.intent_parser import IntentParser
from app.rag.answer_generator import AnswerGenerator
from app.rag.constitution_identifier import ConstitutionIdentifier
from app.rag.qdrant_store import QdrantStore
from app.rag.retriever import KnowledgeRetriever
from app.schemas import ChatRequest, ChatResponse, RetrievedChunk
from app.session_store import SessionStore


class ChatService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = QdrantStore(settings)
        self.sessions = SessionStore(settings)
        self.parser = IntentParser()
        self.clarifier = ClarificationDecider()
        self.identifier = ConstitutionIdentifier(settings, self.store)
        self.retriever = KnowledgeRetriever(settings, self.store)
        self.generator = AnswerGenerator(settings)

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
            self.sessions.append_history(state, "user", request.message)
            self.sessions.append_history(state, "assistant", answer)
            self.sessions.save(request.user_id, request.conversation_id, state)
            return ChatResponse(
                answer=answer,
                need_clarification=False,
                clarification_question=None,
                session_state=self.sessions.to_public_state(state),
                retrieved_chunks=[],
            )

        clarification = self.clarifier.decide(parsed, state)
        if clarification:
            state["last_intent"] = parsed.intent
            self.sessions.append_history(state, "user", request.message)
            self.sessions.append_history(state, "assistant", clarification)
            self.sessions.save(request.user_id, request.conversation_id, state)
            return ChatResponse(
                answer=clarification,
                need_clarification=True,
                clarification_question=clarification,
                session_state=self.sessions.to_public_state(state),
                retrieved_chunks=[],
            )

        if not state.get("constitution"):
            question = "目前还无法判断体质。请补充几个主要症状，比如怕冷怕热、出汗、口干口苦、睡眠、消化和大便情况。"
            self.sessions.append_history(state, "user", request.message)
            self.sessions.append_history(state, "assistant", question)
            self.sessions.save(request.user_id, request.conversation_id, state)
            return ChatResponse(
                answer=question,
                need_clarification=True,
                clarification_question=question,
                session_state=self.sessions.to_public_state(state),
                retrieved_chunks=[],
            )

        advice_type = parsed.advice_type
        if parsed.intent == "diet_advice":
            advice_type = "季节饮食原则"

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
        self.sessions.append_history(state, "user", request.message)
        self.sessions.append_history(state, "assistant", answer)
        self.sessions.save(request.user_id, request.conversation_id, state)

        return ChatResponse(
            answer=answer,
            need_clarification=False,
            clarification_question=None,
            session_state=self.sessions.to_public_state(state),
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id=item["payload"].get("chunk_id", ""),
                    score=item["score"],
                    type=item["payload"].get("type", ""),
                    fallback_level=item.get("fallback_level"),
                )
                for item in retrieved
            ],
        )

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
