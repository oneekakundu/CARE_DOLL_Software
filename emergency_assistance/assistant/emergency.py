from __future__ import annotations

import logging
from dataclasses import dataclass

from assistant.llm import ModelManager, PromptBuilder
from database.repository import ConversationRepository
from rag.schemas import RetrievedChunk
from rag.vector_store import FaissStore

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AssistantReply:
    response: str
    sources: list[dict[str, object]]
    used_fallback: bool


class ResponseValidator:
    """Validates assistant responses to ensure safety, prevent hallucinations, and enforce disclaimers."""

    DANGER_WORDS = {
        "fire", "smoke", "battery", "unconscious", "bleeding", "trapped",
        "brake", "accident", "shock", "voltage", "water", "submerged",
        "explosion", "spark", "gas", "injury", "flame", "leak", "coolant",
        "crash", "impact", "electrocute", "burning", "fumes"
    }

    TECHNICAL_WORDS = {
        "cut", "disconnect", "cable", "wire", "battery", "disable",
        "fuse", "jack", "tow", "lift", "open", "door", "bonnet",
        "trunk", "manual", "service", "high voltage", "orange", "plug"
    }

    EMERGENCY_DISCLAIMER = (
        "IMPORTANT SAFETY WARNING: Always call local emergency services (e.g., 911) first "
        "in any dangerous situation. Do not attempt procedures you are not trained for."
    )

    @classmethod
    def contains_danger(cls, query: str) -> bool:
        query_lower = query.lower()
        return any(word in query_lower for word in cls.DANGER_WORDS)

    @classmethod
    def contains_technical(cls, query: str) -> bool:
        query_lower = query.lower()
        return any(word in query_lower for word in cls.TECHNICAL_WORDS)

    @classmethod
    def has_emergency_mention(cls, text: str) -> bool:
        text_lower = text.lower()
        mentions = {"911", "112", "999", "emergency services", "emergency number", "call emergency", "first responder"}
        return any(mention in text_lower for mention in mentions)

    @classmethod
    def validate(cls, response: str, query: str, context_available: bool) -> str:
        """Validates and sanitizes the response. Modifies or falls back if safety rules are violated."""
        validated_response = response.strip()

        # 1. If context is missing and query asks for technical procedures, return a safe refusal
        if not context_available and cls.contains_technical(query):
            LOGGER.warning("Context missing for technical query. Refusing answer to prevent hallucination.")
            return (
                f"{cls.EMERGENCY_DISCLAIMER}\n\n"
                "I do not have the specific manufacturer manual indexed for this vehicle. "
                "For your safety, I cannot provide technical instructions or procedures without the verified manual. "
                "Please consult the manufacturer's official documentation or contact trained emergency personnel."
            )

        # 2. If danger is present in query, ensure the response explicitly mentions calling emergency services
        if cls.contains_danger(query) and not cls.has_emergency_mention(validated_response):
            LOGGER.warning("Response did not mention emergency services for a dangerous scenario. Prepending safety disclaimer.")
            validated_response = f"{cls.EMERGENCY_DISCLAIMER}\n\n{validated_response}"

        # 3. Prevent obvious hallucinations or weak answers
        if (
            "hallucinate" in validated_response.lower()
            or "i do not know" in validated_response.lower()
            or "i don't know" in validated_response.lower()
            or len(validated_response) < 15
        ):
            if cls.contains_danger(query):
                return (
                    f"{cls.EMERGENCY_DISCLAIMER}\n\n"
                    "1. Move to a safe location away from the vehicle.\n"
                    "2. Call emergency services immediately (911).\n"
                    "3. Wait for professional assistance."
                )
            else:
                return "I do not have sufficient information in the indexed manuals to answer your question safely."

        return validated_response


class EmergencyAssistant:
    def __init__(self, retriever: FaissStore, llm: ModelManager, repository: ConversationRepository, top_k: int) -> None:
        self._retriever = retriever
        self._llm = llm
        self._repository = repository
        self._top_k = top_k

    def answer(self, query: str) -> AssistantReply:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Please describe what is happening.")

        LOGGER.info("Processing emergency query: '%s...'", clean_query[:50])
        matches = self._retriever.retrieve(clean_query, self._top_k)
        
        # Check if RAG context is actually found/available
        context_available = len(matches) > 0
        context = ""
        if context_available:
            context = "\n\n".join(
                f"[{m.chunk.document_name}, page {m.chunk.page}, section {m.chunk.section}] {m.chunk.text}"
                for m in matches
            )

        try:
            # Generate via ModelManager (includes checks)
            raw_response = self._llm.generate(PromptBuilder.build(clean_query, context))
            fallback = False
        except Exception as error:
            LOGGER.warning("Ollama generation failed or unavailable: %s. Using safe fallback.", error)
            raw_response = self._safe_fallback(clean_query)
            fallback = True

        # Run response validation
        validated_response = ResponseValidator.validate(raw_response, clean_query, context_available)

        # Log query & response to SQLite audit log
        try:
            self._repository.save(clean_query, validated_response)
        except Exception as error:
            LOGGER.error("Failed to log conversation to database: %s", error)

        return AssistantReply(validated_response, self._sources(matches), fallback)

    @staticmethod
    def _sources(matches: list[RetrievedChunk]) -> list[dict[str, object]]:
        return [
            {
                "source": m.chunk.document_name,
                "page": m.chunk.page,
                "section": m.chunk.section,
                "score": round(m.score, 3)
            }
            for m in matches
        ]

    @staticmethod
    def _safe_fallback(query: str) -> str:
        """Determined, offline-first fallback instructions when models are offline."""
        urgent = ResponseValidator.contains_danger(query)
        first_step = (
            "Call your local emergency number (e.g., 911) now and state your exact location."
            if urgent
            else "Move to a safe location away from traffic if you can do so safely."
        )
        return (
            f"1. {first_step}\n"
            "2. Do not re-enter, move, or touch a damaged EV, especially if there is smoke, heat, or leaking fluid.\n"
            "3. Keep other people away and follow instructions from emergency responders.\n"
            "4. If it is safe, share your location and the vehicle's condition with responders."
        )
