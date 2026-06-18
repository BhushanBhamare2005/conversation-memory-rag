from __future__ import annotations

import re
from typing import Any, Dict, List

from app.chatbot.answer_generator import AnswerGenerator
from app.retrieval.retriever import MemoryRetriever


class ConversationalMemoryChatbot:
    def __init__(self, retriever: MemoryRetriever, answer_generator: AnswerGenerator | None = None) -> None:
        self.retriever = retriever
        self.answer_generator = answer_generator or AnswerGenerator()

    def ask(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        retrieval = self.retriever.retrieve(query, top_k=top_k)
        normalized_query = self._normalize_query(query)
        intent = self._detect_intent(normalized_query, retrieval.intent)

        if intent == "persona_summary":
            result = self.answer_generator.generate_persona_summary(
                persona_facts=retrieval.retrieved_persona_facts,
                evidence=self._collect_evidence(retrieval.retrieved_persona_facts),
                sources=self._collect_sources(retrieval.retrieved_persona_facts),
            )
        elif intent == "persona_habits":
            habits = self._filter_persona_items(retrieval.retrieved_persona_facts, {"habit", "interest", "recurring_behavior", "personal_fact"})
            result = self.answer_generator.generate_habits_answer(
                habits=habits,
                evidence=self._collect_evidence(habits),
                sources=self._collect_sources(habits),
            )
        elif intent == "persona_interests":
            interests = self._filter_persona_items(retrieval.retrieved_persona_facts, {"interest"})
            result = self.answer_generator.generate_interests_answer(
                interests=interests,
                evidence=self._collect_evidence(interests),
                sources=self._collect_sources(interests),
            )
        elif intent == "persona_goals":
            goals = self._filter_persona_items(retrieval.retrieved_persona_facts, {"goal"})
            result = self.answer_generator.generate_goals_answer(
                goals=goals,
                evidence=self._collect_evidence(goals),
                sources=self._collect_sources(goals),
            )
        elif intent == "communication_style":
            style_data = [item for item in retrieval.sources if str(item.get("layer", "")) == "global_profile"]
            result = self.answer_generator.generate_communication_style_answer(
                style_data=style_data,
                evidence=self._collect_style_evidence(retrieval.retrieved_chunks, retrieval.retrieved_persona_facts),
                sources=self._collect_sources(style_data),
            )
        elif intent == "recent_events":
            result = self.answer_generator.generate_recent_events_answer(
                events=[item for item in retrieval.sources if str(item.get("layer", "")) in {"checkpoint_summary", "topic_summary"}],
                evidence=self._collect_recent_evidence(retrieval.sources, retrieval.retrieved_topics, retrieval.retrieved_chunks),
                sources=self._collect_event_sources(retrieval.sources, retrieval.retrieved_topics, retrieval.retrieved_chunks),
            )
        elif intent == "topic_summary":
            result = self.answer_generator.generate_topic_summary_answer(
                topics=retrieval.retrieved_topics,
                evidence=self._collect_topic_evidence(retrieval.retrieved_topics, retrieval.retrieved_chunks),
                sources=self._collect_sources(retrieval.retrieved_topics),
            )
        else:
            result = self.answer_generator.generate_persona_summary(
                persona_facts=retrieval.retrieved_persona_facts,
                evidence=self._collect_evidence(retrieval.retrieved_persona_facts),
                sources=self._collect_sources(retrieval.retrieved_persona_facts),
            )

        result["intent"] = intent
        return result

    def _detect_intent(self, query: str, fallback: str) -> str:
        lowered = self._normalize_query(query)
        rules = [
            ("persona_summary", ["what kind of person", "kind of person is this user", "who is this user", "what are they like", "describe the user", "tell me about this user", "persona summary"]),
            ("persona_habits", ["what are their habits", "habits", "routine", "usually", "often", "always", "tend to"]),
            ("persona_interests", ["what interests", "interests", "hobbies", "enjoy", "like doing", "like to"]),
            ("persona_goals", ["what are their goals", "goals", "want to", "future", "dream", "plan"]),
            ("communication_style", ["how do they talk", "communication style", "style", "voice", "tone", "talk"]),
            ("recent_events", ["recent", "latest", "what happened", "last conversation", "recently"]),
            ("topic_summary", ["topic", "topics", "discussed", "talked about", "conversation about"]),
        ]
        for intent, phrases in rules:
            if any(phrase in lowered for phrase in phrases):
                return intent
        fallback_map = {
            "persona": "persona_summary",
            "general": "persona_summary",
            "style": "communication_style",
            "goal": "persona_goals",
            "checkpoint": "recent_events",
            "topic": "topic_summary",
        }
        return fallback_map.get(fallback, "persona_summary")

    def _normalize_query(self, query: str) -> str:
        text = str(query or "")
        text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        text = re.sub(r"[^\w\s?']+", " ", text)
        text = re.sub(r"\s+", " ", text).strip().strip('"').strip("'")
        return text.lower()

    def _filter_persona_items(self, items: List[Dict[str, Any]], categories: set[str]) -> List[Dict[str, Any]]:
        filtered = []
        for item in items:
            metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
            category = self._normalize_category(metadata.get("category") or item.get("category") or "")
            if category in categories:
                filtered.append(item)
        return filtered

    def _collect_evidence(self, items: List[Dict[str, Any]]) -> List[str]:
        evidence: List[str] = []
        for item in items:
            metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
            quotes = metadata.get("evidence", []) or item.get("evidence", []) or []
            for quote in quotes:
                text = self._clean_text(quote)
                if text:
                    evidence.append(text)
            if not quotes:
                value = self._clean_text(metadata.get("value") or item.get("value") or item.get("content") or "")
                if value:
                    evidence.append(value)
        return self._dedupe(evidence)

    def _collect_style_evidence(self, chunks: List[Dict[str, Any]], persona_items: List[Dict[str, Any]]) -> List[str]:
        evidence = []
        for item in persona_items:
            metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
            if str(metadata.get("category") or item.get("category") or "").lower() == "communication_style":
                evidence.extend(metadata.get("evidence", []) or [])
        for chunk in chunks:
            content = str(chunk.get("content") or "")
            for sentence in content.split(". "):
                text = self._clean_text(sentence)
                if text and any(keyword in text.lower() for keyword in ["how", "what", "you", "do you", "?", "tell me"]):
                    evidence.append(text)
        return self._dedupe(evidence)

    def _collect_recent_evidence(self, sources: List[Dict[str, Any]], topics: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> List[str]:
        evidence: List[str] = []
        for source in sources:
            if str(source.get("layer", "")) == "checkpoint_summary":
                evidence.extend(source.get("important_facts", []) or [])
                evidence.extend(source.get("important_events", []) or [])
                evidence.append(source.get("summary", ""))
        for topic in topics:
            evidence.extend(topic.get("metadata", {}).get("evidence", []) or [])
            evidence.append(topic.get("metadata", {}).get("summary", ""))
        for chunk in chunks:
            evidence.append(chunk.get("content", ""))
        return self._dedupe(evidence)

    def _collect_topic_evidence(self, topics: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> List[str]:
        evidence: List[str] = []
        for topic in topics:
            metadata = topic.get("metadata", {})
            evidence.extend(metadata.get("evidence", []) or [])
            evidence.extend(metadata.get("key_facts", []) or [])
            if metadata.get("summary"):
                evidence.append(metadata["summary"])
        for chunk in chunks:
            evidence.append(chunk.get("content", ""))
        return self._dedupe(evidence)

    def _collect_sources(self, items: List[Dict[str, Any]]) -> List[str]:
        sources: List[str] = []
        for item in items:
            sources.extend(self._source_message_ids(item))
        return self._dedupe(sources)

    def _collect_event_sources(self, sources: List[Dict[str, Any]], topics: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> List[str]:
        message_ids: List[str] = []
        for item in sources + topics + chunks:
            message_ids.extend(self._source_message_ids(item))
        return self._dedupe(message_ids)

    def _source_message_ids(self, item: Dict[str, Any]) -> List[str]:
        metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
        candidates = []
        for key in ("source_message_ids", "message_ids", "start_message", "end_message"):
            value = item.get(key) if isinstance(item, dict) else None
            if not value:
                value = metadata.get(key)
            if isinstance(value, list):
                candidates.extend(str(entry) for entry in value if entry)
            elif value:
                candidates.append(str(value))
        if item.get("layer") == "global_profile" and metadata.get("source_message_ids"):
            candidates.extend(str(entry) for entry in metadata.get("source_message_ids", []) if entry)
        if item.get("layer") == "topic_summary":
            for key in ("start_message", "end_message"):
                value = item.get(key) or metadata.get(key)
                if value:
                    candidates.append(str(value))
        return self._dedupe(candidates)

    def _clean_text(self, text: Any) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip().strip("\"'")
        cleaned = re.sub(r"^(?:personal_facts?|goals?|interests?|habits?|communication_style|retrieved_chunks?|retrieved_topics?|similarity_scores?)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*\(0\.\d+\)$", "", cleaned).strip()
        if cleaned.lower().startswith(("personal_facts:", "goals:", "interests:", "habits:", "communication_style:", "retrieved_chunks:", "retrieved_topics:")):
            return ""
        return cleaned

    def _normalize_category(self, category: Any) -> str:
        mapping = {
            "personal_facts": "personal_fact",
            "personal_fact": "personal_fact",
            "goals": "goal",
            "goal": "goal",
            "interests": "interest",
            "interest": "interest",
            "habits": "habit",
            "habit": "habit",
            "communication_style": "communication_style",
            "recurring_behaviors": "recurring_behavior",
            "recurring_behavior": "recurring_behavior",
        }
        return mapping.get(str(category or "").lower(), str(category or "").lower())

    def _dedupe(self, items: Iterable[str]) -> List[str]:
        ordered: List[str] = []
        seen = set()
        for item in items:
            text = self._clean_text(item)
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            ordered.append(text)
        return ordered
