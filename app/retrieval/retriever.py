from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from app.models import RetrievedDocument, RetrievalResult
from app.retrieval.faiss_store import FaissMemoryStore


class MemoryRetriever:
    def __init__(
        self,
        topic_store: FaissMemoryStore,
        raw_store: FaissMemoryStore,
        checkpoint_store: FaissMemoryStore,
        persona_store: FaissMemoryStore,
        profile_store: FaissMemoryStore,
    ) -> None:
        self.topic_store = topic_store
        self.raw_store = raw_store
        self.checkpoint_store = checkpoint_store
        self.persona_store = persona_store
        self.profile_store = profile_store

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        normalized_query = self._normalize_query(query)
        intent = self._detect_intent(normalized_query)
        topic_hits = self.topic_store.search(normalized_query, top_k=top_k)
        raw_hits = self.raw_store.search(normalized_query, top_k=max(top_k, 6))
        checkpoint_hits = self.checkpoint_store.search(normalized_query, top_k=max(top_k, 4))
        persona_hits = self.persona_store.search(normalized_query, top_k=max(top_k, 6))
        profile_hits = self.profile_store.search(normalized_query, top_k=1)

        ranked_hits = self._rank_hits(intent, topic_hits, raw_hits, checkpoint_hits, persona_hits, profile_hits)
        context = self._build_context_bundle(topic_hits, raw_hits, checkpoint_hits, persona_hits, profile_hits, ranked_hits)
        answer = self._generate_answer(query, intent, context)
        evidence = self._compose_evidence(context)
        sources = self._build_sources(intent, context)

        return RetrievalResult(
            answer=answer,
            intent=intent,
            evidence=evidence,
            sources=sources,
            retrieved_topics=[self._structured_topic(hit) for hit in topic_hits],
            retrieved_chunks=[self._structured_chunk(hit) for hit in raw_hits],
            retrieved_persona_facts=[self._structured_persona_fact(hit) for hit in persona_hits],
            similarity_scores=ranked_hits,
        )

    def _detect_intent(self, query: str) -> str:
        lowered = self._normalize_query(query)
        if any(phrase in lowered for phrase in ["what kind of person", "who is this user", "what are they like", "describe the user", "tell me about this user", "persona summary"]):
            return "persona_summary"
        if any(phrase in lowered for phrase in ["what are their habits", "habits", "routine", "usually", "often", "always", "tend to"]):
            return "persona_habits"
        if any(phrase in lowered for phrase in ["what interests", "interests", "hobbies", "enjoy", "like to", "like doing"]):
            return "persona_interests"
        if any(phrase in lowered for phrase in ["what are their goals", "goals", "goal", "want to", "future", "dream", "plan"]):
            return "persona_goals"
        if any(phrase in lowered for phrase in ["how do they talk", "communication style", "style", "voice", "tone", "talk"]):
            return "communication_style"
        if any(phrase in lowered for phrase in ["recent", "latest", "what happened", "last conversation", "recently"]):
            return "recent_events"
        if any(phrase in lowered for phrase in ["topic", "topics", "discussed", "talked about", "conversation about"]):
            return "topic_summary"
        return "general"

    def _normalize_query(self, query: str) -> str:
        text = str(query or "")
        text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        text = re.sub(r"[^\w\s?']+", " ", text)
        text = re.sub(r"\s+", " ", text).strip().strip('"').strip("'")
        return text.lower()

    def _rank_hits(
        self,
        intent: str,
        topic_hits: List[RetrievedDocument],
        raw_hits: List[RetrievedDocument],
        checkpoint_hits: List[RetrievedDocument],
        persona_hits: List[RetrievedDocument],
        profile_hits: List[RetrievedDocument],
    ) -> List[Dict[str, Any]]:
        layer_weights = {
            "topic_summary": 1.0,
            "raw_chunk": 0.9,
            "checkpoint_summary": 0.95,
            "persona_fact": 1.08,
            "global_profile": 1.12,
        }
        intent_weights = {
            "persona_summary": {"persona_fact": 1.45, "global_profile": 1.35, "topic_summary": 1.15, "checkpoint_summary": 1.0, "raw_chunk": 0.95},
            "persona_habits": {"persona_fact": 1.45, "global_profile": 1.05, "topic_summary": 0.95, "checkpoint_summary": 0.95, "raw_chunk": 0.9},
            "persona_interests": {"persona_fact": 1.45, "global_profile": 1.05, "topic_summary": 0.95, "checkpoint_summary": 0.95, "raw_chunk": 0.9},
            "persona_goals": {"persona_fact": 1.45, "global_profile": 1.2, "topic_summary": 1.1, "checkpoint_summary": 1.0, "raw_chunk": 0.9},
            "communication_style": {"global_profile": 1.45, "persona_fact": 1.15, "raw_chunk": 1.0, "topic_summary": 0.95, "checkpoint_summary": 0.9},
            "recent_events": {"checkpoint_summary": 1.35, "raw_chunk": 1.15, "topic_summary": 1.0, "persona_fact": 0.9, "global_profile": 0.85},
            "topic_summary": {"topic_summary": 1.35, "checkpoint_summary": 1.0, "raw_chunk": 0.98, "persona_fact": 0.85, "global_profile": 0.8},
            "general": {"topic_summary": 1.0, "checkpoint_summary": 1.0, "persona_fact": 1.0, "raw_chunk": 0.95, "global_profile": 1.0},
        }
        scored: List[Dict[str, Any]] = []
        for hit in topic_hits + raw_hits + checkpoint_hits + persona_hits + profile_hits:
            layer = hit.doc_type
            score = hit.score * layer_weights.get(layer, 1.0) * intent_weights.get(intent, intent_weights["general"]).get(layer, 1.0)
            scored.append(
                {
                    "type": layer,
                    "id": hit.doc_id,
                    "similarity": round(hit.score, 4),
                    "weighted_score": round(score, 4),
                    "content": hit.content,
                    "metadata": hit.metadata,
                }
            )
        return sorted(scored, key=lambda item: item["weighted_score"], reverse=True)

    def _build_context_bundle(
        self,
        topic_hits: List[RetrievedDocument],
        raw_hits: List[RetrievedDocument],
        checkpoint_hits: List[RetrievedDocument],
        persona_hits: List[RetrievedDocument],
        profile_hits: List[RetrievedDocument],
        ranked_hits: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "topics": topic_hits,
            "chunks": raw_hits,
            "checkpoints": checkpoint_hits,
            "persona": persona_hits,
            "profile": profile_hits,
            "ranked": ranked_hits,
        }

    def _normalize_category(self, category: str) -> str:
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

    def _structured_persona_fact(self, hit: RetrievedDocument) -> Dict[str, Any]:
        metadata = hit.metadata or {}
        evidence = self._dedupe(metadata.get("evidence", []))
        source_message_ids = self._dedupe(
            [str(item) for item in metadata.get("source_message_ids", []) if item]
            + [str(item) for item in metadata.get("message_ids", []) if item]
        )
        confidence = float(metadata.get("confidence", hit.score) or hit.score or 0.0)
        return {
            "category": self._normalize_category(metadata.get("category") or hit.doc_type),
            "value": self._clean_text(metadata.get("value") or hit.content),
            "evidence": evidence[:3],
            "source_message_ids": source_message_ids,
            "confidence": round(min(0.99, max(0.0, confidence)), 4),
        }

    def _structured_topic(self, hit: RetrievedDocument) -> Dict[str, Any]:
        metadata = hit.metadata or {}
        evidence = self._dedupe(metadata.get("evidence", []))
        key_facts = self._dedupe(metadata.get("key_facts", []))
        source_message_ids = self._dedupe([str(item) for item in [metadata.get("start_message"), metadata.get("end_message")] if item])
        return {
            "topic_id": hit.doc_id,
            "title": self._clean_text(metadata.get("title") or hit.doc_id),
            "summary": self._clean_text(metadata.get("summary") or hit.content),
            "evidence": evidence[:3],
            "key_facts": key_facts[:3],
            "source_message_ids": source_message_ids,
            "confidence": round(min(0.99, max(0.0, float(metadata.get("confidence", hit.score) or hit.score or 0.0))), 4),
        }

    def _structured_chunk(self, hit: RetrievedDocument) -> Dict[str, Any]:
        metadata = hit.metadata or {}
        source_message_ids = self._dedupe([str(item) for item in metadata.get("message_ids", []) if item])
        return {
            "chunk_id": hit.doc_id,
            "excerpt": self._clean_text(hit.content),
            "source_message_ids": source_message_ids,
            "range": metadata.get("range"),
            "confidence": round(min(0.99, max(0.0, float(hit.score or 0.0))), 4),
        }

    def _generate_answer(self, query: str, intent: str, context: Dict[str, Any]) -> str:
        topics = context["topics"]
        chunks = context["chunks"]
        checkpoints = context["checkpoints"]
        persona = context["persona"]
        profile_hits = context["profile"]

        profile = profile_hits[0].metadata if profile_hits else {}
        summary = profile.get("summary", "")
        communication_style = profile.get("communication_style", {})
        dominant_themes = profile.get("dominant_themes", [])
        notable_facts = profile.get("notable_facts", [])

        if intent == "persona_summary":
            facts = self._combine_supporting_phrases(notable_facts[:4] or [hit.metadata.get("value", hit.content) for hit in persona[:4]])
            style = communication_style.get("style", "conversational")
            engagement = communication_style.get("engagement", "medium")
            return (
                f"The user appears ambitious and goal-oriented. {summary} "
                f"Communication style: {style} and {engagement}."
            ) if summary else f"The user appears ambitious and goal-oriented. They are {facts}."

        if intent == "communication_style":
            tone = communication_style.get("tone", "friendly")
            style = communication_style.get("style", "conversational")
            engagement = communication_style.get("engagement", "high")
            characteristics = self._style_characteristics(communication_style)
            return (
                f"The user communicates in a {tone}, highly engaging, {style} manner. "
                f"Their engagement level is {engagement}. {characteristics}"
            )

        if intent == "recent_events":
            recent = [self._simplify_checkpoint_hit(hit) for hit in checkpoints[:3]]
            return f"Recently, the conversation focused on {self._combine_supporting_phrases(recent)}."

        if intent == "topic_summary":
            titles = [self._topic_title(hit) for hit in topics[:4]]
            return f"The main topics discussed were {self._combine_supporting_phrases(titles)}."

        if intent == "persona_goals":
            goals = [hit.metadata.get("value") or hit.content for hit in persona if hit.metadata.get("category") == "goal"]
            theme_text = self._combine_supporting_phrases((goals[:3] + dominant_themes[:2]) or dominant_themes[:3])
            return f"The user's goals center on {theme_text}."

        if intent == "general" and summary:
            return summary

        topic_titles = [self._topic_title(hit) for hit in topics[:3]]
        persona_values = [hit.metadata.get("value") or hit.content for hit in persona[:4]]
        chunk_phrases = [self._simplify_chunk_hit(hit) for hit in chunks[:2]]
        return self._combine_supporting_phrases(persona_values + topic_titles + chunk_phrases)

    def _compose_evidence(self, context: Dict[str, Any]) -> List[str]:
        evidence: List[str] = []
        for hit in context["topics"][:3]:
            topic_evidence = hit.metadata.get("evidence", []) or []
            evidence.extend(topic_evidence)
            if not topic_evidence:
                evidence.append(self._simplify_topic_hit(hit))
        for hit in context["persona"][:4]:
            persona_evidence = hit.metadata.get("evidence", []) or []
            evidence.extend(persona_evidence)
            if not persona_evidence:
                evidence.append(self._simplify_persona_hit(hit))
        for hit in context["chunks"][:3]:
            evidence.append(self._simplify_chunk_hit(hit))
        for hit in context["checkpoints"][:2]:
            checkpoint_evidence = (hit.metadata.get("important_facts", []) or []) + (hit.metadata.get("important_events", []) or [])
            evidence.extend(checkpoint_evidence)
            if not checkpoint_evidence:
                evidence.append(self._simplify_checkpoint_hit(hit))
        profile_hits = context["profile"]
        if profile_hits:
            profile_summary = profile_hits[0].metadata.get("summary", profile_hits[0].content)
            if profile_summary:
                evidence.append(profile_summary)
        return [item for item in evidence if item]

    def _build_sources(self, intent: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        for hit in context["topics"]:
            sources.append(
                {
                    "layer": "topic_summary",
                    "id": hit.doc_id,
                    "title": hit.metadata.get("title", hit.doc_id),
                    "summary": hit.metadata.get("summary", hit.content),
                    "evidence": self._dedupe(hit.metadata.get("evidence", [])),
                    "key_facts": self._dedupe(hit.metadata.get("key_facts", [])),
                    "source_message_ids": self._dedupe([hit.metadata.get("start_message"), hit.metadata.get("end_message")]),
                    "similarity": round(hit.score, 4),
                    "intent": intent,
                }
            )
        for hit in context["chunks"]:
            sources.append(
                {
                    "layer": "raw_chunk",
                    "id": hit.doc_id,
                    "range": hit.metadata.get("range"),
                    "excerpt": hit.content,
                    "source_message_ids": self._dedupe(hit.metadata.get("message_ids", [])),
                    "similarity": round(hit.score, 4),
                    "intent": intent,
                }
            )
        for hit in context["checkpoints"]:
            sources.append(
                {
                    "layer": "checkpoint_summary",
                    "id": hit.doc_id,
                    "range": hit.metadata.get("message_range") or hit.metadata.get("range"),
                    "summary": hit.metadata.get("summary", hit.content),
                    "important_events": self._dedupe(hit.metadata.get("important_events", [])),
                    "important_facts": self._dedupe(hit.metadata.get("important_facts", [])),
                    "recurring_patterns": self._dedupe(hit.metadata.get("recurring_patterns", [])),
                    "similarity": round(hit.score, 4),
                    "intent": intent,
                }
            )
        for hit in context["persona"]:
            sources.append(
                {
                    "layer": "persona_fact",
                    "id": hit.doc_id,
                    "category": self._normalize_category(hit.metadata.get("category") or hit.doc_type),
                    "value": hit.metadata.get("value") or hit.content,
                    "evidence": self._dedupe(hit.metadata.get("evidence", [])),
                    "source_message_ids": self._dedupe(hit.metadata.get("source_message_ids", [])),
                    "similarity": round(hit.score, 4),
                    "intent": intent,
                }
            )
        for hit in context["profile"]:
            sources.append(
                {
                    "layer": "global_profile",
                    "id": hit.doc_id,
                    "summary": hit.metadata.get("summary", hit.content),
                    "communication_style": hit.metadata.get("communication_style", {}),
                    "dominant_themes": self._dedupe(hit.metadata.get("dominant_themes", [])),
                    "notable_facts": self._dedupe(hit.metadata.get("notable_facts", [])),
                    "source_message_ids": self._dedupe(hit.metadata.get("source_message_ids", [])),
                    "similarity": round(hit.score, 4),
                    "intent": intent,
                }
            )
        return sources

    def _topic_title(self, hit: RetrievedDocument) -> str:
        return str(hit.metadata.get("title") or hit.metadata.get("topic_id") or hit.doc_id)

    def _simplify_topic_hit(self, hit: RetrievedDocument) -> str:
        return f"{self._topic_title(hit)} ({hit.metadata.get('confidence', round(hit.score, 4))})"

    def _simplify_chunk_hit(self, hit: RetrievedDocument) -> str:
        content = hit.content.replace("\n", " ")
        return content[:140]

    def _simplify_checkpoint_hit(self, hit: RetrievedDocument) -> str:
        summary = hit.metadata.get("summary") or hit.content
        return summary[:140]

    def _simplify_persona_hit(self, hit: RetrievedDocument) -> str:
        return str(hit.metadata.get("value") or hit.content)

    def _style_characteristics(self, communication_style: Dict[str, Any]) -> str:
        question_frequency = float(communication_style.get("question_frequency", 0.0) or 0.0)
        engagement = communication_style.get("engagement", "medium")
        verbosity = communication_style.get("verbosity", "balanced")
        bullets: List[str] = []
        if question_frequency >= 0.2:
            bullets.append("Frequently asks follow-up questions.")
        if engagement in {"high", "medium"}:
            bullets.append("Shows interest in other people.")
        if verbosity in {"balanced", "verbose"}:
            bullets.append("Uses conversational language.")
        bullets.append("Maintains dialogue flow.")
        return "Communication characteristics: " + " ".join(bullets)

    def _clean_text(self, text: Any) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip().strip('"').strip("'")
        cleaned = re.sub(r"^(?:personal_facts?|goals?|interests?|habits?|communication_style|retrieved_chunks?|retrieved_topics?|similarity_scores?)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*\(0\.\d+\)$", "", cleaned).strip()
        return cleaned

    def _dedupe(self, items: Iterable[Any]) -> List[str]:
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

    def _combine_supporting_phrases(self, phrases: List[str]) -> str:
        cleaned: List[str] = []
        seen = set()
        for phrase in phrases:
            phrase = phrase.strip(" .")
            if not phrase:
                continue
            normalized = phrase.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(phrase)
        if not cleaned:
            return "the available evidence does not support a stronger claim"
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"
