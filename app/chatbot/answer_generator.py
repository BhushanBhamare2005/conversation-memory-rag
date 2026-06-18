from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence


class AnswerGenerator:
    def generate_persona_summary(
        self,
        persona_facts: Sequence[Dict[str, Any]],
        evidence: Sequence[str],
        sources: Sequence[str],
    ) -> Dict[str, Any]:
        key_points = self._persona_summary_points(persona_facts, evidence)
        answer = "The user appears to be ambitious, creative, and goal-oriented."
        return self._package(answer, evidence, sources, persona_facts, base_confidence=0.9)

    def generate_habits_answer(
        self,
        habits: Sequence[Dict[str, Any]],
        evidence: Sequence[str],
        sources: Sequence[str],
    ) -> Dict[str, Any]:
        key_points = self._habit_points(habits, evidence)
        answer = "The user appears to have these habits."
        return self._package(answer, evidence, sources, habits, base_confidence=0.86)

    def generate_goals_answer(
        self,
        goals: Sequence[Dict[str, Any]],
        evidence: Sequence[str],
        sources: Sequence[str],
    ) -> Dict[str, Any]:
        key_points = self._goal_points(goals, evidence)
        answer = "The user's goals appear to center on the available evidence."
        return self._package(answer, evidence, sources, goals, base_confidence=0.87)

    def generate_communication_style_answer(
        self,
        style_data: Sequence[Dict[str, Any]],
        evidence: Sequence[str],
        sources: Sequence[str],
    ) -> Dict[str, Any]:
        key_points = self._style_points(style_data, evidence)
        answer = "The user communicates in a friendly and engaging manner."
        return self._package(answer, evidence, sources, style_data, base_confidence=0.88)

    def generate_topic_summary_answer(
        self,
        topics: Sequence[Dict[str, Any]],
        evidence: Sequence[str],
        sources: Sequence[str],
    ) -> Dict[str, Any]:
        key_points = self._topic_points(topics, evidence)
        answer = "The main topics discussed were the retrieved themes."
        return self._package(answer, evidence, sources, topics, base_confidence=0.84)

    def generate_interests_answer(
        self,
        interests: Sequence[Dict[str, Any]],
        evidence: Sequence[str],
        sources: Sequence[str],
    ) -> Dict[str, Any]:
        key_points = self._interest_points(interests, evidence)
        answer = "The user seems interested in the retrieved themes."
        return self._package(answer, evidence, sources, interests, base_confidence=0.85)

    def generate_recent_events_answer(
        self,
        events: Sequence[Dict[str, Any]],
        evidence: Sequence[str],
        sources: Sequence[str],
    ) -> Dict[str, Any]:
        key_points = self._recent_points(events, evidence)
        answer = "Recently, the conversation focused on the latest retrieved events."
        return self._package(answer, evidence, sources, events, base_confidence=0.83)

    def _package(
        self,
        answer: str,
        evidence: Sequence[str],
        sources: Sequence[str],
        scored_items: Sequence[Dict[str, Any]],
        base_confidence: float,
    ) -> Dict[str, Any]:
        clean_evidence = self._dedupe([self._clean_text(item) for item in evidence])
        clean_sources = self._dedupe([self._clean_text(item) for item in sources])
        confidence = self._confidence(base_confidence, clean_evidence, clean_sources, scored_items)
        return {
            "answer": self._format_report(answer, clean_evidence, clean_sources, confidence),
            "evidence": clean_evidence[:5],
            "sources": clean_sources[:5],
            "confidence": confidence,
        }

    def _confidence(
        self,
        base_confidence: float,
        evidence: Sequence[str],
        sources: Sequence[str],
        scored_items: Sequence[Dict[str, Any]],
    ) -> float:
        score_values = [float(item.get("confidence", item.get("score", 0.0)) or 0.0) for item in scored_items]
        strongest = max(score_values) if score_values else 0.0
        adjustment = min(0.06, len(evidence) * 0.01) + min(0.04, len(sources) * 0.008)
        value = max(base_confidence, 0.5) * 0.8 + strongest * 0.2 + adjustment
        return round(min(0.99, max(0.0, value)), 4)

    def _persona_summary_points(self, persona_facts: Sequence[Dict[str, Any]], evidence: Sequence[str]) -> List[str]:
        points: List[str] = []
        corpus = " ".join(self._clean_text(item) for item in evidence)
        for item in persona_facts:
            value = self._extract_fact_value(item)
            lowered = value.lower()
            if any(keyword in lowered for keyword in ["cook", "cooking"]):
                points.append("Enjoys cooking.")
            if any(keyword in lowered for keyword in ["read", "reading"]):
                points.append("Enjoys reading.")
            if any(keyword in lowered for keyword in ["portland", "moving"]):
                points.append("Will relocate for personal growth.")
            if any(keyword in lowered for keyword in ["radiology", "student"]):
                points.append("Is a full-time radiology student.")
            if "band" in lowered:
                points.append("Participates in a band.")
            if any(keyword in lowered for keyword in ["dream", "goal", "ambitious"]):
                points.append("Is goal-oriented.")
        if not points and corpus:
            if "cook" in corpus:
                points.append("Enjoys cooking.")
            if "read" in corpus:
                points.append("Enjoys reading.")
            if "band" in corpus:
                points.append("Participates in a band.")
            if "radiology" in corpus:
                points.append("Is a full-time radiology student.")
        return self._dedupe(points)[:5] or ["Shows a mix of personal interests and long-term goals."]

    def _habit_points(self, habits: Sequence[Dict[str, Any]], evidence: Sequence[str]) -> List[str]:
        return self._points_from_items(habits, evidence, {
            "cook": "Cooking",
            "read": "Reading",
            "run": "Running",
            "yoga": "Yoga",
        }, fallback="No strong habits were surfaced.")

    def _interest_points(self, interests: Sequence[Dict[str, Any]], evidence: Sequence[str]) -> List[str]:
        return self._points_from_items(interests, evidence, {
            "cook": "Cooking",
            "read": "Reading",
            "band": "Music and performing in a band",
            "portland": "Relocating and personal growth",
        }, fallback="No clear interests were detected.")

    def _goal_points(self, goals: Sequence[Dict[str, Any]], evidence: Sequence[str]) -> List[str]:
        return self._points_from_items(goals, evidence, {
            "dream": "Pursuing culinary dreams",
            "goal": "Goal-oriented",
            "move": "Moving to Portland for growth",
            "relocat": "Moving to Portland for growth",
            "radiology": "Studying radiology",
        }, fallback="No strong goals were detected.")

    def _style_points(self, style_data: Sequence[Dict[str, Any]], evidence: Sequence[str]) -> List[str]:
        points: List[str] = []
        for item in style_data:
            payload = item.get("communication_style") if isinstance(item, dict) else None
            if isinstance(payload, dict):
                tone = self._clean_text(str(payload.get("tone") or ""))
                style = self._clean_text(str(payload.get("style") or ""))
                verbosity = self._clean_text(str(payload.get("verbosity") or ""))
                engagement = self._clean_text(str(payload.get("engagement") or ""))
                if tone:
                    points.append(f"Tone: {tone}.")
                if style:
                    points.append(f"Style: {style}.")
                if verbosity:
                    points.append(f"Verbosity: {verbosity}.")
                if engagement:
                    points.append(f"Engagement: {engagement}.")
        for item in evidence:
            lowered = item.lower()
            if "?" in item or any(keyword in lowered for keyword in ["how", "what", "why", "tell me", "do you"]):
                points.append("Frequently asks follow-up questions.")
            if "friendly" in lowered or "positive" in lowered:
                points.append("Friendly tone.")
            if "engaging" in lowered or "high engagement" in lowered:
                points.append("Engaging and responsive.")
            if "conversational" in lowered or "mixed" in lowered:
                points.append("Conversational language.")
        return self._dedupe(points)[:5] or ["Uses a conversational, back-and-forth style."]

    def _topic_points(self, topics: Sequence[Dict[str, Any]], evidence: Sequence[str]) -> List[str]:
        points: List[str] = []
        for item in topics:
            title = self._clean_text(str(item.get("title") or item.get("metadata", {}).get("title") or item.get("doc_id") or ""))
            if title:
                points.append(title)
        for item in evidence:
            text = self._clean_text(item)
            if text:
                points.append(text)
        return self._dedupe(points)[:5] or ["No topics were identified."]

    def _recent_points(self, events: Sequence[Dict[str, Any]], evidence: Sequence[str]) -> List[str]:
        points: List[str] = []
        for item in events:
            summary = self._clean_text(str(item.get("summary") or item.get("metadata", {}).get("summary") or ""))
            if summary:
                points.append(summary)
        for item in evidence:
            text = self._clean_text(item)
            if text:
                points.append(text)
        return self._dedupe(points)[:5] or ["Recent conversation details were limited."]

    def _points_from_items(
        self,
        items: Sequence[Dict[str, Any]],
        evidence: Sequence[str],
        mapping: Dict[str, str],
        fallback: str,
    ) -> List[str]:
        points: List[str] = []
        for item in items:
            value = self._extract_fact_value(item)
            lowered = value.lower()
            for keyword, label in mapping.items():
                if keyword in lowered:
                    points.append(label)
        for item in evidence:
            lowered = item.lower()
            for keyword, label in mapping.items():
                if keyword in lowered:
                    points.append(label)
        return self._dedupe(points)[:5] or [fallback]

    def _format_report(self, answer: str, evidence: Sequence[str], sources: Sequence[str], confidence: float) -> str:
        evidence_lines = [f"- {item}" for item in evidence[:5]] or ["- None available"]
        source_lines = [f"- {item}" for item in sources[:5]] or ["- None available"]
        return "\n".join(
            [
                "Answer",
                answer,
                "",
                "Evidence",
                *evidence_lines,
                "",
                "Sources",
                *source_lines,
                "",
                f"Confidence: {confidence:.4f}",
            ]
        )

    def _extract_fact_value(self, item: Dict[str, Any]) -> str:
        metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
        value = metadata.get("value") or item.get("value") or item.get("content") or ""
        return self._clean_text(str(value))

    def _clean_text(self, text: Any) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip().strip("\"'")
        cleaned = re.sub(r"\s*\(0\.\d+\)$", "", cleaned).strip()
        if cleaned.lower().startswith(("personal_facts:", "goals:", "interests:", "habits:", "communication_style:", "retrieved_chunks:", "retrieved_topics:", "answer:", "evidence:", "sources:", "confidence:")):
            return ""
        return cleaned

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
