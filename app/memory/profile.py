from __future__ import annotations

from collections import Counter
from typing import List

from app.models import GlobalUserProfile, PersonaProfile, TopicCheckpoint


def build_global_user_profile(
    persona: PersonaProfile,
    topic_checkpoints: List[TopicCheckpoint],
    style_report: object,
    source_message_ids: List[str] | None = None,
) -> GlobalUserProfile:
    dominant_themes = _collect_dominant_themes(persona, topic_checkpoints)
    notable_facts = _collect_notable_facts(persona)
    communication_style = _style_snapshot(style_report)
    summary = _compose_summary(dominant_themes, notable_facts, communication_style)
    confidence = _profile_confidence(persona, topic_checkpoints, communication_style)
    return GlobalUserProfile(
        summary=summary,
        dominant_themes=dominant_themes,
        notable_facts=notable_facts,
        communication_style=communication_style,
        confidence=confidence,
        source_message_ids=source_message_ids or [],
    )


def _collect_dominant_themes(persona: PersonaProfile, topic_checkpoints: List[TopicCheckpoint]) -> List[str]:
    themes: List[str] = []
    for checkpoint in topic_checkpoints[:4]:
        themes.append(checkpoint.title)
    for item in persona.goals[:3] + persona.interests[:3] + persona.personal_facts[:3]:
        themes.append(item.value)
    return _dedupe(themes)[:6]


def _collect_notable_facts(persona: PersonaProfile) -> List[str]:
    facts: List[str] = []
    for category in [persona.personal_facts, persona.goals, persona.interests, persona.recurring_behaviors]:
        for item in category[:3]:
            if item.value.strip().lower() in {"full-time", "full time", "fulltime", "cook"}:
                continue
            facts.append(item.value)
    return _dedupe(facts)[:8]


def _style_snapshot(style_report: object) -> dict[str, object]:
    return {
        "tone": getattr(style_report, "tone", "neutral"),
        "style": getattr(style_report, "style", "conversational"),
        "verbosity": getattr(style_report, "verbosity", "moderate"),
        "engagement": getattr(style_report, "engagement_level", "medium"),
        "average_message_length": getattr(style_report, "average_message_length", 0.0),
        "question_frequency": getattr(style_report, "question_frequency", 0.0),
        "emoji_frequency": getattr(style_report, "emoji_frequency", 0.0),
        "formality_score": getattr(style_report, "formality_score", 0.0),
        "vocabulary_diversity": getattr(style_report, "vocabulary_richness", 0.0),
        "sentiment_distribution": getattr(style_report, "sentiment_distribution", {}),
    }


def _compose_summary(dominant_themes: List[str], notable_facts: List[str], communication_style: dict[str, object]) -> str:
    style = communication_style.get("style", "conversational")
    engagement = communication_style.get("engagement", "medium")
    tone = communication_style.get("tone", "neutral")
    theme_text = _join_phrases(dominant_themes[:3])
    fact_text = _join_phrases(notable_facts[:4])
    return (
        f"The user is a {tone}, {style}, and {engagement} communicator. "
        f"Their main themes are {theme_text}. "
        f"Key facts include {fact_text}."
    ).strip()


def _profile_confidence(persona: PersonaProfile, topic_checkpoints: List[TopicCheckpoint], communication_style: dict[str, object]) -> float:
    persona_strength = sum(len(category) for category in [persona.personal_facts, persona.interests, persona.goals])
    topic_strength = len(topic_checkpoints)
    style_strength = 1 if communication_style.get("engagement") else 0
    confidence = 0.68 + min(0.18, persona_strength * 0.02) + min(0.08, topic_strength * 0.01) + style_strength * 0.03
    return round(min(0.99, confidence), 4)


def _join_phrases(values: List[str]) -> str:
    cleaned = _dedupe(values)
    if not cleaned:
        return "general conversation and personal memory"
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        normalized = value.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(value)
    return ordered
