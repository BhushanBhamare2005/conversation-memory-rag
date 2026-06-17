from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from statistics import mean
from typing import Any, Iterable

from app.analysis.style import build_style_report
from app.models import MessageRecord, MemoryCheckpoint, PersonaProfile, StyleReport, TopicBoundary


def _parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def analyze_communication_style(speaker: str, messages: list[MessageRecord]) -> StyleReport:
    texts = [message.text for message in messages]
    style = build_style_report(texts, [message.timestamp for message in messages])
    return StyleReport(
        tone=style.tone,
        style=style.style,
        verbosity=style.verbosity,
        engagement_level=style.engagement_level,
        average_message_length=style.average_message_length,
        emoji_frequency=style.emoji_frequency,
        question_frequency=style.question_frequency,
        formality_score=style.formality_score,
        sentiment_distribution=style.sentiment_distribution,
        response_latency_seconds=style.response_latency_seconds,
        vocabulary_richness=style.vocabulary_richness,
    )


def compute_average_latency(messages: list[MessageRecord]) -> float | None:
    if len(messages) < 2:
        return None
    timestamps = []
    for message in messages:
        parsed = _parse_timestamp(message.timestamp)
        if parsed is not None:
            timestamps.append(parsed)
    if len(timestamps) < 2:
        return None
    gaps = [(timestamps[index] - timestamps[index - 1]).total_seconds() for index in range(1, len(timestamps))]
    return round(sum(gaps) / len(gaps), 4) if gaps else None


def build_topic_transition_graph(topics: list[TopicBoundary]) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for index in range(1, len(topics)):
        edges.append(
            {
                "source": topics[index - 1].topic_id,
                "target": topics[index].topic_id,
                "weight": round((topics[index - 1].confidence + topics[index].confidence) / 2, 4),
            }
        )
    return edges


def build_message_volume_series(messages: list[MessageRecord]) -> list[dict[str, object]]:
    grouped: defaultdict[str, int] = defaultdict(int)
    for message in messages:
        timestamp = _parse_timestamp(message.timestamp)
        bucket = timestamp.date().isoformat() if timestamp else message.timestamp[:10]
        grouped[bucket] += 1
    return [{"date": key, "count": grouped[key]} for key in sorted(grouped)]


def build_topic_distribution(topics: list[TopicBoundary]) -> list[dict[str, object]]:
    return [
        {
            "topic_id": topic.topic_id,
            "topic_name": getattr(topic, "title", getattr(topic, "topic_name", topic.topic_id)),
            "messages": getattr(topic, "message_count", 0),
            "confidence": topic.confidence,
        }
        for topic in topics
    ]


def build_persona_trait_distribution(persona_profiles: dict[str, PersonaProfile]) -> list[dict[str, object]]:
    distribution: list[dict[str, object]] = []
    for speaker, profile in persona_profiles.items():
        trait_count = sum(
            len(category.items)
            for category in [
                profile.habits,
                profile.personal_facts,
                profile.interests,
                profile.goals,
                profile.personality_traits,
                profile.communication_style,
                profile.recurring_behaviors,
            ]
        )
        distribution.append({"speaker": speaker, "trait_count": trait_count})
    return distribution


def build_conversation_heatmap(messages: list[MessageRecord]) -> list[dict[str, object]]:
    bins: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for index, message in enumerate(messages):
        row = f"row_{index // 20 + 1}"
        bins[row][message.speaker] += 1
    heatmap: list[dict[str, object]] = []
    for row, counter in bins.items():
        for speaker, count in counter.items():
            heatmap.append({"row": row, "speaker": speaker, "count": count})
    return heatmap


def build_analytics(
    messages: list[MessageRecord],
    topics: list[TopicBoundary],
    checkpoints: list[MemoryCheckpoint],
    persona_profiles: dict[str, PersonaProfile],
) -> dict[str, object]:
    return {
        "total_messages": len(messages),
        "total_topics": len(topics),
        "total_checkpoints": len(checkpoints),
        "total_persona_traits": sum(
            len(category.items)
            for profile in persona_profiles.values()
            for category in [
                profile.habits,
                profile.personal_facts,
                profile.interests,
                profile.goals,
                profile.personality_traits,
                profile.communication_style,
                profile.recurring_behaviors,
            ]
        ),
        "message_volume_over_time": build_message_volume_series(messages),
        "topic_distribution": build_topic_distribution(topics),
        "persona_trait_distribution": build_persona_trait_distribution(persona_profiles),
        "conversation_heatmap": build_conversation_heatmap(messages),
        "topic_transition_graph": build_topic_transition_graph(topics),
    }


def build_memory_layer_summary(memory_layers: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for layer_name, layer_info in memory_layers.items():
        summary.append(
            {
                "layer": layer_name,
                "documents": layer_info.get("documents", 0),
                "confidence": layer_info.get("confidence"),
            }
        )
    return summary


def build_evaluation_metrics(bundle: dict[str, Any], retrieval_result: dict[str, Any] | None = None) -> dict[str, float]:
    messages = bundle.get("messages", [])
    topics = bundle.get("topics", [])
    topic_checkpoints = bundle.get("topic_checkpoints", [])
    memory_checkpoints = bundle.get("memory_checkpoints", [])
    persona = bundle.get("persona", {})
    global_profile = bundle.get("global_profile", {})

    topic_confidences = [float(topic.get("confidence", 0.0)) for topic in topics] or [0.0]
    persona_confidences = [float(item.get("confidence", 0.0)) for values in persona.values() for item in values] or [0.0]

    topic_segmentation_quality = min(1.0, (mean(topic_confidences) * 0.6) + min(0.4, len(topics) / max(1, len(messages) / 4)))
    persona_precision = min(1.0, (mean(persona_confidences) * 0.7) + min(0.3, sum(1 for values in persona.values() if values) / 7))
    retrieval_accuracy = 0.0
    source_attribution_coverage = 0.0
    answer_quality = 0.0

    if retrieval_result:
        sources = retrieval_result.get("sources", []) or []
        evidence = retrieval_result.get("evidence", []) or []
        similarity_scores = retrieval_result.get("similarity_scores", []) or []
        retrieval_accuracy = min(1.0, mean([float(item.get("similarity", 0.0)) for item in similarity_scores[:5]]) if similarity_scores else 0.0)
        attributed = sum(1 for source in sources if source.get("source_message_ids") or source.get("range") or source.get("title"))
        source_attribution_coverage = attributed / max(1, len(sources))
        answer_quality = min(1.0, (len(evidence) / 6) * 0.35 + source_attribution_coverage * 0.35 + min(0.3, len(retrieval_result.get("answer", "")) / 500))
    else:
        retrieval_accuracy = min(1.0, len(topic_checkpoints) / max(1, len(memory_checkpoints) + 1))
        source_attribution_coverage = min(1.0, len(topic_checkpoints) / max(1, len(topics) or 1))
        answer_quality = min(1.0, float(global_profile.get("confidence", 0.0)) or 0.0)

    return {
        "topic_segmentation_quality": round(topic_segmentation_quality, 4),
        "persona_precision": round(persona_precision, 4),
        "retrieval_accuracy": round(retrieval_accuracy, 4),
        "source_attribution_coverage": round(source_attribution_coverage, 4),
        "answer_quality": round(answer_quality, 4),
    }
