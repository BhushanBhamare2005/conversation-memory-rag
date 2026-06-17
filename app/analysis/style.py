from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Optional

from app.models import StyleReport
from app.utils.text import count_emojis, estimate_formality, lexicon_sentiment, tokenise, vocabulary_richness


def _tone_from_sentiment(score: float) -> str:
    if score > 0.01:
        return "positive"
    if score < -0.01:
        return "concerned"
    return "neutral"


def _verbosity_label(average_length: float) -> str:
    if average_length < 40:
        return "concise"
    if average_length < 120:
        return "balanced"
    return "verbose"


def _engagement_label(question_frequency: float) -> str:
    if question_frequency > 0.35:
        return "high"
    if question_frequency > 0.15:
        return "medium"
    return "low"


def build_style_report(
    messages: Iterable[str],
    timestamps: Optional[List[datetime]] = None,
) -> StyleReport:
    text_list = [message for message in messages if message]
    average_length = sum(len(message) for message in text_list) / max(1, len(text_list))
    emoji_frequency = sum(count_emojis(message) for message in text_list) / max(1, len(text_list))
    question_frequency = sum(1 for message in text_list if "?" in message) / max(1, len(text_list))
    formality_score = estimate_formality(text_list)
    sentiment_values = [lexicon_sentiment(message) for message in text_list]
    positive = sum(1 for score in sentiment_values if score > 0.01)
    negative = sum(1 for score in sentiment_values if score < -0.01)
    neutral = len(sentiment_values) - positive - negative
    vocabulary_score = vocabulary_richness(text_list)

    latency = None
    if timestamps and len(timestamps) > 1:
        deltas = [
            (timestamps[index] - timestamps[index - 1]).total_seconds()
            for index in range(1, len(timestamps))
            if timestamps[index] >= timestamps[index - 1]
        ]
        if deltas:
            latency = sum(deltas) / len(deltas)

    return StyleReport(
        tone=_tone_from_sentiment(sum(sentiment_values)),
        style="informal" if formality_score < 0.45 else "formal" if formality_score > 0.65 else "mixed",
        verbosity=_verbosity_label(average_length),
        engagement_level=_engagement_label(question_frequency),
        average_message_length=round(average_length, 2),
        emoji_frequency=round(emoji_frequency, 4),
        question_frequency=round(question_frequency, 4),
        formality_score=round(formality_score, 4),
        sentiment_distribution={
            "positive": round(positive / max(1, len(sentiment_values)), 4),
            "neutral": round(neutral / max(1, len(sentiment_values)), 4),
            "negative": round(negative / max(1, len(sentiment_values)), 4),
        },
        response_latency_seconds=round(latency, 2) if latency is not None else None,
        vocabulary_richness=round(vocabulary_score, 4),
    )
