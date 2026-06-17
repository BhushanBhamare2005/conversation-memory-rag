from __future__ import annotations

from collections import Counter
from typing import Iterable, List

from app.utils.text import extract_keywords, representative_lines, summarize_texts


def build_summary(texts: List[str], max_sentences: int = 3) -> str:
    return summarize_texts(texts, max_sentences=max_sentences)


def build_key_facts(texts: List[str], top_n: int = 5) -> List[str]:
    lines = representative_lines(texts, max_items=top_n)
    return [line for line in lines if line]


def build_entities(texts: Iterable[str]) -> List[str]:
    entities: Counter[str] = Counter()
    for text in texts:
        for token in text.split():
            clean = token.strip(",.;:!?()[]{}\"'")
            if clean and clean[:1].isupper() and len(clean) > 2:
                entities[clean] += 1
    return [entity for entity, _ in entities.most_common(8)]


def build_recurring_patterns(texts: List[str]) -> List[str]:
    keywords = extract_keywords(texts, top_n=6)
    return [f"Recurring theme: {keyword}" for keyword in keywords]


def build_important_events(texts: List[str]) -> List[str]:
    event_markers = ("will", "going to", "started", "moving", "plan", "need", "want", "decided")
    events = [text for text in texts if any(marker in text.lower() for marker in event_markers)]
    return events[:5] if events else build_key_facts(texts, top_n=5)
