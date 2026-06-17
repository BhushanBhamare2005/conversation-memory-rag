from __future__ import annotations

from dataclasses import asdict
from typing import List

from app.config import SETTINGS
from app.memory.summarization import build_entities, build_important_events, build_key_facts, build_recurring_patterns, build_summary
from app.models import MemoryCheckpoint, MessageRecord, TopicCheckpoint
from app.utils.text import extract_keywords


def build_topic_checkpoints(topics: List[dict], messages: List[MessageRecord]) -> List[TopicCheckpoint]:
    checkpoints: List[TopicCheckpoint] = []
    message_lookup = {message.message_id: message for message in messages}
    for topic in topics:
        start_message = message_lookup[topic["start_message"]]
        end_message = message_lookup[topic["end_message"]]
        related_texts = [message.text for message in messages if start_message.timestamp <= message.timestamp <= end_message.timestamp]
        checkpoints.append(
            TopicCheckpoint(
                topic_id=topic["topic_id"],
                title=topic.get("title", topic["topic_id"]),
                start_message=start_message.message_id,
                end_message=end_message.message_id,
                message_range=f"{start_message.source_row + 1}-{end_message.source_row + 1}",
                confidence=float(topic["confidence"]),
                keywords=topic.get("keywords", extract_keywords(related_texts, top_n=SETTINGS.topic.max_keywords)),
                summary=topic.get("summary", build_summary(related_texts)),
                key_facts=build_key_facts(related_texts),
                entities=build_entities(related_texts),
                evidence=related_texts[:5],
            )
        )
    return checkpoints


def build_memory_checkpoints(messages: List[MessageRecord], interval: int | None = None) -> List[MemoryCheckpoint]:
    checkpoint_interval = interval or SETTINGS.memory.checkpoint_interval
    checkpoints: List[MemoryCheckpoint] = []
    for index in range(0, len(messages), checkpoint_interval):
        slice_messages = messages[index : index + checkpoint_interval]
        if not slice_messages:
            continue
        texts = [message.text for message in slice_messages]
        checkpoints.append(
            MemoryCheckpoint(
                checkpoint_id=f"cp_{index // checkpoint_interval + 1}",
                message_range=f"{index + 1}-{index + len(slice_messages)}",
                summary=build_summary(texts, max_sentences=4),
                important_events=build_important_events(texts),
                important_facts=build_key_facts(texts),
                recurring_patterns=build_recurring_patterns(texts),
            )
        )
    return checkpoints
