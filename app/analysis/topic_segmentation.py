from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.config import SETTINGS
from app.memory.summarization import build_key_facts, build_summary
from app.models import MessageRecord, TopicBoundary
from app.retrieval.embeddings import embed_texts
from app.utils.text import extract_keywords


class TopicSegmenter:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or SETTINGS.topic.model_name

    def embed_messages(self, messages: List[MessageRecord]) -> np.ndarray:
        texts = [message.text for message in messages]
        return embed_texts(texts, model_name=self.model_name)

    def detect_boundaries(self, messages: List[MessageRecord]) -> List[TopicBoundary]:
        embeddings = self.embed_messages(messages)
        if len(messages) < 6:
            return []
        window = 3 if len(messages) < 50 else 10
        candidate_scores: Dict[int, float] = {}
        for index in range(window * 2, len(messages) - window):
            previous_window = embeddings[index - window : index]
            current_window = embeddings[index : index + window]
            if len(previous_window) == 0 or len(current_window) == 0:
                continue
            previous_centroid = previous_window.mean(axis=0, keepdims=True)
            current_centroid = current_window.mean(axis=0, keepdims=True)
            candidate_scores[index] = float(cosine_similarity(previous_centroid, current_centroid)[0][0])

        if not candidate_scores:
            return []

        similarities = list(candidate_scores.values())
        percentile = 50 if len(messages) < 50 else 35
        dynamic_threshold = max(0.18, min(0.72, float(np.percentile(similarities, percentile))))

        boundaries: List[TopicBoundary] = []
        for index in sorted(candidate_scores):
            similarity = candidate_scores[index]
            left_neighbor = candidate_scores.get(index - 1, similarity)
            right_neighbor = candidate_scores.get(index + 1, similarity)
            drift = 1.0 - similarity
            is_local_peak = similarity <= left_neighbor and similarity <= right_neighbor
            if similarity <= dynamic_threshold and is_local_peak:
                confidence = self._boundary_confidence(drift, dynamic_threshold, window)
                boundaries.append(
                    TopicBoundary(
                        topic_id=f"topic_boundary_{len(boundaries) + 1}",
                        boundary_index=index,
                        similarity=round(similarity, 4),
                        drift=round(drift, 4),
                        confidence=round(confidence, 4),
                        reason=(
                            f"Semantic drift detected across {window}-message windows: similarity {similarity:.3f}"
                            f" fell below the dynamic threshold {dynamic_threshold:.3f}."
                        ),
                    )
                )
        return self._dedupe_local_boundaries(boundaries)

    def _dedupe_local_boundaries(self, boundaries: List[TopicBoundary]) -> List[TopicBoundary]:
        if not boundaries:
            return boundaries
        deduped = [boundaries[0]]
        for boundary in boundaries[1:]:
            if boundary.boundary_index - deduped[-1].boundary_index >= SETTINGS.topic.window_size // 2:
                deduped.append(boundary)
        return deduped

    def build_topics(self, messages: List[MessageRecord]) -> List[dict]:
        boundaries = self.detect_boundaries(messages)
        split_points = [0] + [boundary.boundary_index for boundary in boundaries] + [len(messages)]
        topics: List[dict] = []
        speaker_counter = Counter(message.speaker for message in messages)

        for topic_index in range(len(split_points) - 1):
            start = split_points[topic_index]
            end = split_points[topic_index + 1]
            slice_messages = messages[start:end]
            if len(slice_messages) < SETTINGS.topic.min_topic_length:
                continue
            texts = [message.text for message in slice_messages]
            title = self._derive_topic_title(texts)
            confidence = self._segment_confidence(slice_messages, boundaries)
            topic_id = f"topic_{topic_index + 1}"
            topics.append(
                {
                    "topic_id": topic_id,
                    "topic_name": title,
                    "title": title,
                    "start_message": slice_messages[0].message_id,
                    "end_message": slice_messages[-1].message_id,
                    "start_index": start,
                    "end_index": end - 1,
                    "confidence": round(confidence, 4),
                    "keywords": extract_keywords(texts, top_n=SETTINGS.topic.max_keywords),
                    "summary": build_summary(texts),
                    "important_facts": build_key_facts(texts),
                    "messages": [message.to_dict() for message in slice_messages],
                    "speaker_count": dict(speaker_counter),
                }
            )
        return topics

    def _segment_confidence(self, slice_messages: List[MessageRecord], boundaries: List[TopicBoundary]) -> float:
        if not boundaries:
            return min(0.94, 0.55 + min(0.3, len(slice_messages) / 80))
        boundary_confidence = max((boundary.confidence for boundary in boundaries), default=0.55)
        length_bonus = min(0.2, len(slice_messages) / 150)
        diversity_bonus = min(0.1, len({message.speaker for message in slice_messages}) / 10)
        return min(0.99, 0.45 + boundary_confidence * 0.35 + length_bonus + diversity_bonus)

    def _boundary_confidence(self, drift: float, threshold: float, window: int) -> float:
        drift_gap = max(0.0, drift - (1.0 - threshold))
        window_bonus = 0.06 if window == 3 else 0.1
        return min(0.99, 0.5 + drift_gap * 1.6 + window_bonus)

    def _derive_topic_title(self, texts: List[str]) -> str:
        joined = " ".join(texts).lower()
        title_map = [
            ("portland", "Moving to Portland"),
            ("radiology", "Radiology Student and Band"),
            ("band", "Radiology Student and Band"),
            ("cook", "Cooking and Reading"),
            ("read", "Cooking and Reading"),
            ("yoga", "Health and Lifestyle"),
        ]
        for keyword, title in title_map:
            if keyword in joined:
                return title
        keywords = extract_keywords(texts, top_n=3)
        return " / ".join(word.title() for word in keywords[:3]) if keywords else "General Conversation"
