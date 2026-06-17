from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class MessageRecord:
    message_id: str
    timestamp: datetime
    speaker: str
    text: str
    conversation_id: str
    source_row: int
    turn_index: int

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


@dataclass
class TopicBoundary:
    topic_id: str
    boundary_index: int
    similarity: float
    drift: float
    confidence: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TopicCheckpoint:
    topic_id: str
    title: str
    start_message: str
    end_message: str
    message_range: str
    confidence: float
    keywords: List[str]
    summary: str
    key_facts: List[str]
    entities: List[str]
    evidence: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryCheckpoint:
    checkpoint_id: str
    message_range: str
    summary: str
    important_events: List[str]
    important_facts: List[str]
    recurring_patterns: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceItem:
    category: str
    value: str
    evidence: List[str]
    confidence: float
    source_message_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PersonaProfile:
    habits: List[EvidenceItem] = field(default_factory=list)
    personal_facts: List[EvidenceItem] = field(default_factory=list)
    interests: List[EvidenceItem] = field(default_factory=list)
    goals: List[EvidenceItem] = field(default_factory=list)
    personality_traits: List[EvidenceItem] = field(default_factory=list)
    communication_style: List[EvidenceItem] = field(default_factory=list)
    recurring_behaviors: List[EvidenceItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "habits": [item.to_dict() for item in self.habits],
            "personal_facts": [item.to_dict() for item in self.personal_facts],
            "interests": [item.to_dict() for item in self.interests],
            "goals": [item.to_dict() for item in self.goals],
            "personality_traits": [item.to_dict() for item in self.personality_traits],
            "communication_style": [item.to_dict() for item in self.communication_style],
            "recurring_behaviors": [item.to_dict() for item in self.recurring_behaviors],
        }


@dataclass
class GlobalUserProfile:
    summary: str
    dominant_themes: List[str]
    notable_facts: List[str]
    communication_style: Dict[str, Any]
    confidence: float
    source_message_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StyleReport:
    tone: str
    style: str
    verbosity: str
    engagement_level: str
    average_message_length: float
    emoji_frequency: float
    question_frequency: float
    formality_score: float
    sentiment_distribution: Dict[str, float]
    response_latency_seconds: Optional[float]
    vocabulary_richness: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedDocument:
    doc_id: str
    doc_type: str
    content: str
    score: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    answer: str
    intent: str
    evidence: List[str]
    sources: List[Dict[str, Any]]
    retrieved_topics: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_persona_facts: List[Dict[str, Any]] = field(default_factory=list)
    similarity_scores: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
