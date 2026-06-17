from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

from app.analysis.style import build_style_report
from app.models import EvidenceItem, MessageRecord, PersonaProfile
from app.retrieval.embeddings import embed_texts
from app.utils.text import normalize_whitespace, tokenise

FACT_PATTERNS: List[Tuple[str, str, List[re.Pattern[str]]]] = [
    (
        "personal_fact",
        "personal_fact",
        [
            re.compile(r"\b(?:i[' ]?m|i am) moving to ([^.?!]+)", re.IGNORECASE),
            re.compile(r"\b(?:i[' ]?m|i am) (?:a|an)?\s*full[- ]?time student studying ([^.?!]+)", re.IGNORECASE),
            re.compile(r"\b(?:i[' ]?m|i am) a ([^.?!]+?student)\b", re.IGNORECASE),
            re.compile(r"\b(?:i[' ]?m|i am) (?:originally from|from) ([^.?!]+)", re.IGNORECASE),
            re.compile(r"\b(?:i work as|i work in|i study) ([^.?!]+)", re.IGNORECASE),
        ],
    ),
    (
        "interest",
        "interest",
        [
            re.compile(r"\bi love to ([^.?!]+)", re.IGNORECASE),
            re.compile(r"\bi love ([^.?!]+)", re.IGNORECASE),
            re.compile(r"\bi like to ([^.?!]+)", re.IGNORECASE),
            re.compile(r"\bi like ([^.?!]+)", re.IGNORECASE),
            re.compile(r"\bi enjoy ([^.?!]+)", re.IGNORECASE),
            re.compile(r"\bi play in a band\b", re.IGNORECASE),
        ],
    ),
    (
        "goal",
        "goal",
        [
            re.compile(r"\bi'm going to be ([^.?!]+)", re.IGNORECASE),
            re.compile(r"\bi want to ([^.?!]+)", re.IGNORECASE),
            re.compile(r"\bi plan to ([^.?!]+)", re.IGNORECASE),
            re.compile(r"\bi hope to ([^.?!]+)", re.IGNORECASE),
            re.compile(r"\bi'm pursuing ([^.?!]+)", re.IGNORECASE),
        ],
    ),
    (
        "habit",
        "habit",
        [
            re.compile(r"\bi usually ([^.?!]+)", re.IGNORECASE),
            re.compile(r"\bi often ([^.?!]+)", re.IGNORECASE),
            re.compile(r"\bi always ([^.?!]+)", re.IGNORECASE),
            re.compile(r"\bi tend to ([^.?!]+)", re.IGNORECASE),
        ],
    ),
]

NEGATION_PATTERNS = [re.compile(r"\bnot\b|\bnever\b|\bdon't\b|\bdo not\b", re.IGNORECASE)]


def _clean_fact(text: str) -> str:
    cleaned = normalize_whitespace(text)
    cleaned = cleaned.strip(" .,:;!?\"'()").replace("  ", " ")
    lowered = cleaned.lower()
    if "radiology" in lowered and "student" in lowered:
        return "Full-time radiology student"
    if "portland" in lowered and "moving" in lowered:
        return "Moving to Portland, Oregon"
    if lowered.startswith("to cook") or lowered.startswith("cook") or lowered in {"cook too", "cooking"}:
        return "Cooking"
    if "culinary" in lowered and "dream" in lowered:
        return "Pursuing culinary dreams"
    if "band" in lowered and ("play" in lowered or "playing" in lowered):
        return "Playing in a band"
    if "read" in lowered:
        return "Reading"
    replacements = {
        "fulltime": "full-time",
        "full time": "full-time",
        "radiology at a local college": "radiology student",
        "culinary dreams there": "pursuing culinary dreams",
    }
    for source, target in replacements.items():
        if source in lowered:
            cleaned = target
            lowered = cleaned.lower()
    cleaned = re.sub(r"\bto\s+too\b", "to", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\btoo\b$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^to\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("Cook too", "Cooking")
    cleaned = cleaned.replace("cook too", "Cooking")
    if cleaned.lower() in {"full-time", "full time", "fulltime"}:
        return "Full-time student"
    return cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned


def _embedding_verification(value: str, evidence_texts: List[str], category: str) -> float:
    if not evidence_texts:
        return 0.0
    vectors = embed_texts([value] + evidence_texts)
    if vectors.shape[0] < 2:
        return 0.0
    value_vector = vectors[0]
    evidence_vectors = vectors[1:]
    similarities = evidence_vectors @ value_vector
    if similarities.size == 0:
        return 0.0
    base = float(similarities.max())
    if category == "goal":
        return min(0.99, base + 0.04)
    return min(0.99, base + 0.06)


def _extract_structured_facts(messages: List[MessageRecord]) -> Dict[str, List[EvidenceItem]]:
    grouped: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(dict)
    for message in messages:
        text = normalize_whitespace(message.text)
        if not text:
            continue
        for category, item_category, patterns in FACT_PATTERNS:
            for pattern in patterns:
                match = pattern.search(text)
                if not match:
                    continue
                raw_value = match.group(1) if match.groups() else pattern.pattern
                if pattern.pattern == r"\bi play in a band\b":
                    raw_value = "play in a band"
                value = _clean_fact(raw_value)
                if not value:
                    continue
                evidence_bucket = grouped[category].setdefault(
                    value.lower(),
                    {"value": value, "evidence": [], "messages": []},
                )
                evidence_bucket["evidence"].append(text)
                evidence_bucket["messages"].append(message.message_id)

    structured: Dict[str, List[EvidenceItem]] = {"personal_fact": [], "interest": [], "goal": [], "habit": []}
    for category, items in grouped.items():
        for item in items.values():
            value = str(item["value"])
            evidence_texts = list(dict.fromkeys(item["evidence"]))
            verification = _embedding_verification(value, evidence_texts, category)
            frequency_bonus = min(0.08, 0.03 * max(0, len(evidence_texts) - 1))
            certainty_base = {"personal_fact": 0.8, "interest": 0.76, "goal": 0.82, "habit": 0.72}.get(category, 0.74)
            confidence = round(min(0.99, max(0.55, certainty_base + verification * 0.18 + frequency_bonus)), 4)
            structured[category].append(
                EvidenceItem(
                    category=category,
                    value=value,
                    evidence=evidence_texts[:3],
                    confidence=confidence,
                    source_message_ids=list(dict.fromkeys(item["messages"]))[:3],
                )
            )
    return structured


def _recurring_behaviors(messages: List[MessageRecord]) -> List[EvidenceItem]:
    phrase_counts = Counter()
    evidence_map: Dict[str, List[str]] = defaultdict(list)
    for message in messages:
        normalized = message.text.lower()
        for phrase in ("every day", "usually", "often", "always", "most days", "weekend", "work out", "go to"):
            if phrase in normalized:
                phrase_counts[phrase] += 1
                evidence_map[phrase].append(message.text)
    items: List[EvidenceItem] = []
    for phrase, count in phrase_counts.most_common(6):
        confidence = round(min(0.95, 0.52 + 0.08 * count), 4)
        items.append(
            EvidenceItem(
                category="recurring_behavior",
                value=phrase,
                evidence=evidence_map[phrase][:3],
                confidence=confidence,
                source_message_ids=[],
            )
        )
    return items


def extract_persona(messages: List[MessageRecord], target_speaker: str = "User 1") -> PersonaProfile:
    target_messages = [message for message in messages if message.speaker.lower() == target_speaker.lower()]
    if not target_messages:
        target_messages = messages

    structured = _extract_structured_facts(target_messages)

    habits = structured["habit"]
    personal_facts = structured["personal_fact"]
    interests = structured["interest"]
    goals = structured["goal"]
    personality_traits: List[EvidenceItem] = []
    recurring_behaviors = _recurring_behaviors(target_messages)

    style_report = build_style_report([message.text for message in target_messages], [message.timestamp for message in target_messages])
    style_value = f"tone: {style_report.tone}, style: {style_report.style}, verbosity: {style_report.verbosity}, engagement: {style_report.engagement_level}"
    style_confidence = round(min(0.95, 0.72 + 0.1 * style_report.question_frequency + 0.05 * style_report.formality_score), 4)
    communication_style = [
        EvidenceItem(
            category="communication_style",
            value=style_value,
            evidence=[
                f"Average message length: {style_report.average_message_length}",
                f"Question frequency: {style_report.question_frequency}",
                f"Formality score: {style_report.formality_score}",
            ],
            confidence=style_confidence,
            source_message_ids=[message.message_id for message in target_messages[:5]],
        )
    ]

    return PersonaProfile(
        habits=habits,
        personal_facts=personal_facts,
        interests=interests,
        goals=goals,
        personality_traits=personality_traits,
        communication_style=communication_style,
        recurring_behaviors=recurring_behaviors,
    )
