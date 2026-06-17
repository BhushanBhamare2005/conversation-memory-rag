from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)
SPEAKER_PATTERN = re.compile(
    r"^\s*(User\s*\d+|Speaker\s*\d+|[A-Za-z][A-Za-z0-9 _-]{0,30}):\s*(.+)$"
)
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'\-]+")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def split_sentences(text: str) -> List[str]:
    clean_text = normalize_whitespace(text)
    if not clean_text:
        return []
    parts = SENTENCE_PATTERN.split(clean_text)
    return [part.strip() for part in parts if part.strip()]


def tokenise(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


def extract_keywords(texts: Iterable[str], top_n: int = 8) -> List[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        for token in tokenise(text):
            if token not in ENGLISH_STOP_WORDS and len(token) > 2:
                counter[token] += 1
    return [token for token, _ in counter.most_common(top_n)]


def count_emojis(text: str) -> int:
    return len(EMOJI_PATTERN.findall(text or ""))


def vocabulary_richness(texts: Iterable[str]) -> float:
    tokens = [token for text in texts for token in tokenise(text)]
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 4)


def estimate_formality(texts: Iterable[str]) -> float:
    formal_markers = ("please", "thank", "appreciate", "could you", "would you", "kindly")
    informal_markers = ("gonna", "wanna", "lol", "btw", "hey", "yeah", "omg")
    total = 0
    score = 0
    for text in texts:
        lowered = text.lower()
        total += 1
        if any(marker in lowered for marker in formal_markers):
            score += 1
        if any(marker in lowered for marker in informal_markers):
            score -= 1
    if total == 0:
        return 0.5
    normalized = (score + total) / (2 * total)
    return round(min(1.0, max(0.0, normalized)), 4)


def lexicon_sentiment(text: str) -> float:
    positive_words = {
        "good",
        "great",
        "awesome",
        "love",
        "amazing",
        "happy",
        "excited",
        "cool",
        "wonderful",
        "enjoy",
    }
    negative_words = {
        "bad",
        "sad",
        "tired",
        "angry",
        "hate",
        "stress",
        "worried",
        "sorry",
        "hard",
        "difficult",
    }
    tokens = tokenise(text)
    if not tokens:
        return 0.0
    pos = sum(1 for token in tokens if token in positive_words)
    neg = sum(1 for token in tokens if token in negative_words)
    return (pos - neg) / max(1, len(tokens))


def summarize_texts(texts: List[str], max_sentences: int = 3) -> str:
    sentences = []
    for text in texts:
        sentences.extend(split_sentences(text))
    if not sentences:
        return ""
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    ranked = sorted(sentences, key=lambda sentence: (len(sentence), sentence), reverse=True)
    selected = ranked[:max_sentences]
    return " ".join(selected)


def representative_lines(texts: List[str], max_items: int = 5) -> List[str]:
    cleaned = [normalize_whitespace(text) for text in texts if normalize_whitespace(text)]
    if len(cleaned) <= max_items:
        return cleaned
    lengths = sorted(cleaned, key=len, reverse=True)
    return lengths[:max_items]
