from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
SAMPLE_DATA_PATH = DATA_DIR / "sample" / "conversations.csv"


@dataclass(frozen=True)
class TopicConfig:
    model_name: str = "all-MiniLM-L6-v2"
    window_size: int = 10
    drift_threshold: float = 0.18
    min_topic_length: int = 4
    max_keywords: int = 8


@dataclass(frozen=True)
class MemoryConfig:
    checkpoint_interval: int = 100
    raw_chunk_size: int = 25
    retrieval_top_k: int = 5
    embedding_batch_size: int = 32


@dataclass(frozen=True)
class AppSettings:
    topic: TopicConfig = TopicConfig()
    memory: MemoryConfig = MemoryConfig()


SETTINGS = AppSettings()
