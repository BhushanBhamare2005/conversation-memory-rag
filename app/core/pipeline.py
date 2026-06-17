from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from app.analysis.topic_segmentation import TopicSegmenter
from app.analysis.style import build_style_report
from app.config import SETTINGS
from app.ingestion.conversation_parser import chunk_messages, load_messages
from app.memory.checkpoints import build_memory_checkpoints, build_topic_checkpoints
from app.memory.profile import build_global_user_profile
from app.memory.persona import extract_persona
from app.models import GlobalUserProfile, MessageRecord, PersonaProfile, StyleReport
from app.retrieval.faiss_store import FaissMemoryStore
from app.retrieval.retriever import MemoryRetriever
from app.utils.text import normalize_whitespace


class MemoryPipeline:
    def __init__(self) -> None:
        self.segmenter = TopicSegmenter()

    def build(self, csv_path: str | Path, output_dir: str | Path | None = None) -> Dict[str, object]:
        messages = load_messages(csv_path)
        topics = self.segmenter.build_topics(messages)
        topic_checkpoints = build_topic_checkpoints(topics, messages)
        memory_checkpoints = build_memory_checkpoints(messages)
        persona = extract_persona(messages)
        style_report = build_style_report([message.text for message in messages], [message.timestamp for message in messages])
        global_profile = build_global_user_profile(persona, topic_checkpoints, style_report, [message.message_id for message in messages])

        raw_store = self._build_raw_store(messages)
        topic_store = self._build_topic_store(topic_checkpoints)
        checkpoint_store = self._build_checkpoint_store(memory_checkpoints)
        persona_store = self._build_persona_store(persona)
        profile_store = self._build_profile_store(global_profile)
        retriever = MemoryRetriever(topic_store, raw_store, checkpoint_store, persona_store, profile_store)

        bundle = {
            "messages": [message.to_dict() for message in messages],
            "topics": topics,
            "topic_checkpoints": [checkpoint.to_dict() for checkpoint in topic_checkpoints],
            "memory_checkpoints": [checkpoint.to_dict() for checkpoint in memory_checkpoints],
            "persona": persona.to_dict(),
            "global_profile": global_profile.to_dict(),
            "memory_layers": self._memory_layers(messages, topics, topic_checkpoints, memory_checkpoints, persona, global_profile),
            "retriever": retriever,
            "stores": {
                "topic_store": topic_store,
                "raw_store": raw_store,
                "checkpoint_store": checkpoint_store,
                "persona_store": persona_store,
                "profile_store": profile_store,
            },
        }

        if output_dir is not None:
            self.save_bundle(bundle, output_dir)
        return bundle

    def save_bundle(self, bundle: Dict[str, object], output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "messages": bundle["messages"],
            "topics": bundle["topics"],
            "topic_checkpoints": bundle["topic_checkpoints"],
            "memory_checkpoints": bundle["memory_checkpoints"],
            "persona": bundle["persona"],
            "global_profile": bundle.get("global_profile"),
            "memory_layers": bundle.get("memory_layers"),
        }
        (path / "memory_bundle.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        bundle["stores"]["topic_store"].save(path / "topic_store")
        bundle["stores"]["raw_store"].save(path / "raw_store")
        bundle["stores"]["checkpoint_store"].save(path / "checkpoint_store")
        bundle["stores"]["persona_store"].save(path / "persona_store")
        bundle["stores"]["profile_store"].save(path / "profile_store")

    def _build_raw_store(self, messages: List[MessageRecord]) -> FaissMemoryStore:
        store = FaissMemoryStore()
        documents = []
        for chunk_index, chunk in enumerate(chunk_messages(messages, chunk_size=SETTINGS.memory.raw_chunk_size), start=1):
            text = "\n".join(f"{message.speaker}: {message.text}" for message in chunk)
            documents.append(
                {
                    "doc_id": f"raw_{chunk_index}",
                    "doc_type": "raw_chunk",
                    "content": text,
                    "metadata": {
                        "message_ids": [message.message_id for message in chunk],
                        "range": f"{chunk[0].source_row + 1}-{chunk[-1].source_row + 1}",
                    },
                }
            )
        store.add_documents(documents)
        return store

    def _build_topic_store(self, topic_checkpoints: List[object]) -> FaissMemoryStore:
        store = FaissMemoryStore()
        documents = []
        for checkpoint in topic_checkpoints:
            documents.append(
                {
                    "doc_id": checkpoint.topic_id,
                    "doc_type": "topic_summary",
                    "content": f"{checkpoint.summary} Keywords: {', '.join(checkpoint.keywords)}",
                    "metadata": checkpoint.to_dict(),
                }
            )
        store.add_documents(documents)
        return store

    def _build_checkpoint_store(self, memory_checkpoints: List[object]) -> FaissMemoryStore:
        store = FaissMemoryStore()
        documents = []
        for checkpoint in memory_checkpoints:
            documents.append(
                {
                    "doc_id": checkpoint.checkpoint_id,
                    "doc_type": "checkpoint_summary",
                    "content": f"{checkpoint.summary} Important facts: {'; '.join(checkpoint.important_facts)}",
                    "metadata": checkpoint.to_dict(),
                }
            )
        store.add_documents(documents)
        return store

    def _build_persona_store(self, persona: PersonaProfile) -> FaissMemoryStore:
        store = FaissMemoryStore()
        documents = []
        for category_name, items in persona.to_dict().items():
            for index, item in enumerate(items, start=1):
                documents.append(
                    {
                        "doc_id": f"persona_{category_name}_{index}",
                        "doc_type": "persona_fact",
                        "content": f"{category_name}: {item['value']} | evidence: {'; '.join(item['evidence'])}",
                        "metadata": item,
                    }
                )
        store.add_documents(documents)
        return store

    def _build_profile_store(self, global_profile: GlobalUserProfile) -> FaissMemoryStore:
        store = FaissMemoryStore()
        store.add_documents(
            [
                {
                    "doc_id": "global_profile_1",
                    "doc_type": "global_profile",
                    "content": global_profile.summary,
                    "metadata": global_profile.to_dict(),
                }
            ]
        )
        return store

    def _memory_layers(
        self,
        messages: List[MessageRecord],
        topics: List[dict],
        topic_checkpoints: List[object],
        memory_checkpoints: List[object],
        persona: PersonaProfile,
        global_profile: GlobalUserProfile,
    ) -> Dict[str, Dict[str, int]]:
        return {
            "raw_message_chunks": {"documents": max(1, len(messages) // SETTINGS.memory.raw_chunk_size + (1 if len(messages) % SETTINGS.memory.raw_chunk_size else 0))},
            "topic_summaries": {"documents": len(topics)},
            "checkpoint_summaries": {"documents": len(memory_checkpoints)},
            "persona_memory": {"documents": sum(len(values) for values in persona.to_dict().values())},
            "global_user_profile": {"documents": 1, "confidence": round(global_profile.confidence, 4)},
        }
