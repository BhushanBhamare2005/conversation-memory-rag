from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.chatbot.engine import ConversationalMemoryChatbot
from app.config import SAMPLE_DATA_PATH, SETTINGS
from app.core.pipeline import MemoryPipeline

router = APIRouter()
pipeline = MemoryPipeline()
STATE: Dict[str, Any] = {}


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = Field(default=5, ge=1, le=20)


@router.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.post("/build")
def build_from_sample() -> Dict[str, Any]:
    bundle = pipeline.build(SAMPLE_DATA_PATH, output_dir=Path("artifacts"))
    STATE.update(bundle)
    chatbot = ConversationalMemoryChatbot(bundle["retriever"])
    STATE["chatbot"] = chatbot
    return {
        "status": "built",
        "messages": len(bundle["messages"]),
        "topics": len(bundle["topics"]),
        "topic_checkpoints": len(bundle["topic_checkpoints"]),
        "memory_checkpoints": len(bundle["memory_checkpoints"]),
        "persona_traits": sum(len(values) for values in bundle["persona"].values()),
        "global_profile_confidence": bundle.get("global_profile", {}).get("confidence"),
    }


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    destination = Path("artifacts") / file.filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(await file.read())
    bundle = pipeline.build(destination, output_dir=Path("artifacts"))
    STATE.update(bundle)
    STATE["chatbot"] = ConversationalMemoryChatbot(bundle["retriever"])
    return {"status": "uploaded", "file": file.filename}


@router.post("/chat")
def chat(request: ChatRequest) -> Dict[str, Any]:
    chatbot = STATE.get("chatbot")
    if chatbot is None:
        bundle = pipeline.build(SAMPLE_DATA_PATH, output_dir=Path("artifacts"))
        STATE.update(bundle)
        chatbot = ConversationalMemoryChatbot(bundle["retriever"])
        STATE["chatbot"] = chatbot
    return chatbot.ask(request.query, top_k=request.top_k)


@router.get("/stats")
def stats() -> Dict[str, Any]:
    if not STATE:
        return {"status": "empty"}
    return {
        "messages": len(STATE.get("messages", [])),
        "topics": len(STATE.get("topics", [])),
        "topic_checkpoints": len(STATE.get("topic_checkpoints", [])),
        "memory_checkpoints": len(STATE.get("memory_checkpoints", [])),
        "persona": STATE.get("persona", {}),
        "global_profile": STATE.get("global_profile", {}),
        "memory_layers": STATE.get("memory_layers", {}),
    }
