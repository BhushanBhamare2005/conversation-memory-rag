from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

from app.models import MessageRecord
from app.utils.text import normalize_whitespace, SPEAKER_PATTERN


def _detect_header(csv_path: Path) -> bool:
    first_line = ""
    with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                first_line = stripped.lower()
                break
    return first_line.startswith("date") or first_line.startswith("timestamp") or first_line.startswith("conversation")


def load_conversation_frame(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    has_header = _detect_header(path)
    frame = pd.read_csv(
        path,
        header=0 if has_header else None,
        dtype=str,
        keep_default_na=False,
        engine="python",
    )
    if not has_header:
        if frame.shape[1] == 1:
            frame.columns = ["conversation"]
        elif frame.shape[1] >= 2:
            columns = ["date", "conversation"] + [f"extra_{index}" for index in range(frame.shape[1] - 2)]
            frame.columns = columns[: frame.shape[1]]
    return frame


def _parse_timestamp(raw_date: str | None, row_index: int) -> datetime:
    if raw_date:
        parsed = pd.to_datetime(raw_date, errors="coerce")
        if pd.notna(parsed):
            return parsed.to_pydatetime()
    return datetime(2024, 1, 1) + timedelta(days=row_index)


def split_conversation_turns(conversation: str) -> List[tuple[str, str]]:
    lines = [normalize_whitespace(line) for line in (conversation or "").splitlines()]
    turns: List[tuple[str, str]] = []
    current_speaker: Optional[str] = None
    current_text: List[str] = []
    fallback_speaker = 1

    for line in lines:
        if not line:
            continue
        match = SPEAKER_PATTERN.match(line)
        if match:
            if current_speaker and current_text:
                turns.append((current_speaker, " ".join(current_text).strip()))
            current_speaker = match.group(1).strip()
            current_text = [match.group(2).strip()]
            continue
        if current_speaker is None:
            current_speaker = f"User {fallback_speaker}"
            fallback_speaker = 2 if fallback_speaker == 1 else 1
        current_text.append(line)

    if current_speaker and current_text:
        turns.append((current_speaker, " ".join(current_text).strip()))

    if not turns and conversation.strip():
        fallback_lines = [line.strip() for line in conversation.split("  ") if line.strip()]
        for index, text in enumerate(fallback_lines):
            turns.append((f"User {(index % 2) + 1}", text))

    return turns


def load_messages(csv_path: str | Path) -> List[MessageRecord]:
    frame = load_conversation_frame(csv_path)
    messages: List[MessageRecord] = []
    for row_index, row in frame.iterrows():
        raw_date = row.get("date") if "date" in frame.columns else None
        conversation = row.get("conversation") if "conversation" in frame.columns else row.iloc[0]
        timestamp = _parse_timestamp(raw_date, int(row_index))
        turns = split_conversation_turns(str(conversation))
        for turn_index, (speaker, text) in enumerate(turns):
            messages.append(
                MessageRecord(
                    message_id=f"m_{row_index}_{turn_index}",
                    timestamp=timestamp + timedelta(seconds=turn_index),
                    speaker=speaker,
                    text=normalize_whitespace(text),
                    conversation_id=f"c_{row_index}",
                    source_row=int(row_index),
                    turn_index=turn_index,
                )
            )
    messages.sort(key=lambda item: (item.timestamp, item.source_row, item.turn_index))
    return messages


def chunk_messages(messages: List[MessageRecord], chunk_size: int = 25) -> List[List[MessageRecord]]:
    return [messages[index : index + chunk_size] for index in range(0, len(messages), chunk_size)]


def messages_to_dict(messages: List[MessageRecord]) -> List[dict]:
    return [message.to_dict() for message in messages]
