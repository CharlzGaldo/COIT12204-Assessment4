"""
Communication logger.

Every message event in the system is recorded in three places:
    1. the console (python logging, logger "agent.comms") - readable lines
    2. a JSON-lines audit file (default logs/agent_messages.jsonl) - full detail
    3. an in-memory ring buffer, exposed through the API for the demo/report

Events:
    started    orchestrator began a conversation
    routed     orchestrator delivered a message to an agent
    received   an agent received a message
    sent       an agent produced a message
    error      something failed (timeout, invalid message, crash, ...)
    fallback   the safe default answer was used
    completed  the conversation finished
"""
import json
import logging
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.config import settings
from app.messages import Message

logger = logging.getLogger("agent.comms")


class CommLogger:
    def __init__(self, path: Optional[str] = None, keep: int = 1000) -> None:
        self.path = Path(path) if path else None
        self._records: deque[dict[str, Any]] = deque(maxlen=keep)
        self._lock = threading.Lock()
        self._file_ok = self.path is not None
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                # Read-only filesystem etc. Logging must never crash the app.
                self._file_ok = False

    def log(
        self,
        event: str,
        agent: str,
        trace_id: str,
        message: Optional[Message] = None,
        step: Optional[int] = None,
        level: str = "INFO",
        detail: Optional[str] = None,
        **extra: Any,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "level": level,
            "agent": agent,  # who wrote this log line
            "trace_id": trace_id,
            "step": message.step if message is not None else step,
        }
        if message is not None:
            record.update(
                message_id=message.message_id,
                sender=message.sender,
                receiver=message.receiver,
                intent=message.intent,
                content=message.content,
                constraints=message.constraints.model_dump(exclude_none=True),
            )
        if detail:
            record["detail"] = detail
        record.update(extra)

        with self._lock:
            self._records.append(record)
            if self._file_ok:
                try:
                    with self.path.open("a", encoding="utf-8") as fh:  # type: ignore[union-attr]
                        fh.write(json.dumps(record, default=str) + "\n")
                except OSError:
                    self._file_ok = False

        logger.log(
            logging.getLevelName(level),
            "[trace %s step %s] %-12s %-9s %s",
            trace_id,
            record["step"] if record["step"] is not None else "-",
            agent,
            event.upper(),
            self._describe(message, detail),
        )
        return record

    @staticmethod
    def _describe(message: Optional[Message], detail: Optional[str]) -> str:
        bits = []
        if message is not None:
            bits.append(f"{message.sender}->{message.receiver} {message.intent}")
            summary = message.summary()
            if summary:
                bits.append(f"| {summary}")
        if detail:
            bits.append(f"| {detail}")
        return " ".join(bits)

    # ---- read access (used by API endpoints and tests) -----------------
    def records(self, trace_id: Optional[str] = None, limit: Optional[int] = None) -> list[dict]:
        with self._lock:
            items = [r for r in self._records if trace_id is None or r["trace_id"] == trace_id]
        return items[-limit:] if limit else items

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


# Shared default instance used by the FastAPI app.
comms = CommLogger(settings.message_log_path or None)
