"""
Agent-to-agent message protocol.

Every piece of communication between the orchestrator and the agents is a
`Message`. A message has an envelope (who, to whom, why, constraints, where
in the conversation) and a `content` payload whose shape depends on the
message `intent`.

Validation happens in three layers:
    1. Envelope  - Pydantic `Message` model (types, required fields, no extras)
    2. Content   - the payload must match the schema registered for the intent
    3. Route     - only certain senders may send a given intent to certain
                   receivers (see ROUTES), so e.g. the classifier cannot
                   declare a final result by itself.

Message flow in this system:

    orchestrator --classify_task-------------> classifier
    classifier   --classification_result-----> validator
    validator    --revise_classification----> classifier     (rejected)
    validator    --final_result--------------> orchestrator   (approved)
"""
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

Priority = Literal["High", "Medium", "Low"]

# Names used on the wire. "orchestrator" is a real participant: agents send
# their final answers to it, and it is the only thing that creates the first
# message of a conversation.
ORCHESTRATOR = "orchestrator"
CLASSIFIER = "classifier"
VALIDATOR = "validator"

# Intent names (kept as plain strings so malformed/unknown intents can be
# represented and then rejected by validation instead of failing at parse time).
CLASSIFY_TASK = "classify_task"
CLASSIFICATION_RESULT = "classification_result"
REVISE_CLASSIFICATION = "revise_classification"
FINAL_RESULT = "final_result"


class InvalidMessageError(ValueError):
    """Raised when a message fails envelope, content or route validation."""


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------
class Constraints(BaseModel):
    """Instructions the sender attaches about how the receiver must respond."""

    model_config = ConfigDict(extra="forbid")

    # The receiver must answer with a JSON-serialisable payload.
    format: Literal["json"] = "json"
    # How long the receiver has to handle this message (seconds).
    timeout_seconds: Optional[float] = Field(default=None, gt=0)
    # Maximum number of classification attempts allowed for this conversation.
    max_attempts: Optional[int] = Field(default=None, ge=1)


class Message(BaseModel):
    """The envelope shared by every message in the system."""

    # extra="forbid": a typo like "reciever" is an error, not silently ignored.
    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    trace_id: str = Field(min_length=1)  # one id per conversation
    step: int = Field(default=0, ge=0)  # position in the conversation
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sender: str = Field(min_length=1)
    receiver: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    content: dict[str, Any]
    constraints: Constraints = Field(default_factory=Constraints)

    def summary(self) -> str:
        """Short, human-readable description used in console log lines."""
        c = self.content
        parts = []
        if "priority" in c:
            parts.append(f"priority={c['priority']}")
        if "previous_priority" in c:
            parts.append(f"previous={c['previous_priority']}")
        if "attempt" in c:
            parts.append(f"attempt={c['attempt']}")
        if "attempts" in c:
            parts.append(f"attempts={c['attempts']}")
        if c.get("issues"):
            parts.append(f"issues={c['issues']}")
        if "task_description" in c and not parts:
            text = str(c["task_description"])
            parts.append(f"task={text[:50]!r}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Content schemas (one per intent)
# ---------------------------------------------------------------------------
class _Content(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassifyTaskContent(_Content):
    task_description: str = Field(min_length=1, max_length=2000)
    attempt: int = Field(default=1, ge=1)


class ClassificationResultContent(_Content):
    task_description: str = Field(min_length=1, max_length=2000)
    priority: Priority
    # Deliberately NOT min_length-constrained here: an empty/weak reasoning
    # string is a quality problem for the *validator agent* to catch and send
    # back for revision, not a malformed message.
    reasoning: str
    keyword_scan: str
    attempt: int = Field(ge=1)


class ReviseClassificationContent(_Content):
    task_description: str = Field(min_length=1, max_length=2000)
    previous_priority: Priority
    previous_reasoning: str
    issues: list[str] = Field(min_length=1)  # a rejection must say why
    feedback: str = Field(min_length=1)
    attempt: int = Field(ge=2)  # revisions are always attempt 2 or later


class FinalResultContent(_Content):
    priority: Priority
    reasoning: str = Field(min_length=1)
    validated: bool
    issues: list[str] = Field(default_factory=list)
    attempts: int = Field(ge=1)


CONTENT_MODELS: dict[str, type[BaseModel]] = {
    CLASSIFY_TASK: ClassifyTaskContent,
    CLASSIFICATION_RESULT: ClassificationResultContent,
    REVISE_CLASSIFICATION: ReviseClassificationContent,
    FINAL_RESULT: FinalResultContent,
}

# intent -> (allowed senders, allowed receivers)
ROUTES: dict[str, tuple[set[str], set[str]]] = {
    CLASSIFY_TASK: ({ORCHESTRATOR}, {CLASSIFIER}),
    CLASSIFICATION_RESULT: ({CLASSIFIER}, {VALIDATOR}),
    REVISE_CLASSIFICATION: ({VALIDATOR}, {CLASSIFIER}),
    FINAL_RESULT: ({VALIDATOR}, {ORCHESTRATOR}),
}

# Intents that start a new classification attempt (used for iteration limits).
ATTEMPT_INTENTS = {CLASSIFY_TASK, REVISE_CLASSIFICATION}


def _format_errors(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
        for err in exc.errors()
    )


def validate_message(message: Any) -> Message:
    """Validate envelope + content. Returns a `Message` or raises InvalidMessageError.

    Accepts either a Message or a plain dict (e.g. something an agent
    returned that was never built through the Message model).
    """
    if isinstance(message, dict):
        try:
            message = Message.model_validate(message)
        except ValidationError as exc:
            raise InvalidMessageError(f"invalid envelope: {_format_errors(exc)}") from exc
    elif not isinstance(message, Message):
        raise InvalidMessageError(
            f"expected a Message, got {type(message).__name__}"
        )

    model = CONTENT_MODELS.get(message.intent)
    if model is None:
        raise InvalidMessageError(f"unknown intent {message.intent!r}")

    try:
        model.model_validate(message.content)
    except ValidationError as exc:
        raise InvalidMessageError(
            f"invalid content for intent {message.intent!r}: {_format_errors(exc)}"
        ) from exc
    return message


def check_route(message: Message) -> None:
    """Raise InvalidMessageError if this sender may not send this intent here."""
    senders, receivers = ROUTES[message.intent]
    if message.sender not in senders:
        raise InvalidMessageError(
            f"{message.sender!r} may not send {message.intent!r} "
            f"(allowed senders: {sorted(senders)})"
        )
    if message.receiver not in receivers:
        raise InvalidMessageError(
            f"{message.intent!r} may not be sent to {message.receiver!r} "
            f"(allowed receivers: {sorted(receivers)})"
        )
