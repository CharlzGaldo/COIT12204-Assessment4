"""
Base class shared by all agents.

`handle()` is the single entry point the orchestrator calls. It wraps the
agent-specific `process()` with the behaviour every agent must have:

    - validate the incoming message (envelope + content)
    - check it is actually addressed to this agent and has an intent the
      agent understands
    - log "received" and "sent" (and "error" if anything goes wrong)
    - validate the outgoing message before handing it back
"""
from typing import Any, Optional

from app.comms_log import CommLogger, comms as default_comms
from app.messages import Constraints, InvalidMessageError, Message, validate_message


class BaseAgent:
    name: str = "base"
    # Intents this agent knows how to handle.
    accepts: frozenset[str] = frozenset()

    def __init__(self, comm_logger: Optional[CommLogger] = None) -> None:
        self.comms = comm_logger or default_comms

    def attach_logger(self, comm_logger: CommLogger) -> None:
        """Called by the orchestrator so all components share one logger."""
        self.comms = comm_logger

    # ------------------------------------------------------------------
    def handle(self, message: Message) -> Message:
        try:
            message = validate_message(message)
            if message.receiver != self.name:
                raise InvalidMessageError(
                    f"message addressed to {message.receiver!r} delivered to {self.name!r}"
                )
            if message.intent not in self.accepts:
                raise InvalidMessageError(
                    f"{self.name} cannot handle intent {message.intent!r}"
                )
            self.comms.log("received", self.name, message.trace_id, message=message)

            reply = self.process(message)

            reply = validate_message(reply)
            self.comms.log("sent", self.name, reply.trace_id, message=reply)
            return reply
        except Exception as exc:
            self.comms.log(
                "error",
                self.name,
                getattr(message, "trace_id", "unknown"),
                step=getattr(message, "step", None),
                level="ERROR",
                detail=f"{type(exc).__name__}: {exc}",
            )
            raise

    def process(self, message: Message) -> Message:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------------
    def reply(
        self,
        incoming: Message,
        receiver: str,
        intent: str,
        content: dict[str, Any],
        constraints: Optional[Constraints] = None,
    ) -> Message:
        """Build an outgoing message in the same conversation as `incoming`."""
        return Message(
            trace_id=incoming.trace_id,
            step=incoming.step + 1,
            sender=self.name,
            receiver=receiver,
            intent=intent,
            content=content,
            constraints=constraints or incoming.constraints,
        )
