"""
Internal state management for the priority-classification agent.

Tracks:
    - last_suggestion: the most recent classification result
    - error_count: how many agent runs have failed
    - last_tool_call: name/result of the last tool invocation
    - history: an ordered log of every action the agent has taken

State updates are logged so the required "logging of state
transitions" can be demonstrated (see README / demo video notes).
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# Named logger for state-related log lines specifically, separate from
# agent.core and agent.api so log output is easy to filter/read.
logger = logging.getLogger("agent.state")


@dataclass
class ActionRecord:
    """One entry in the agent's history — a single logged action."""
    timestamp: str
    action: str          # e.g. "tool_call", "success", "error"
    detail: Any = None    # whatever extra info is relevant to that action


@dataclass
class AgentState:
    """
    Holds everything the agent "remembers" about its own recent activity.

    This is intentionally a plain in-memory dataclass rather than a
    database-backed model — persistence (SQLite/JSON/Redis) is listed as
    a bonus extension in the assessment brief, not a required feature.
    """

    # The 4 tracked state variables required by the assessment brief:
    last_suggestion: Optional[dict] = None
    error_count: int = 0
    last_tool_call: Optional[dict] = None
    history: list[ActionRecord] = field(default_factory=list)

    def _log_transition(self, action: str, detail: Any = None) -> None:
        """
        Internal helper: append an ActionRecord to history AND emit a log
        line. Every public "record_*" method below routes through here so
        there's exactly one place that defines what a "state transition"
        looks like.
        """
        record = ActionRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=action,
            detail=detail,
        )
        self.history.append(record)
        logger.info("state transition: %s | detail=%s", action, detail)

    def record_tool_call(self, tool_name: str, tool_result: Any) -> None:
        """Call this right after invoking a LangChain tool."""
        self.last_tool_call = {"tool": tool_name, "result": tool_result}
        self._log_transition("tool_call", self.last_tool_call)

    def record_success(self, suggestion: dict) -> None:
        """Call this when the agent produces a valid classification."""
        self.last_suggestion = suggestion
        self._log_transition("success", suggestion)

    def record_error(self, error_message: str) -> None:
        """Call this whenever the agent run fails and falls back."""
        self.error_count += 1
        self._log_transition("error", {"message": error_message, "error_count": self.error_count})

    def to_dict(self) -> dict:
        """
        Serialise state to plain dicts/lists so it can be returned directly
        as a JSON response from the /api/agent/state endpoint.
        """
        return {
            "last_suggestion": self.last_suggestion,
            "error_count": self.error_count,
            "last_tool_call": self.last_tool_call,
            "history": [
                {"timestamp": h.timestamp, "action": h.action, "detail": h.detail}
                for h in self.history
            ],
        }


# Single shared instance used by the FastAPI app for the whole process
# lifetime. All requests hitting the same running server share this state,
# which is what lets /api/agent/state show a running history across calls.
# (Swap for a persistence-backed store as a bonus extension.)
agent_state = AgentState()
