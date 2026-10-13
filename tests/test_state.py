"""
Tests for AgentState: the plain in-memory object that tracks the agent's
last suggestion, error count, last tool call, and full action history.

These tests don't touch LangChain, Gemini, or FastAPI at all — they only
exercise the dataclass logic in app/state.py in isolation.
"""
from app.state import AgentState


def test_initial_state_is_empty():
    """A freshly-created AgentState should have no history yet."""
    state = AgentState()
    assert state.last_suggestion is None
    assert state.error_count == 0
    assert state.last_tool_call is None
    assert state.history == []


def test_record_tool_call_updates_state_and_logs_transition():
    """record_tool_call() should set last_tool_call AND append to history."""
    state = AgentState()
    state.record_tool_call("check_urgency_keywords", "No urgency keywords detected.")

    assert state.last_tool_call == {
        "tool": "check_urgency_keywords",
        "result": "No urgency keywords detected.",
    }
    # Exactly one history entry should have been added, tagged "tool_call".
    assert len(state.history) == 1
    assert state.history[0].action == "tool_call"


def test_record_success_updates_last_suggestion():
    """record_success() should set last_suggestion and log a "success" event."""
    state = AgentState()
    suggestion = {"priority": "High", "reasoning": "test"}
    state.record_success(suggestion)

    assert state.last_suggestion == suggestion
    assert any(h.action == "success" for h in state.history)


def test_record_error_increments_error_count():
    """error_count should increment on every call, and each call logged separately."""
    state = AgentState()
    state.record_error("boom")
    state.record_error("boom again")

    assert state.error_count == 2
    error_events = [h for h in state.history if h.action == "error"]
    assert len(error_events) == 2


def test_to_dict_serialises_history():
    """to_dict() must return plain dicts/lists so FastAPI can JSON-encode it directly."""
    state = AgentState()
    state.record_tool_call("tool", "result")
    as_dict = state.to_dict()

    assert as_dict["last_tool_call"] == {"tool": "tool", "result": "result"}
    assert isinstance(as_dict["history"], list)
    assert as_dict["history"][0]["action"] == "tool_call"
