"""
Tests for the agent wrapper (classify_priority), the check_urgency_keywords
tool, and the fallback logic that kicks in when the LLM call fails.

No real Gemini calls happen anywhere in this file — the LLM is always
either a FakeLLM/FailingLLM object we control, or monkeypatched. This is
what the assessment brief calls "Mocked LLM calls" and "Fallback logic".
"""
import pytest
from langchain_core.runnables import RunnableLambda

from app.agent import PriorityClassification, classify_priority
from app.state import AgentState
from app.tools import check_urgency_keywords


class FakeLLM:
    """
    Stands in for ChatGoogleGenerativeAI in tests. No network calls.

    Only implements the one method classify_priority() actually calls:
    with_structured_output(). Everything downstream of that (the chain,
    .invoke()) still runs for real — we're only faking the LLM boundary,
    not the whole pipeline.
    """

    def __init__(self, response: PriorityClassification):
        self._response = response

    def with_structured_output(self, schema):
        # RunnableLambda wraps a plain function as a full LangChain
        # Runnable, which is required for `PROMPT | structured_model` (the
        # `|` chain operator) to work the same way it would with a real
        # chat model. It just ignores whatever input it's given and always
        # returns our pre-set fake response.
        return RunnableLambda(lambda _inputs: self._response)


class FailingLLM:
    """Simulates an LLM call that raises (network error, bad API key, etc.)."""

    def with_structured_output(self, schema):
        raise RuntimeError("simulated network failure")


# ---- Tool behaviour ----------------------------------------------------
# These two tests check check_urgency_keywords in complete isolation from
# the rest of the agent — no LLM, no state, just the keyword-scan logic.

def test_check_urgency_keywords_detects_keyword():
    result = check_urgency_keywords.invoke({"task_description": "This is URGENT, please help"})
    assert "urgent" in result.lower()


def test_check_urgency_keywords_no_match():
    result = check_urgency_keywords.invoke({"task_description": "Update the README sometime"})
    assert "no urgency keywords" in result.lower()


# ---- Agent wrapper: success path ---------------------------------------

def test_classify_priority_success_updates_state():
    """
    Happy path: tool runs, fake LLM returns a valid classification, and
    the AgentState we pass in gets updated with both the tool call and
    the successful suggestion.
    """
    state = AgentState()
    fake_llm = FakeLLM(PriorityClassification(priority="High", reasoning="Task is explicitly marked urgent."))

    result = classify_priority("This is urgent, asap!", llm=fake_llm, state=state)

    assert result["priority"] == "High"
    assert result["fallback"] is False
    assert state.error_count == 0
    assert state.last_tool_call["tool"] == "check_urgency_keywords"
    assert state.last_suggestion["priority"] == "High"


def test_classify_priority_uses_injected_get_llm(monkeypatch):
    """
    Instead of passing llm= directly, this test monkeypatches
    app.agent.get_llm — the function classify_priority() calls internally
    when no llm is explicitly passed. This exercises the "real" code path
    (get_llm()) without needing a real API key.
    """
    fake_llm = FakeLLM(PriorityClassification(priority="Low", reasoning="The task says there is no rush at all."))
    monkeypatch.setattr("app.agent.get_llm", lambda: fake_llm)  # ClassifierAgent calls core.get_llm()

    result = classify_priority("No rush on this one")

    assert result["priority"] == "Low"
    assert result["fallback"] is False


# ---- Agent wrapper: fallback / error handling --------------------------

def test_classify_priority_falls_back_on_llm_failure():
    """
    When the LLM raises, classify_priority() must NOT propagate the
    exception — it should catch it, log it, record it in state, and return
    the safe FALLBACK_RESULT with fallback=True instead.
    """
    state = AgentState()
    result = classify_priority("urgent task", llm=FailingLLM(), state=state)

    assert result["fallback"] is True
    assert result["priority"] == "Medium"  # fallback default
    assert result["fallback_reason"].startswith("agent_failure")
    assert state.error_count == 1  # one recorded error per failed conversation
    assert any(h.action == "error" for h in state.history)


def test_classify_priority_rejects_empty_description():
    """Blank/whitespace-only input should raise ValueError before any LLM call happens."""
    state = AgentState()
    with pytest.raises(ValueError):
        classify_priority("   ", llm=FakeLLM(PriorityClassification(priority="Low", reasoning="irrelevant")), state=state)
