"""
Orchestrator behaviour: routing, step tracking, iteration limits, invalid
receivers, timeouts, retries, bad replies, and logging. The LLM is always a
ScriptedLLM; agents and orchestrator run for real.
"""
import time

import pytest

from app.agents.base import BaseAgent
from app.agents.classifier import ClassifierAgent
from app.agents.validator import ValidatorAgent
from app.fakes import (
    Delay, MalformedClassifier, MisroutingClassifier, ScriptedLLM, SpoofingClassifier, classification,
)
from app.messages import CLASSIFICATION_RESULT, CLASSIFY_TASK, Message
from app.orchestrator import Orchestrator, OrchestratorConfig

URGENT = "URGENT: payment gateway is blocking checkout, fix asap"
GOOD = "The payment gateway outage blocks all customer purchases."
BAD_LOW = classification("Low", GOOD)
GOOD_HIGH = classification("High", GOOD)


# ---- happy path / agent-to-agent flow ------------------------------------
def test_happy_path_routes_classifier_then_validator(make_orchestrator):
    result = make_orchestrator(ScriptedLLM([GOOD_HIGH])).run(URGENT)
    assert result.status == "approved" and result.validated
    assert result.priority == "High"
    assert result.iterations == 1 and result.steps == 3
    assert result.path == [
        "orchestrator->classifier:classify_task",
        "classifier->validator:classification_result",
        "validator->orchestrator:final_result",
    ]


def test_rejection_triggers_revision_loop_with_feedback(make_orchestrator):
    llm = ScriptedLLM([BAD_LOW, GOOD_HIGH])
    result = make_orchestrator(llm).run(URGENT)
    assert result.status == "approved" and result.priority == "High"
    assert result.iterations == 2 and result.steps == 5
    assert "validator->classifier:revise_classification" in result.path
    # the validator's feedback actually reached the LLM prompt on attempt 2
    assert "ignores_urgency_keywords" in llm.prompts[1]
    assert "None (first attempt)" in llm.prompts[0]


def test_state_updated_on_success(make_orchestrator, state):
    make_orchestrator(ScriptedLLM([GOOD_HIGH])).run(URGENT)
    assert state.last_suggestion["priority"] == "High"
    assert state.error_count == 0 and state.last_tool_call["tool"] == "check_urgency_keywords"


# ---- iteration / step limits ------------------------------------------------
def test_max_iterations_enforced_then_fallback(make_orchestrator, state):
    llm = ScriptedLLM([BAD_LOW])  # never improves
    result = make_orchestrator(llm, max_iterations=2).run(URGENT)
    assert result.status == "fallback" and result.priority == "Medium"
    assert result.fallback_reason.startswith("max_iterations")
    assert result.iterations == 2 and llm.calls == 2  # no third LLM call
    assert result.last_issues  # says why the validator kept rejecting
    assert state.error_count == 1


def test_max_steps_enforced(make_orchestrator):
    result = make_orchestrator(ScriptedLLM([BAD_LOW]), max_iterations=10, max_steps=4).run(URGENT)
    assert result.status == "fallback" and result.fallback_reason.startswith("max_steps")
    assert result.steps == 4


# ---- timeouts / retries / failures ------------------------------------------------
def test_slow_agent_times_out_then_falls_back(make_orchestrator):
    llm = ScriptedLLM([Delay(1.0, GOOD_HIGH)])
    start = time.monotonic()
    result = make_orchestrator(llm, agent_timeout=0.1, max_agent_retries=1).run(URGENT)
    assert result.status == "fallback" and result.fallback_reason.startswith("agent_failure")
    assert "AgentTimeout" in result.fallback_reason
    assert time.monotonic() - start < 0.9


def test_transient_failure_is_retried_and_recovers(make_orchestrator, comm_logger):
    llm = ScriptedLLM([RuntimeError("503 unavailable"), GOOD_HIGH])
    result = make_orchestrator(llm, max_agent_retries=1).run(URGENT)
    assert result.status == "approved" and llm.calls == 2
    errors = [r for r in comm_logger.records() if r["event"] == "error"]
    assert errors and "503 unavailable" in errors[0]["detail"]


def test_persistent_llm_failure_falls_back(make_orchestrator, state):
    result = make_orchestrator(ScriptedLLM([RuntimeError("down")]), max_agent_retries=2).run(URGENT)
    assert result.status == "fallback" and "RuntimeError" in result.fallback_reason
    assert state.error_count == 1


def test_run_timeout_enforced(make_orchestrator):
    llm = ScriptedLLM([Delay(0.3, BAD_LOW)])
    result = make_orchestrator(llm, run_timeout=0.5, agent_timeout=5, max_iterations=10).run(URGENT)
    assert result.status == "fallback" and result.fallback_reason.startswith("run_timeout")


# ---- invalid messages / receivers ---------------------------------------------------
def _orch(classifier, comm_logger, state, **cfg):
    config = OrchestratorConfig(agent_timeout=2, run_timeout=10, retry_delay=0, **cfg)
    return Orchestrator([classifier, ValidatorAgent()], config, comm_logger, state)


def test_invalid_receiver_handled_and_logged(comm_logger, state):
    clf = MisroutingClassifier(llm=ScriptedLLM([GOOD_HIGH]), state=state)
    result = _orch(clf, comm_logger, state).run(URGENT)
    assert result.status == "fallback" and result.fallback_reason.startswith("invalid_receiver")
    logged = [r for r in comm_logger.records() if r["event"] == "error"]
    assert any("invalid_receiver" in r["detail"] and "validatr" in r["detail"] for r in logged)


def test_malformed_reply_retried_then_fallback(comm_logger, state):
    llm = ScriptedLLM([GOOD_HIGH])
    clf = MalformedClassifier(llm=llm, state=state)
    result = _orch(clf, comm_logger, state, max_agent_retries=1).run(URGENT)
    assert result.status == "fallback" and "agent_failure" in result.fallback_reason
    assert llm.calls == 2  # original try + one retry


def test_spoofed_sender_rejected(comm_logger, state):
    clf = SpoofingClassifier(llm=ScriptedLLM([GOOD_HIGH]), state=state)
    result = _orch(clf, comm_logger, state, max_agent_retries=0).run(URGENT)
    assert result.status == "fallback" and "claims sender" in result.fallback_reason


class NoopAgent(BaseAgent):
    name = "classifier"
    accepts = frozenset({CLASSIFY_TASK})

    def process(self, message):
        return self.reply(message, "validator", CLASSIFICATION_RESULT,
                          {"task_description": "x", "priority": "High", "reasoning": GOOD,
                           "keyword_scan": "x", "attempt": 1})


def test_missing_agent_registration_is_invalid_receiver(comm_logger, state):
    orch = Orchestrator([NoopAgent()], OrchestratorConfig(agent_timeout=2), comm_logger, state)
    result = orch.run("anything")  # validator was never registered
    assert result.fallback_reason.startswith("invalid_receiver")


def test_blank_input_raises(make_orchestrator):
    with pytest.raises(ValueError):
        make_orchestrator(ScriptedLLM([GOOD_HIGH])).run("   ")


# ---- logging -------------------------------------------------------------------------
def test_every_hop_is_logged_with_one_trace_id(make_orchestrator, comm_logger):
    result = make_orchestrator(ScriptedLLM([GOOD_HIGH])).run(URGENT)
    records = comm_logger.records(trace_id=result.trace_id)
    events = [(r["event"], r["agent"]) for r in records]
    assert events[0] == ("started", "orchestrator") and events[-1] == ("completed", "orchestrator")
    for expected in [("received", "classifier"), ("sent", "classifier"),
                     ("received", "validator"), ("sent", "validator"), ("routed", "orchestrator")]:
        assert expected in events
    steps = [r["step"] for r in records if r["event"] == "routed"]
    assert steps == sorted(steps) == [1, 2, 3]


def test_jsonl_file_written(tmp_path):
    import json

    from app.comms_log import CommLogger

    log = CommLogger(path=str(tmp_path / "sub" / "log.jsonl"))
    log.log("started", "orchestrator", "abc", step=0, detail="hello")
    line = json.loads((tmp_path / "sub" / "log.jsonl").read_text().strip())
    assert line["event"] == "started" and line["trace_id"] == "abc"
