"""Message protocol: envelope, per-intent content, and route validation."""
import pytest

from app.messages import (
    CLASSIFICATION_RESULT, CLASSIFY_TASK, FINAL_RESULT, REVISE_CLASSIFICATION,
    InvalidMessageError, Message, check_route, validate_message,
)


def msg(**overrides):
    base = dict(
        trace_id="t1", sender="orchestrator", receiver="classifier",
        intent=CLASSIFY_TASK, content={"task_description": "Fix the bug", "attempt": 1},
    )
    base.update(overrides)
    return Message(**base)


def test_valid_message_passes():
    m = validate_message(msg())
    assert m.message_id and m.timestamp and m.constraints.format == "json"


def test_unknown_intent_rejected():
    with pytest.raises(InvalidMessageError, match="unknown intent"):
        validate_message(msg(intent="make_coffee"))


def test_missing_content_field_rejected():
    with pytest.raises(InvalidMessageError, match="task_description"):
        validate_message(msg(content={"attempt": 1}))


def test_extra_content_field_rejected():
    with pytest.raises(InvalidMessageError):
        validate_message(msg(content={"task_description": "x", "surprise": True}))


def test_invalid_priority_value_rejected():
    bad = msg(
        sender="classifier", receiver="validator", intent=CLASSIFICATION_RESULT,
        content={"task_description": "x", "priority": "Urgent!!", "reasoning": "r",
                 "keyword_scan": "none", "attempt": 1},
    )
    with pytest.raises(InvalidMessageError, match="priority"):
        validate_message(bad)


def test_revision_must_explain_why():
    bad = msg(
        sender="validator", receiver="classifier", intent=REVISE_CLASSIFICATION,
        content={"task_description": "x", "previous_priority": "Low", "previous_reasoning": "r",
                 "issues": [], "feedback": "f", "attempt": 2},
    )
    with pytest.raises(InvalidMessageError, match="issues"):
        validate_message(bad)


def test_dict_with_typo_in_envelope_rejected():
    raw = dict(trace_id="t", sender="a", reciever="b", intent=CLASSIFY_TASK, content={})
    with pytest.raises(InvalidMessageError, match="invalid envelope"):
        validate_message(raw)


def test_non_message_rejected():
    with pytest.raises(InvalidMessageError, match="expected a Message"):
        validate_message("hello")


def test_route_allows_correct_pairing():
    check_route(msg())  # orchestrator -> classifier : classify_task


def test_route_rejects_wrong_sender():
    forged = msg(
        sender="classifier", receiver="orchestrator", intent=FINAL_RESULT,
        content={"priority": "High", "reasoning": "r", "validated": True, "attempts": 1},
    )
    validate_message(forged)  # structurally fine...
    with pytest.raises(InvalidMessageError, match="may not send"):  # ...but illegal route
        check_route(forged)


def test_route_rejects_wrong_receiver():
    with pytest.raises(InvalidMessageError, match="may not be sent to"):
        check_route(msg(receiver="validator"))
