"""Validator agent rules and message behaviour."""
from app.agents.validator import (
    CONTRADICTS_STATED, IGNORES_KEYWORDS, INVALID_PRIORITY, WEAK_REASONING,
    ValidatorAgent, review, stated_priority,
)
from app.messages import CLASSIFICATION_RESULT, FINAL_RESULT, REVISE_CLASSIFICATION, Message

GOOD = "Production outage affects all customers right now."


def codes(issues):
    return {i.split(":")[0] for i in issues}


def test_good_answer_has_no_issues():
    assert review("Server down, fix ASAP", "High", GOOD) == []


def test_short_reasoning_flagged():
    assert WEAK_REASONING in codes(review("Update docs", "Low", "ok"))


def test_overlong_reasoning_flagged():
    assert WEAK_REASONING in codes(review("Update docs", "Low", "x" * 700))


def test_invalid_priority_flagged():
    assert INVALID_PRIORITY in codes(review("Update docs", "Urgent", GOOD))


def test_contradicting_stated_priority_flagged():
    issues = review("Refactor module, low priority", "High", GOOD)
    assert CONTRADICTS_STATED in codes(issues)


def test_matching_stated_priority_ok():
    assert review("Refactor module, low priority", "Low", GOOD) == []


def test_negated_priority_not_read_as_stated():
    assert stated_priority("This is not high priority") is None


def test_not_urgent_counts_as_stated_low():
    assert stated_priority("Tidy the wiki, not urgent") == "Low"


def test_conflicting_statements_ignored():
    assert stated_priority("high priority but also low priority") is None


def test_low_with_many_keywords_flagged():
    issues = review("URGENT: checkout is blocking, fix asap", "Low", GOOD)
    assert IGNORES_KEYWORDS in codes(issues)


def test_low_with_keywords_but_stated_low_is_ok():
    # "not urgent" contains the keyword 'urgent'; the stated level wins.
    assert review("Tidy wiki. Not urgent, no rush, critical to nobody", "Low", GOOD) == []


def _result(priority, reasoning, task="URGENT: checkout is blocking, fix asap", attempt=1):
    return Message(
        trace_id="t", step=2, sender="classifier", receiver="validator",
        intent=CLASSIFICATION_RESULT,
        content={"task_description": task, "priority": priority, "reasoning": reasoning,
                 "keyword_scan": "x", "attempt": attempt},
    )


def test_agent_approves_with_final_result_to_orchestrator():
    reply = ValidatorAgent().handle(_result("High", GOOD))
    assert (reply.receiver, reply.intent) == ("orchestrator", FINAL_RESULT)
    assert reply.content["validated"] is True and reply.step == 3


def test_agent_rejects_with_revision_to_classifier():
    reply = ValidatorAgent().handle(_result("Low", GOOD, attempt=1))
    assert (reply.receiver, reply.intent) == ("classifier", REVISE_CLASSIFICATION)
    assert reply.content["attempt"] == 2
    assert reply.content["issues"] and reply.content["feedback"]
