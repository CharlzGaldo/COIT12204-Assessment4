"""
Validator agent.

Role: independently review the Classifier's answer using deterministic
rules, then either approve it or send it back with specific feedback.

Why deterministic rules instead of a second LLM call?
    - fast, free and repeatable (good for testing and for the demo)
    - it re-runs the keyword scan itself instead of trusting the
      Classifier's copy, so it is a genuinely independent check
    - an LLM reviewer can be added later behind the same message protocol

Handles:
    classification_result  (from classifier)
Sends:
    final_result           (to orchestrator)  - approved
    revise_classification  (to classifier)    - rejected, with feedback
"""
import re
from typing import Optional

from app.agents.base import BaseAgent
from app.comms_log import CommLogger
from app.messages import (
    CLASSIFICATION_RESULT,
    CLASSIFIER,
    FINAL_RESULT,
    ORCHESTRATOR,
    REVISE_CLASSIFICATION,
    VALIDATOR,
    Message,
)
from app.tools import check_urgency_keywords

MIN_REASONING_CHARS = 15
MAX_REASONING_CHARS = 600

# Phrases where the task text states its own priority. The lookbehinds stop
# "not high priority" from being read as "high priority".
_STATED_PATTERNS = [
    (re.compile(r"(?<!not )(?<!non-)\bhigh[- ](?:priority|urgency)\b", re.I), "High"),
    (re.compile(r"(?<!not )(?<!non-)\bmedium[- ](?:priority|urgency)\b", re.I), "Medium"),
    (re.compile(r"(?<!not )\blow[- ](?:priority|urgency)\b|\bnot urgent\b|\bno rush\b", re.I), "Low"),
]

# Issue codes (stable strings so they can be asserted in tests and logs).
INVALID_PRIORITY = "invalid_priority"
WEAK_REASONING = "weak_reasoning"
CONTRADICTS_STATED = "contradicts_stated_priority"
IGNORES_KEYWORDS = "ignores_urgency_keywords"


def stated_priority(task_description: str) -> Optional[str]:
    """Return the priority the task text explicitly states, if exactly one."""
    levels = {level for pattern, level in _STATED_PATTERNS if pattern.search(task_description)}
    return levels.pop() if len(levels) == 1 else None  # none or ambiguous -> None


def count_keywords(scan: str) -> int:
    if scan.startswith("No urgency"):
        return 0
    return len(scan.split(":", 1)[1].split(",")) if ":" in scan else 0


def review(task_description: str, priority: str, reasoning: str) -> list[str]:
    """Return a list of human-readable issues. Empty list means approved."""
    issues: list[str] = []

    if priority not in ("High", "Medium", "Low"):
        issues.append(f"{INVALID_PRIORITY}: {priority!r} is not High, Medium or Low.")

    text = reasoning.strip()
    if len(text) < MIN_REASONING_CHARS:
        issues.append(
            f"{WEAK_REASONING}: reasoning is {len(text)} characters; write at least "
            f"{MIN_REASONING_CHARS} characters that justify the priority."
        )
    elif len(text) > MAX_REASONING_CHARS:
        issues.append(
            f"{WEAK_REASONING}: reasoning is {len(text)} characters; keep it under "
            f"{MAX_REASONING_CHARS}."
        )

    stated = stated_priority(task_description)
    if stated and stated != priority:
        issues.append(
            f"{CONTRADICTS_STATED}: the task states its own priority as {stated} "
            f"but you answered {priority}."
        )

    # Independent keyword scan (does not trust the classifier's copy).
    scan = check_urgency_keywords.invoke({"task_description": task_description})
    n_keywords = count_keywords(scan)
    if priority == "Low" and n_keywords >= 2 and not stated:
        issues.append(
            f"{IGNORES_KEYWORDS}: {n_keywords} urgency keywords were found "
            f"({scan}) yet the task was classified Low."
        )
    return issues


class ValidatorAgent(BaseAgent):
    name = VALIDATOR
    accepts = frozenset({CLASSIFICATION_RESULT})

    def __init__(self, comm_logger: Optional[CommLogger] = None) -> None:
        super().__init__(comm_logger)

    def process(self, message: Message) -> Message:
        c = message.content
        issues = review(c["task_description"], c["priority"], c["reasoning"])

        if not issues:
            return self.reply(
                message,
                receiver=ORCHESTRATOR,
                intent=FINAL_RESULT,
                content={
                    "priority": c["priority"],
                    "reasoning": c["reasoning"].strip(),
                    "validated": True,
                    "issues": [],
                    "attempts": c["attempt"],
                },
            )

        return self.reply(
            message,
            receiver=CLASSIFIER,
            intent=REVISE_CLASSIFICATION,
            content={
                "task_description": c["task_description"],
                "previous_priority": c["priority"],
                "previous_reasoning": c["reasoning"],
                "issues": issues,
                "feedback": " ".join(issues),
                "attempt": c["attempt"] + 1,
            },
        )
