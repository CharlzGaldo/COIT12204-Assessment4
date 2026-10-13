"""
Classifier agent.

Role: decide the priority (High / Medium / Low) of a task and explain why.
It never decides whether its own answer is good enough - that is the
Validator's job - which keeps the two roles cleanly separated.

Handles:
    classify_task          (from orchestrator) first attempt
    revise_classification  (from validator)    retry with feedback
Sends:
    classification_result  (to validator)
"""
from typing import Optional

from app import agent as core  # module reference so tests can monkeypatch core.get_llm
from app.agents.base import BaseAgent
from app.comms_log import CommLogger
from app.messages import (
    CLASSIFICATION_RESULT,
    CLASSIFIER,
    CLASSIFY_TASK,
    REVISE_CLASSIFICATION,
    VALIDATOR,
    Message,
)
from app.state import AgentState, agent_state


class ClassifierAgent(BaseAgent):
    name = CLASSIFIER
    accepts = frozenset({CLASSIFY_TASK, REVISE_CLASSIFICATION})

    def __init__(
        self,
        llm=None,
        state: AgentState = agent_state,
        comm_logger: Optional[CommLogger] = None,
    ) -> None:
        super().__init__(comm_logger)
        self.llm = llm
        self.state = state

    def process(self, message: Message) -> Message:
        content = message.content
        task = content["task_description"]
        feedback = content.get("feedback")  # only present on revisions
        attempt = content.get("attempt", 1)

        result = core.run_classification(
            task, llm=self.llm, state=self.state, feedback=feedback
        )

        return self.reply(
            message,
            receiver=VALIDATOR,
            intent=CLASSIFICATION_RESULT,
            content={
                "task_description": task,
                "priority": result["priority"],
                "reasoning": result["reasoning"],
                "keyword_scan": result["keyword_scan"],
                "attempt": attempt,
            },
        )
