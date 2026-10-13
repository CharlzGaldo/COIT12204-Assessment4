"""
Fake LLMs for tests and the offline demo script (scripts/demo.py).

They stand in for ChatGoogleGenerativeAI at the one boundary the system
uses: `.with_structured_output(schema)` returning something chainable.
Everything else (prompt, tool, agents, orchestrator) runs for real.
"""
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableLambda

from app.agent import PriorityClassification
from app.agents.classifier import ClassifierAgent
from app.messages import CLASSIFICATION_RESULT


@dataclass
class Delay:
    """Script item: sleep `seconds`, then return `response` (simulates a slow LLM)."""

    seconds: float
    response: Any


def classification(priority: str, reasoning: str) -> PriorityClassification:
    return PriorityClassification(priority=priority, reasoning=reasoning)


class ScriptedLLM:
    """Returns scripted answers in order; the last item repeats forever.

    Each script item may be a PriorityClassification, an Exception instance
    (raised), or a Delay. `prompts` records what the model was asked, so
    tests can check that reviewer feedback really reached the LLM.
    """

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls = 0
        self.prompts: list[str] = []

    def with_structured_output(self, schema):
        return RunnableLambda(self._next)

    def _next(self, prompt_value):
        to_string = getattr(prompt_value, "to_string", None)
        self.prompts.append(to_string() if to_string else str(prompt_value))
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, Delay):
            time.sleep(item.seconds)
            item = item.response
        if isinstance(item, Exception):
            raise item
        return item


# ---- Deliberately faulty agents (for failure-case demos and tests) --------
class MisroutingClassifier(ClassifierAgent):
    """Bug simulation: sends its result to an agent that does not exist."""

    def process(self, message):
        reply = super().process(message)
        return reply.model_copy(update={"receiver": "validatr"})


class MalformedClassifier(ClassifierAgent):
    """Bug simulation: returns a dict with a typo'd field instead of a Message."""

    def process(self, message):
        super().process(message)  # the LLM call itself succeeds...
        return {"trace_id": message.trace_id, "sender": "classifier", "reciever": "validator",
                "intent": CLASSIFICATION_RESULT, "content": {}}


class SpoofingClassifier(ClassifierAgent):
    """Bug simulation: claims to be the validator."""

    def process(self, message):
        return super().process(message).model_copy(update={"sender": "validator"})
