"""
Priority-classification core + public entry point.

This module now has two jobs:

1. `run_classification()` - the single-agent logic from Assessment 3 (run the
   keyword tool, then call Gemini with structured output). The Classifier
   agent (app/agents/classifier.py) calls this. It does NOT swallow errors
   any more: failures propagate so the orchestrator can decide whether to
   retry or fall back.

2. `classify_priority()` - the function the FastAPI endpoint calls. It builds
   a two-agent system (Classifier + Validator), runs it through the
   Orchestrator and returns a plain dict.

Keeping the LLM as an injectable parameter is what makes this testable:
tests pass in a fake object with the same `.with_structured_output()` shape.
"""
import logging
from typing import Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.config import settings
from app.state import AgentState, agent_state
from app.tools import check_urgency_keywords

logger = logging.getLogger("agent.core")


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------
class PriorityClassification(BaseModel):
    """Structured output schema the LLM must return."""

    priority: Literal["High", "Medium", "Low"] = Field(
        description="The classified priority level of the task."
    )
    reasoning: str = Field(
        description="A short explanation for why this priority was chosen."
    )


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
# {feedback} is "None (first attempt)." on the first pass. When the Validator
# agent rejects an answer, the orchestrator routes its feedback back to the
# Classifier, which puts it here for the next attempt.
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a task-triage assistant. Classify the priority of the "
            "given task as High, Medium, or Low, and briefly justify your "
            "choice. Use the urgency-keyword scan result as supporting "
            "evidence, but weigh the overall meaning of the task too. "
            "If the task description itself explicitly states its own "
            "priority or urgency level (e.g. 'medium urgency', 'low "
            "priority', 'not urgent'), treat that stated level as a strong "
            "signal and follow it unless the rest of the description "
            "clearly contradicts it — do not default to a higher priority "
            "just because a word like 'urgency' appears in the text. "
            "If a reviewer has given feedback on a previous attempt, "
            "address every point in it and write a reasoning of at least "
            "one full sentence.",
        ),
        (
            "human",
            "Task description: {task_description}\n\n"
            "Urgency keyword scan result: {tool_result}\n\n"
            "Reviewer feedback on your previous attempt: {feedback}",
        ),
    ]
)

NO_FEEDBACK = "None (first attempt)."

# Safe default used by the orchestrator when the conversation cannot produce
# a validated answer (LLM down, timeout, validator never satisfied, ...).
FALLBACK_PRIORITY = "Medium"


def get_llm() -> ChatGoogleGenerativeAI:
    """Construct the real Gemini chat model. Not called during tests."""
    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to your .env file (see .env.example)."
        )
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0,
        # Network-level timeout + a single client retry. The orchestrator
        # adds its own per-agent timeout on top as a backstop.
        timeout=settings.llm_timeout,
        max_retries=1,
    )


def run_classification(
    task_description: str,
    llm=None,
    state: AgentState = agent_state,
    feedback: Optional[str] = None,
) -> dict:
    """Run the tool + LLM once. Returns {priority, reasoning, keyword_scan}.

    Raises on any failure (bad key, network, malformed output); callers
    (the Classifier agent / orchestrator) handle retries and fallback.
    """
    if not task_description or not task_description.strip():
        raise ValueError("task_description must not be empty")

    tool_result = check_urgency_keywords.invoke({"task_description": task_description})
    state.record_tool_call("check_urgency_keywords", tool_result)

    model = llm or get_llm()
    chain = PROMPT | model.with_structured_output(PriorityClassification)
    result: PriorityClassification = chain.invoke(
        {
            "task_description": task_description,
            "tool_result": tool_result,
            "feedback": feedback or NO_FEEDBACK,
        }
    )
    return {
        "priority": result.priority,
        "reasoning": result.reasoning,
        "keyword_scan": tool_result,
    }


def classify_priority(
    task_description: str,
    llm=None,
    state: AgentState = agent_state,
    config=None,
) -> dict:
    """Classify a task with the two-agent system. Public entry point.

    Returns a dict with at least {priority, reasoning, fallback}, plus
    orchestration details (validated, iterations, steps, trace_id, ...).
    """
    # Imported here (not at module top) to avoid a circular import: the
    # agents import this module for run_classification()/get_llm().
    from app.orchestrator import build_default_orchestrator

    orchestrator = build_default_orchestrator(llm=llm, state=state, config=config)
    return orchestrator.run(task_description).to_dict()
