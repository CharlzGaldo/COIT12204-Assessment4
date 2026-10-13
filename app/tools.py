"""
Tools available to the priority-classification agent.

A LangChain "tool" is just a regular Python function decorated with
@tool — the decorator adds the metadata (name, description, expected
args) that lets an LLM know the tool exists and how to call it, and
gives us a consistent `.invoke({...})` calling convention.
"""
from langchain_core.tools import tool

# Simple, explicit keyword list rather than an LLM call — this keeps the
# tool fast, free, deterministic, and easy to unit test on its own.
URGENCY_KEYWORDS = [
    "urgent", "asap", "immediately", "overdue", "deadline today",
    "critical", "blocking", "emergency", "due today", "past due",
]


@tool
def check_urgency_keywords(task_description: str) -> str:
    """Scan a task description for urgency keywords and report which ones were found.

    Args:
        task_description: The raw text description of the task to check.

    Returns:
        A short string listing which urgency keywords (if any) were found.

    Note: this docstring is not just documentation — LangChain uses it
    (plus the type hints) to describe the tool to the LLM if the tool is
    ever bound with .bind_tools() for the model to call autonomously.
    """
    text = task_description.lower()
    found = [kw for kw in URGENCY_KEYWORDS if kw in text]
    if not found:
        return "No urgency keywords detected."
    return f"Urgency keywords detected: {', '.join(found)}"
