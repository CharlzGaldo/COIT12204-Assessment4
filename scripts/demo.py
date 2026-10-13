"""
Demo runner: exercises the multi-agent system and prints the communication log.

Offline scenarios use scripted fake LLMs, so they need no API key and always
behave the same - ideal for screenshots and the demo video. Everything except
the LLM (prompt, tool, agents, message validation, orchestrator, timeouts,
logging) is the real code.

Usage (from the project root):
    python -m scripts.demo                    # run every scenario
    python -m scripts.demo --list             # list scenario names
    python -m scripts.demo revision_loop      # run one scenario
    python -m scripts.demo --live "Server is down in production, fix ASAP"
                                              # real Gemini (needs GOOGLE_API_KEY)

Every run also appends full JSON records to logs/agent_messages.jsonl.
"""
import argparse
import logging
import sys

from app.agent import classify_priority
from app.agents.classifier import ClassifierAgent
from app.agents.validator import ValidatorAgent
from app.comms_log import comms
from app.fakes import (
    Delay, MalformedClassifier, MisroutingClassifier, ScriptedLLM, classification,
)
from app.orchestrator import Orchestrator, OrchestratorConfig
from app.state import AgentState

URGENT = "URGENT: payment gateway is blocking checkout, fix asap"
REASON = "The payment gateway outage blocks every customer purchase."


def _fast(**kw):
    base = dict(max_iterations=3, agent_timeout=1.5, run_timeout=15, max_agent_retries=1)
    base.update(kw)
    return OrchestratorConfig(**base)


def _custom(classifier_cls, llm, config):
    state = AgentState()
    return Orchestrator(
        [classifier_cls(llm=llm, state=state), ValidatorAgent()], config, comms, state
    ).run


def happy_path():
    """Interaction 1: classifier -> validator approves on the first attempt."""
    return classify_priority(
        "Server is down in production, fix ASAP",
        llm=ScriptedLLM([classification("High", "Production outage; urgent keywords present.")]),
        config=_fast(),
    )


def revision_loop():
    """Interaction 2: validator rejects a 'Low', classifier revises using feedback."""
    llm = ScriptedLLM([classification("Low", REASON), classification("High", REASON)])
    return classify_priority(URGENT, llm=llm, config=_fast())


def stated_priority_conflict():
    """Interaction 3: classifier ignores the priority the task states; validator catches it."""
    task = "Rename the old config keys, low priority, whenever there is time"
    llm = ScriptedLLM([
        classification("High", "Config changes can break things, so treat as important."),
        classification("Low", "The task explicitly states it is low priority."),
    ])
    return classify_priority(task, llm=llm, config=_fast())


def weak_reasoning():
    """Validator rejects a one-word justification."""
    llm = ScriptedLLM([classification("Medium", "ok"),
                       classification("Medium", "Routine maintenance with no stated deadline.")])
    return classify_priority("Update the internal wiki pages", llm=llm, config=_fast())


def fail_max_iterations():
    """FAILURE 1: classifier never satisfies the validator -> iteration limit -> fallback."""
    llm = ScriptedLLM([classification("Low", REASON)])
    return classify_priority(URGENT, llm=llm, config=_fast(max_iterations=3))


def fail_timeout():
    """FAILURE 2: LLM hangs; orchestrator times the agent out, retries, then falls back."""
    llm = ScriptedLLM([Delay(5, classification("High", REASON))])
    return classify_priority(URGENT, llm=llm, config=_fast(agent_timeout=0.5))


def fail_invalid_receiver():
    """FAILURE 3: a buggy agent addresses a message to 'validatr' (does not exist)."""
    return _custom(MisroutingClassifier, ScriptedLLM([classification("High", REASON)]), _fast())(URGENT)


def fail_malformed_message():
    """FAILURE 4: an agent returns a message with a misspelt field; retried then fallback."""
    return _custom(MalformedClassifier, ScriptedLLM([classification("High", REASON)]), _fast())(URGENT)


def fail_llm_error():
    """FAILURE 5: LLM keeps raising (e.g. 503); retries exhausted -> fallback."""
    llm = ScriptedLLM([RuntimeError("503 Service Unavailable")])
    return classify_priority(URGENT, llm=llm, config=_fast())


SCENARIOS = {f.__name__: f for f in [
    happy_path, revision_loop, stated_priority_conflict, weak_reasoning,
    fail_max_iterations, fail_timeout, fail_invalid_receiver,
    fail_malformed_message, fail_llm_error,
]}


def show(name, result):
    r = result if isinstance(result, dict) else result.to_dict()
    print(f"\n>>> RESULT [{name}]: priority={r['priority']} status={r['status']} "
          f"validated={r['validated']} iterations={r['iterations']} steps={r['steps']}")
    if r["fallback_reason"]:
        print(f"    fallback_reason: {r['fallback_reason']}")
    print("    path: " + "  ".join(r["path"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", nargs="?", help="scenario name (default: run all)")
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    parser.add_argument("--live", metavar="TEXT", help="classify TEXT with the real Gemini model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    if args.list:
        for name, fn in SCENARIOS.items():
            print(f"{name:26} {fn.__doc__}")
        return 0
    if args.live:
        show("live", classify_priority(args.live))
        return 0

    names = [args.scenario] if args.scenario else list(SCENARIOS)
    for name in names:
        if name not in SCENARIOS:
            print(f"Unknown scenario {name!r}. Use --list.", file=sys.stderr)
            return 2
        print(f"\n{'=' * 78}\nSCENARIO: {name} - {SCENARIOS[name].__doc__}\n{'=' * 78}")
        show(name, SCENARIOS[name]())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
