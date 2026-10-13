# Mapping the Assessment 4 brief to this repository

Use this as a checklist while writing the report. The *analysis* — what you
observed, what surprised you, what you changed — must be your own.

## Brief requirement → where it lives

| Brief requirement | Evidence in repo |
|---|---|
| Two specialised agents with clear roles | `app/agents/classifier.py`, `app/agents/validator.py` |
| Structured I/O + message schema per agent | `app/messages.py` (`CONTENT_MODELS`, `ROUTES`) |
| Logging of received and sent messages | `BaseAgent.handle()` → `CommLogger` ("received"/"sent" events) |
| Messages validated / logged / routed | `validate_message`, `check_route`, `Orchestrator._loop` |
| Orchestrator: route, track steps, max iterations, invalid receivers, log all | `app/orchestrator.py` |
| Output validation | `app/agents/validator.py::review` + Pydantic schemas |
| Max iterations / timeouts / fallback / error logging | `OrchestratorConfig`, `run_with_timeout`, `Orchestrator._fallback`, `CommLogger` |
| Pattern | Supervisor–worker hub with a classifier→validator pipeline and a feedback loop |

## Evidence to generate yourself (screenshots / video)

Run `python -m scripts.demo` and the web UI, then screenshot:
1. `revision_loop` console output (agent-to-agent interaction with feedback)
2. The web UI "Agent Communication Log" panel after a real Gemini classification
3. `logs/agent_messages.jsonl` open in an editor (shows full JSON messages)
4. Each failure scenario's `ERROR` / `FALLBACK` lines
5. `pytest -v` passing

## Suggested report content

**3 agent-to-agent interactions** — `happy_path`, `revision_loop`, `stated_priority_conflict`
(also run a few with `--live` so you have real-LLM examples).

**2+ failure cases** — pick from `fail_max_iterations`, `fail_timeout`,
`fail_invalid_receiver`, `fail_malformed_message`, `fail_llm_error`. Better still,
find a failure of your own with the live model (for example a task the validator
wrongly rejects) and explain it.

**2 improvements after debugging** — you need to have actually hit these. Ideas to
look for while testing live: validator rules that are too strict or too loose
(false rejections that end in fallback); whether feedback text actually changes the
LLM's answer; whether the 20s agent timeout suits Cloud Run cold starts; what
happens on a retry after a timeout (the abandoned thread keeps running).

**Future work ideas** — LLM-based reviewer behind the same protocol; persisting
the JSONL log; per-agent circuit breakers; async execution so timeouts can cancel
the work rather than abandon it; cost/latency metrics per hop.

## Honest limits to mention
- Timeouts abandon a hung call rather than cancelling it (Python threads can't be killed).
- The validator is rule-based, so it checks consistency, not whether the priority is *truly* right.
- Agent state in `AgentState` is a process-wide singleton shared across requests.
