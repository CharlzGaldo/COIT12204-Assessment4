# Task Priority Classification — Two-Agent System (COIT12204 Assessment 4)

Extends the Assessment 3 single-agent classifier into a **two-agent system**
coordinated by an **orchestrator**:

| Component | Role |
|-----------|------|
| **Classifier agent** | Uses the keyword tool + Gemini (structured output) to assign High / Medium / Low with reasoning. |
| **Validator agent** | Independently reviews the answer with deterministic rules; approves it or sends it back with specific feedback. |
| **Orchestrator** | Supervisor hub. Routes every message, tracks steps, enforces limits, handles invalid receivers, retries, falls back. |

# View the application on web
https://priority-agent-458025859336.australia-southeast1.run.app

# View the video demo
(add your Assessment 4 video link here)

# Self-Deploy
Read the Deploy.md file

## How the agents communicate

```
orchestrator --classify_task-----------> classifier
classifier   --classification_result---> validator
validator    --revise_classification--> classifier      (rejected, with feedback)
validator    --final_result------------> orchestrator   (approved)
```

Agents never call each other directly: every message passes through the
orchestrator's queue, so every hop is validated, logged and counted in one place.

### Message schema (`app/messages.py`)

```json
{
  "message_id": "9f2c1a7b3d10",
  "trace_id": "79866148",
  "step": 3,
  "timestamp": "2026-10-13T04:39:32+00:00",
  "sender": "validator",
  "receiver": "classifier",
  "intent": "revise_classification",
  "content": {
    "task_description": "URGENT: payment gateway is blocking checkout, fix asap",
    "previous_priority": "Low",
    "previous_reasoning": "...",
    "issues": ["ignores_urgency_keywords: 3 urgency keywords were found ..."],
    "feedback": "ignores_urgency_keywords: ...",
    "attempt": 2
  },
  "constraints": { "format": "json", "timeout_seconds": 20.0, "max_attempts": 3 }
}
```

Validation happens in three layers: **envelope** (Pydantic, unknown fields
rejected), **content** (a schema per `intent`), and **route** (only certain
senders may send an intent to certain receivers — e.g. only the validator can
send `final_result`).

### Resilience and guardrails

| Mechanism | Where | Behaviour |
|-----------|-------|-----------|
| Output validation | Validator agent + Pydantic | Rejects invalid priority, weak/overlong reasoning, answers that contradict a priority stated in the task, or `Low` despite ≥2 urgency keywords |
| Max iterations | Orchestrator | `MAX_ITERATIONS` classification attempts, then fallback |
| Max steps | Orchestrator | `MAX_STEPS` routed messages per conversation |
| Timeouts | Orchestrator (`run_with_timeout`) + Gemini client | Per-agent timeout and whole-run deadline |
| Retries | Orchestrator | `MAX_AGENT_RETRIES` for crashes, timeouts, malformed replies |
| Invalid receiver / route / sender | Orchestrator | Logged as an error; conversation ends with fallback |
| Fallback | Orchestrator | Safe `Medium` answer with `fallback=true` and a `fallback_reason` |
| Error logging | `CommLogger` | Every failure logged by both agent and orchestrator |

### Logging

Every event (`started`, `routed`, `received`, `sent`, `error`, `fallback`,
`completed`) is written to the console, to `logs/agent_messages.jsonl`
(full message contents), and to an in-memory buffer served by the API:

```
GET /api/agent/trace/{trace_id}   all events of one conversation
GET /api/agent/messages?limit=50  most recent events
GET /api/agent/state              legacy AgentState (tool calls, errors, history)
```

## Project layout

```
app/
  config.py        settings (API key, model, timeouts, limits)
  messages.py      Message envelope, per-intent content schemas, route table, validation
  comms_log.py     CommLogger (console + JSONL + in-memory)
  resilience.py    run_with_timeout, AgentTimeout, AgentFailure
  orchestrator.py  Orchestrator, OrchestratorConfig, RunResult
  agent.py         shared LLM logic (prompt, get_llm, run_classification) + classify_priority()
  agents/
    base.py        BaseAgent: validate in/out, log received/sent/error
    classifier.py  ClassifierAgent
    validator.py   ValidatorAgent (rules + feedback)
  tools.py         check_urgency_keywords tool
  state.py         AgentState (unchanged from Assessment 3)
  fakes.py         ScriptedLLM + deliberately faulty agents (tests and demo)
  main.py          FastAPI app and endpoints
  static/index.html  browser UI, now with an Agent Communication Log panel
scripts/demo.py    runs scripted scenarios and prints the communication log
tests/             pytest suite (no API key or network needed)
docs/REPORT_NOTES.md  maps the brief's requirements to evidence in this repo
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your GOOGLE_API_KEY
uvicorn app.main:app --reload   # open http://127.0.0.1:8000/
```

```bash
curl -X POST http://127.0.0.1:8000/api/agent/classify_priority \
  -H "Content-Type: application/json" \
  -d '{"task_description": "Server is down in production, fix ASAP"}'
# then: curl http://127.0.0.1:8000/api/agent/trace/<trace_id from the response>
```

## Demo scenarios (for logs, screenshots and the video)

Offline — no API key needed; scripted fake LLMs, real agents/orchestrator:

```bash
python -m scripts.demo --list            # list scenarios
python -m scripts.demo                   # run all
python -m scripts.demo revision_loop     # one scenario
python -m scripts.demo --live "Server is down in production, fix ASAP"   # real Gemini
```

| Scenario | Shows |
|----------|-------|
| `happy_path` | classifier → validator → approved (3 messages) |
| `revision_loop` | validator rejects, classifier revises using the feedback (5 messages) |
| `stated_priority_conflict` | validator catches an answer that ignores the task's stated priority |
| `weak_reasoning` | validator rejects a one-word justification |
| `fail_max_iterations` | validator never satisfied → iteration limit → fallback |
| `fail_timeout` | hung LLM → agent timeout → retry → fallback |
| `fail_invalid_receiver` | message addressed to a non-existent agent |
| `fail_malformed_message` | agent returns a message with a misspelt field |
| `fail_llm_error` | LLM keeps raising → retries exhausted → fallback |

## Tests

```bash
pytest -v
```


