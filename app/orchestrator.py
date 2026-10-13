"""
Orchestrator (supervisor) for the Classifier + Validator system.

Coordination pattern: supervisor-worker hub. Agents never call each other
directly. Every message - including agent-to-agent ones - goes through the
orchestrator's queue, so every hop is validated, logged, counted and
controlled in one place.

Per message the orchestrator:
    1. enforces the step limit and the overall run deadline
    2. stamps the step number and validates the message (envelope + content)
    3. resolves the receiver (unknown receiver -> logged error + fallback)
    4. checks the route is legal for that intent (ROUTES)
    5. enforces the max-iteration limit for classification attempts
    6. dispatches to the agent with a timeout and bounded retries
    7. validates the reply (and that the sender really is that agent)

Any condition it cannot recover from ends the conversation with a safe
fallback answer instead of an exception, and says why (fallback_reason).
"""
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from app.agent import FALLBACK_PRIORITY
from app.agents.base import BaseAgent
from app.agents.classifier import ClassifierAgent
from app.agents.validator import ValidatorAgent
from app.comms_log import CommLogger, comms as default_comms
from app.config import settings
from app.messages import (
    ATTEMPT_INTENTS,
    CLASSIFY_TASK,
    FINAL_RESULT,
    ORCHESTRATOR,
    REVISE_CLASSIFICATION,
    Constraints,
    InvalidMessageError,
    Message,
    check_route,
    validate_message,
)
from app.resilience import AgentFailure, AgentTimeout, run_with_timeout
from app.state import AgentState, agent_state


@dataclass
class OrchestratorConfig:
    max_iterations: int = field(default_factory=lambda: settings.max_iterations)
    max_steps: int = field(default_factory=lambda: settings.max_steps)
    agent_timeout: float = field(default_factory=lambda: settings.agent_timeout)
    run_timeout: float = field(default_factory=lambda: settings.run_timeout)
    max_agent_retries: int = field(default_factory=lambda: settings.max_agent_retries)
    retry_delay: float = 0.0  # seconds to wait between agent retries


@dataclass
class RunResult:
    trace_id: str
    priority: str
    reasoning: str
    status: str  # "approved" or "fallback"
    iterations: int
    steps: int
    path: list[str]  # e.g. ["orchestrator->classifier:classify_task", ...]
    validated: bool = False
    fallback_reason: Optional[str] = None
    last_issues: list[str] = field(default_factory=list)

    @property
    def fallback(self) -> bool:
        return self.status == "fallback"

    def to_dict(self) -> dict:
        return {
            "priority": self.priority,
            "reasoning": self.reasoning,
            "fallback": self.fallback,
            "validated": self.validated,
            "status": self.status,
            "fallback_reason": self.fallback_reason,
            "iterations": self.iterations,
            "steps": self.steps,
            "trace_id": self.trace_id,
            "path": self.path,
            "last_issues": self.last_issues,
        }


class _Stop(Exception):
    """Internal: end the conversation with a fallback. Carries the reason."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass
class _Run:
    """Mutable bookkeeping for one conversation."""

    trace_id: str
    steps: int = 0
    iterations: int = 0
    path: list[str] = field(default_factory=list)
    last_issues: list[str] = field(default_factory=list)


class Orchestrator:
    def __init__(
        self,
        agents: list[BaseAgent],
        config: Optional[OrchestratorConfig] = None,
        comm_logger: Optional[CommLogger] = None,
        state: AgentState = agent_state,
    ) -> None:
        self.config = config or OrchestratorConfig()
        self.comms = comm_logger or default_comms
        self.state = state
        self.agents: dict[str, BaseAgent] = {}
        for agent in agents:
            self.register(agent)

    def register(self, agent: BaseAgent) -> None:
        agent.attach_logger(self.comms)  # one shared audit log
        self.agents[agent.name] = agent

    # ------------------------------------------------------------------
    def run(self, task_description: str) -> RunResult:
        if not task_description or not task_description.strip():
            raise ValueError("task_description must not be empty")

        run = _Run(trace_id=uuid.uuid4().hex[:8])
        deadline = time.monotonic() + self.config.run_timeout
        self.comms.log(
            "started",
            ORCHESTRATOR,
            run.trace_id,
            step=0,
            detail=(
                f"max_iterations={self.config.max_iterations} "
                f"max_steps={self.config.max_steps} "
                f"agent_timeout={self.config.agent_timeout}s"
            ),
        )

        try:
            result = self._loop(run, task_description, deadline)
        except _Stop as stop:
            result = self._fallback(run, stop)
        except Exception as exc:  # noqa: BLE001 - last-resort safety net
            result = self._fallback(run, _Stop("orchestrator_error", f"{type(exc).__name__}: {exc}"))

        self.comms.log(
            "completed",
            ORCHESTRATOR,
            run.trace_id,
            step=run.steps,
            detail=(
                f"status={result.status} priority={result.priority} "
                f"iterations={result.iterations} steps={result.steps}"
            ),
        )
        return result

    # ------------------------------------------------------------------
    def _loop(self, run: _Run, task: str, deadline: float) -> RunResult:
        cfg = self.config
        queue: deque[Message] = deque(
            [
                Message(
                    trace_id=run.trace_id,
                    sender=ORCHESTRATOR,
                    receiver="classifier",
                    intent=CLASSIFY_TASK,
                    content={"task_description": task, "attempt": 1},
                    constraints=Constraints(
                        timeout_seconds=cfg.agent_timeout, max_attempts=cfg.max_iterations
                    ),
                )
            ]
        )

        while queue:
            if time.monotonic() >= deadline:
                raise _Stop("run_timeout", f"conversation exceeded {cfg.run_timeout}s")
            if run.steps >= cfg.max_steps:
                raise _Stop("max_steps", f"more than {cfg.max_steps} messages routed")

            message = queue.popleft()
            run.steps += 1
            message = message.model_copy(update={"step": run.steps})  # authoritative step

            # 1. validate envelope + content
            try:
                message = validate_message(message)
            except InvalidMessageError as exc:
                self.comms.log("error", ORCHESTRATOR, run.trace_id, message=message,
                               level="ERROR", detail=f"invalid_message: {exc}")
                raise _Stop("invalid_message", str(exc)) from exc

            # 2. resolve receiver
            receiver = message.receiver
            agent = self.agents.get(receiver)
            if receiver != ORCHESTRATOR and agent is None:
                self.comms.log("error", ORCHESTRATOR, run.trace_id, message=message,
                               level="ERROR",
                               detail=f"invalid_receiver: no agent named {receiver!r} "
                                      f"(registered: {sorted(self.agents)})")
                raise _Stop("invalid_receiver", f"no agent named {receiver!r}")

            # 3. check the route is legal for this intent
            try:
                check_route(message)
            except InvalidMessageError as exc:
                self.comms.log("error", ORCHESTRATOR, run.trace_id, message=message,
                               level="ERROR", detail=f"invalid_route: {exc}")
                raise _Stop("invalid_route", str(exc)) from exc

            run.path.append(f"{message.sender}->{message.receiver}:{message.intent}")

            # 4. messages addressed to the orchestrator end the conversation
            if receiver == ORCHESTRATOR:
                self.comms.log("routed", ORCHESTRATOR, run.trace_id, message=message)
                return self._finish(run, message)

            # 5. iteration limit (each classify/revise message = one attempt)
            if message.intent in ATTEMPT_INTENTS:
                if run.iterations >= cfg.max_iterations:
                    raise _Stop(
                        "max_iterations",
                        f"validator still unsatisfied after {cfg.max_iterations} attempts",
                    )
                run.iterations += 1
                if message.intent == REVISE_CLASSIFICATION:
                    run.last_issues = list(message.content["issues"])

            self.comms.log("routed", ORCHESTRATOR, run.trace_id, message=message)

            # 6. dispatch with timeout + retries
            reply = self._dispatch(agent, message, run, deadline)
            queue.append(reply)

        raise _Stop("no_result", "message queue emptied without a final result")

    # ------------------------------------------------------------------
    def _dispatch(self, agent: BaseAgent, message: Message, run: _Run, deadline: float) -> Message:
        cfg = self.config
        attempts = 1 + cfg.max_agent_retries
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _Stop("run_timeout", f"conversation exceeded {cfg.run_timeout}s")
            timeout = min(message.constraints.timeout_seconds or cfg.agent_timeout, remaining)

            try:
                reply = run_with_timeout(agent.handle, timeout, message)
                reply = validate_message(reply)
                if reply.sender != agent.name:
                    raise InvalidMessageError(
                        f"reply claims sender {reply.sender!r} but came from {agent.name!r}"
                    )
                if reply.trace_id != run.trace_id:
                    raise InvalidMessageError("reply belongs to a different conversation")
                return reply
            except AgentTimeout as exc:
                last_error = exc
                kind = "timeout"
            except InvalidMessageError as exc:
                last_error = exc
                kind = "invalid_reply"
            except Exception as exc:  # noqa: BLE001 - any agent crash
                last_error = exc
                kind = "agent_error"

            self.comms.log(
                "error", ORCHESTRATOR, run.trace_id, message=message, level="ERROR",
                detail=f"{kind} from {agent.name} (try {attempt}/{attempts}): "
                       f"{type(last_error).__name__}: {last_error}",
            )
            if attempt < attempts and cfg.retry_delay:
                time.sleep(cfg.retry_delay)

        raise _Stop(
            "agent_failure",
            f"{agent.name} failed after {attempts} tries: {type(last_error).__name__}: {last_error}",
        ) from AgentFailure(str(last_error))

    # ------------------------------------------------------------------
    def _finish(self, run: _Run, message: Message) -> RunResult:
        if message.intent != FINAL_RESULT:  # route table makes this unreachable; defensive
            raise _Stop("invalid_route", f"unexpected {message.intent!r} sent to orchestrator")
        c = message.content
        result = RunResult(
            trace_id=run.trace_id,
            priority=c["priority"],
            reasoning=c["reasoning"],
            status="approved",
            validated=c["validated"],
            iterations=run.iterations,
            steps=run.steps,
            path=run.path,
        )
        self.state.record_success(
            {"priority": result.priority, "reasoning": result.reasoning, "fallback": False}
        )
        return result

    def _fallback(self, run: _Run, stop: _Stop) -> RunResult:
        reason = f"{stop.code}: {stop.detail}"
        self.comms.log(
            "fallback", ORCHESTRATOR, run.trace_id, step=run.steps, level="WARNING",
            detail=f"using safe default {FALLBACK_PRIORITY} ({reason})",
        )
        self.state.record_error(reason)
        return RunResult(
            trace_id=run.trace_id,
            priority=FALLBACK_PRIORITY,
            reasoning=(
                f"Fallback classification: the agents could not produce a validated "
                f"answer ({stop.code}), so a safe default of {FALLBACK_PRIORITY} was used."
            ),
            status="fallback",
            validated=False,
            fallback_reason=reason,
            iterations=run.iterations,
            steps=run.steps,
            path=run.path,
            last_issues=run.last_issues,
        )


def build_default_orchestrator(
    llm=None,
    state: AgentState = agent_state,
    config: Optional[OrchestratorConfig] = None,
    comm_logger: Optional[CommLogger] = None,
) -> Orchestrator:
    """Wire up the standard system: Classifier + Validator under one Orchestrator."""
    comm_logger = comm_logger or default_comms
    return Orchestrator(
        agents=[
            ClassifierAgent(llm=llm, state=state, comm_logger=comm_logger),
            ValidatorAgent(comm_logger=comm_logger),
        ],
        config=config,
        comm_logger=comm_logger,
        state=state,
    )
