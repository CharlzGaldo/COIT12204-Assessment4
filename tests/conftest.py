"""Shared test setup. Runs before any test module imports the app."""
import os

# Disable the on-disk message log during tests (the in-memory log still works).
os.environ.setdefault("MESSAGE_LOG_PATH", "")

import pytest  # noqa: E402

from app.comms_log import CommLogger  # noqa: E402
from app.orchestrator import OrchestratorConfig, build_default_orchestrator  # noqa: E402
from app.state import AgentState  # noqa: E402


@pytest.fixture
def comm_logger():
    return CommLogger(path=None)


@pytest.fixture
def state():
    return AgentState()


@pytest.fixture
def make_orchestrator(comm_logger, state):
    """Factory: make_orchestrator(llm, **config_overrides) -> Orchestrator with fast defaults."""

    def _make(llm, **overrides):
        cfg = dict(max_iterations=3, max_steps=12, agent_timeout=2.0, run_timeout=10.0,
                   max_agent_retries=1, retry_delay=0.0)
        cfg.update(overrides)
        return build_default_orchestrator(
            llm=llm, state=state, config=OrchestratorConfig(**cfg), comm_logger=comm_logger
        )

    return _make
