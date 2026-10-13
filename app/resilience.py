"""
Resilience helpers shared by the orchestrator and agents:
custom exception types and a hard timeout wrapper.
"""
import threading
from typing import Any, Callable


class AgentTimeout(Exception):
    """An agent did not finish handling a message in time."""


class AgentFailure(Exception):
    """An agent failed (crash, timeout, or invalid reply) even after retries."""


def run_with_timeout(fn: Callable[..., Any], timeout: float, *args: Any, **kwargs: Any) -> Any:
    """Run fn(*args, **kwargs) and raise AgentTimeout if it exceeds `timeout` seconds.

    The call runs in a daemon thread. Python cannot forcibly kill a thread,
    so a timed-out call is *abandoned* (its eventual result is ignored)
    rather than cancelled. Daemon=True means a stuck call can never keep the
    server process from shutting down.
    """
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise AgentTimeout(f"no response within {timeout:.1f}s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]
