import time

import pytest

from app.resilience import AgentTimeout, run_with_timeout


def test_returns_value_when_fast():
    assert run_with_timeout(lambda x: x * 2, 1.0, 21) == 42


def test_raises_timeout_when_slow():
    start = time.monotonic()
    with pytest.raises(AgentTimeout):
        run_with_timeout(time.sleep, 0.1, 5)
    assert time.monotonic() - start < 1.0  # did not wait for the full sleep


def test_reraises_original_exception():
    def boom():
        raise KeyError("nope")

    with pytest.raises(KeyError):
        run_with_timeout(boom, 1.0)
