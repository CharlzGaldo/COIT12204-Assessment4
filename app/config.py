"""
Application configuration.

Loads settings from environment variables (see .env.example).
Keeping this in one place means tests can monkeypatch a single
object instead of hunting for os.environ calls scattered around.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Reads the .env file in the project root (if present) and loads its
# key=value pairs into os.environ, so os.getenv() below can see them.
# This has to run BEFORE the Settings dataclass reads os.getenv(), which
# is why it's called here at module import time, before the class body.
# override=False means real environment variables (e.g. ones you `export`
# yourself in the terminal) always win over the .env file.
load_dotenv(override=False)


@dataclass
class Settings:
    # Read once at import time. If you change .env while the server is
    # running, you'll need to restart it for the new values to take effect.
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    # gemini-3.1-flash-lite is the current stable (GA), cost-effective model
    # as of late 2026 — good fit for a simple classification task. Gemini
    # 1.5 and 2.0 are fully retired; Gemini 2.5 is stable but scheduled for
    # shutdown, so avoid pinning to it for anything long-lived.
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # ---- Multi-agent / orchestrator settings (Assessment 4) --------------
    # Seconds the Gemini client itself waits for a response (network level).
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "15"))
    # Seconds the orchestrator waits for ONE agent to handle ONE message.
    # Slightly larger than llm_timeout so the LLM client normally times out
    # first and the orchestrator timeout is the backstop.
    agent_timeout: float = float(os.getenv("AGENT_TIMEOUT", "20"))
    # Hard ceiling (seconds) for one whole classify -> validate conversation.
    run_timeout: float = float(os.getenv("RUN_TIMEOUT", "60"))
    # Max classification attempts (1 initial + revisions) before fallback.
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "3"))
    # Max messages the orchestrator will route in one conversation.
    max_steps: int = int(os.getenv("MAX_STEPS", "12"))
    # Extra tries for an agent that crashes / times out / returns junk.
    max_agent_retries: int = int(os.getenv("MAX_AGENT_RETRIES", "1"))
    # JSON-lines file for the full message audit log. Set to an empty
    # string (MESSAGE_LOG_PATH=) to disable file logging.
    message_log_path: str = os.getenv("MESSAGE_LOG_PATH", "logs/agent_messages.jsonl")


# Single shared settings instance imported everywhere else
# (app/agent.py uses settings.google_api_key and settings.gemini_model).
settings = Settings()
