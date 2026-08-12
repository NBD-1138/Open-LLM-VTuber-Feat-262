from __future__ import annotations

import logging
from typing import Optional

_log = logging.getLogger("live.llm")
_api_key_env: Optional[str] = None


def configure(api_key_env: str) -> None:
    """Persist the environment variable name used for LLM credentials.

    The Twitch integration currently forwards chat to the primary backend for
    generation, so we only log the intent here to avoid hard failures when the
    key is absent.
    """
    global _api_key_env
    _api_key_env = api_key_env or None
    if _api_key_env:
        _log.debug("LLM configured to use env var %s", _api_key_env)
    else:
        _log.debug("LLM disabled (no api_key_env provided).")


async def generate_reply(*_, **__) -> str:
    """Placeholder async generator. Twitch replies are produced upstream."""
    _log.debug("generate_reply called; returning empty response (handled upstream).")
    return ""
