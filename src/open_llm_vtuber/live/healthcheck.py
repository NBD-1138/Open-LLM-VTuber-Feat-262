from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger("live.healthcheck")


async def run_all(cfg, services: Dict[str, Any]) -> Dict[str, Any]:
    """Return a very simple health snapshot for live services."""
    snapshot: Dict[str, Any] = {"live_backend": "ok"}

    twitch = services.get("twitch_chat")
    if twitch is not None:
        status = getattr(twitch, "get_talkback_status", None)
        if callable(status):
            try:
                snapshot["twitch_chat"] = status()
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("Failed to gather twitch status: %s", exc)
                snapshot["twitch_chat"] = {"error": str(exc)}
        else:
            snapshot["twitch_chat"] = {"info": "status unavailable"}

    bridge = services.get("bridge")
    if bridge is not None:
        client_count = getattr(bridge, "client_count", None)
        if client_count is not None:
            snapshot.setdefault("bridge", {})["clients"] = client_count

    return snapshot
