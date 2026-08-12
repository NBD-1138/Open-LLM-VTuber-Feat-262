from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

import aiohttp

from .publisher import LivePublisher

EVENTSUB_WS = "wss://eventsub.wss.twitch.tv/ws"


class EventSubWS:
    def __init__(self, twitch_conf, publisher: LivePublisher, verbose: bool = False):
        self.cfg = twitch_conf
        self.publisher = publisher
        self.log = logging.getLogger("eventsub")
        self._verbose = verbose
        self.connected: bool = False
        self.last_heartbeat: Optional[float] = None
        self._stop = asyncio.Event()
        self._session: Optional[aiohttp.ClientSession] = None

    def _log_event(self, message: str, *args) -> None:
        if self._verbose:
            self.log.info(message, *args)
        else:
            self.log.debug(message, *args)

    async def stop(self) -> None:
        self._stop.set()
        if self._session:
            await self._session.close()

    async def run(self) -> None:
        self._session = aiohttp.ClientSession()
        backoff = 1
        try:
            while not self._stop.is_set():
                try:
                    await self._connect_once()
                    backoff = 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.log.error("EventSub connection error: %s", exc)
                    self.connected = False
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60)
        finally:
            if self._session:
                await self._session.close()
                self._session = None

    async def _connect_once(self) -> None:
        assert self._session is not None
        async with self._session.ws_connect(EVENTSUB_WS, heartbeat=15) as ws:
            self.connected = True
            self._log_event("EventSub WebSocket connected.")
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(ws, msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise ws.exception() or RuntimeError("EventSub WS error")
        self.connected = False

    async def _handle_message(self, ws: aiohttp.ClientWebSocketResponse, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.log.debug("Discarding malformed EventSub frame: %s", raw)
            return

        message_type = data.get("metadata", {}).get("message_type")
        if message_type == "session_welcome":
            session = data.get("payload", {}).get("session", {})
            session_id = session.get("id")
            self.last_heartbeat = time.time()
            self._log_event("EventSub session welcome (%s)", session_id)
        elif message_type == "session_keepalive":
            self.last_heartbeat = time.time()
            self.log.debug("EventSub keepalive")
        elif message_type == "notification":
            await self.publisher.publish_eventsub(data)
        else:
            self.log.debug("Unhandled EventSub frame: %s", message_type)
