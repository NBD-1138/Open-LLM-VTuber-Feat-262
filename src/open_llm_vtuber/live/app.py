from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from .config import AppConf, load_conf
from .eventsub_ws import EventSubWS
from .moderation import ModEngine
from .oauth import TwitchOAuthManager, TwitchTokenBundle, TwitchTokenStore
from .publisher import LivePublisher
from .twitch_chat import ChatBot, TokenRefreshError
from ..websocket_handler import WebSocketHandler

logger = logging.getLogger("live.runtime")
TOKEN_VALIDATION_INTERVAL_SECONDS = 3600


class LiveRuntime:
    """Manage Twitch chat/event integrations within the main server process."""

    def __init__(
        self,
        cfg: AppConf,
        ws_handler: WebSocketHandler,
        backend_base_url: str,
    ):
        self.cfg = cfg
        self.ws_handler = ws_handler
        self.backend_base_url = backend_base_url.rstrip("/")
        self.token_store = TwitchTokenStore()
        self.oauth = TwitchOAuthManager(
            cfg.twitch,
            self.token_store,
            self.backend_base_url,
            logger=logger.getChild("oauth"),
        )
        self.publisher = LivePublisher(
            ws_handler,
            platform="twitch",
            talkback_config=cfg.talkback,
        )
        self.ws_handler.register_live_publisher(self.publisher)
        self.mod_engine = ModEngine(cfg.moderation)
        self.chat_bot: Optional[ChatBot] = None
        self.eventsub: Optional[EventSubWS] = None
        self._tasks: list[asyncio.Task] = []
        self._status_task: Optional[asyncio.Task] = None
        self._started = False
        self._status_override: Optional[str] = None
        self._status_detail: Optional[str] = None
        self._last_status_signature: Optional[str] = None
        self._load_persisted_tokens()

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.twitch.enabled)

    def _load_persisted_tokens(self) -> None:
        bundle = self.token_store.load()
        if not bundle:
            return

        self.cfg.twitch.oauth_token = bundle.access_token
        self.cfg.twitch.refresh_token = bundle.refresh_token
        logger.info(
            "Loaded Twitch OAuth tokens from %s; conf.yaml was left unchanged.",
            self.token_store.display_path,
        )

    async def start(self) -> None:
        if self._started or not self.enabled:
            await self.broadcast_status()
            return
        self._status_override = "connecting"
        self._status_detail = "Starting Twitch live integrations."
        await self.broadcast_status()
        if self.chat_bot is None:
            self.chat_bot = ChatBot(
                self.cfg.twitch,
                self.mod_engine,
                self.publisher,
                self.cfg.talkback,
                persist_tokens=self._persist_tokens,
            )
        if self.eventsub is None and self.cfg.twitch.eventsub.enabled:
            self.eventsub = EventSubWS(
                self.cfg.twitch, self.publisher, verbose=self.cfg.talkback.debug
            )
        self._started = True
        logger.info("Starting live integrations (twitch enabled=%s)", bool(self.chat_bot))
        if self.chat_bot:
            self._tasks.append(
                asyncio.create_task(
                    self._run_chat_bot(),
                    name="twitch-chat",
                )
            )
            self._tasks.append(
                asyncio.create_task(
                    self._token_validation_loop(),
                    name="twitch-token-validation",
                )
            )
        if self.eventsub:
            self._tasks.append(
                asyncio.create_task(
                    self._run_eventsub(),
                    name="eventsub",
                )
            )
        self._status_task = asyncio.create_task(
            self._status_loop(),
            name="live-status",
        )
        await self.broadcast_status()

    async def stop(self) -> None:
        if not self._started and self._status_task is None:
            return
        self._status_override = "stopping"
        self._status_detail = "Stopping Twitch live integrations."
        await self.broadcast_status()
        logger.info("Stopping live integrations")
        if self._status_task:
            self._status_task.cancel()
        for task in self._tasks:
            task.cancel()
        if self._status_task:
            await asyncio.gather(self._status_task, return_exceptions=True)
            self._status_task = None
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self.chat_bot:
            await self.chat_bot.shutdown()
        if self.eventsub:
            await self.eventsub.stop()
        self.chat_bot = None
        self.eventsub = None
        self._started = False
        self._status_override = "disconnected" if self.enabled else "disabled"
        self._status_detail = (
            "Twitch live integrations stopped."
            if self.enabled
            else "Twitch integration disabled in live_config."
        )
        await self.broadcast_status()

    async def _persist_tokens(self, access_token: str, refresh_token: str) -> None:
        self.cfg.twitch.oauth_token = access_token
        self.cfg.twitch.refresh_token = refresh_token
        self.token_store.save(
            TwitchTokenBundle(
                access_token=access_token,
                refresh_token=refresh_token,
            )
        )
        logger.info(
            "Refreshed Twitch tokens saved to %s; conf.yaml was left unchanged.",
            self.token_store.display_path,
        )

    async def _run_chat_bot(self) -> None:
        if not self.chat_bot:
            return
        try:
            await self.chat_bot.run_forever()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._status_override = "error"
            self._status_detail = f"Twitch chat error: {exc}"
            await self.broadcast_status()
            raise

    async def _run_eventsub(self) -> None:
        if not self.eventsub:
            return
        try:
            await self.eventsub.run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("EventSub runtime failed: %s", exc)
            await self.broadcast_status()
            raise

    async def _status_loop(self) -> None:
        try:
            while True:
                await self.broadcast_status()
                await asyncio.sleep(2)
        except asyncio.CancelledError:
            pass

    async def _token_validation_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(TOKEN_VALIDATION_INTERVAL_SECONDS)
                if not self.chat_bot:
                    continue
                try:
                    await self.chat_bot.refresh_credentials()
                except TokenRefreshError as exc:
                    self._status_override = "error"
                    self._status_detail = (
                        f"Twitch token validation failed: {exc}"
                    )
                    await self.broadcast_status()
        except asyncio.CancelledError:
            pass

    async def begin_twitch_authorization(self, *, force_verify: bool = True) -> str:
        authorize_url = self.oauth.build_authorize_url(force_verify=force_verify)
        self._status_detail = "Waiting for Twitch authorization in the browser."
        await self.broadcast_status()
        return authorize_url

    async def record_twitch_authorization_failure(self, detail: str) -> None:
        self.oauth.clear_pending()
        self._status_detail = detail
        await self.broadcast_status()

    async def complete_twitch_authorization(
        self,
        *,
        code: str,
        state: str,
    ) -> dict:
        result = await self.oauth.complete_authorization(code=code, state=state)
        if self.chat_bot:
            self.chat_bot._sync_twitchio_token_state()

        self._status_override = None
        self._status_detail = (
            f"Twitch authorization completed. Tokens saved to {self.token_store.display_path}."
        )

        if self.enabled and self._started:
            await self.stop()
            await self.start()
        else:
            await self.broadcast_status()
        return result

    def get_status_messages(self) -> list[dict]:
        return [self._build_twitch_status_payload()]

    async def broadcast_status(self) -> None:
        payload = self._build_twitch_status_payload()
        signature = json.dumps(payload, sort_keys=True)
        if signature == self._last_status_signature:
            return
        self._last_status_signature = signature
        await self.ws_handler.broadcast_json(payload)

    def _build_twitch_status_payload(self) -> dict:
        authenticated = bool((self.cfg.twitch.oauth_token or "").strip())
        if not self.enabled:
            status = "disabled"
            detail = self._status_detail or "Twitch integration disabled in live_config."
        elif not self.cfg.twitch.client_id or not self.cfg.twitch.channel:
            status = "unconfigured"
            detail = self._status_detail or "Twitch client credentials or channel are incomplete."
        elif not authenticated:
            status = "unauthenticated"
            detail = self._status_detail or "No Twitch OAuth token configured."
        elif self._status_override == "error":
            status = "error"
            detail = self._status_detail or "Twitch integration encountered an error."
        elif self._status_override == "stopping":
            status = "stopping"
            detail = self._status_detail or "Stopping Twitch live integrations."
        elif self.chat_bot and self.chat_bot.is_connected():
            status = "connected"
            detail = "Connected to Twitch chat."
        elif self._started:
            status = "connecting"
            detail = self._status_detail or "Connecting to Twitch chat."
        else:
            status = "disconnected"
            detail = self._status_detail or "Twitch integration is idle."

        broadcaster_id = ""
        if self.chat_bot and self.chat_bot.api.broadcaster_id:
            broadcaster_id = self.chat_bot.api.broadcaster_id

        talkback_tts = self.publisher.get_talkback_tts_status()
        return {
            "type": "live/twitch/status",
            "enabled": self.enabled,
            "status": status,
            "authenticated": authenticated,
            "channel": self.cfg.twitch.channel,
            "broadcaster_id": broadcaster_id,
            "talkback_enabled": bool(self.cfg.talkback.enabled),
            "read_chat_aloud": bool(talkback_tts["enabled"]),
            "chat_tts_volume": 1,
            "notify_subscriptions": bool(
                self.cfg.talkback.notifications.announce_subscriptions
            ),
            "notify_first_observed_chatters": bool(
                self.cfg.talkback.notifications.announce_low_viewer_joins
            ),
            "first_observed_chatter_viewer_threshold": int(
                self.cfg.talkback.notifications.low_viewer_threshold
            ),
            "redemptions_enabled": bool(self.cfg.channel_points.enabled),
            "self_moderation_enabled": bool(self.cfg.moderation.enabled),
            "debug": bool(self.cfg.twitch.debug or self.cfg.talkback.debug),
            "restart_required_fields": [],
            "reauth_available": bool(self.oauth.can_start),
            "auth_flow_pending": bool(self.oauth.auth_flow_pending),
            "oauth_redirect_uri": self.oauth.redirect_uri,
            "token_storage": (
                self.token_store.display_path
                if self.token_store.exists()
                else "conf.yaml"
            ),
            "detail": detail,
        }


def load_live_runtime(
    config_path: str,
    ws_handler: WebSocketHandler,
    backend_base_url: str,
) -> LiveRuntime:
    """Utility helper to load live configuration and create a runtime manager."""
    cfg = load_conf(config_path)
    return LiveRuntime(cfg, ws_handler, backend_base_url)
