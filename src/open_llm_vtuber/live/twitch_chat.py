from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from typing import Any, Awaitable, Callable, Dict, Optional, Set, Tuple

import aiohttp
from twitchio import Message
from twitchio.ext import commands

from .config import TalkbackConf, TwitchConf
from .moderation import LINK_RE, ModEngine
from .publisher import LivePublisher

VALIDATE_URL = "https://id.twitch.tv/oauth2/validate"
TOKEN_URL = "https://id.twitch.tv/oauth2/token"
REFRESH_THRESHOLD_SECONDS = 1800  # refresh if <30 minutes remain


class TokenRefreshError(Exception):
    """Raised when the Twitch token cannot be refreshed."""


def _clean_logger_message(token: str) -> str:
    sample = token[:6] + "..." if token else "(empty)"
    return sample


async def ensure_fresh_user_token(
    cfg: TwitchConf,
    logger: logging.Logger,
    threshold_seconds: int = REFRESH_THRESHOLD_SECONDS,
    persist_callback: Optional[Callable[[str, str], Awaitable[None] | None]] = None,
) -> Dict[str, Any]:
    """
    Validate cfg.oauth_token; if invalid or expiring soon, refresh using cfg.refresh_token.
    Mutates cfg.oauth_token / cfg.refresh_token in-place on success and returns the validation payload.
    """

    loop = asyncio.get_running_loop()

    def _validate() -> Optional[Dict[str, Any]]:
        token = (cfg.oauth_token or "").strip()
        if not token:
            logger.warning("No oauth_token provided; skipping validation/refresh.")
            return None

        import requests

        try:
            r = requests.get(
                VALIDATE_URL,
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            if r.status_code != 200:
                raise TokenRefreshError(f"validate:{r.status_code} {r.text}")
            return r.json()
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning(
                "Token validation failed for %s: %s", _clean_logger_message(token), exc
            )
            return None

    validation = await loop.run_in_executor(None, _validate)

    if validation:
        expires_in = int(validation.get("expires_in", 0))
        if expires_in > threshold_seconds:
            logger.debug(
                "Token valid (expires_in=%ss) - no refresh needed.", expires_in
            )
            return validation
        logger.info("Token expiring soon (expires_in=%ss); refreshing.", expires_in)

    # Attempt refresh using requests in thread executor to avoid blocking the event loop.
    import requests

    if not (cfg.client_id and cfg.client_secret and cfg.refresh_token):
        raise TokenRefreshError(
            "Cannot refresh Twitch token: missing client_id/client_secret/refresh_token."
        )

    def _refresh() -> Optional[Dict[str, Any]]:
        r = requests.post(
            TOKEN_URL,
            data={
                "client_id": cfg.client_id,
                "client_secret": cfg.client_secret,
                "grant_type": "refresh_token",
                "refresh_token": cfg.refresh_token,
            },
            timeout=15,
        )
        if r.status_code != 200:
            raise TokenRefreshError(f"refresh:{r.status_code} {r.text}")
        refreshed = r.json()
        return refreshed

    try:
        refreshed = await loop.run_in_executor(None, _refresh)
    except Exception as exc:  # pragma: no cover - network failure
        raise TokenRefreshError(f"Token refresh failed: {exc}") from exc

    if not refreshed:
        raise TokenRefreshError("Token refresh returned empty payload.")

    new_access = refreshed.get("access_token")
    new_refresh = refreshed.get("refresh_token") or cfg.refresh_token
    if not new_access:
        raise TokenRefreshError(
            f"Token refresh payload missing access_token: {refreshed}"
        )

    cfg.oauth_token = new_access
    cfg.refresh_token = new_refresh
    logger.info(
        "Refreshed Twitch OAuth token (prefix=%s).", _clean_logger_message(new_access)
    )

    if persist_callback:
        result = persist_callback(new_access, new_refresh)
        if asyncio.iscoroutine(result):
            await result

    validation = await loop.run_in_executor(None, _validate)
    if not validation:
        raise TokenRefreshError("Unable to validate refreshed access token.")
    return validation


class PerMinuteLimiter:
    """Simple token bucket that limits actions per rolling minute."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max(0, max_per_minute)
        self._history: list[float] = []

    def _prune(self) -> None:
        now = time.monotonic()
        self._history = [ts for ts in self._history if now - ts < 60]

    def acquire(self) -> bool:
        self._prune()
        if self.max_per_minute <= 0:
            return False
        if len(self._history) >= self.max_per_minute:
            return False
        self._history.append(time.monotonic())
        return True

    @property
    def remaining(self) -> int:
        self._prune()
        return max(self.max_per_minute - len(self._history), 0)


class TwitchAPIClient:
    BASE_URL = "https://api.twitch.tv/helix"

    def __init__(self, cfg: TwitchConf, logger: logging.Logger):
        self.cfg = cfg
        self.log = logger.getChild("api")
        self.session: Optional[aiohttp.ClientSession] = None
        self.broadcaster_id: Optional[str] = None
        self.moderator_id: Optional[str] = None

    async def ensure_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.cfg.oauth_token}",
            "Client-Id": self.cfg.client_id,
            "Content-Type": "application/json",
        }

    async def fetch_user(self, login: str) -> Optional[Dict[str, Any]]:
        session = await self.ensure_session()
        params = {"login": login}
        async with session.get(
            f"{self.BASE_URL}/users", params=params, headers=self._headers()
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                self.log.warning(
                    "Failed to fetch user %s: %s %s", login, resp.status, text
                )
                return None
            data = await resp.json()
            items = data.get("data", [])
            return items[0] if items else None

    async def resolve_broadcaster(self, login: str) -> None:
        user = await self.fetch_user(login)
        if user:
            self.broadcaster_id = user.get("id")
        else:
            self.log.warning("Unable to resolve broadcaster id for %s", login)

    async def resolve_moderator(self, login: str) -> None:
        user = await self.fetch_user(login)
        if user:
            self.moderator_id = user.get("id")
        else:
            self.log.warning("Unable to resolve moderator id for %s", login)

    async def _moderation_endpoint(
        self, payload: Dict[str, Any], *, ban: bool = False
    ) -> bool:
        if not (self.broadcaster_id and self.moderator_id):
            self.log.debug(
                "Moderation endpoint skipped: broadcaster or moderator id missing."
            )
            return False
        session = await self.ensure_session()
        params = {
            "broadcaster_id": self.broadcaster_id,
            "moderator_id": self.moderator_id,
        }
        endpoint = f"{self.BASE_URL}/moderation/bans"
        try:
            async with session.post(
                endpoint, params=params, headers=self._headers(), json=payload
            ) as resp:
                if resp.status not in (200, 204):
                    text = await resp.text()
                    self.log.warning(
                        "Moderation request failed: %s %s", resp.status, text
                    )
                    return False
        except Exception as exc:  # pragma: no cover - network failure
            self.log.error("Moderation request raised %s", exc)
            return False
        return True

    async def timeout(self, user_id: str, duration: int, reason: str) -> bool:
        payload = {
            "data": {"user_id": user_id, "duration": duration, "reason": reason[:500]}
        }
        return await self._moderation_endpoint(payload)

    async def ban(self, user_id: str, reason: str) -> bool:
        payload = {"data": {"user_id": user_id, "reason": reason[:500]}}
        return await self._moderation_endpoint(payload, ban=True)

    async def close(self) -> None:
        if self.session and not self.session.closed:
            await self.session.close()
        self.session = None


class ChatBot(commands.Bot):
    SEND_RATE_WINDOW = 30  # seconds
    SEND_RATE_COUNT = 15

    def __init__(
        self,
        cfg: TwitchConf,
        mod_engine: ModEngine,
        publisher: LivePublisher,
        talkback_conf: TalkbackConf,
        persist_tokens: Optional[Callable[[str, str], Awaitable[None] | None]] = None,
    ):
        normalized_channel = cfg.channel.lstrip("#")
        super().__init__(
            token=cfg.oauth_token, prefix="!", initial_channels=[normalized_channel]
        )

        self.cfg = cfg
        self.mod = mod_engine
        self.publisher = publisher
        self._persist_tokens_cb = persist_tokens
        self.talkback_cfg = talkback_conf

        self.log = logging.getLogger("twitch.chat")
        if cfg.debug:
            self.log.setLevel(logging.DEBUG)
        else:
            self.log.setLevel(logging.WARNING)

        self._ready = asyncio.Event()
        self._primary_channel: Optional[Any] = None
        self._send_queue: asyncio.Queue[Tuple[Any, str]] = asyncio.Queue()
        self._sender_task: Optional[asyncio.Task] = None
        self._token_refresh_lock = asyncio.Lock()

        self._forward_limiter = PerMinuteLimiter(
            talkback_conf.rate_limit.max_forwards_per_minute
        )
        self._system_forward_limiter = PerMinuteLimiter(
            talkback_conf.rate_limit.max_forwards_per_minute
        )
        self._talkback_enabled = bool(talkback_conf.enabled)
        self._sanitize_mentions = talkback_conf.sanitize.strip_mentions
        self._sanitize_links = talkback_conf.sanitize.strip_links
        self._sanitize_badwords = talkback_conf.sanitize.strip_badwords
        self._badword_list = [
            word
            for word in self.mod.cfg.filters.get("badwords", [])
            if isinstance(word, str) and word
        ]
        self._viewer_threshold = talkback_conf.notifications.low_viewer_threshold
        self._announce_joins = talkback_conf.notifications.announce_low_viewer_joins
        self._announce_subscriptions = (
            talkback_conf.notifications.announce_subscriptions
        )
        self._active_viewers: Set[str] = set()

        self._token_info: Optional[Dict[str, Any]] = None
        self.api = TwitchAPIClient(cfg, self.log)

    def _sync_twitchio_token_state(self) -> None:
        """Propagate a refreshed OAuth token into TwitchIO's cached clients."""
        token = (self.cfg.oauth_token or "").replace("oauth:", "").strip()
        if not token:
            return

        http_client = getattr(self, "_http", None)
        if http_client is not None:
            try:
                http_client.token = token
            except Exception:
                pass

        connection = getattr(self, "_connection", None)
        if connection is not None:
            try:
                connection._token = token  # type: ignore[attr-defined]
            except Exception:
                pass

    # ------------------------------------------------------------------ Lifecycle

    async def run_forever(self) -> None:
        self.log.debug("Starting Twitch chat bot.")
        try:
            self._token_info = await self.refresh_credentials()
        except TokenRefreshError as exc:
            self.log.error("Unable to validate Twitch credentials: %s", exc)
            raise
        if self.cfg.broadcaster_login:
            await self.api.resolve_broadcaster(self.cfg.broadcaster_login.lower())
        if self._token_info and self._token_info.get("login"):
            await self.api.resolve_moderator(self._token_info["login"])
        elif self.cfg.broadcaster_login:
            await self.api.resolve_moderator(self.cfg.broadcaster_login.lower())

        self._sync_twitchio_token_state()

        try:
            await self.start()
        except Exception as exc:
            self.log.exception("Twitch bot encountered an unrecoverable error: %s", exc)
            raise

    async def shutdown(self) -> None:
        self.log.debug("Shutting down Twitch bot.")
        if self._sender_task:
            self._sender_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sender_task
        await self.api.close()
        connection = getattr(self, "_connection", None)
        keeper = getattr(connection, "_keeper", None) if connection else None
        if connection is not None and keeper is None:
            # TwitchIO close() assumes _keeper exists; provide a cancellable placeholder task.
            try:
                placeholder = connection._loop.create_future()  # type: ignore[attr-defined]
            except Exception:
                placeholder = asyncio.get_running_loop().create_future()
            connection._keeper = placeholder  # type: ignore[attr-defined]
        await super().close()

    async def wait_until_ready(self) -> None:
        await self._ready.wait()

    def is_connected(self) -> bool:
        return self._ready.is_set()

    async def refresh_credentials(
        self,
        *,
        threshold_seconds: int = REFRESH_THRESHOLD_SECONDS,
    ) -> Dict[str, Any]:
        """Validate or refresh the Twitch OAuth session and sync TwitchIO state."""
        async with self._token_refresh_lock:
            token_info = await ensure_fresh_user_token(
                self.cfg,
                self.log,
                threshold_seconds=threshold_seconds,
                persist_callback=self._persist_tokens_cb,
            )
            self._token_info = token_info
            self._sync_twitchio_token_state()
            return token_info

    def get_talkback_status(self) -> Dict[str, Any]:
        return {
            "enabled": self._talkback_enabled,
            "forward_rate": {
                "max_per_minute": self._forward_limiter.max_per_minute,
                "remaining": self._forward_limiter.remaining,
            },
            "announce_subscriptions": self._announce_subscriptions,
            "announce_low_viewer_joins": self._announce_joins,
        }

    async def set_talkback_enabled(self, enabled: bool) -> None:
        self._talkback_enabled = bool(enabled)
        self.log.debug("Talkback enabled set to %s", self._talkback_enabled)

    # ------------------------------------------------------------------ Sending utilities

    async def enqueue_message(self, text: str, channel: Optional[Any] = None) -> None:
        if not text:
            return
        if channel is None:
            if not self._primary_channel:
                self._primary_channel = self.get_channel(
                    self.cfg.channel
                ) or self.get_channel(self.cfg.channel.lstrip("#"))
            channel = self._primary_channel
        if channel is None:
            self.log.debug("Dropping outbound message, no channel available: %s", text)
            return
        await self._send_queue.put((channel, text))

    async def _sender_loop(self) -> None:
        window: list[float] = []
        try:
            while True:
                channel, text = await self._send_queue.get()
                if not channel:
                    self._send_queue.task_done()
                    continue
                while True:
                    now = time.monotonic()
                    window = [ts for ts in window if now - ts < self.SEND_RATE_WINDOW]
                    if len(window) < self.SEND_RATE_COUNT:
                        break
                    await asyncio.sleep(0.25)
                try:
                    await channel.send(text)
                    window.append(time.monotonic())
                except Exception as exc:  # pragma: no cover - network failure
                    self.log.warning("Failed to send Twitch message: %s", exc)
                finally:
                    self._send_queue.task_done()
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------ Event handlers

    async def event_ready(self) -> None:
        self.log.debug("Connected to Twitch chat as %s", self.nick)
        if not self._sender_task:
            self._sender_task = asyncio.create_task(
                self._sender_loop(), name="twitch-sender"
            )
        self._primary_channel = self.get_channel(self.cfg.channel) or self.get_channel(
            self.cfg.channel.lstrip("#")
        )
        self._ready.set()

    async def event_message(self, message: Message) -> None:
        if message.echo:
            return

        user = (message.author.name if message.author else "unknown").lower()
        if message.author and message.author.is_mod:
            self._active_viewers.add(user)
        if message.content:
            await self._handle_chat_message(message)

    async def event_join(self, channel, user):  # type: ignore[override]
        username = getattr(user, "name", str(user))
        normalized = username.lower()
        if normalized == self.nick.lower():
            return
        self._active_viewers.add(normalized)
        if (
            self._talkback_enabled
            and self._announce_joins
            and len(self._active_viewers) <= self._viewer_threshold
        ):
            notification = f"(Twitch) {username} joined the stream."
            await self._emit_system_notification(
                notification,
                metadata={"user": username, "event": "viewer_join"},
                forward_to_talkback=True,
            )

    async def event_part(  # type: ignore[override]
        self,
        channel: Optional[Any] = None,
        user: Optional[Any] = None,
        *args: Any,
        **_kwargs: Any,
    ) -> None:
        """
        TwitchIO has historically passed either (channel, user) or just (user) to event_part.
        Some edge-cases have been observed where no positional args are supplied at all.
        Normalise all of these into a single user object so we can maintain viewer state.
        """

        user_obj = user or channel
        if user_obj is None and args:
            user_obj = args[-1]
        if user_obj is None:
            return
        username = getattr(user_obj, "name", str(user_obj))
        normalized = (
            username.lower() if isinstance(username, str) else str(username).lower()
        )
        if normalized:
            self._active_viewers.discard(normalized)

    async def event_raw_usernotice(self, channel, tags):  # type: ignore[override]
        msg_id = tags.get("msg-id") if tags else None
        if not msg_id or not self._announce_subscriptions:
            return
        if msg_id in {"sub", "resub", "subgift", "anonsubgift"}:
            username = tags.get("login") or tags.get("display-name") or "someone"
            await self._emit_system_notification(
                f"(Twitch) {username} just subscribed!",
                metadata={"user": username, "event": msg_id},
                forward_to_talkback=True,
            )

    # ------------------------------------------------------------------ Helpers

    async def _emit_system_notification(
        self,
        text: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        forward_to_talkback: bool = False,
    ) -> None:
        self.log.debug("Emitting system notification: %s", text)
        await self.publisher.send_notification(
            text, category="system", metadata=metadata
        )
        if forward_to_talkback:
            await self._forward_system_notification(text, metadata)

    async def _forward_system_notification(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._talkback_enabled:
            return
        if not text:
            return
        if not self._system_forward_limiter.acquire():
            self.log.debug("Skipping system forward (rate limit reached).")
            return

        payload_meta: Dict[str, Any] = {
            "platform": "twitch",
            "source": "twitch",
            "category": "system",
            "message_kind": "notification",
            "notification": True,
            "assistant_recorded": True,
        }
        if metadata:
            payload_meta.update(metadata)

        await self.publisher.forward_to_backend(text, payload_meta)

    def _sanitize_text(self, text: str) -> str:
        sanitized = text.replace("\n", " ")
        if self._sanitize_links:
            sanitized = LINK_RE.sub("", sanitized)
        if self._sanitize_mentions:
            sanitized = re.sub(r"@\w+", "", sanitized)
        if self._sanitize_badwords and self._badword_list:
            for word in self._badword_list:
                if not word:
                    continue
                sanitized = re.sub(
                    re.escape(word), "***", sanitized, flags=re.IGNORECASE
                )
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        return sanitized

    async def _handle_chat_message(self, message: Message) -> None:
        user = message.author.name if message.author else "unknown"
        text = message.content or ""
        self.log.debug("Incoming message from %s: %s", user, text)

        violated, reason = self.mod.check(user, text)
        if violated:
            await self._apply_moderation(message, user, reason or "policy violation")
            return

        sanitized = self._sanitize_text(text)
        forward_text = f"(Twitch) {user}: {sanitized}"
        display_text = forward_text
        metadata = {
            "platform": "twitch",
            "message_id": getattr(message, "id", None),
            "message_kind": "chat",
            "spoken_text": sanitized,
            "user": user,
        }
        await self.publisher.send_chat(user, display_text, metadata=metadata)

        if self._talkback_enabled and self._forward_limiter.acquire():
            meta = {
                "source": "twitch",
                "user": user,
                "platform": "twitch",
                "message_id": getattr(message, "id", None),
                "message_kind": "chat",
            }
            await self.publisher.forward_to_backend(forward_text, meta)
        else:
            self.log.debug(
                "Skipping talkback forward (enabled=%s remaining=%s).",
                self._talkback_enabled,
                self._forward_limiter.remaining,
            )

    async def _apply_moderation(self, message: Message, user: str, reason: str) -> None:
        action, duration = self.mod.apply(user)
        self.log.warning(
            "Moderation action=%s duration=%s for %s", action, duration, user
        )
        if action == "warn":
            warn_text = self.mod.cfg.warn_text
            await self.enqueue_message(f"@{user} {warn_text}", message.channel)
            return

        user_id = getattr(message.author, "id", None)
        if not user_id:
            self.log.warning("Cannot moderate %s: missing user id.", user)
            return

        if action == "timeout" and duration:
            success = await self.api.timeout(user_id, duration, reason)
            if success:
                await self.enqueue_message(
                    f"/timeout {user} {duration}", message.channel
                )
        elif action == "ban":
            success = await self.api.ban(user_id, reason)
            if success:
                await self.enqueue_message(f"/ban {user}", message.channel)
