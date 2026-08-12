from __future__ import annotations

import asyncio
import json
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlencode

AUTHORIZE_URL = "https://id.twitch.tv/oauth2/authorize"
TOKEN_URL = "https://id.twitch.tv/oauth2/token"
VALIDATE_URL = "https://id.twitch.tv/oauth2/validate"
DEFAULT_TOKEN_STORE_PATH = Path("private") / "twitch_tokens.json"


class TwitchOAuthError(Exception):
    """Raised when a Twitch OAuth authorization step fails."""


@dataclass
class TwitchTokenBundle:
    access_token: str
    refresh_token: str
    expires_in: Optional[int] = None
    scope: Optional[list[str]] = None
    token_type: Optional[str] = None


class TwitchTokenStore:
    """Persist Twitch tokens outside conf.yaml in an untracked local file."""

    def __init__(self, path: str | Path = DEFAULT_TOKEN_STORE_PATH):
        self.path = Path(path)

    @property
    def display_path(self) -> str:
        try:
            return self.path.as_posix()
        except Exception:
            return str(self.path)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> Optional[TwitchTokenBundle]:
        if not self.path.exists():
            return None

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return None
        access_token = str(payload.get("oauth_token") or "").strip()
        refresh_token = str(payload.get("refresh_token") or "").strip()
        if not access_token or not refresh_token:
            return None

        raw_scope = payload.get("scope")
        scope = [str(item) for item in raw_scope] if isinstance(raw_scope, list) else None
        expires_in = payload.get("expires_in")
        return TwitchTokenBundle(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=int(expires_in) if isinstance(expires_in, int) else None,
            scope=scope,
            token_type=str(payload.get("token_type")) if payload.get("token_type") else None,
        )

    def save(self, bundle: TwitchTokenBundle) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "oauth_token": bundle.access_token,
            "refresh_token": bundle.refresh_token,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if bundle.expires_in is not None:
            payload["expires_in"] = bundle.expires_in
        if bundle.scope is not None:
            payload["scope"] = list(bundle.scope)
        if bundle.token_type:
            payload["token_type"] = bundle.token_type

        self.path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )


class TwitchOAuthManager:
    """Manage Twitch OAuth authorization-code flow for the live runtime."""

    def __init__(
        self,
        cfg,
        token_store: TwitchTokenStore,
        backend_base_url: str,
        logger: Optional[logging.Logger] = None,
    ):
        self.cfg = cfg
        self.token_store = token_store
        self.backend_base_url = backend_base_url.rstrip("/")
        self.logger = logger or logging.getLogger("live.oauth")
        self._pending_state: Optional[str] = None
        self._pending_started_at: Optional[str] = None

    @property
    def redirect_uri(self) -> str:
        return f"{self.backend_base_url}/twitch/callback"

    @property
    def auth_flow_pending(self) -> bool:
        return bool(self._pending_state)

    @property
    def can_start(self) -> bool:
        return bool(
            (self.cfg.client_id or "").strip()
            and (self.cfg.client_secret or "").strip()
        )

    def build_authorize_url(self, *, force_verify: bool = True) -> str:
        if not self.can_start:
            raise TwitchOAuthError(
                "Twitch OAuth requires both client_id and client_secret in live_config.twitch."
            )

        state = secrets.token_urlsafe(24)
        self._pending_state = state
        self._pending_started_at = datetime.now(timezone.utc).isoformat()

        params = {
            "client_id": self.cfg.client_id,
            "force_verify": "true" if force_verify else "false",
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.cfg.scopes),
            "state": state,
        }
        return f"{AUTHORIZE_URL}?{urlencode(params)}"

    def clear_pending(self) -> None:
        self._pending_state = None
        self._pending_started_at = None

    def _exchange_code_sync(self, code: str) -> TwitchTokenBundle:
        import requests

        response = requests.post(
            TOKEN_URL,
            data={
                "client_id": self.cfg.client_id,
                "client_secret": self.cfg.client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": self.redirect_uri,
            },
            timeout=20,
        )
        if response.status_code != 200:
            raise TwitchOAuthError(
                f"Twitch OAuth token exchange failed: {response.status_code} {response.text}"
            )

        payload = response.json()
        access_token = str(payload.get("access_token") or "").strip()
        refresh_token = str(payload.get("refresh_token") or "").strip()
        if not access_token or not refresh_token:
            raise TwitchOAuthError("Twitch OAuth response did not include both access and refresh tokens.")

        raw_scope = payload.get("scope")
        scope = [str(item) for item in raw_scope] if isinstance(raw_scope, list) else None
        expires_in = payload.get("expires_in")
        return TwitchTokenBundle(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=int(expires_in) if isinstance(expires_in, int) else None,
            scope=scope,
            token_type=str(payload.get("token_type")) if payload.get("token_type") else None,
        )

    def _validate_token_sync(self, access_token: str) -> Dict[str, Any]:
        import requests

        response = requests.get(
            VALIDATE_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=15,
        )
        if response.status_code != 200:
            raise TwitchOAuthError(
                f"Twitch OAuth token validation failed: {response.status_code} {response.text}"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise TwitchOAuthError("Twitch token validation returned malformed data.")
        return payload

    async def complete_authorization(self, *, code: str, state: str) -> Dict[str, Any]:
        if not self._pending_state or state != self._pending_state:
            raise TwitchOAuthError(
                "Twitch OAuth state mismatch. Start the authorization flow again from the app."
            )

        loop = asyncio.get_running_loop()
        bundle = await loop.run_in_executor(None, self._exchange_code_sync, code)
        validation = await loop.run_in_executor(
            None,
            self._validate_token_sync,
            bundle.access_token,
        )

        self.cfg.oauth_token = bundle.access_token
        self.cfg.refresh_token = bundle.refresh_token
        self.token_store.save(bundle)
        self.clear_pending()
        self.logger.info(
            "Stored Twitch OAuth tokens in %s after browser authorization.",
            self.token_store.display_path,
        )
        return {
            "bundle": bundle,
            "validation": validation,
        }
