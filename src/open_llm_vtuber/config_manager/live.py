from pydantic import Field, model_validator
from typing import Any, Dict, ClassVar, List

from .i18n import I18nMixin, Description


class BiliBiliLiveConfig(I18nMixin):
    """Configuration for BiliBili Live platform."""

    room_ids: List[int] = Field([], alias="room_ids")
    sessdata: str = Field("", alias="sessdata")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "room_ids": Description.from_str(
            "List of BiliBili live room IDs to monitor"
        ),
        "sessdata": Description.from_str(
            "SESSDATA cookie value for authenticated requests (optional)"
        ),
    }


def _default_twitch_config() -> Dict[str, Any]:
    return {
        "enabled": False,
        "debug": False,
        "client_id": "",
        "client_secret": "",
        "broadcaster_login": "",
        "channel": "",
        "oauth_token": "",
        "refresh_token": "",
        "scopes": [
            "chat:read",
            "chat:edit",
            "moderator:manage:banned_users",
            "channel:read:subscriptions",
        ],
        "eventsub": {
            "transport": "websocket",
            "webhook_public_base": "",
            "enabled": True,
        },
    }


def _default_moderation_config() -> Dict[str, Any]:
    return {
        "enabled": False,
        "decay_hours": 6,
        "warn_text": "Please keep chat friendly.",
        "timeout_sequence_seconds": [30, 600],
        "filters": {
            "badwords": [],
            "block_links": True,
            "block_slurs": True,
            "block_confusables": True,
            "block_mass_caps": True,
            "block_repeats": False,
        },
    }


def _default_bridge_config() -> Dict[str, Any]:
    return {
        "ws_host": "127.0.0.1",
        "ws_port": 9876,
    }


def _default_logging_config() -> Dict[str, Any]:
    return {
        "dir": "logs/live",
        "level": "INFO",
        "rotate_when": "midnight",
        "backup_count": 7,
    }


def _default_sandbox_config() -> Dict[str, Any]:
    return {
        "enabled": False,
        "interval_seconds": 60,
    }


def _default_channel_points_config() -> Dict[str, Any]:
    return {
        "enabled": False,
        "actions": {},
    }


def _default_talkback_config() -> Dict[str, Any]:
    return {
        "enabled": False,
        "max_chars": 320,
        "rate_limit": {
            "max_replies_per_minute": 6,
            "max_forwards_per_minute": 12,
        },
        "context": {
            "last_n_messages": 12,
        },
        "sanitize": {
            "strip_links": True,
            "strip_mentions": True,
            "strip_badwords": True,
        },
        "expressions": {
            "allowed": ["neutral", "happy", "angry"],
        },
        "acknowledgements": {
            "enabled": True,
            "template": "@{user} answered on stream",
        },
        "twitch_reply": {
            "mirror_full_text": False,
        },
        "language": "en",
        "ignore_commands": True,
        "debug": False,
        "notifications": {
            "announce_subscriptions": False,
            "announce_low_viewer_joins": False,
            "low_viewer_threshold": 5,
        },
        "tts": {
            "enabled": False,
            "engine": "edge_tts",
            "voice": "en-US-AvaMultilingualNeural",
        },
    }


class LiveConfig(I18nMixin):
    """Configuration for live streaming platforms integration."""

    twitch: Dict[str, Any] = Field(default_factory=_default_twitch_config, alias="twitch")
    moderation: Dict[str, Any] = Field(
        default_factory=_default_moderation_config,
        alias="moderation",
    )
    bridge: Dict[str, Any] = Field(
        default_factory=_default_bridge_config,
        alias="bridge",
    )
    logging: Dict[str, Any] = Field(
        default_factory=_default_logging_config,
        alias="logging",
    )
    sandbox: Dict[str, Any] = Field(
        default_factory=_default_sandbox_config,
        alias="sandbox",
    )
    channel_points: Dict[str, Any] = Field(
        default_factory=_default_channel_points_config,
        alias="channel_points",
    )
    talkback: Dict[str, Any] = Field(
        default_factory=_default_talkback_config,
        alias="talkback",
    )
    bilibili_live: BiliBiliLiveConfig = Field(
        BiliBiliLiveConfig(),
        alias="bilibili_live",
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "twitch": Description.from_str("Configuration for Twitch live integration"),
        "moderation": Description.from_str("Moderation rules for live chat"),
        "bridge": Description.from_str(
            "Bridge configuration for live runtime services"
        ),
        "logging": Description.from_str(
            "Logging settings for live runtime services"
        ),
        "sandbox": Description.from_str(
            "Sandbox controls for live runtime automation"
        ),
        "channel_points": Description.from_str(
            "Channel points reaction configuration"
        ),
        "talkback": Description.from_str(
            "Talkback and chat forwarding configuration"
        ),
        "bilibili_live": Description.from_str(
            "Configuration for BiliBili Live platform"
        ),
    }

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_talkback_llm(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        talkback = data.get("talkback")
        if not isinstance(talkback, dict):
            return data

        talkback = dict(talkback)
        legacy_llm = talkback.pop("llm", None)
        if "max_chars" not in talkback and isinstance(legacy_llm, dict):
            legacy_max_chars = legacy_llm.get("max_chars")
            if isinstance(legacy_max_chars, int):
                talkback["max_chars"] = legacy_max_chars

        data = dict(data)
        data["talkback"] = talkback
        return data
