from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .schema import ConfigValidationError, validate_conf


@dataclass
class EventSubConf:
    transport: str
    webhook_public_base: Optional[str]
    enabled: bool


@dataclass
class TwitchConf:
    enabled: bool
    debug: bool
    client_id: str
    client_secret: str
    broadcaster_login: str
    channel: str
    oauth_token: str
    refresh_token: str
    scopes: List[str]
    eventsub: EventSubConf


@dataclass
class ModConf:
    enabled: bool
    decay_hours: int
    warn_text: str
    timeout_sequence_seconds: List[int]
    filters: Dict[str, Any]


@dataclass
class BridgeConf:
    ws_host: str
    ws_port: int


@dataclass
class LogConf:
    dir: str
    level: str
    rotate_when: str
    backup_count: int


@dataclass
class SandboxConf:
    enabled: bool
    interval_seconds: int


@dataclass
class ChannelPointsConf:
    enabled: bool
    actions: Dict[str, Dict[str, Any]]


@dataclass
class TalkbackRateLimitConf:
    max_replies_per_minute: int
    max_forwards_per_minute: int


@dataclass
class TalkbackContextConf:
    last_n_messages: int


@dataclass
class TalkbackSanitizeConf:
    strip_links: bool
    strip_mentions: bool
    strip_badwords: bool


@dataclass
class TalkbackAcknowledgementsConf:
    enabled: bool
    template: str


@dataclass
class TalkbackNotificationsConf:
    announce_subscriptions: bool
    announce_low_viewer_joins: bool
    low_viewer_threshold: int


@dataclass
class TalkbackTTSConf:
    enabled: bool
    engine: str
    voice: str


@dataclass
class TalkbackConf:
    enabled: bool
    max_chars: int
    rate_limit: TalkbackRateLimitConf
    context: TalkbackContextConf
    sanitize: TalkbackSanitizeConf
    expressions_allowed: List[str]
    acknowledgements: TalkbackAcknowledgementsConf
    twitch_reply_mirror_full_text: bool
    language: str
    ignore_commands: bool
    debug: bool
    notifications: TalkbackNotificationsConf
    tts: TalkbackTTSConf


@dataclass
class AppConf:
    twitch: TwitchConf
    moderation: ModConf
    bridge: BridgeConf
    logging: LogConf
    sandbox: SandboxConf
    channel_points: ChannelPointsConf
    talkback: TalkbackConf
    config_path: Path


def _default_live_config() -> Dict[str, Any]:
    return {
        "twitch": {
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
        },
        "moderation": {
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
        },
        "bridge": {
            "ws_host": "127.0.0.1",
            "ws_port": 9876,
        },
        "logging": {
            "dir": "logs/live",
            "level": "INFO",
            "rotate_when": "midnight",
            "backup_count": 7,
        },
        "sandbox": {
            "enabled": False,
            "interval_seconds": 60,
        },
        "channel_points": {
            "enabled": False,
            "actions": {},
        },
        "talkback": {
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
        },
        "bilibili_live": {
            "room_ids": [1991478060],
            "sessdata": "",
        },
    }


def _merge_missing(target: Dict[str, Any], defaults: Dict[str, Any]) -> None:
    for key, value in defaults.items():
        if key not in target:
            target[key] = deepcopy(value)
            continue
        current = target[key]
        if isinstance(current, dict) and isinstance(value, dict):
            _merge_missing(current, value)


def _normalize_legacy_talkback_config(live_cfg: Dict[str, Any]) -> None:
    talkback = live_cfg.get("talkback")
    if not isinstance(talkback, dict):
        return

    legacy_llm = talkback.pop("llm", None)
    if "max_chars" in talkback:
        return
    if not isinstance(legacy_llm, dict):
        return

    legacy_max_chars = legacy_llm.get("max_chars")
    if isinstance(legacy_max_chars, int):
        talkback["max_chars"] = legacy_max_chars


def load_conf(path: str | Path = "config.yaml") -> AppConf:
    """Load configuration file and return structured configuration."""
    primary = Path(path)
    if not primary.exists():
        fallback = Path("conf.yaml")
        if primary != fallback and fallback.exists():
            primary = fallback
        else:
            raise FileNotFoundError(
                f"Configuration file not found at {primary.resolve()}"
            )

    with primary.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ConfigValidationError("Configuration root must be a mapping.")

    normalized = deepcopy(raw)
    live_cfg = normalized.setdefault("live_config", {})
    if not isinstance(live_cfg, dict):
        raise ConfigValidationError("live_config must be a mapping.")

    _normalize_legacy_talkback_config(live_cfg)
    _merge_missing(live_cfg, _default_live_config())

    validated = validate_conf(normalized)

    live = validated["live_config"]

    twitch_raw = live["twitch"]
    twitch_eventsub = twitch_raw["eventsub"]
    channel_points = live.get("channel_points", {})
    talkback = live["talkback"]
    moderation_raw = live["moderation"]
    bridge_raw = live["bridge"]
    logging_raw = live["logging"]
    sandbox_raw = live["sandbox"]

    twitch_conf = TwitchConf(
        enabled=bool(twitch_raw["enabled"]),
        debug=bool(twitch_raw.get("debug", False)),
        client_id=str(twitch_raw["client_id"]),
        client_secret=str(twitch_raw["client_secret"]),
        broadcaster_login=str(twitch_raw["broadcaster_login"]),
        channel=str(twitch_raw["channel"]),
        oauth_token=str(twitch_raw["oauth_token"]),
        refresh_token=str(twitch_raw["refresh_token"]),
        scopes=list(twitch_raw["scopes"]),
        eventsub=EventSubConf(
            transport=twitch_eventsub["transport"],
            webhook_public_base=twitch_eventsub.get("webhook_public_base"),
            enabled=bool(twitch_eventsub["enabled"]),
        ),
    )

    mod_conf = ModConf(
        enabled=bool(moderation_raw["enabled"]),
        decay_hours=int(moderation_raw["decay_hours"]),
        warn_text=str(moderation_raw["warn_text"]),
        timeout_sequence_seconds=list(moderation_raw["timeout_sequence_seconds"]),
        filters=dict(moderation_raw["filters"]),
    )

    bridge_conf = BridgeConf(
        ws_host=str(bridge_raw["ws_host"]),
        ws_port=int(bridge_raw["ws_port"]),
    )

    log_conf = LogConf(
        dir=str(logging_raw["dir"]),
        level=str(logging_raw["level"]).upper(),
        rotate_when=str(logging_raw["rotate_when"]),
        backup_count=int(logging_raw["backup_count"]),
    )

    sandbox_conf = SandboxConf(
        enabled=bool(sandbox_raw["enabled"]),
        interval_seconds=int(sandbox_raw["interval_seconds"]),
    )

    channel_points_conf = ChannelPointsConf(
        enabled=bool(channel_points.get("enabled", False)),
        actions=dict(channel_points.get("actions", {})),
    )

    tts_cfg = talkback.get("tts", {})
    talkback_conf = TalkbackConf(
        enabled=bool(talkback["enabled"]),
        max_chars=int(talkback["max_chars"]),
        rate_limit=TalkbackRateLimitConf(
            max_replies_per_minute=int(
                talkback["rate_limit"]["max_replies_per_minute"]
            ),
            max_forwards_per_minute=int(
                talkback["rate_limit"]["max_forwards_per_minute"]
            ),
        ),
        context=TalkbackContextConf(
            last_n_messages=int(talkback["context"]["last_n_messages"])
        ),
        sanitize=TalkbackSanitizeConf(
            strip_links=bool(talkback["sanitize"]["strip_links"]),
            strip_mentions=bool(talkback["sanitize"]["strip_mentions"]),
            strip_badwords=bool(talkback["sanitize"]["strip_badwords"]),
        ),
        expressions_allowed=list(talkback["expressions"]["allowed"]),
        acknowledgements=TalkbackAcknowledgementsConf(
            enabled=bool(talkback["acknowledgements"]["enabled"]),
            template=str(talkback["acknowledgements"]["template"]),
        ),
        twitch_reply_mirror_full_text=bool(
            talkback["twitch_reply"]["mirror_full_text"]
        ),
        language=str(talkback["language"]),
        ignore_commands=bool(talkback["ignore_commands"]),
        debug=bool(talkback.get("debug", False)),
        notifications=TalkbackNotificationsConf(
            announce_subscriptions=bool(
                talkback["notifications"]["announce_subscriptions"]
            ),
            announce_low_viewer_joins=bool(
                talkback["notifications"]["announce_low_viewer_joins"]
            ),
            low_viewer_threshold=int(
                talkback["notifications"]["low_viewer_threshold"]
            ),
        ),
        tts=TalkbackTTSConf(
            enabled=bool(tts_cfg.get("enabled", False)),
            engine=str(tts_cfg.get("engine", "edge_tts")),
            voice=str(tts_cfg.get("voice", "en-US-AvaMultilingualNeural")),
        ),
    )

    return AppConf(
        twitch=twitch_conf,
        moderation=mod_conf,
        bridge=bridge_conf,
        logging=log_conf,
        sandbox=sandbox_conf,
        channel_points=channel_points_conf,
        talkback=talkback_conf,
        config_path=primary,
    )


__all__ = [
    "AppConf",
    "BridgeConf",
    "ChannelPointsConf",
    "ConfigValidationError",
    "EventSubConf",
    "LogConf",
    "ModConf",
    "SandboxConf",
    "TalkbackConf",
    "TalkbackAcknowledgementsConf",
    "TalkbackContextConf",
    "TalkbackNotificationsConf",
    "TalkbackRateLimitConf",
    "TalkbackSanitizeConf",
    "TalkbackTTSConf",
    "TwitchConf",
    "load_conf",
]
