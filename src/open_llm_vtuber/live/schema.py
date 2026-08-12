from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


class ConfigValidationError(Exception):
    """Raised when the configuration file fails validation."""


_BOOL_KEYS = (
    ("live_config", "twitch", "enabled"),
    ("live_config", "moderation", "enabled"),
    ("live_config", "moderation", "filters", "block_links"),
    ("live_config", "moderation", "filters", "block_slurs"),
    ("live_config", "moderation", "filters", "block_confusables"),
    ("live_config", "moderation", "filters", "block_mass_caps"),
    ("live_config", "moderation", "filters", "block_repeats"),
    ("live_config", "sandbox", "enabled"),
    ("live_config", "talkback", "enabled"),
    ("live_config", "talkback", "sanitize", "strip_links"),
    ("live_config", "talkback", "sanitize", "strip_mentions"),
    ("live_config", "talkback", "sanitize", "strip_badwords"),
    ("live_config", "talkback", "acknowledgements", "enabled"),
    ("live_config", "talkback", "twitch_reply", "mirror_full_text"),
    ("live_config", "talkback", "ignore_commands"),
    ("live_config", "talkback", "notifications", "announce_subscriptions"),
    ("live_config", "talkback", "notifications", "announce_low_viewer_joins"),
    ("live_config", "talkback", "tts", "enabled"),
)


def _get_path(data: Dict[str, Any], path: Sequence[str]) -> Any:
    node: Any = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(".".join(path))
        node = node[key]
    return node


def _ensure_type(value: Any, expected: Tuple[type, ...]) -> bool:
    return isinstance(value, expected)


def validate_conf(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration structure and return normalized copy."""
    errors: List[str] = []

    if not isinstance(data, dict):
        raise ConfigValidationError("Configuration root must be a mapping.")

    def require(path: Sequence[str], expected: Tuple[type, ...]) -> Any:
        try:
            value = _get_path(data, path)
        except KeyError:
            errors.append(f"Missing config key: {'.'.join(path)}")
            return None
        if not _ensure_type(value, expected):
            errors.append(
                f"Invalid type for {'.'.join(path)} (expected {expected}, got {type(value).__name__})"
            )
        return value

    def require_str(path: Sequence[str]) -> Optional[str]:
        value = require(path, (str,))
        if isinstance(value, str) and not value.strip():
            errors.append(f"Config value {'.'.join(path)} must not be empty.")
        return value if isinstance(value, str) else None

    def require_int(path: Sequence[str]) -> Optional[int]:
        value = require(path, (int,))
        return value if isinstance(value, int) else None

    def require_bool(path: Sequence[str]) -> Optional[bool]:
        value = require(path, (bool,))
        return value if isinstance(value, bool) else None

    live_cfg_obj = require(("live_config",), (dict,))
    live_cfg = live_cfg_obj if isinstance(live_cfg_obj, dict) else {}

    # Twitch
    twitch_cfg = live_cfg.get("twitch") if isinstance(live_cfg, dict) else None
    twitch_enabled = (
        bool(twitch_cfg.get("enabled")) if isinstance(twitch_cfg, dict) else False
    )
    for key in (
        "client_id",
        "client_secret",
        "broadcaster_login",
        "channel",
        "oauth_token",
        "refresh_token",
    ):
        if twitch_enabled:
            require_str(("live_config", "twitch", key))
        else:
            require(("live_config", "twitch", key), (str,))

    scopes = require(("live_config", "twitch", "scopes"), (list,))
    if isinstance(scopes, list):
        for idx, scope in enumerate(scopes):
            if not isinstance(scope, str):
                errors.append(f"twitch.scopes[{idx}] must be a string.")

    eventsub = require(("live_config", "twitch", "eventsub"), (dict,))
    if isinstance(eventsub, dict):
        transport = require_str(("live_config", "twitch", "eventsub", "transport"))
        if transport and transport not in {"websocket", "webhook"}:
            errors.append(
                "twitch.eventsub.transport must be either 'websocket' or 'webhook'."
            )
        require_bool(("live_config", "twitch", "eventsub", "enabled"))
        if transport == "webhook" and eventsub.get("enabled") is True:
            require_str(("live_config", "twitch", "eventsub", "webhook_public_base"))

    # Moderation
    require_bool(("live_config", "moderation", "enabled"))
    require_int(("live_config", "moderation", "decay_hours"))
    require_str(("live_config", "moderation", "warn_text"))
    seq = require(("live_config", "moderation", "timeout_sequence_seconds"), (list,))
    if isinstance(seq, list):
        for idx, value in enumerate(seq):
            if not isinstance(value, int) or value <= 0:
                errors.append(
                    f"moderation.timeout_sequence_seconds[{idx}] must be a positive integer."
                )
    filters = require(("live_config", "moderation", "filters"), (dict,))
    if isinstance(filters, dict):
        badwords = filters.get("badwords")
        if badwords is not None:
            if not isinstance(badwords, list):
                errors.append("moderation.filters.badwords must be a list of strings.")
            else:
                for idx, word in enumerate(badwords):
                    if not isinstance(word, str):
                        errors.append(
                            f"moderation.filters.badwords[{idx}] must be a string."
                        )
    for path in _BOOL_KEYS:
        require_bool(path)

    # Bridge
    require_str(("live_config", "bridge", "ws_host"))
    require_int(("live_config", "bridge", "ws_port"))

    # Logging
    require_str(("live_config", "logging", "dir"))
    require_str(("live_config", "logging", "level"))
    require_str(("live_config", "logging", "rotate_when"))
    require_int(("live_config", "logging", "backup_count"))

    # Sandbox
    interval = require_int(("live_config", "sandbox", "interval_seconds"))
    if isinstance(interval, int) and interval <= 0:
        errors.append("sandbox.interval_seconds must be greater than zero.")

    # Channel points (optional)
    channel_points: Any = None
    if isinstance(live_cfg, dict):
        channel_points = live_cfg.get("channel_points")
    if channel_points is None:
        if isinstance(live_cfg, dict):
            live_cfg["channel_points"] = {"enabled": False, "actions": {}}
    else:
        if not isinstance(channel_points, dict):
            errors.append("channel_points must be a mapping if provided.")
        else:
            enabled = channel_points.get("enabled")
            if enabled is None:
                channel_points["enabled"] = False
                enabled = False
            elif not isinstance(enabled, bool):
                errors.append("channel_points.enabled must be a boolean.")
            actions = channel_points.get("actions")
            if actions is None:
                channel_points["actions"] = {}
            elif not isinstance(actions, dict):
                errors.append("channel_points.actions must be a mapping.")
            else:
                for reward, payload in actions.items():
                    if not isinstance(reward, str):
                        errors.append("channel_points.actions keys must be strings.")
                        continue
                    if not isinstance(payload, dict):
                        errors.append(
                            f"channel_points.actions['{reward}'] must be a mapping."
                        )
                        continue
                    action_type = payload.get("type")
                    if action_type not in {"expression", "motion"}:
                        errors.append(
                            f"channel_points.actions['{reward}'].type must be 'expression' or 'motion'."
                        )
                    if action_type == "expression":
                        key = payload.get("key")
                        if not isinstance(key, str) or not key:
                            errors.append(
                                f"channel_points.actions['{reward}'].key must be a non-empty string."
                            )
                    if action_type == "motion":
                        group = payload.get("group")
                        if not isinstance(group, str) or not group:
                            errors.append(
                                f"channel_points.actions['{reward}'].group must be a non-empty string."
                            )

    # Talkback
    talkback = live_cfg.get("talkback") if isinstance(live_cfg, dict) else None
    if talkback is None:
        errors.append("Missing config key: live_config.talkback")
    elif not isinstance(talkback, dict):
        errors.append("talkback must be a mapping.")
    else:
        max_chars = talkback.get("max_chars")
        if max_chars is None:
            legacy_llm = talkback.get("llm")
            if isinstance(legacy_llm, dict):
                max_chars = legacy_llm.get("max_chars")
        if not isinstance(max_chars, int) or max_chars <= 0:
            errors.append("talkback.max_chars must be a positive integer.")

        rate_limit = talkback.get("rate_limit")
        if not isinstance(rate_limit, dict):
            errors.append("talkback.rate_limit must be a mapping.")
        else:
            max_replies = rate_limit.get("max_replies_per_minute")
            if not isinstance(max_replies, int) or max_replies <= 0:
                errors.append(
                    "talkback.rate_limit.max_replies_per_minute must be a positive integer."
                )
            max_forwards = rate_limit.get("max_forwards_per_minute")
            if not isinstance(max_forwards, int) or max_forwards <= 0:
                errors.append(
                    "talkback.rate_limit.max_forwards_per_minute must be a positive integer."
                )

        context_cfg = talkback.get("context")
        if not isinstance(context_cfg, dict):
            errors.append("talkback.context must be a mapping.")
        else:
            last_n = context_cfg.get("last_n_messages")
            if not isinstance(last_n, int) or last_n <= 0:
                errors.append(
                    "talkback.context.last_n_messages must be a positive integer."
                )

        sanitize_cfg = talkback.get("sanitize")
        if not isinstance(sanitize_cfg, dict):
            errors.append("talkback.sanitize must be a mapping.")

        expressions_cfg = talkback.get("expressions")
        if not isinstance(expressions_cfg, dict):
            errors.append("talkback.expressions must be a mapping.")
        else:
            allowed = expressions_cfg.get("allowed")
            if not isinstance(allowed, list) or not allowed:
                errors.append("talkback.expressions.allowed must be a non-empty list.")
            else:
                for idx, item in enumerate(allowed):
                    if not isinstance(item, str) or not item.strip():
                        errors.append(
                            f"talkback.expressions.allowed[{idx}] must be a non-empty string."
                        )

        acknowledgements_cfg = talkback.get("acknowledgements")
        if not isinstance(acknowledgements_cfg, dict):
            errors.append("talkback.acknowledgements must be a mapping.")
        else:
            template = acknowledgements_cfg.get("template")
            if not isinstance(template, str) or not template.strip():
                errors.append(
                    "talkback.acknowledgements.template must be a non-empty string."
                )

        twitch_reply = talkback.get("twitch_reply")
        if not isinstance(twitch_reply, dict):
            errors.append("talkback.twitch_reply must be a mapping.")

        language = talkback.get("language")
        if language != "en":
            errors.append("talkback.language must be 'en'.")

        debug_flag = talkback.get("debug")
        if debug_flag is not None and not isinstance(debug_flag, bool):
            errors.append("talkback.debug must be a boolean.")

        notifications_cfg = talkback.get("notifications")
        if not isinstance(notifications_cfg, dict):
            errors.append("talkback.notifications must be a mapping.")
        else:
            threshold = notifications_cfg.get("low_viewer_threshold")
            if not isinstance(threshold, int) or threshold < 0:
                errors.append(
                    "talkback.notifications.low_viewer_threshold must be a non-negative integer."
                )

        tts_cfg = talkback.get("tts")
        if not isinstance(tts_cfg, dict):
            errors.append("talkback.tts must be a mapping.")
        else:
            enabled = tts_cfg.get("enabled")
            if not isinstance(enabled, bool):
                errors.append("talkback.tts.enabled must be a boolean.")
            engine = tts_cfg.get("engine")
            if not isinstance(engine, str) or not engine.strip():
                errors.append("talkback.tts.engine must be a non-empty string.")
            elif engine != "edge_tts":
                errors.append("talkback.tts.engine must be 'edge_tts'.")
            voice = tts_cfg.get("voice")
            if not isinstance(voice, str) or not voice.strip():
                errors.append("talkback.tts.voice must be a non-empty string.")

    if errors:
        raise ConfigValidationError("\n".join(sorted(set(errors))))

    return data
