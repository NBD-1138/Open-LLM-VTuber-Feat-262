from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Optional

from .common import (
    AppendOnlyTextFileTailer,
    BaseTelemetryAdapter,
    clip_summary,
    discover_configured_directory,
    isoformat,
    normalize_token_text,
    utcnow,
)
from .models import (
    StructuredTelemetrySnapshot,
    TelemetryCapabilityAdvice,
    TelemetryEvent,
    TelemetryHighlight,
    TelemetrySettings,
)

DEFAULT_MINECRAFT_LOG_DIR = Path.home() / "AppData" / "Roaming" / ".minecraft" / "logs"
MINECRAFT_LOG_FILE_NAME = "latest.log"
POLL_INTERVAL_SECONDS = 1.0
STALE_AFTER_SECONDS = 45

MESSAGE_PREFIX_PATTERN = re.compile(
    r"^\[[^\]]+\]\s+\[[^\]]+\]:\s*(?P<message>.*)$"
)
SETTING_USER_PATTERN = re.compile(r"^Setting user:\s*(?P<player>.+)$", re.IGNORECASE)
CONNECTING_PATTERN = re.compile(
    r"^Connecting to (?P<server>[^,]+)(?:,\s*(?P<port>\d+))?$", re.IGNORECASE
)
DIMENSION_PATTERN = re.compile(
    r"Preparing start region for dimension (?P<dimension>[\w:.\-/]+)",
    re.IGNORECASE,
)
ADVANCEMENT_PATTERN = re.compile(
    r"(?P<player>.+?) has (?P<kind>made the advancement|reached the goal|completed the challenge) \[(?P<advancement>.+?)\]",
    re.IGNORECASE,
)
DISCONNECT_PATTERN = re.compile(
    r"Disconnect(?:ing|ed)(?: from server)?:?\s*(?P<reason>.+)?",
    re.IGNORECASE,
)


def _format_dimension(value: str) -> str:
    token = value.split(":")[-1].replace("_", " ").strip()
    if not token:
        return "Unknown"
    return token.title()


def _normalize_window_session(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = " ".join(value.split()).strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered in {"minecraft", "minecraft launcher"}:
        return None
    return cleaned


@dataclass(slots=True)
class MinecraftRuntimeContext:
    profile_id: Optional[str] = None
    profile_display_name: Optional[str] = None
    detected_process_name: Optional[str] = None
    detected_window_title: Optional[str] = None


class MinecraftStateAccumulator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._session = {
            "playerName": None,
            "sessionType": None,
            "serverAddress": None,
            "worldName": None,
            "dimension": None,
            "windowTitle": None,
            "profileDisplayName": None,
        }
        self._world = {
            "lastAdvancement": None,
            "lastDisconnectReason": None,
            "lastMilestoneAt": None,
        }
        self._last_update_at: Optional[datetime] = None
        self._last_source_timestamp: Optional[datetime] = None
        self._last_event_at: dict[str, datetime] = {}

    def set_runtime_context(self, context: MinecraftRuntimeContext) -> None:
        if context.profile_display_name:
            self._session["profileDisplayName"] = context.profile_display_name
        self._session["windowTitle"] = _normalize_window_session(
            context.detected_window_title
        )
        title_world = self._session["windowTitle"]
        if title_world and not self._session["worldName"] and not self._session["serverAddress"]:
            self._session["worldName"] = title_world
        self._touch(None)

    def snapshot(self) -> StructuredTelemetrySnapshot:
        now = utcnow()
        stale = True
        if self._last_update_at is not None:
            stale = (now - self._last_update_at).total_seconds() > STALE_AFTER_SECONDS
        confidence = 0.0
        if self._last_update_at is not None:
            confidence = 0.78 if not stale else 0.45

        highlights: list[TelemetryHighlight] = []
        if self._session.get("playerName"):
            highlights.append(
                TelemetryHighlight(
                    key="player",
                    label="Player",
                    value=str(self._session["playerName"]),
                )
            )
        if self._session.get("sessionType"):
            highlights.append(
                TelemetryHighlight(
                    key="mode",
                    label="Mode",
                    value=str(self._session["sessionType"]).replace("_", " ").title(),
                )
            )
        session_focus = (
            self._session.get("serverAddress")
            or self._session.get("worldName")
            or self._session.get("windowTitle")
        )
        if session_focus:
            highlights.append(
                TelemetryHighlight(
                    key="session",
                    label="Session",
                    value=str(session_focus),
                )
            )
        if self._session.get("dimension"):
            highlights.append(
                TelemetryHighlight(
                    key="dimension",
                    label="Dimension",
                    value=str(self._session["dimension"]),
                )
            )
        if self._world.get("lastAdvancement"):
            highlights.append(
                TelemetryHighlight(
                    key="advancement",
                    label="Last milestone",
                    value=str(self._world["lastAdvancement"]),
                    level="good",
                )
            )

        return StructuredTelemetrySnapshot(
            source_timestamp=isoformat(self._last_source_timestamp),
            last_update_timestamp=isoformat(self._last_update_at),
            confidence=round(confidence, 2),
            stale=stale,
            session=dict(self._session),
            ship={},
            navigation={},
            combat={},
            missions={},
            world=dict(self._world),
            automation={},
            highlights=highlights,
        )

    def apply_log_line(
        self,
        raw_line: str,
        settings: TelemetrySettings,
    ) -> list[TelemetryEvent]:
        del settings
        events: list[TelemetryEvent] = []
        text = raw_line.strip()
        if not text:
            return events

        match = MESSAGE_PREFIX_PATTERN.match(text)
        message = (match.group("message") if match else text).strip()
        if not message:
            return events

        user_match = SETTING_USER_PATTERN.match(message)
        if user_match:
            player_name = clip_summary(user_match.group("player"), limit=64)
            if player_name != self._session.get("playerName"):
                self._session["playerName"] = player_name
                events.append(
                    TelemetryEvent(
                        event_type="minecraft.session.started",
                        summary=f"Minecraft ready for {player_name}.",
                        novelty_key=f"minecraft.session:{player_name.lower()}",
                        importance=4,
                        urgency=2,
                        confidence=0.8,
                        conversational=True,
                    )
                )

        connection_match = CONNECTING_PATTERN.match(message)
        if connection_match:
            server = clip_summary(connection_match.group("server"), limit=96)
            port = connection_match.group("port")
            address = f"{server}:{port}" if port else server
            if address != self._session.get("serverAddress"):
                self._session["sessionType"] = "multiplayer"
                self._session["serverAddress"] = address
                self._session["worldName"] = None
                events.append(
                    TelemetryEvent(
                        event_type="minecraft.server.connecting",
                        summary=f"Connecting to {address}.",
                        novelty_key=f"minecraft.server:{address.lower()}",
                        importance=4,
                        urgency=3,
                        confidence=0.82,
                        conversational=True,
                    )
                )

        if "integrated server" in message.lower():
            if self._session.get("sessionType") != "singleplayer":
                self._session["sessionType"] = "singleplayer"
                events.append(
                    TelemetryEvent(
                        event_type="minecraft.world.opened",
                        summary="Opened a singleplayer Minecraft world.",
                        novelty_key="minecraft.world.singleplayer",
                        importance=4,
                        urgency=2,
                        confidence=0.78,
                        conversational=True,
                    )
                )

        dimension_match = DIMENSION_PATTERN.search(message)
        if dimension_match:
            dimension = _format_dimension(dimension_match.group("dimension"))
            if dimension != self._session.get("dimension"):
                self._session["dimension"] = dimension
                events.append(
                    TelemetryEvent(
                        event_type="minecraft.dimension.changed",
                        summary=f"Entered {dimension}.",
                        novelty_key=f"minecraft.dimension:{dimension.lower()}",
                        importance=4,
                        urgency=2,
                        confidence=0.76,
                        cooldown_ms=20_000,
                        conversational=True,
                    )
                )

        advancement_match = ADVANCEMENT_PATTERN.search(message)
        if advancement_match:
            advancement = clip_summary(advancement_match.group("advancement"), limit=96)
            self._world["lastAdvancement"] = advancement
            self._world["lastMilestoneAt"] = isoformat(utcnow())
            events.append(
                TelemetryEvent(
                    event_type="minecraft.advancement",
                    summary=f"Unlocked advancement: {advancement}.",
                    novelty_key=f"minecraft.advancement:{advancement.lower()}",
                    importance=5,
                    urgency=2,
                    confidence=0.88,
                    cooldown_ms=25_000,
                    conversational=True,
                )
            )

        disconnect_match = DISCONNECT_PATTERN.search(message)
        if disconnect_match and self._event_ready("disconnect", 15_000):
            reason = clip_summary(disconnect_match.group("reason") or "Disconnected")
            self._world["lastDisconnectReason"] = reason
            events.append(
                TelemetryEvent(
                    event_type="minecraft.session.ended",
                    summary="Left the current Minecraft session.",
                    novelty_key="minecraft.session.ended",
                    importance=3,
                    urgency=2,
                    confidence=0.74,
                    cooldown_ms=15_000,
                    conversational=True,
                )
            )

        self._touch(None)
        return events

    def _touch(self, source_timestamp: Optional[datetime]) -> None:
        self._last_update_at = utcnow()
        if source_timestamp is not None:
            self._last_source_timestamp = source_timestamp

    def _event_ready(self, key: str, cooldown_ms: int) -> bool:
        now = utcnow()
        previous = self._last_event_at.get(key)
        if previous and (now - previous).total_seconds() * 1000 < cooldown_ms:
            return False
        self._last_event_at[key] = now
        return True


class MinecraftTelemetryAdapter(BaseTelemetryAdapter):
    adapter_id = "minecraft-log"
    game_id = "minecraft"
    game_display_name = "Minecraft"
    source_kind = "log"
    supported_process_names = ("minecraft.exe", "minecraft launcher", "javaw.exe", "java.exe")

    def __init__(
        self,
        on_event: Callable[[TelemetryEvent], Awaitable[None]],
        on_state_change: Callable[[], Awaitable[None]],
    ) -> None:
        super().__init__(on_event, on_state_change)
        self._state = MinecraftStateAccumulator()
        self._tailer = AppendOnlyTextFileTailer()
        self._runtime_context = MinecraftRuntimeContext()

    @classmethod
    def match_score(
        cls,
        *,
        profile_id: Optional[str],
        profile_display_name: Optional[str],
        profile_process_names: Iterable[str],
        detected_process_name: Optional[str],
        detected_window_title: Optional[str],
    ) -> int:
        del profile_process_names
        normalized_profile = " ".join(
            filter(
                None,
                [
                    normalize_token_text(profile_id or ""),
                    normalize_token_text(profile_display_name or ""),
                ],
            )
        )
        normalized_title = normalize_token_text(detected_window_title or "")
        normalized_process = normalize_token_text(detected_process_name or "")

        if normalized_process == "minecraft.exe":
            return 100
        if "minecraft" in normalized_title:
            return 95
        if "minecraft" in normalized_profile and normalized_process in {"javaw.exe", "java.exe"}:
            return 90
        if "minecraft" in normalized_profile:
            return 70
        return 0

    async def update_runtime_context(
        self,
        *,
        profile_id: Optional[str],
        profile_display_name: Optional[str],
        profile_process_names: Iterable[str],
        detected_process_name: Optional[str],
        detected_window_title: Optional[str],
    ) -> None:
        del profile_process_names
        self._runtime_context = MinecraftRuntimeContext(
            profile_id=profile_id,
            profile_display_name=profile_display_name,
            detected_process_name=detected_process_name,
            detected_window_title=detected_window_title,
        )
        self._state.set_runtime_context(self._runtime_context)
        await self._emit_state_if_changed()

    async def _reset_runtime_state(self) -> None:
        self._tailer.reset()
        self._state.reset()

    def _current_structured_snapshot(self) -> StructuredTelemetrySnapshot:
        return self._state.snapshot()

    async def _run(self) -> None:
        while not self._closed:
            directory, status, label = discover_configured_directory(
                self._settings, DEFAULT_MINECRAFT_LOG_DIR
            )
            self._snapshot.status = "running"
            self._snapshot.error = None
            self._snapshot.source_directory_status = status
            self._snapshot.source_directory_label = label

            if directory is None or not directory.exists():
                self._snapshot.current_source = None
                await self._emit_state_if_changed()
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                continue

            log_path = directory / MINECRAFT_LOG_FILE_NAME
            if not log_path.exists():
                self._snapshot.current_source = None
                self._snapshot.source_directory_status = "missing"
                await self._emit_state_if_changed()
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                continue

            result = await self._tailer.poll(log_path)
            self._snapshot.current_source = result.current_file
            for line in result.lines:
                for event in self._state.apply_log_line(line, self._settings):
                    await self._emit_event(event)
            await self._emit_state_if_changed()
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    def build_capability_advice(
        self, capabilities: Iterable[dict[str, Any]]
    ) -> dict[str, TelemetryCapabilityAdvice]:
        del capabilities
        return {}
