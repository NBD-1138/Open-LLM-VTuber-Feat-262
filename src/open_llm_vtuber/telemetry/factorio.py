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

DEFAULT_FACTORIO_DATA_DIR = Path.home() / "AppData" / "Roaming" / "Factorio"
FACTORIO_LOG_FILE_NAME = "factorio-current.log"
POLL_INTERVAL_SECONDS = 1.0
STALE_AFTER_SECONDS = 60

LOAD_MAP_PATTERN = re.compile(r"Loading map (?P<path>.+?)(?:\s*\(|$)", re.IGNORECASE)
SCENARIO_PATTERN = re.compile(
    r"Loading scenario (?P<scenario>.+?)(?:\s*\(|$)", re.IGNORECASE
)
HOST_PATTERN = re.compile(r"Hosting game at (?P<endpoint>\S+)", re.IGNORECASE)
JOIN_PATTERN = re.compile(
    r"(?:Joining|Connecting to) game at (?P<endpoint>\S+)", re.IGNORECASE
)
AUTOSAVE_PATTERN = re.compile(r"Saving to (?P<save>.+?)(?:\s*\(|$)", re.IGNORECASE)
RESEARCH_PATTERN = re.compile(
    r"(?:Technology|Research)(?:\s+(?:finished|completed|researched))?[: ]+(?P<research>[A-Za-z0-9 _\-/]+)",
    re.IGNORECASE,
)
ROCKET_PATTERN = re.compile(r"rocket.*launch", re.IGNORECASE)


def _basename_without_extension(value: str) -> str:
    raw = value.strip().strip('"')
    name = Path(raw).name or raw
    return clip_summary(Path(name).stem or name, limit=96)


def _extract_title_focus(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = " ".join(value.split()).strip()
    if not cleaned:
        return None
    segments = [segment.strip() for segment in re.split(r"\s+-\s+", cleaned) if segment.strip()]
    for segment in reversed(segments):
        if "factorio" not in segment.lower():
            return segment
    return None if "factorio" in cleaned.lower() else cleaned


@dataclass(slots=True)
class FactorioRuntimeContext:
    profile_id: Optional[str] = None
    profile_display_name: Optional[str] = None
    detected_process_name: Optional[str] = None
    detected_window_title: Optional[str] = None


class FactorioStateAccumulator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._session = {
            "saveName": None,
            "scenario": None,
            "sessionType": None,
            "serverEndpoint": None,
            "windowTitle": None,
            "profileDisplayName": None,
        }
        self._world = {
            "lastResearch": None,
            "lastAutosave": None,
            "rocketLaunchCount": 0,
        }
        self._last_update_at: Optional[datetime] = None
        self._last_source_timestamp: Optional[datetime] = None

    def set_runtime_context(self, context: FactorioRuntimeContext) -> None:
        if context.profile_display_name:
            self._session["profileDisplayName"] = context.profile_display_name
        self._session["windowTitle"] = context.detected_window_title
        title_focus = _extract_title_focus(context.detected_window_title)
        if title_focus and not self._session.get("saveName"):
            self._session["saveName"] = title_focus
        self._touch(None)

    def snapshot(self) -> StructuredTelemetrySnapshot:
        now = utcnow()
        stale = True
        if self._last_update_at is not None:
            stale = (now - self._last_update_at).total_seconds() > STALE_AFTER_SECONDS
        confidence = 0.0
        if self._last_update_at is not None:
            confidence = 0.72 if not stale else 0.42

        highlights: list[TelemetryHighlight] = []
        if self._session.get("saveName"):
            highlights.append(
                TelemetryHighlight(
                    key="save",
                    label="Save",
                    value=str(self._session["saveName"]),
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
        if self._session.get("serverEndpoint"):
            highlights.append(
                TelemetryHighlight(
                    key="server",
                    label="Server",
                    value=str(self._session["serverEndpoint"]),
                )
            )
        if self._session.get("scenario"):
            highlights.append(
                TelemetryHighlight(
                    key="scenario",
                    label="Scenario",
                    value=str(self._session["scenario"]),
                )
            )
        if self._world.get("lastResearch"):
            highlights.append(
                TelemetryHighlight(
                    key="research",
                    label="Last research",
                    value=str(self._world["lastResearch"]),
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
        text = raw_line.strip()
        if not text:
            return []

        events: list[TelemetryEvent] = []

        load_map = LOAD_MAP_PATTERN.search(text)
        if load_map:
            save_name = _basename_without_extension(load_map.group("path"))
            if save_name != self._session.get("saveName"):
                self._session["saveName"] = save_name
                self._session["sessionType"] = self._session.get("sessionType") or "singleplayer"
                events.append(
                    TelemetryEvent(
                        event_type="factorio.save.loaded",
                        summary=f"Loaded Factorio save {save_name}.",
                        novelty_key=f"factorio.save:{save_name.lower()}",
                        importance=4,
                        urgency=2,
                        confidence=0.77,
                        conversational=True,
                    )
                )

        scenario_match = SCENARIO_PATTERN.search(text)
        if scenario_match:
            self._session["scenario"] = clip_summary(
                scenario_match.group("scenario"), limit=96
            )

        host_match = HOST_PATTERN.search(text)
        if host_match:
            endpoint = clip_summary(host_match.group("endpoint"), limit=96)
            self._session["sessionType"] = "hosted_multiplayer"
            self._session["serverEndpoint"] = endpoint
            events.append(
                TelemetryEvent(
                    event_type="factorio.multiplayer.hosted",
                    summary=f"Hosting a Factorio game at {endpoint}.",
                    novelty_key=f"factorio.host:{endpoint.lower()}",
                    importance=4,
                    urgency=2,
                    confidence=0.75,
                    conversational=True,
                )
            )

        join_match = JOIN_PATTERN.search(text)
        if join_match:
            endpoint = clip_summary(join_match.group("endpoint"), limit=96)
            self._session["sessionType"] = "multiplayer"
            self._session["serverEndpoint"] = endpoint
            events.append(
                TelemetryEvent(
                    event_type="factorio.multiplayer.joined",
                    summary=f"Joined a Factorio game at {endpoint}.",
                    novelty_key=f"factorio.join:{endpoint.lower()}",
                    importance=4,
                    urgency=3,
                    confidence=0.78,
                    conversational=True,
                )
            )

        autosave_match = AUTOSAVE_PATTERN.search(text)
        if autosave_match:
            self._world["lastAutosave"] = clip_summary(
                _basename_without_extension(autosave_match.group("save")),
                limit=96,
            )

        research_match = RESEARCH_PATTERN.search(text)
        if research_match:
            research = clip_summary(research_match.group("research"), limit=96)
            lowered = research.lower()
            if "factorio.cpp" not in lowered and "research queue" not in lowered:
                self._world["lastResearch"] = research
                events.append(
                    TelemetryEvent(
                        event_type="factorio.research.completed",
                        summary=f"Research completed: {research}.",
                        novelty_key=f"factorio.research:{research.lower()}",
                        importance=5,
                        urgency=2,
                        confidence=0.76,
                        cooldown_ms=25_000,
                        conversational=True,
                    )
                )

        if ROCKET_PATTERN.search(text):
            self._world["rocketLaunchCount"] = int(
                self._world.get("rocketLaunchCount") or 0
            ) + 1
            events.append(
                TelemetryEvent(
                    event_type="factorio.rocket.launched",
                    summary="A rocket launch was detected in Factorio.",
                    novelty_key=f"factorio.rocket:{self._world['rocketLaunchCount']}",
                    importance=6,
                    urgency=3,
                    confidence=0.7,
                    cooldown_ms=60_000,
                    conversational=True,
                )
            )

        self._touch(None)
        return events

    def _touch(self, source_timestamp: Optional[datetime]) -> None:
        self._last_update_at = utcnow()
        if source_timestamp is not None:
            self._last_source_timestamp = source_timestamp


class FactorioTelemetryAdapter(BaseTelemetryAdapter):
    adapter_id = "factorio-log"
    game_id = "factorio"
    game_display_name = "Factorio"
    source_kind = "log"
    supported_process_names = ("factorio.exe", "factorio")

    def __init__(
        self,
        on_event: Callable[[TelemetryEvent], Awaitable[None]],
        on_state_change: Callable[[], Awaitable[None]],
    ) -> None:
        super().__init__(on_event, on_state_change)
        self._state = FactorioStateAccumulator()
        self._tailer = AppendOnlyTextFileTailer()
        self._runtime_context = FactorioRuntimeContext()

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

        if normalized_process == "factorio.exe":
            return 100
        if "factorio" in normalized_title:
            return 96
        if "factorio" in normalized_profile:
            return 85
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
        self._runtime_context = FactorioRuntimeContext(
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
                self._settings, DEFAULT_FACTORIO_DATA_DIR
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

            log_path = directory / FACTORIO_LOG_FILE_NAME
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
